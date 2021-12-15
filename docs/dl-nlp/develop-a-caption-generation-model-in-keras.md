# 如何利用小实验在 Keras 中开发标题生成模型

> 原文： [https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/](https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/)

标题生成是一个具有挑战性的人工智能问题，其中必须为照片生成文本描述。

它既需要计算机视觉的方法来理解图像的内容，也需要来自自然语言处理领域的语言模型，以便将图像的理解转化为正确的单词。最近，深度学习方法已经在该问题的示例上获得了现有技术的结果。

在您自己的数据上开发标题生成模型可能很困难，主要是因为数据集和模型太大而需要数天才能进行训练。另一种方法是使用较小数据集的小样本来探索模型配置。

在本教程中，您将了解如何使用标准照片字幕数据集的小样本来探索不同的深度模型设计。

完成本教程后，您将了解：

*   如何为照片字幕建模准备数据。
*   如何设计基线和测试工具来评估模型的技能和控制其随机性。
*   如何评估模型技能，特征提取模型和单词嵌入等属性，以提升模型技能。

让我们开始吧。

*   **2019 年 2 月 2 日**：提供了 Flickr8k_Dataset 数据集的直接链接，因为官方网站被删除了。

![How to Use Small Experiments to Develop a Caption Generation Model in Keras](img/7d34b218f89d903c2711e5c2dc7e3027.jpg)

如何使用小实验开发 Keras 中的标题生成模型
照片由 [Per](https://www.flickr.com/photos/perry-pics/5968641588/) ，保留一些权利。

## 教程概述

本教程分为 6 个部分;他们是：

1.  数据准备
2.  基线标题生成模型
3.  网络大小参数
4.  配置特征提取模型
5.  词嵌入模型
6.  结果分析

### Python 环境

本教程假设您安装了 Python SciPy 环境，理想情况下使用 Python 3。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅本教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

我建议在带 GPU 的系统上运行代码。

您可以在 Amazon Web Services 上以低成本方式访问 GPU。在本教程中学习如何：

*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

让我们潜入。

## 数据准备

首先，我们需要准备数据集来训练模型。

我们将使用 Flickr8K 数据集，该数据集包含超过 8,000 张照片及其描述。

您可以从此处下载数据集：

*   [将图像描述框架化为排名任务：数据，模型和评估指标](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)。

**UPDATE（April / 2019）**：官方网站似乎已被删除（虽然表格仍然有效）。以下是我的[数据集 GitHub 存储库](https://github.com/jbrownlee/Datasets)的一些直接下载链接：

*   [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
*   [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

将照片和说明分别解压缩到`Flicker8k_Dataset`和`Flickr8k_text`目录中的当前工作目录中。

数据准备分为两部分，它们是：

1.  准备文本
2.  准备照片

### 准备文本

数据集包含每张照片的多个描述，描述文本需要一些最小的清洁。

首先，我们将加载包含所有描述的文件。

```py
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
```

每张照片都有唯一的标识符。这用于照片文件名和描述的文本文件中。接下来，我们将逐步浏览照片说明列表并保存每张照片的第一个描述。下面定义了一个名为`load_descriptions()`的函数，给定加载的文档文本，它将返回照片标识符的字典到描述。

```py
# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# store the first description for each image
		if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
```

接下来，我们需要清理描述文本。

描述已经被分词并且易于使用。我们将通过以下方式清理文本，以减少我们需要使用的单词词汇量：

*   将所有单词转换为小写。
*   删除所有标点符号。
*   删除所有长度不超过一个字符的单词（例如“a”）。

下面定义`clean_descriptions()`函数，给定描述图像标识符的字典，逐步执行每个描述并清理文本。

```py
import string

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc in descriptions.items():
		# tokenize
		desc = desc.split()
		# convert to lower case
		desc = [word.lower() for word in desc]
		# remove punctuation from each token
		desc = [w.translate(table) for w in desc]
		# remove hanging 's' and 'a'
		desc = [word for word in desc if len(word)>1]
		# store as string
		descriptions[key] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
all_tokens = ' '.join(descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
```

最后，我们将图像标识符和描述字典保存到名为`descriptionss.txt`的新文件中，每行有一个图像标识符和描述。

下面定义了`save_doc()`函数，该函数给出了包含标识符到描述和文件名的映射的字典，将映射保存到文件。

```py
# save descriptions to file, one per line
def save_doc(descriptions, filename):
	lines = list()
	for key, desc in descriptions.items():
		lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# save descriptions
save_doc(descriptions, 'descriptions.txt')
```

综合这些，下面提供了完整的列表。

```py
import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# store the first description for each image
		if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc in descriptions.items():
		# tokenize
		desc = desc.split()
		# convert to lower case
		desc = [word.lower() for word in desc]
		# remove punctuation from each token
		desc = [w.translate(table) for w in desc]
		# remove hanging 's' and 'a'
		desc = [word for word in desc if len(word)>1]
		# store as string
		descriptions[key] =  ' '.join(desc)

# save descriptions to file, one per line
def save_doc(descriptions, filename):
	lines = list()
	for key, desc in descriptions.items():
		lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
all_tokens = ' '.join(descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
# save descriptions
save_doc(descriptions, 'descriptions.txt')
```

首先运行示例打印已加载的照片描述数（8,092）和干净词汇表的大小（4,484 个单词）。

```py
Loaded: 8092
Vocabulary Size: 4484
```

然后将干净的描述写入'`descriptionss.txt`'。看一下文件，我们可以看到描述已准备好进行建模。

看一下文件，我们可以看到描述已准备好进行建模。

```py
3621647714_fc67ab2617 man is standing on snow with trees and mountains all around him
365128300_6966058139 group of people are rafting on river rapids
2751694538_fffa3d307d man and boy sit in the driver seat
537628742_146f2c24f8 little girl running in field
2320125735_27fe729948 black and brown dog with blue collar goes on alert by soccer ball in the grass
...
```

### 准备照片

我们将使用预先训练的模型来解释照片的内容。

有很多型号可供选择。在这种情况下，我们将使用 2014 年赢得 ImageNet 竞赛的牛津视觉几何组或 VGG 模型。在此处了解有关模型的更多信息：

*   [用于大规模视觉识别的超深卷积网络](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

Keras 直接提供这种预先训练的模型。请注意，第一次使用此模型时，Keras 将从 Internet 下载模型权重，大约为 500 兆字节。这可能需要几分钟，具体取决于您的互联网连接。

我们可以将此模型用作更广泛的图像标题模型的一部分。问题是，它是一个大型模型，每次我们想要测试一个新的语言模型配置（下游）是多余的时，通过网络运行每张照片。

相反，我们可以使用预先训练的模型预先计算“照片功能”并将其保存到文件中。然后，我们可以稍后加载这些功能，并将它们作为数据集中给定照片的解释提供给我们的模型。通过完整的 VGG 模型运行照片也没有什么不同，只是我们提前完成了一次。

这是一种优化，可以更快地训练我们的模型并消耗更少的内存。

我们可以使用 VGG 类在 Keras 中加载 VGG 模型。我们将加载没有顶部的模型;这意味着没有网络末端的层用于解释从输入中提取的特征并将它们转换为类预测。我们对照片的图像网络分类不感兴趣，我们将训练自己对图像特征的解释。

Keras 还提供了用于将加载的照片整形为模型的优选尺寸的工具（例如，3 通道 224×224 像素图像）。

下面是一个名为`extract_features()`的函数，给定目录名称将加载每张照片，为 VGG 准备并从 VGG 模型中收集预测的特征。图像特征是具有形状（7,7,512）的三维数组。

该函数返回图像标识符的字典到图像特征。

```py
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	in_layer = Input(shape=(224, 224, 3))
	model = VGG16(include_top=False, input_tensor=in_layer)
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features
```

我们可以调用此函数来准备用于测试模型的照片数据，然后将生成的字典保存到名为“`features.pkl`”的文件中。

下面列出了完整的示例。

```py
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	in_layer = Input(shape=(224, 224, 3))
	model = VGG16(include_top=False, input_tensor=in_layer)
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
```

运行此数据准备步骤可能需要一段时间，具体取决于您的硬件，可能需要一个小时的 CPU 与现代工作站。

在运行结束时，您将提取的特征存储在'`features.pkl`'中供以后使用。

## 基线标题生成模型

在本节中，我们将定义一个基线模型，用于生成照片的标题以及如何对其进行评估，以便将其与此基线的变体进行比较。

本节分为 5 部分：

1.  加载数据。
2.  适合模型。
3.  评估模型。
4.  完整的例子
5.  “A”与“A”测试
6.  生成照片标题

### 1.加载数据

我们不会在所有字幕数据上，甚至在大量数据样本上使用该模型。

在本教程中，我们感兴趣的是快速测试一组标题模型的不同配置，以查看对此数据有何用处。这意味着我们需要快速评估一个模型配置。为此，我们将在 100 张照片和标题上训练模型，然后在训练数据集和 100 张照片和标题的新测试集上进行评估。

首先，我们需要加载预定义的照片子集。提供的数据集具有用于训练，测试和开发的单独集合，这些集合实际上只是不同的照片标识符组。我们将加载开发集并使用前 100 个列表标识符和第二个 100 标识符（例如从 100 到 200）作为测试集。

下面的函数`load_set()`将加载一组预定义的标识符，我们将使用'`Flickr_8k.devImages.txt`'文件名作为参数调用它。

```py
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
```

接下来，我们需要将集合拆分为训练集和测试集。

我们将首先通过对标识符进行排序来对它们进行排序，以确保我们始终在机器和运行中对它们进行一致的分割，然后将前 100 个用于训练，接下来的 100 个用于测试。

下面的`train_test_split()`函数将在加载的标识符集作为输入的情况下创建此拆分。

```py
# split a dataset into train/test elements
def train_test_split(dataset):
	# order keys so the split is consistent
	ordered = sorted(dataset)
	# return split dataset as two new sets
	return set(ordered[:100]), set(ordered[100:200])
```

现在，我们可以使用预定义的一组训练或测试标识符加载照片描述。

下面是函数 _load_clean_descriptions（）_，它为来自'`descriptionss.txt`'的已清除文本描述加载给定的一组标识符，并将标识符字典返回给文本。

我们将开发的模型将生成给定照片的标题，并且标题将一次生成一个单词。将提供先前生成的单词的序列作为输入。因此，我们需要一个“_ 第一个字 _”来启动生成过程和'_ 最后一个字 _'来表示标题的结束。为此，我们将使用字符串'`startseq`'和'`endseq`'。

```py
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# store
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions
```

接下来，我们可以加载给定数据集的照片功能。

下面定义了一个名为`load_photo_features()`的函数，它加载了整套照片描述，然后返回给定照片标识符集的感兴趣子集。这不是非常有效，因为所有照片功能的加载字典大约是 700 兆字节。然而，这将使我们快速起步。

请注意，如果您有更好的方法，请在下面的评论中分享。

```py
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
```

我们可以暂停一下，测试迄今为止开发的所有内容

完整的代码示例如下所示。

```py
from pickle import load

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
	# order keys so the split is consistent
	ordered = sorted(dataset)
	# return split dataset as two new sets
	return set(ordered[:100]), set(ordered[100:200])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# store
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load dev set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
dataset = load_set(filename)
print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
print('Train=%d, Test=%d' % (len(train), len(test)))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
# photo features
train_features = load_photo_features('features.pkl', train)
test_features = load_photo_features('features.pkl', test)
print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
```

运行此示例首先在开发数据集中加载 1,000 个照片标识符。选择训练和测试集并用于过滤一组干净的照片描述和准备好的图像特征。

我们快到了。

```py
Dataset: 1,000
Train=100, Test=100
Descriptions: train=100, test=100
Photos: train=100, test=100
```

描述文本需要先编码为数字，然后才能像输入中那样呈现给模型，或者与模型的预测进行比较。

编码数据的第一步是创建从单词到唯一整数值​​的一致映射。 Keras 提供了 Tokenizer 类，可以从加载的描述数据中学习这种映射。

下面定义 _create_tokenizer（）_，它将在给定加载的照片描述文本的情况下适合 Tokenizer。

```py
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# prepare tokenizer
tokenizer = create_tokenizer(descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
```

我们现在可以对文本进行编码。

每个描述将分为单词。该模型将提供一个单词和照片，并生成下一个单词。然后，将描述的前两个单词作为输入提供给模型，以生成下一个单词。这就是模型的训练方式。

例如，输入序列“_ 在场 _ 中运行的小女孩”将被分成 6 个输入 - 输出对来训练模型：

```py
X1,		X2 (text sequence), 						y (word)
photo	startseq, 									little
photo	startseq, little,							girl
photo	startseq, little, girl, 					running
photo	startseq, little, girl, running, 			in
photo	startseq, little, girl, running, in, 		field
photo	startseq, little, girl, running, in, field, endseq
```

稍后，当模型用于生成描述时，生成的单词将被连接并递归地提供作为输入以生成图像的标题。

下面给出标记器`create_sequences()`的函数，单个干净的描述，照片的特征和最大描述长度将为训练模型准备一组输入 - 输出对。调用此函数将返回`X1`和`X2`，用于图像数据和输入序列数据的数组以及输出字的`y`值。

输入序列是整数编码的，输出字是单热编码的，以表示在整个可能单词的词汇表中预期单词的概率分布。

```py
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, desc, image, max_length):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	# integer encode the description
	seq = tokenizer.texts_to_sequences([desc])[0]
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# select
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	# Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]
```

### 2.适合模型

我们几乎准备好适应这个模型。

已经讨论了模型的部分内容，但让我们重新进行迭代。

该模型基于文章“ [Show and Tell：A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) ”，2015 年。

该模型包括三个部分：

*   **照片功能提取器**。这是在 ImageNet 数据集上预训练的 16 层 VGG 模型。我们使用 VGG 模型（没有顶部）预处理照片，并将使用此模型预测的提取特征作为输入。
*   **序列处理器**。这是用于处理文本输入的单词嵌入层，后跟 LSTM 层。 LSTM 输出由 Dense 层一次解释一个输出。
*   **口译员（缺少更好的名字）**。特征提取器和序列处理器都输出固定长度的向量，该向量是最大序列的长度。它们连接在一起并由 LSTM 和 Dense 层处理，然后进行最终预测。

在基础模型中使用保守数量的神经元。具体来说，在特征提取器之后的 128 Dense 层，在序列处理器之后是 50 维单词嵌入，接着是 256 单元 LSTM 和 128 神经元密集，最后是 500 单元 LSTM，接着是网络末端的 500 神经元密集。

该模型预测了词汇表中的概率分布，因此使用 softmax 激活函数，并且在拟合网络时最小化分类交叉熵损失函数。

函数`define_model()`定义基线模型，给定词汇量的大小和照片描述的最大长度。 Keras 功能 API 用于定义模型，因为它提供了定义采用两个输入流并组合它们的模型所需的灵活性。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='plot.png')
	return model
```

要了解模型的结构，特别是层的形状，请参阅下面列出的摘要。

```py
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 7, 7, 512)     0
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 25)            0
____________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalMa (None, 512)           0           input_1[0][0]
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 25, 50)        18300       input_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           65664       global_max_pooling2d_1[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 25, 256)       314368      embedding_1[0][0]
____________________________________________________________________________________________________
repeat_vector_1 (RepeatVector)   (None, 25, 128)       0           dense_1[0][0]
____________________________________________________________________________________________________
time_distributed_1 (TimeDistribu (None, 25, 128)       32896       lstm_1[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 25, 256)       0           repeat_vector_1[0][0]
                                                                   time_distributed_1[0][0]
____________________________________________________________________________________________________
lstm_2 (LSTM)                    (None, 500)           1514000     concatenate_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 500)           250500      lstm_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 366)           183366      dense_3[0][0]
====================================================================================================
Total params: 2,379,094
Trainable params: 2,379,094
Non-trainable params: 0
____________________________________________________________________________________________________
```

我们还创建了一个图表来可视化网络结构，更好地帮助理解两个输入流。

![Plot of the Baseline Captioning Deep Learning Model](img/eb076553435ca4d2a366c4b5e7d90a61.jpg)

基线标题深度学习模型的情节

我们将使用数据生成器训练模型。鉴于字幕和提取的照片特征可能作为单个数据集适合存储器，因此严格来说不需要这样做。然而，当您在整个数据集上训练最终模型时，这是一种很好的做法。

调用时，生成器将产生结果。在 Keras 中，它将产生一批输入 - 输出样本，用于估计误差梯度并更新模型权重。

函数`data_generator()`定义数据生成器，给定加载的照片描述字典，照片特征，整数编码序列的分词器以及数据集中的最大序列长度。

生成器永远循环，并在被问及时保持产生批量的输入 - 输出对。我们还有一个`n_step`参数，它允许我们调整每批次要生成的输入输出对的图像数量。平均序列有 10 个字，即 10 个输入 - 输出对，良好的批量大小可能是 30 个样本，大约 2 到 3 个图像值。

```py
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length, n_step):
	# loop until we finish training
	while 1:
		# loop over photo identifiers in the dataset
		keys = list(descriptions.keys())
		for i in range(0, len(keys), n_step):
			Ximages, XSeq, y = list(), list(),list()
			for j in range(i, min(len(keys), i+n_step)):
				image_id = keys[j]
				# retrieve photo feature input
				image = features[image_id][0]
				# retrieve text input
				desc = descriptions[image_id]
				# generate input-output pairs
				in_img, in_seq, out_word = create_sequences(tokenizer, desc, image, max_length)
				for k in range(len(in_img)):
					Ximages.append(in_img[k])
					XSeq.append(in_seq[k])
					y.append(out_word[k])
			# yield this batch of samples to the model
			yield [[array(Ximages), array(XSeq)], array(y)]
```

通过调用`fit_generator()`并将其传递给数据生成器以及所需的所有参数，可以拟合模型。在拟合模型时，我们还可以指定每个时期运行的批次数和时期数。

```py
model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose)
```

对于这些实验，我们将每批使用 2 个图像，每个时期使用 50 个批次（或 100 个图像），以及 50 个训练时期。您可以在自己的实验中尝试不同的配置。

### 3.评估模型

现在我们知道如何准备数据并定义模型，我们必须定义一个测试工具来评估给定的模型。

我们将通过在数据集上训练模型来评估模型，生成训练数据集中所有照片的描述，使用成本函数评估这些预测，然后多次重复此评估过程。

结果将是模型的技能分数分布，我们可以通过计算平均值和标准差来总结。这是评估深度学习模型的首选方式。看这篇文章：

*   [如何评估深度学习模型的技巧](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

首先，我们需要能够使用训练有素的模型生成照片的描述。

这包括传入开始描述标记'`startseq`'，生成一个单词，然后以生成的单词作为输入递归调用模型，直到到达序列标记结尾'`endseq`'或达到最大描述长度。

以下名为`generate_desc()`的函数实现此行为，并在给定训练模型和给定准备照片作为输入的情况下生成文本描述。它调用函数`word_for_id()`以将整数预测映射回一个字。

```py
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
```

我们将为训练数据集和测试数据集中的所有照片生成预测。

以下名为`evaluate_model()`的函数将针对给定的照片描述和照片特征数据集评估训练模型。使用语料库 BLEU 分数收集和评估实际和预测的描述，该分数总结了生成的文本与预期文本的接近程度。

```py
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		actual.append([desc.split()])
		predicted.append(yhat.split())
	# calculate BLEU score
	bleu = corpus_bleu(actual, predicted)
	return bleu
```

BLEU 分数用于文本翻译，用于针对一个或多个参考翻译评估翻译文本。事实上，我们可以访问我们可以比较的每个图像的多个参考描述，但为了简单起见，我们将使用数据集中每张照片的第一个描述（例如清理版本）。

您可以在此处了解有关 BLEU 分数的更多信息：

*   维基百科 [BLEU（双语评估替补）](https://en.wikipedia.org/wiki/BLEU)

NLTK Python 库在 [_corpus_bleu（）_ 函数](http://www.nltk.org/api/nltk.translate.html)中实现 BLEU 分数计算。接近 1.0 的较高分数更好，接近零的分数更差。

最后，我们需要做的就是在循环中多次定义，拟合和评估模型，然后报告最终的平均分数。

理想情况下，我们会重复实验 30 次或更多次，但这对我们的小型测试工具来说需要很长时间。相反，将评估模型 3 次。它会更快，但平均分数会有更高的差异。

下面定义了模型评估循环。在运行结束时，训练和测试集的 BLEU 分数的分布被保存到文件中。

```py
# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
	# define the model
	model = define_model(vocab_size, max_length)
	# fit model
	model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose)
	# evaluate model on training data
	train_score = evaluate_model(model, train_descriptions, train_features, tokenizer, max_length)
	test_score = evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
	# store
	train_results.append(train_score)
	test_results.append(test_score)
	print('>%d: train=%f test=%f' % ((i+1), train_score, test_score))
# save results to file
df = DataFrame()
df['train'] = train_results
df['test'] = test_results
print(df.describe())
df.to_csv(model_name+'.csv', index=False)
```

我们按如下方式对运行进行参数化，允许我们命名每次运行并将结果保存到单独的文件中。

```py
# define experiment
model_name = 'baseline1'
verbose = 2
n_epochs = 50
n_photos_per_update = 2
n_batches_per_epoch = int(len(train) / n_photos_per_update)
n_repeats = 3
```

### 4.完成示例

下面列出了完整的示例。

```py
from os import listdir
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
	# order keys so the split is consistent
	ordered = sorted(dataset)
	# return split dataset as two new sets
	return set(ordered[:100]), set(ordered[100:200])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# store
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, desc, image, max_length):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	# integer encode the description
	seq = tokenizer.texts_to_sequences([desc])[0]
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# select
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	# Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='plot.png')
	return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length, n_step):
	# loop until we finish training
	while 1:
		# loop over photo identifiers in the dataset
		keys = list(descriptions.keys())
		for i in range(0, len(keys), n_step):
			Ximages, XSeq, y = list(), list(),list()
			for j in range(i, min(len(keys), i+n_step)):
				image_id = keys[j]
				# retrieve photo feature input
				image = features[image_id][0]
				# retrieve text input
				desc = descriptions[image_id]
				# generate input-output pairs
				in_img, in_seq, out_word = create_sequences(tokenizer, desc, image, max_length)
				for k in range(len(in_img)):
					Ximages.append(in_img[k])
					XSeq.append(in_seq[k])
					y.append(out_word[k])
			# yield this batch of samples to the model
			yield [[array(Ximages), array(XSeq)], array(y)]

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		actual.append([desc.split()])
		predicted.append(yhat.split())
	# calculate BLEU score
	bleu = corpus_bleu(actual, predicted)
	return bleu

# load dev set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
dataset = load_set(filename)
print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
# photo features
train_features = load_photo_features('features.pkl', train)
test_features = load_photo_features('features.pkl', test)
print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max(len(s.split()) for s in list(train_descriptions.values()))
print('Description Length: %d' % max_length)

# define experiment
model_name = 'baseline1'
verbose = 2
n_epochs = 50
n_photos_per_update = 2
n_batches_per_epoch = int(len(train) / n_photos_per_update)
n_repeats = 3

# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
	# define the model
	model = define_model(vocab_size, max_length)
	# fit model
	model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose)
	# evaluate model on training data
	train_score = evaluate_model(model, train_descriptions, train_features, tokenizer, max_length)
	test_score = evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
	# store
	train_results.append(train_score)
	test_results.append(test_score)
	print('>%d: train=%f test=%f' % ((i+1), train_score, test_score))
# save results to file
df = DataFrame()
df['train'] = train_results
df['test'] = test_results
print(df.describe())
df.to_csv(model_name+'.csv', index=False)
```

首先运行该示例打印已加载的训练数据的摘要统计信息。

```py
Dataset: 1,000
Descriptions: train=100, test=100
Photos: train=100, test=100
Vocabulary Size: 366
Description Length: 25
```

该示例在 GPU 硬件上需要大约 20 分钟，在 CPU 硬件上需要更长时间。

在运行结束时，训练集上报告的平均 BLEU 为 0.06，测试集上报告为 0.04。结果存储在`baseline1.csv`中。

```py
          train      test
count  3.000000  3.000000
mean   0.060617  0.040978
std    0.023498  0.025105
min    0.042882  0.012101
25%    0.047291  0.032658
50%    0.051701  0.053215
75%    0.069484  0.055416
max    0.087268  0.057617
```

这提供了用于与备用配置进行比较的基线模型。

### “A”与“A”测试

在我们开始测试模型的变化之前，了解测试装置是否稳定非常重要。

也就是说，5 次运行的模型的总结技巧是否足以控制模型的随机性。

我们可以通过在 A / B 测试区域中所谓的 A 对 A 测试再次运行实验来了解这一点。如果我们再次进行相同的实验，我们期望获得相同的结果;如果我们不这样做，可能需要额外的重复来控制方法的随机性和数据集。

以下是该算法的第二次运行的结果。

```py
          train      test
count  3.000000  3.000000
mean   0.036902  0.043003
std    0.020281  0.017295
min    0.018522  0.026055
25%    0.026023  0.034192
50%    0.033525  0.042329
75%    0.046093  0.051477
max    0.058660  0.060624
```

我们可以看到该运行获得了非常相似的均值和标准差 BLEU 分数。具体而言，在训练上的平均 BLEU 为 0.03 对 0.06，对于测试为 0.04 至 0.04。

线束有点吵，但足够稳定，可以进行比较。

模特有什么好处吗？

### 生成照片标题

我们希望该模型训练不足，甚至可能在配置下，但是它可以生成任何类型的可读文本吗？

重要的是，基线模型具有一定的能力，以便我们可以将基线的 BLEU 分数与产生什么样的描述质量的想法联系起来。

让我们训练一个模型并从训练和测试集生成一些描述作为健全性检查。

将重复次数更改为 1，将运行名称更改为“`baseline_generate`”。

```py
model_name = 'baseline_generate'
n_repeats = 1
```

然后更新`evaluate_model()`函数以仅评估数据集中的前 5 张照片并打印描述，如下所示。

```py
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		actual.append([desc.split()])
		predicted.append(yhat.split())
		print('Actual:    %s' % desc)
		print('Predicted: %s' % yhat)
		if len(actual) >= 5:
			break
	# calculate BLEU score
	bleu = corpus_bleu(actual, predicted)
	return bleu
```

重新运行示例。

您应该看到训练的结果如下所示（具体结果将根据算法的随机性质而变化）：

```py
Actual:    startseq boy bites hard into treat while he sits outside endseq
Predicted: startseq boy boy while while he while outside endseq

Actual:    startseq man in field backed by american flags endseq
Predicted: startseq man in in standing city endseq

Actual:    startseq two girls are walking down dirt road in park endseq
Predicted: startseq man walking down down road in endseq

Actual:    startseq girl laying on the tree with boy kneeling before her endseq
Predicted: startseq boy while in up up up water endseq

Actual:    startseq boy in striped shirt is jumping in front of water fountain endseq
Predicted: startseq man is is shirt is on on on on bike endseq
```

您应该在测试数据集上看到如下结果：

```py
Actual:    startseq three people are looking into photographic equipment endseq
Predicted: startseq boy racer on on on on bike endseq

Actual:    startseq boy is leaning on chair whilst another boy pulls him around with rope endseq
Predicted: startseq girl in playing on on on sword endseq

Actual:    startseq black and brown dog jumping in midair near field endseq
Predicted: startseq dog dog running running running and dog in grass endseq

Actual:    startseq dog places his head on man face endseq
Predicted: startseq brown dog dog to to to to to to to ball endseq

Actual:    startseq man in green hat is someplace up high endseq
Predicted: startseq man in up up his waves endseq
```

我们可以看到描述并不完美，有些是粗略的，但通常模型会生成一些可读的文本。一个很好的改善起点。

接下来，让我们看一些实验来改变不同子模型的大小或容量。

## 网络大小参数

在本节中，我们将了解网络结构的总体变化如何影响模型技能。

我们将看看模型大小的以下几个方面：

1.  '编码器'的固定向量输出的大小。
2.  序列编码器模型的大小。
3.  语言模型的大小。

让我们潜入。

### 固定长度向量的大小

在基线模型中，照片特征提取器和文本序列编码器都输出 128 个元素向量。然后将这些向量连接起来以由语言模型处理。

来自每个子模型的 128 个元素向量包含有关输入序列和照片的所有已知信息。我们可以改变这个向量的大小，看它是否会影响模型技能

首先，我们可以将大小从 128 个元素减少到 64 个元素。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(64, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(64, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

我们将此模型命名为“`size_sm_fixed_vec`”。

```py
model_name = 'size_sm_fixed_vec'
```

运行此实验会产生以下 BLEU 分数，可能是测试集上基线的小增益。

```py
          train      test
count  3.000000  3.000000
mean   0.204421  0.063148
std    0.026992  0.003264
min    0.174769  0.059391
25%    0.192849  0.062074
50%    0.210929  0.064757
75%    0.219246  0.065026
max    0.227564  0.065295
```

我们还可以将固定长度向量的大小从 128 增加到 256 个单位。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(256, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

我们将此配置命名为“`size_lg_fixed_vec`”。

```py
model_name = 'size_lg_fixed_vec'
```

运行此实验显示 BLEU 分数表明该模型并没有更好。

有可能通过更多数据和/或更长时间的训练，我们可能会看到不同的故事。

```py
          train      test
count  3.000000  3.000000
mean   0.023517  0.027813
std    0.009951  0.010525
min    0.012037  0.021737
25%    0.020435  0.021737
50%    0.028833  0.021737
75%    0.029257  0.030852
max    0.029682  0.039966
```

### 序列编码器大小

我们可以调用子模型来解释生成到目前为止的序列编码器的单词的输入序列。

首先，我们可以尝试降低序列编码器的代表表现力是否会影响模型技能。我们可以将 LSTM 层中的内存单元数从 256 减少到 128。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(128, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'size_sm_seq_model'
```

运行这个例子，我们可以看到两列训练上的小凹凸和基线测试。这可能是小训练集大小的神器。

```py
          train      test
count  3.000000  3.000000
mean   0.074944  0.053917
std    0.014263  0.013264
min    0.066292  0.039142
25%    0.066713  0.048476
50%    0.067134  0.057810
75%    0.079270  0.061304
max    0.091406  0.064799
```

换句话说，我们可以将 LSTM 层的数量从一个增加到两个，看看是否会产生显着的差异。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = LSTM(256, return_sequences=True)(emb3)
	emb5 = TimeDistributed(Dense(128, activation='relu'))(emb4)
	# merge inputs
	merged = concatenate([fe3, emb5])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'size_lg_seq_model'
```

运行此实验表明 BLEU 在训练和测试装置上都有不错的碰撞。

```py
          train      test
count  3.000000  3.000000
mean   0.094937  0.096970
std    0.022394  0.079270
min    0.069151  0.046722
25%    0.087656  0.051279
50%    0.106161  0.055836
75%    0.107830  0.122094
max    0.109499  0.188351
```

我们还可以尝试通过将其从 50 维加倍到 100 维来增加单词嵌入的表示能力。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 100, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'size_em_seq_model'
```

我们在训练数据集上看到一个大的运动，但测试数据集上的运动可能很少。

```py
count  3.000000  3.000000
mean   0.112743  0.050935
std    0.017136  0.006860
min    0.096121  0.043741
25%    0.103940  0.047701
50%    0.111759  0.051661
75%    0.121055  0.054533
max    0.130350  0.057404
```

### 语言模型的大小

我们可以参考从连接序列和照片特征输入中学习的模型作为语言模型。它负责生成单词。

首先，我们可以通过将 LSTM 和密集层切割为 500 到 256 个神经元来研究对模型技能的影响。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(256)(merged)
	lm3 = Dense(256, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'size_sm_lang_model'
```

我们可以看到，这对 BLEU 对训练和测试数据集的影响都很小，同样可能与数据集的小尺寸有关。

```py
          train      test
count  3.000000  3.000000
mean   0.063632  0.056059
std    0.018521  0.009064
min    0.045127  0.048916
25%    0.054363  0.050961
50%    0.063599  0.053005
75%    0.072884  0.059630
max    0.082169  0.066256
```

我们还可以通过添加相同大小的第二个 LSTM 层来查看加倍语言模型容量的影响。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500, return_sequences=True)(merged)
	lm3 = LSTM(500)(lm2)
	lm4 = Dense(500, activation='relu')(lm3)
	outputs = Dense(vocab_size, activation='softmax')(lm4)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'size_lg_lang_model'
```

同样，我们看到 BLEU 中的微小运动，可能是噪声和数据集大小的人为因素。测试的改进

测试数据集的改进可能是个好兆头。这可能是一个值得探索的变化。

```py
          train      test
count  3.000000  3.000000
mean   0.043838  0.067658
std    0.037580  0.045813
min    0.017990  0.015757
25%    0.022284  0.050252
50%    0.026578  0.084748
75%    0.056763  0.093608
max    0.086948  0.102469
```

在更小的数据集上调整模型大小具有挑战性。

## 配置特征提取模型

使用预先训练的 VGG16 模型提供了一些额外的配置点。

基线模型从 VGG 模型中移除了顶部，包括全局最大池化层，然后将特征的编码提供给 128 元素向量。

在本节中，我们将查看对基线模型的以下修改：

1.  在 VGG 模型之后使用全局平均池层。
2.  不使用任何全局池。

### 全球平均汇集

我们可以用 GlobalAveragePooling2D 替换 GlobalMaxPooling2D 层以实现平均池化。

开发全局平均合并以减少图像分类问题的过拟合，但可以在解释从图像中提取的特征方面提供一些益处。

有关全球平均合并的更多信息，请参阅论文：

*   [网络网络](https://arxiv.org/abs/1312.4400)，2013 年。

下面列出了更新的`define_model()`函数和实验名称。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalAveragePooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'fe_avg_pool'
```

结果表明训练数据集得到了显着改善，这可能是过拟合的标志。我们也看到了测试技巧的小幅提升。这可能是一个值得探索的变化。

我们也看到了测试技巧的小幅提升。这可能是一个值得探索的变化。

```py
          train      test
count  3.000000  3.000000
mean   0.834627  0.060847
std    0.083259  0.040463
min    0.745074  0.017705
25%    0.797096  0.042294
50%    0.849118  0.066884
75%    0.879404  0.082418
max    0.909690  0.097952
```

### 没有合并

我们可以删除 GlobalMaxPooling2D 并展平 3D 照片功能并将其直接送入 Dense 层。

我不认为这是一个很好的模型设计，但值得测试这个假设。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = Flatten()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'fe_flat'
```

令人惊讶的是，我们看到训练数据的小幅提升和测试数据的大幅提升。这对我来说是令人惊讶的，可能值得进一步调查。

```py
          train      test
count  3.000000  3.000000
mean   0.055988  0.135231
std    0.017566  0.079714
min    0.038605  0.044177
25%    0.047116  0.106633
50%    0.055627  0.169089
75%    0.064679  0.180758
max    0.073731  0.192428
```

我们可以尝试重复此实验，并提供更多容量来解释提取的照片功能。在 Flatten 层之后添加具有 500 个神经元的新 Dense 层。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = Flatten()(inputs1)
	fe2 = Dense(500, activation='relu')(fe1)
	fe3 = Dense(128, activation='relu')(fe2)
	fe4 = RepeatVector(max_length)(fe3)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe4, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'fe_flat2'
```

这导致更改不太令人印象深刻，并且测试数据集上的 BLEU 结果可能更差。

```py
          train      test
count  3.000000  3.000000
mean   0.060126  0.029487
std    0.030300  0.013205
min    0.031235  0.020850
25%    0.044359  0.021887
50%    0.057483  0.022923
75%    0.074572  0.033805
max    0.091661  0.044688
```

## 词嵌入模型

模型的关键部分是序列学习模型，它必须解释到目前为止为照片生成的单词序列。

在该子模型的输入处是单词嵌入和改进单词嵌入而不是从头开始学习它作为模型的一部分（如在基线模型中）的好方法是使用预训练的单词嵌入。

在本节中，我们将探讨在模型上使用预先训练的单词嵌入的影响。特别：

1.  训练 Word2Vec 模型
2.  训练 Word2Vec 模型+微调

### 训练有素的 word2vec 嵌入

用于从文本语料库预训练单词嵌入的有效学习算法是 word2vec 算法。

您可以在此处了解有关 word2vec 算法的更多信息：

*   [Word2Vec Google 代码项目](https://code.google.com/archive/p/word2vec/)

我们可以使用此算法使用数据集中的已清理照片描述来训练新的独立单词向量集。

[Gensim 库](https://radimrehurek.com/gensim/models/word2vec.html)提供对算法实现的访问，我们可以使用它来预先训练嵌入。

首先，我们必须像以前一样加载训练数据集的干净照片描述。

接下来，我们可以在所有干净的描述中使用 word2vec 模型。我们应该注意，这包括比训练数据集中使用的 50 更多的描述。这些实验的更公平的模型应该只训练训练数据集中的那些描述。

一旦适合，我们可以将单词和单词向量保存为 ASCII 文件，可能用于以后的检查或可视化。

```py
# train word2vec model
lines = [s.split() for s in train_descriptions.values()]
model = Word2Vec(lines, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'custom_embedding.txt'
model.wv.save_word2vec_format(filename, binary=False)
```

单词嵌入保存到文件'`custom_embedding.txt`'。

现在，我们可以将嵌入加载到内存中，只检索词汇表中单词的单词向量，然后将它们保存到新文件中。

```py
# load the whole embedding into memory
embedding = dict()
file = open('custom_embedding.txt')
for line in file:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embedding[word] = coefs
file.close()
print('Embedding Size: %d' % len(embedding))

# summarize vocabulary
all_tokens = ' '.join(train_descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))

# get the vectors for words in our vocab
cust_embedding = dict()
for word in vocabulary:
	# check if word in embedding
	if word not in embedding:
		continue
	cust_embedding[word] = embedding[word]
print('Custom Embedding %d' % len(cust_embedding))

# save
dump(cust_embedding, open('word2vec_embedding.pkl', 'wb'))
print('Saved Embedding')
```

下面列出了完整的示例。

```py
# prepare word vectors for captioning model

from numpy import asarray
from pickle import dump
from gensim.models import Word2Vec

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
	# order keys so the split is consistent
	ordered = sorted(dataset)
	# return split dataset as two new sets
	return set(ordered[:100]), set(ordered[100:200])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# store
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions

# load dev set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
dataset = load_set(filename)
print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
print('Train=%d, Test=%d' % (len(train), len(test)))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# train word2vec model
lines = [s.split() for s in train_descriptions.values()]
model = Word2Vec(lines, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'custom_embedding.txt'
model.wv.save_word2vec_format(filename, binary=False)

# load the whole embedding into memory
embedding = dict()
file = open('custom_embedding.txt')
for line in file:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embedding[word] = coefs
file.close()
print('Embedding Size: %d' % len(embedding))

# summarize vocabulary
all_tokens = ' '.join(train_descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))

# get the vectors for words in our vocab
cust_embedding = dict()
for word in vocabulary:
	# check if word in embedding
	if word not in embedding:
		continue
	cust_embedding[word] = embedding[word]
print('Custom Embedding %d' % len(cust_embedding))

# save
dump(cust_embedding, open('word2vec_embedding.pkl', 'wb'))
print('Saved Embedding')
```

运行此示例将创建存储在文件'`word2vec_embedding.pkl`'中的单词到单词向量的新字典映射。

```py
Dataset: 1000
Train=100, Test=100
Descriptions: train=100
Vocabulary size: 365
Embedding Size: 366
Vocabulary Size: 365
Custom Embedding 365
Saved Embedding
```

接下来，我们可以加载此嵌入并使用单词向量作为嵌入层中的固定权重。

下面提供`load_embedding()`函数，它加载自定义 word2vec 嵌入并返回新的嵌入层以供在模型中使用。

```py
# load a word embedding
def load_embedding(tokenizer, vocab_size, max_length):
	# load the tokenizer
	embedding = load(open('word2vec_embedding.pkl', 'rb'))
	dimensions = 100
	trainable = False
	# create a weight matrix for words in training docs
	weights = zeros((vocab_size, dimensions))
	# walk words in order of tokenizer vocab to ensure vectors are in the right index
	for word, i in tokenizer.word_index.items():
		if word not in embedding:
			continue
		weights[i] = embedding[word]
	layer = Embedding(vocab_size, dimensions, weights=[weights], input_length=max_length, trainable=trainable, mask_zero=True)
	return layer
```

我们可以通过直接从`define_model()`函数调用函数在我们的模型中使用它。

```py
# define the captioning model
def define_model(tokenizer, vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)
	fe2 = Dense(128, activation='relu')(fe1)
	fe3 = RepeatVector(max_length)(fe2)
	# embedding
	inputs2 = Input(shape=(max_length,))
	emb2 = load_embedding(tokenizer, vocab_size, max_length)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model_name = 'seq_w2v_fixed'
```

我们可以在训练数据集上看到一些提升，也许在测试数据集上没有真正显着的变化。

```py
          train      test
count  3.000000  3.000000
mean   0.096780  0.047540
std    0.055073  0.008445
min    0.033511  0.038340
25%    0.078186  0.043840
50%    0.122861  0.049341
75%    0.128414  0.052140
max    0.133967  0.054939
```

### 训练有素的 word2vec 嵌入微调

我们可以重复之前的实验，并允许模型在拟合模型时调整单词向量。

下面列出了允许微调嵌入层的更新的`load_embedding()`功能。

```py
# load a word embedding
def load_embedding(tokenizer, vocab_size, max_length):
	# load the tokenizer
	embedding = load(open('word2vec_embedding.pkl', 'rb'))
	dimensions = 100
	trainable = True
	# create a weight matrix for words in training docs
	weights = zeros((vocab_size, dimensions))
	# walk words in order of tokenizer vocab to ensure vectors are in the right index
	for word, i in tokenizer.word_index.items():
		if word not in embedding:
			continue
		weights[i] = embedding[word]
	layer = Embedding(vocab_size, dimensions, weights=[weights], input_length=max_length, trainable=trainable, mask_zero=True)
	return layer

model_name = 'seq_w2v_tuned'
```

同样，我们认为在基线模型中使用这些预先训练的字嵌入向量并没有太大差异。

```py
          train      test
count  3.000000  3.000000
mean   0.065297  0.042712
std    0.080194  0.007697
min    0.017675  0.034593
25%    0.019003  0.039117
50%    0.020332  0.043641
75%    0.089108  0.046772
max    0.157885  0.049904
```

## 结果分析

我们对来自 8,000 张照片的 Flickr8k 训练数据集的非常小的样本（1.6％）进行了一些实验。

样本可能太小，模型没有经过足够长时间的训练，并且每个模型的 3 次重复会导致过多的变化。这些方面也可以通过设计实验来评估，例如：

1.  模型技能是否随着数据集的大小而缩放？
2.  更多的时代会带来更好的技能吗？
3.  更多重复会产生一个方差较小的技能吗？

尽管如此，我们对如何为更全面的数据集配置模型有一些想法。

以下是本教程中进行的实验的平均结果摘要。

查看结果图表很有帮助。如果我们有更多的重复，每个分数分布的盒子和胡须图可能是一个很好的可视化。这里我们使用一个简单的条形图。请记住，较大的 BLEU 分数更好。

训练数据集的结果：

![Bar Chart of Experiment vs Model Skill on the Training Dataset](img/06296eb24404348507d2d48f948c0313.jpg)

实验条形图与训练数据集的模型技巧

测试数据集上的结果：

![Bar Chart of Experiment vs Model Skill on the Test Dataset](img/e69be5d66733a6d148d48cf818a04539.jpg)

实验条形图与测试数据集的模型技巧

从仅查看测试数据集的平均结果，我们可以建议：

*   在照片特征提取器（fe_flat 在 0.135231）之后可能不需要合并。
*   在照片特征提取器（fe_avg_pool 为 0.060847）之后，平均合并可能比最大合并更有优势。
*   也许在子模型之后的较小尺寸的固定长度向量是一个好主意（size_sm_fixed_vec 在 0.063148）。
*   也许在语言模型中添加更多层可以带来一些好处（size_lg_lang_model 为 0.067658）。
*   也许在序列模型中添加更多层可以带来一些好处（size_lg_seq_model 为 0.09697）。

我还建议探索这些建议的组合。

我们还可以查看结果的分布。

下面是一些代码，用于加载每个实验的保存结果，并在训练和测试集上创建结果的盒子和须状图以供审查。

```py
from os import listdir
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot

# load all .csv results into a dataframe
train, test = DataFrame(), DataFrame()
directory = 'results'
for name in listdir(directory):
	if not name.endswith('csv'):
		continue
	filename = directory + '/' + name
	data = read_csv(filename, header=0)
	experiment = name.split('.')[0]
	train[experiment] = data['train']
	test[experiment] = data['test']

# plot results on train
train.boxplot(vert=False)
pyplot.show()
# plot results on test
test.boxplot(vert=False)
pyplot.show()
```

在训练数据集上分配结果。

![Box and Whisker Plot of Experiment vs Model Skill on the Training Dataset](img/85c1ebb9ffbbcd5c80fc50b1c3b07ef9.jpg)

训练数据集中实验与模型技巧的盒子和晶须图

在测试数据集上分配结果。

![Box and Whisker Plot of Experiment vs Model Skill on the Test Dataset](img/7c30315b0ae45ddd0d650b0ed6c36d67.jpg)

测试数据集的实验与模型技巧的盒子和晶须图

对这些分布的审查表明：

*   平面上的利差很大;也许平均合并可能更安全。
*   较大的语言模型的传播很大，并且在错误/危险的方向上倾斜。
*   较大序列模型上的扩散很大，并且向右倾斜。
*   较小的固定长度向量大小可能有一些好处。

我预计增加重复到 5,10 或 30 会稍微收紧这些分布。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [Show and Tell：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。
*   [显示，参与和讲述：视觉注意的神经图像标题生成](https://arxiv.org/abs/1502.03044)，2016。
*   [网络网络](https://arxiv.org/abs/1312.4400)，2013 年。

### 相关字幕项目

*   [caption_generator：图片字幕项目](https://github.com/anuragmishracse/caption_generator)
*   [Keras 图片标题](https://github.com/LemonATsu/Keras-Image-Caption)
*   [神经图像字幕（NIC）](https://github.com/oarriaga/neural_image_captioning)
*   [Keras 深度学习图像标题检索](https://deeplearningmania.quora.com/Keras-deep-learning-for-image-caption-retrieval)
*   [DataLab Cup 2：图像字幕](http://www.cs.nthu.edu.tw/~shwu/courses/ml/competitions/02_Image-Caption/02_Image-Caption.html)

### 其他

*   [将图像描述框架化为排名任务：数据，模型和评估指标](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)。

### API

*   [Keras VGG16 API](https://keras.io/applications/#vgg16)
*   [Gensim word2vec API](https://radimrehurek.com/gensim/models/word2vec.html)

## 摘要

在本教程中，您了解了如何使用照片字幕数据集的一小部分样本来探索不同的模型设计。

具体来说，你学到了：

*   如何为照片字幕建模准备数据。
*   如何设计基线和测试工具来评估模型的技能和控制其随机性。
*   如何评估模型技能，特征提取模型和单词嵌入等属性，以提升模型技能。

你能想到什么实验？
你还尝试了什么？
您可以在训练和测试数据集上获得哪些最佳结果？

请在下面的评论中告诉我。