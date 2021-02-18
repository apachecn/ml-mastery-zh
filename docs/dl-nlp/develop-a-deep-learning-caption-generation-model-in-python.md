# 如何从头开发深度学习图片标题生成器

> 原文： [https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)

#### 开发一个深度学习模型自动
用 Keras 逐步描述 Python 中的照片。

字幕生成是一个具有挑战性的人工智能问题，必须为给定的照片生成文本描述。

它既需要计算机视觉的方法来理解图像的内容，也需要来自自然语言处理领域的语言模型，以便将图像的理解转化为正确的单词。最近，深度学习方法已经在这个问题的例子上取得了最新的成果。

深度学习方法已经证明了关于字幕生成问题的最新结果。这些方法最令人印象深刻的是，可以定义单个端到端模型来预测标题，给定照片，而不是需要复杂的数据准备或专门设计模型的管道。

在本教程中，您将了解如何从头开发照片字幕深度学习模型。

完成本教程后，您将了解：

*   如何准备用于训练深度学习模型的照片和文本数据。
*   如何设计和训练深度学习字幕生成模型。
*   如何评估训练标题生成模型并使用它来标注全新的照片。

**注**：摘录自：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。
看一下，如果你想要更多的分步教程，在使用文本数据时充分利用深度学习方法。

让我们开始吧。

*   **2017 年 11 月更新**：添加了关于 Keras 2.1.0 和 2.1.1 中引入的影响本教程中代码的错误的说明。
*   **2017 年 12 月更新**：在解释如何将描述保存到文件时更新了函数名称中的拼写错误，感谢 Minel。
*   **Update Apr / 2018** ：增加了一个新的部分，展示了如何使用渐进式加载为具有最小 RAM 的工作站训练模型。
*   **2002 年 2 月更新**：提供了 Flickr8k_Dataset 数据集的直接链接，因为官方网站已被删除。

![How to Develop a Deep Learning Caption Generation Model in Python from Scratch](img/7c3093e713bfc0f44e9aa591c5ae3415.jpg)

如何从头开始在 Python 中开发深度学习字幕生成模型
照片由[生活在蒙罗维亚](https://www.flickr.com/photos/livinginmonrovia/8069637650/)，保留一些权利。

## 教程概述

本教程分为 6 个部分;他们是：

1.  照片和标题数据集
2.  准备照片数据
3.  准备文本数据
4.  开发深度学习模型
5.  逐步加载训练（ **NEW** ）
6.  评估模型
7.  生成新标题

### Python 环境

本教程假设您安装了 Python SciPy 环境，理想情况下使用 Python 3。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.2 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅本教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

我建议在带 GPU 的系统上运行代码。您可以在 Amazon Web Services 上以低成本方式访问 GPU。在本教程中学习如何：

*   [如何设置 Amazon AWS EC2 GPU 以训练 Keras 深度学习模型（循序渐进）](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

让我们潜入。

## 照片和标题数据集

Flickr8K 数据集是开始使用图像字幕时使用的一个很好的数据集。

原因是因为它是现实的并且相对较小，因此您可以使用 CPU 在工作站上下载它并构建模型。

数据集的确切描述在论文“[框架图像描述作为排名任务：数据，模型和评估指标](https://www.jair.org/media/3994/live-3994-7274-jair.pdf)”从 2013 年开始。

作者将数据集描述如下：

> 我们为基于句子的图像描述和搜索引入了一个新的基准集合，包括 8,000 个图像，每个图像与五个不同的标题配对，提供对显着实体和事件的清晰描述。
> 
> ...
> 
> 图像是从六个不同的 Flickr 组中选择的，并且往往不包含任何知名人物或位置，而是手动选择以描绘各种场景和情况。

- [框架图像描述作为排名任务：数据，模型和评估指标](https://www.jair.org/media/3994/live-3994-7274-jair.pdf)，2013。

数据集可免费获得。您必须填写申请表，并通过电子邮件将链接发送给您。我很乐意为您链接，但电子邮件地址明确要求：“_ 请不要重新分发数据集 _”。

您可以使用以下链接来请求数据集：

*   [数据集申请表](https://illinois.edu/fb/sec/1713398)

在短时间内，您将收到一封电子邮件，其中包含指向两个文件的链接：

*   **Flickr8k_Dataset.zip** （1 千兆字节）所有照片的存档。
*   **Flickr8k_text.zip** （2.2 兆字节）照片所有文字说明的档案。

**UPDATE（2019 年 2 月）**：官方网站似乎已被删除（虽然表格仍然有效）。以下是我的[数据集 GitHub 存储库](https://github.com/jbrownlee/Datasets)的一些直接下载链接：

*   [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
*   [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

下载数据集并将其解压缩到当前工作目录中。您将有两个目录：

*   **Flicker8k_Dataset** ：包含 8092 张 JPEG 格式的照片。
*   **Flickr8k_text** ：包含许多包含不同照片描述来源的文件。

数据集具有预定义的训练数据集（6,000 个图像），开发数据集（1,000 个图像）和测试数据集（1,000 个图像）。

可用于评估模型技能的一个衡量标准是 BLEU 分数。作为参考，下面是在测试数据集上评估时对于熟练模型的一些球场 BLEU 分数（取自 2017 年论文“[将图像放入图像标题生成器](https://arxiv.org/abs/1703.09137)”中）：

*   BLEU-1：0.401 至 0.578。
*   BLEU-2：0.176 至 0.390。
*   BLEU-3：0.099 至 0.260。
*   BLEU-4：0.059 至 0.170。

我们在评估模型时会更晚地描述 BLEU 指标。

接下来，我们来看看如何加载图像。

## 准备照片数据

我们将使用预先训练的模型来解释照片的内容。

有很多型号可供选择。在这种情况下，我们将使用 2014 年赢得 ImageNet 竞赛的 Oxford Visual Geometry Group 或 VGG 模型。在此处了解有关该模型的更多信息：

*   [用于大规模视觉识别的超深卷积网络](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

Keras 直接提供这种预先训练的模型。请注意，第一次使用此模型时，Keras 将从 Internet 下载模型权重，大约为 500 兆字节。这可能需要几分钟，具体取决于您的互联网连接。

我们可以将此模型用作更广泛的图像标题模型的一部分。问题是，它是一个大型模型，每次我们想要测试一个新的语言模型配置（下游）是多余的时，通过网络运行每张照片。

相反，我们可以使用预先训练的模型预先计算“照片功能”并将其保存到文件中。然后，我们可以稍后加载这些功能，并将它们作为数据集中给定照片的解释提供给我们的模型。通过完整的 VGG 模型运行照片也没有什么不同;我们只是提前做过一次。

这是一种优化，可以更快地训练我们的模型并消耗更少的内存。

我们可以使用 VGG 类在 Keras 中加载 VGG 模型。我们将从加载的模型中删除最后一层，因为这是用于预测照片分类的模型。我们对图像分类不感兴趣，但我们对分类前的照片内部表示感兴趣。这些是模型从照片中提取的“特征”。

Keras 还提供了用于将加载的照片整形为模型的优选尺寸的工具（例如，3 通道 224×224 像素图像）。

下面是一个名为 _extract_features（）_ 的函数，给定目录名称，将加载每张照片，为 VGG 准备，并从 VGG 模型中收集预测的特征。图像特征是 1 维 4,096 元素向量。

该函数返回图像标识符的字典到图像特征。

```py
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
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

我们可以调用此函数来准备用于测试模型的照片数据，然后将生成的字典保存到名为“ _features.pkl_ ”的文件中。

下面列出了完整的示例。

```py
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
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

在运行结束时，您将提取的特征存储在' _features.pkl_ '中供以后使用。该文件大小约为 127 兆字节。

## 准备文本数据

数据集包含每张照片的多个描述，描述文本需要一些最小的清洁。

如果您不熟悉清理文本数据，请参阅此帖子：

*   [如何使用 Python 清理机器学习文本](https://machinelearningmastery.com/clean-text-machine-learning-python/)

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

每张照片都有唯一的标识符。此标识符用于照片文件名和描述的文本文件中。

接下来，我们将逐步浏览照片说明列表。下面定义了一个函数 _load_descriptions（）_，给定加载的文档文本，它将返回描述的照片标识符字典。每个照片标识符映射到一个或多个文本描述的列表。

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
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
```

接下来，我们需要清理描述文本。描述已经被分词并且易于使用。

我们将通过以下方式清理文本，以减少我们需要使用的单词词汇量：

*   将所有单词转换为小写。
*   删除所有标点符号。
*   删除所有长度不超过一个字符的单词（例如“a”）。
*   删除包含数字的所有单词。

下面定义 _clean_descriptions（）_ 函数，给定描述图像标识符的字典，逐步执行每个描述并清理文本。

```py
import string

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)
```

清理完毕后，我们可以总结一下词汇量的大小。

理想情况下，我们想要一个既富有表现力又尽可能小的词汇。较小的词汇量将导致较小的模型将更快地训练。

作为参考，我们可以将干净的描述转换为一个集合并打印其大小，以了解我们的数据集词汇表的大小。

```py
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
```

最后，我们可以将图像标识符和描述字典保存到名为 _descriptionss.txt_ 的新文件中，每行一个图像标识符和描述。

下面定义 _save_descriptions（）_ 函数，给定包含标识符到描述和文件名的映射的字典，将映射保存到文件。

```py
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# save descriptions
save_descriptions(descriptions, 'descriptions.txt')
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
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
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
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')
```

首先运行该示例打印加载的照片描述的数量（8,092）和清晰词汇的大小（8,763 个单词）。

```py
Loaded: 8,092
Vocabulary Size: 8,763
```

最后，干净的描述写入' _descriptionss.txt_ '。

看一下这个文件，我们可以看到这些描述已经准备好进行建模了。文件中的描述顺序可能有所不同。

```py
2252123185_487f21e336 bunch on people are seated in stadium
2252123185_487f21e336 crowded stadium is full of people watching an event
2252123185_487f21e336 crowd of people fill up packed stadium
2252123185_487f21e336 crowd sitting in an indoor stadium
2252123185_487f21e336 stadium full of people watch game
...
```

## 开发深度学习模型

在本节中，我们将定义深度学习模型并将其拟合到训练数据集上。

本节分为以下几部分：

1.  加载数据中。
2.  定义模型。
3.  适合模型。
4.  完整的例子。

### 加载数据中

首先，我们必须加载准备好的照片和文本数据，以便我们可以使用它来适应模型。

我们将训练训练数据集中所有照片和标题的数据。在训练期间，我们将监控模型在开发数据集上的表现，并使用该表现来决定何时将模型保存到文件。

训练和开发数据集已分别在 _Flickr_8k.trainImages.txt_ 和 _Flickr_8k.devImages.txt_ 文件中预定义，两者都包含照片文件名列表。从这些文件名中，我们可以提取照片标识符并使​​用这些标识符来过滤每组的照片和说明。

下面的函数 _load_set（）_ 将在给定训练或开发集文件名的情况下加载一组预定义的标识符。

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

现在，我们可以使用预定义的一系列训练或开发标识符来加载照片和描述。

下面是函数 _load_clean_descriptions（）_，它为来自' _descriptionss.txt_ '的已清除文本描述加载给定的一组标识符，并将标识符字典返回给文本描述列表。

我们将开发的模型将生成给定照片的标题，并且标题将一次生成一个单词。将提供先前生成的单词的序列作为输入。因此，我们需要一个'_ 第一个字 _'来启动生成过程，'_ 最后一个字 _'来表示标题的结尾。

为此，我们将使用字符串'`startseq`'和'`endseq`'。这些令牌在加载时会添加到已加载的描述中。在我们对文本进行编码之前，现在执行此操作非常重要，这样才能正确编码令牌。

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
```

接下来，我们可以加载给定数据集的照片功能。

下面定义了一个名为 _load_photo_features（）_ 的函数，它加载了整套照片描述，然后返回给定照片标识符集的感兴趣子集。

这不是很有效;尽管如此，这将使我们快速起步。

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
```

运行此示例首先在测试数据集中加载 6,000 个照片标识符。然后，这些功能用于过滤和加载已清理的描述文本和预先计算的照片功能。

我们快到了。

```py
Dataset: 6,000
Descriptions: train=6,000
Photos: train=6,000
```

描述文本需要先编码为数字，然后才能像输入中那样呈现给模型，或者与模型的预测进行比较。

编码数据的第一步是创建从单词到唯一整数值​​的一致映射。 Keras 提供`Tokenizer`类，可以从加载的描述数据中学习这种映射。

下面定义 _to_lines（）_ 将描述字典转换为字符串列表和 _create_tokenizer（）_ 函数，在给定加载的照片描述文本的情况下，它将适合 Tokenizer。

```py
# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
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

以下函数命名为 _create_sequences（）_，给定分词器，最大序列长度以及所有描述和照片的字典，将数据转换为输入 - 输出数据对以训练模型。模型有两个输入数组：一个用于照片功能，另一个用于编码文本。模型有一个输出，它是文本序列中编码的下一个单词。

输入文本被编码为整数，其将被馈送到字嵌入层。照片功能将直接送到模型的另一部分。该模型将输出预测，该预测将是词汇表中所有单词的概率分布。

因此，输出数据将是每个单词的单热编码版本，表示在除了实际单词位置之外的所有单词位置具有 0 值的理想化概率分布，其具有值 1。

```py
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)
```

我们需要计算最长描述中的最大字数。名为 _max_length（）_ 的短辅助函数定义如下。

```py
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
```

我们现在已经足够加载训练和开发数据集的数据，并将加载的数据转换为输入 - 输出对，以适应深度学习模型。

### 定义模型

我们将基于 Marc Tanti 等人描述的“_ 合并模型 _”来定义深度学习。在 2017 年的论文中：

*   [将图像放在图像标题生成器](https://arxiv.org/abs/1703.09137)中的位置，2017。
*   [循环神经网络（RNN）在图像标题生成器中的作用是什么？](https://arxiv.org/abs/1708.02043) ，2017。

有关此架构的温和介绍，请参阅帖子：

*   [使用编码器 - 解码器模型的注入和合并架构生成字幕](https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/)

作者提供了一个很好的模型示意图，如下所示。

![Schematic of the Merge Model For Image Captioning](img/a5a04b56f81f1075fd690ba33b5bc864.jpg)

图像标题合并模型的示意图

我们将分三个部分描述该模型：

*   **照片功能提取器**。这是在 ImageNet 数据集上预训练的 16 层 VGG 模型。我们已经使用 VGG 模型预处理了照片（没有输出层），并将使用此模型预测的提取特征作为输入。
*   **序列处理器**。这是用于处理文本输入的单词嵌入层，后面是长短期记忆（LSTM）循环神经网络层。
*   **解码器**（缺少一个更好的名字）。特征提取器和序列处理器都输出固定长度的向量。这些被合并在一起并由 Dense 层处理以进行最终预测。

Photo Feature Extractor 模型要求输入照片要素是 4,096 个元素的向量。这些由 Dense 层处理以产生照片的 256 个元素表示。

序列处理器模型期望具有预定义长度（34 个字）的输入序列被馈送到嵌入层，该嵌入层使用掩码来忽略填充值。接下来是具有 256 个存储器单元的 LSTM 层。

两个输入模型都产生 256 个元素向量。此外，两个输入模型都以 50％的丢失形式使用正则化。这是为了减少过度拟合训练数据集，因为这种模型配置学得非常快。

解码器模型使用加法运算合并来自两个输入模型的向量。然后将其馈送到密集 256 神经元层，然后馈送到最终输出密集层，该密集层对序列中的下一个字的整个输出词汇表进行 softmax 预测。

下面名为 _ 的函数 define_model（）_ 定义并返回准备好的模型。

```py
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
```

要了解模型的结构，特别是层的形状，请参阅下面列出的摘要。

```py
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 34)            0
____________________________________________________________________________________________________
input_1 (InputLayer)             (None, 4096)          0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 34, 256)       1940224     input_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4096)          0           input_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 34, 256)       0           embedding_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           1048832     dropout_1[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 256)           525312      dropout_2[0][0]
____________________________________________________________________________________________________
add_1 (Add)                      (None, 256)           0           dense_1[0][0]
                                                                   lstm_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 256)           65792       add_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 7579)          1947803     dense_2[0][0]
====================================================================================================
Total params: 5,527,963
Trainable params: 5,527,963
Non-trainable params: 0
____________________________________________________________________________________________________
```

我们还创建了一个图表来可视化网络结构，更好地帮助理解两个输入流。

![Plot of the Caption Generation Deep Learning Model](img/3a9ec93ec57895a672f3fd9adac0be96.jpg)

标题生成深度学习模型的情节

### 适合模型

现在我们知道如何定义模型，我们可以将它放在训练数据集上。

该模型学习快速且快速地适应训练数据集。因此，我们将监控训练模型在保持开发数据集上的技能。当开发数据集上的模型技能在时代结束时得到改善时，我们将整个模型保存到文件中。

在运行结束时，我们可以使用训练数据集中具有最佳技能的已保存模型作为我们的最终模型。

我们可以通过在 Keras 中定义`ModelCheckpoint`并指定它来监控验证数据集上的最小损失并将模型保存到文件名中具有训练和验证损失的文件来实现。

```py
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```

然后我们可以通过 _ 回调 _ 参数在 _fit（）_ 的调用中指定检查点。我们还必须通过`validation_data`参数在 _fit（）_ 中指定开发数据集。

我们只适用于 20 个时代的模型，但考虑到训练数据的数量，每个时代在现代硬件上可能需要 30 分钟。

```py
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
```

### 完整的例子

下面列出了在训练数据上拟合模型的完整示例。

```py
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# train dataset

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset

# load test set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# fit model

# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
```

首先运行该示例将打印已加载的训练和开发数据集的摘要。

```py
Dataset: 6,000
Descriptions: train=6,000
Photos: train=6,000
Vocabulary Size: 7,579
Description Length: 34
Dataset: 1,000
Descriptions: test=1,000
Photos: test=1,000
```

在对模型进行总结之后，我们可以了解训练和验证（开发）输入 - 输出对的总数。

```py
Train on 306,404 samples, validate on 50,903 samples
```

然后运行该模型，将最佳模型保存到.h5 文件中。

在我的运行中，最佳验证结果已保存到文件中：

*   _model-ep002-loss3.245-val_loss3.612.h5_

该模型在第 2 迭代结束时保存，训练数据集损失 3.245，开发数据集损失 3.612

您的具体结果会有所不同。

让我知道你在下面的评论中得到了什么。

如果您在 AWS 上运行该示例，请将模型文件复制回当前工作目录。如果您需要 AWS 上的命令帮助，请参阅帖子：

*   [10 个亚马逊网络服务深度学习命令行方案](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/)

你得到的错误如下：

```py
Memory Error
```

如果是这样，请参阅下一节。

## 训练与渐进式装载

**注意**：如果您在上一节中没有任何问题，请跳过本节。本节适用于那些没有足够内存来训练模型的人，如上一节所述（例如，出于任何原因不能使用 AWS EC2）。

标题模型的训练确实假设你有很多 RAM。

上一节中的代码不具有内存效率，并假设您在具有 32GB 或 64GB RAM 的大型 EC2 实例上运行。如果您在 8GB RAM 的工作站上运行代码，则无法训练模型。

解决方法是使用渐进式加载。这篇文章在帖子中标题为“ _Progressive Loading_ ”的倒数第二节中详细讨论过：

*   [如何准备用于训练深度学习模型的照片标题数据集](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/)

我建议您继续阅读该部分。

如果您想使用渐进式加载来训练此模型，本节将向您展示如何。

第一步是我们必须定义一个可以用作数据生成器的函数。

我们将保持简单，并使数据生成器每批产生一张照片的数据。这将是为照片及其描述生成的所有序列。

_data_generator（）_ 下面的函数将是数据生成器，将采用加载的文本描述，照片功能，标记器和最大长度。在这里，我假设您可以将这些训练数据放入内存中，我相信 8GB 的 RAM 应该更有能力。

这是如何运作的？阅读上面刚才提到的引入数据生成器的帖子。

```py
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]
```

您可以看到我们正在调用 _create_sequence（）_ 函数来为单个照片而不是整个数据集创建一批数据。这意味着我们必须更新 _create_sequences（）_ 函数以删除 for 循环的“迭代所有描述”。

更新的功能如下：

```py
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
```

我们现在拥有我们所需要的一切。

注意，这是一个非常基本的数据生成器。它提供的大内存节省是在拟合模型之前不在存储器中具有训练和测试数据的展开序列，这些样本（例如来自 _create_sequences（）_ 的结果）是根据每张照片的需要创建的。

一些关于进一步改进这种数据生成器的袖口想法包括：

*   随机化每个时代的照片顺序。
*   使用照片 ID 列表并根据需要加载文本和照片数据，以进一步缩短内存。
*   每批产生不止一张照片的样品。

我过去经历过这些变化。如果您这样做以及如何参与评论，请告诉我们。

您可以通过直接调用数据生成器来检查数据生成器，如下所示：

```py
# test the data generator
generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
```

运行此完整性检查将显示一批批量序列的样子，在这种情况下，47 个样本将为第一张照片进行训练。

```py
(47, 4096)
(47, 34)
(47, 7579)
```

最后，我们可以在模型上使用 _fit_generator（）_ 函数来使用此数据生成器训练模型。

在这个简单的例子中，我们将丢弃开发数据集和模型检查点的加载，并在每个训练时期之后简单地保存模型。然后，您可以在训练后返回并加载/评估每个已保存的模型，以找到我们可以在下一部分中使用的最低损失的模型。

使用数据生成器训练模型的代码如下：

```py
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')
```

而已。您现在可以使用渐进式加载来训练模型并节省大量 RAM。这也可能慢得多。

下面列出了用于训练字幕生成模型的渐进式加载（使用数据生成器）的完整更新示例。

```py
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')
```

也许评估每个保存的模型，并选择保持数据集中损失最小的最终模型。下一节可能有助于此。

您是否在教程中使用了这个新增功能？
你是怎么去的？

## 评估模型

一旦模型适合，我们就可以评估其预测测试数据集的预测技巧。

我们将通过生成测试数据集中所有照片的描述并使用标准成本函数评估这些预测来评估模型。

首先，我们需要能够使用训练有素的模型生成照片的描述。

这包括传入开始描述标记'`startseq`'，生成一个单词，然后以生成的单词作为输入递归调用模型，直到到达序列标记结尾'`endseq`'或达到最大描述长度。

以下名为 _generate_desc（）_ 的函数实现此行为，并在给定训练模型和给定准备照片作为输入的情况下生成文本描述。它调用函数 _word_for_id（）_ 以将整数预测映射回一个字。

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

我们将为测试数据集和训练数据集中的所有照片生成预测。

以下名为 _evaluate_model（）_ 的函数将针对给定的照片描述和照片特征数据集评估训练模型。使用语料库 BLEU 分数收集和评估实际和预测的描述，该分数总结了生成的文本与预期文本的接近程度。

```py
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
```

BLEU 分数用于文本翻译，用于针对一个或多个参考翻译评估翻译文本。

在这里，我们将每个生成的描述与照片的所有参考描述进行比较。然后，我们计算 1,2,3 和 4 累积 n-gram 的 BLEU 分数。

您可以在此处了解有关 BLEU 分数的更多信息：

*   [计算 Python 中文本的 BLEU 分数的温和介绍](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)

[NLTK Python 库在 _corpus_bleu（）_ 函数中实现 BLEU 得分](http://www.nltk.org/api/nltk.translate.html)计算。接近 1.0 的较高分数更好，接近零的分数更差。

我们可以将所有这些与上一节中的函数一起用于加载数据。我们首先需要加载训练数据集以准备 Tokenizer，以便我们可以将生成的单词编码为模型的输入序列。使用与训练模型时使用的完全相同的编码方案对生成的单词进行编码至关重要。

然后，我们使用这些函数来加载测试数据集。

下面列出了完整的示例。

```py
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

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
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model-ep002-loss3.245-val_loss3.612.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
```

运行该示例将打印 BLEU 分数。

我们可以看到分数在问题的熟练模型的预期范围的顶部和接近顶部。所选的模型配置决不会优化。

```py
BLEU-1: 0.579114
BLEU-2: 0.344856
BLEU-3: 0.252154
BLEU-4: 0.131446
```

## 生成新标题

现在我们知道如何开发和评估字幕生成模型，我们如何使用它？

几乎我们为全新照片生成字幕所需的一切都在模型文件中。

我们还需要 Tokenizer 在生成序列时为模型编码生成的单词，以及在我们定义模型时使用的输入序列的最大长度（例如 34）。

我们可以硬编码最大序列长度。通过文本编码，我们可以创建标记生成器并将其保存到文件中，以便我们可以在需要时快速加载它而无需整个 Flickr8K 数据集。另一种方法是使用我们自己的词汇表文件并在训练期间映射到整数函数。

我们可以像以前一样创建 Tokenizer 并将其保存为 pickle 文件 _tokenizer.pkl_ 。下面列出了完整的示例。

```py
from keras.preprocessing.text import Tokenizer
from pickle import dump

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
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

我们现在可以在需要时加载 tokenizer 而无需加载整个注释的训练数据集。

现在，让我们为新照片生成描述。

下面是我在 Flickr 上随机选择的新照片（可在许可许可下获得）。

![Photo of a dog at the beach.](img/1036583bcaf100d850a94df4e70324d4.jpg)

一条狗的照片在海滩的。
照片由 [bambe1964](https://www.flickr.com/photos/bambe1964/7837618434/) 拍摄，部分版权所有。

我们将使用我们的模型为它生成描述。

下载照片并将其保存到本地目录，文件名为“ _example.jpg_ ”。

首先，我们必须从 _tokenizer.pkl_ 加载 Tokenizer，并定义填充输入所需的生成序列的最大长度。

```py
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
```

然后我们必须像以前一样加载模型。

```py
# load the model
model = load_model('model-ep002-loss3.245-val_loss3.612.h5')
```

接下来，我们必须加载要描述的照片并提取特征。

我们可以通过重新定义模型并向其添加 VGG-16 模型来实现这一目标，或者我们可以使用 VGG 模型预测特征并将其用作现有模型的输入。我们将使用后者并使用在数据准备期间使用的 _extract_features（）_ 函数的修改版本，但适用于处理单张照片。

```py
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# load and prepare the photograph
photo = extract_features('example.jpg')
```

然后，我们可以使用在评估模型时定义的 _generate_desc（）_ 函数生成描述。

下面列出了为全新独立照片生成描述的完整示例。

```py
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

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

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep002-loss3.245-val_loss3.612.h5')
# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
```

在这种情况下，生成的描述如下：

```py
startseq dog is running across the beach endseq
```

您可以删除开始和结束标记，您将拥有一个漂亮的自动照片字幕模型的基础。

这就像生活在未来的家伙！

它仍然完全让我感到震惊，我们可以做到这一点。哇。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **替代预训练照片模型**。使用小的 16 层 VGG 模型进行特征提取。考虑探索在 ImageNet 数据集上提供更好表现的更大模型，例如 Inception。
*   **较小的词汇**。在模型的开发中使用了大约八千字的更大词汇。支持的许多单词可能是拼写错误或仅在整个数据集中使用过一次。优化词汇量并缩小尺寸，可能减半。
*   **预先训练过的单词向量**。该模型学习了单词向量作为拟合模型的一部分。通过使用在训练数据集上预训练或在更大的文本语料库（例如新闻文章或维基百科）上训练的单词向量，可以实现更好的表现。
*   **调谐模型**。该模型的配置没有针对该问题进行调整。探索备用配置，看看是否可以获得更好的表现。

你尝试过这些扩展吗？在下面的评论中分享您的结果。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 标题生成论文

*   [Show and Tell：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。
*   [显示，参与和讲述：视觉注意的神经图像标题生成](https://arxiv.org/abs/1502.03044)，2015。
*   [将图像放在图像标题生成器](https://arxiv.org/abs/1703.09137)中的位置，2017。
*   [循环神经网络（RNN）在图像标题生成器中的作用是什么？](https://arxiv.org/abs/1708.02043) ，2017。
*   [图像自动生成描述：模型，数据集和评估措施的调查](https://arxiv.org/abs/1601.03896)，2016。

### Flickr8K 数据集

*   [将图像描述框架化为排名任务：数据，模型和评估指标](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)（主页）
*   [框架图像描述作为排名任务：数据，模型和评估指标](https://www.jair.org/media/3994/live-3994-7274-jair.pdf)，（PDF）2013。
*   [数据集申请表](https://illinois.edu/fb/sec/1713398)
*   [Old Flicrk8K 主页](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)

### API

*   [Keras Model API](https://keras.io/models/model/)
*   [Keras pad_sequences（）API](https://keras.io/preprocessing/sequence/#pad_sequences)
*   [Keras Tokenizer API](https://keras.io/preprocessing/text/#tokenizer)
*   [Keras VGG16 API](https://keras.io/applications/#vgg16)
*   [Gensim word2vec API](https://radimrehurek.com/gensim/models/word2vec.html)
*   [nltk.translate 包 API 文档](http://www.nltk.org/api/nltk.translate.html)

## 摘要

在本教程中，您了解了如何从头开发照片字幕深度学习模型。

具体来说，你学到了：

*   如何准备照片和文本数据，为深度学习模型的训练做好准备。
*   如何设计和训练深度学习字幕生成模型。
*   如何评估训练标题生成模型并使用它来标注全新的照片。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。

**注**：这是以下摘录的章节：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。
看一下，如果你想要更多的分步教程，在使用文本数据时充分利用深度学习方法。