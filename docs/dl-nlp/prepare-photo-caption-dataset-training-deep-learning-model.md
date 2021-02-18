# 如何准备照片标题数据集以训练深度学习模型

> 原文： [https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/)

自动照片字幕是一个问题，其中模型必须在给定照片的情况下生成人类可读的文本描述。

这是人工智能中的一个具有挑战性的问题，需要来自计算机视觉领域的图像理解以及来自自然语言处理领域的语言生成。

现在可以使用深度学习和免费提供的照片数据集及其描述来开发自己的图像标题模型。

在本教程中，您将了解如何准备照片和文本描述，以便开发深度学习自动照片标题生成模型。

完成本教程后，您将了解：

*   关于 Flickr8K 数据集，包含 8,000 多张照片和每张照片最多 5 个字幕。
*   如何为深度学习建模一般加载和准备照片和文本数据。
*   如何在 Keras 中为两种不同类型的深度学习模型专门编码数据。

让我们开始吧。

*   **2017 年 11 月更新**：修复了“_ 整个描述序列模型 _”部分代码中的小拼写错误。谢谢 Moustapha Cheikh 和 Matthew。
*   **2002 年 2 月更新**：提供了 Flickr8k_Dataset 数据集的直接链接，因为官方网站已被删除。

![How to Prepare a Photo Caption Dataset for Training a Deep Learning Model](img/ed876dc6c1e515e527db6e72f03e47ab.jpg)

如何准备照片标题数据集以训练深度学习模型
照片由 [beverlyislike](https://www.flickr.com/photos/beverlyislike/3307325815/) ，保留一些权利。

## 教程概述

本教程分为 9 个部分;他们是：

1.  下载 Flickr8K 数据集
2.  如何加载照片
3.  预先计算照片功能
4.  如何加载描述
5.  准备说明文字
6.  整个描述序列模型
7.  逐字模型
8.  渐进式加载
9.  预先计算照片功能

### Python 环境

本教程假定您已安装 Python 3 SciPy 环境。您可以使用 Python 2，但您可能需要更改一些示例。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 下载 Flickr8K 数据集

Flickr8K 数据集是开始使用图像字幕时使用的一个很好的数据集。

原因是它是现实的并且相对较小，因此您可以使用 CPU 在工作站上下载它并构建模型。

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

*   **Flicker8k_Dataset** ：包含 8092 张 jpeg 格式的照片。
*   **Flickr8k_text** ：包含许多包含不同照片描述来源的文件。

接下来，我们来看看如何加载图像。

## 如何加载照片

在本节中，我们将开发一些代码来加载照片，以便与 Python 中的 Keras 深度学习库一起使用。

图像文件名是唯一的图像标识符。例如，以下是图像文件名的示例：

```py
990890291_afc72be141.jpg
99171998_7cc800ceef.jpg
99679241_adc853a5c0.jpg
997338199_7343367d7f.jpg
997722733_0cb5439472.jpg
```

Keras 提供`load_img()`函数，可用于将图像文件直接作为像素数组加载。

```py
from keras.preprocessing.image import load_img
image = load_img('990890291_afc72be141.jpg')
```

像素数据需要转换为 NumPy 阵列以便在 Keras 中使用。

我们可以使用`img_to_array()`keras 函数来转换加载的数据。

```py
from keras.preprocessing.image import img_to_array
image = img_to_array(image)
```

我们可能想要使用预定义的特征提取模型，例如在 Image net 上训练的最先进的深度图像分类网络。牛津视觉几何组（VGG）模型很受欢迎，可用于 Keras。

牛津视觉几何组（VGG）模型很受欢迎，可用于 Keras。

如果我们决定在模型中使用这个预先训练的模型作为特征提取器，我们可以使用 Keras 中的`preprocess_input()`函数预处理模型的像素数据，例如：

```py
from keras.applications.vgg16 import preprocess_input

# reshape data into a single sample of an image
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
```

我们可能还想强制加载照片以使其具有与 VGG 模型相同的像素尺寸，即 224 x 224 像素。我们可以在调用`load_img()`时这样做，例如：

```py
image = load_img('990890291_afc72be141.jpg', target_size=(224, 224))
```

我们可能想要从图像文件名中提取唯一的图像标识符。我们可以通过将'。'（句点）字符拆分文件名字符串并检索结果数组的第一个元素来实现：

```py
image_id = filename.split('.')[0]
```

我们可以将所有这些结合在一起并开发一个函数，给定包含照片的目录的名称，将加载和预处理 VGG 模型的所有照片，并将它们返回到键入其唯一图像标识符的字典中。

```py
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
	images = dict()
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
		# get image id
		image_id = name.split('.')[0]
		images[image_id] = image
	return images

# load images
directory = 'Flicker8k_Dataset'
images = load_photos(directory)
print('Loaded Images: %d' % len(images))
```

运行此示例将打印已加载图像的数量。运行需要几分钟。

```py
Loaded Images: 8091
```

如果你没有 RAM 来保存所有图像（估计大约 5GB），那么你可以添加一个 if 语句来在加载 100 个图像后提前打破循环，例如：

```py
if (len(images) >= 100):
	break
```

## 预先计算照片功能

可以使用预先训练的模型从数据集中的照片中提取特征并将特征存储到文件中。

这是一种效率，这意味着可以将从照片中提取的特征转换为文本描述的模型的语言部分可以从特征提取模型中单独训练。好处是，非常大的预训练模型不需要加载，保存在存储器中，并且用于在训练语言模型时处理每张照片。

之后，可以将特征提取模型和语言模型放在一起，以便对新照片进行预测。

在本节中，我们将扩展上一节中开发的照片加载行为，以加载所有照片，使用预先训练的 VGG 模型提取其特征，并将提取的特征存储到可以加载并用于训练的新文件中。语言模型。

第一步是加载 VGG 模型。此型号直接在 Keras 中提供，可按如下方式加载。请注意，这会将 500 兆的模型权重下载到您的计算机，这可能需要几分钟。

```py
from keras.applications.vgg16 import VGG16
# load the model
in_layer = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, input_tensor=in_layer, pooling='avg')
print(model.summary())
```

这将加载 VGG 16 层模型。

通过设置 _include_top = False_ ，从模型中删除两个密集输出层以及分类输出层。最终汇集层的输出被视为从图像中提取的特征。

接下来，我们可以像上一节一样遍历图像目录中的所有图像，并在模型上为每个准备好的图像调用`predict()`函数以获取提取的特征。然后可以将这些特征存储在键入图像 id 的字典中。

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

该示例可能需要一些时间才能完成，可能需要一个小时。

提取所有功能后，字典将存储在当前工作目录中的“ _features.pkl_ ”文件中。

然后可以稍后加载这些特征并将其用作训练语言模型的输入。

您可以在 Keras 中尝试其他类型的预训练模型。

## 如何加载描述

花点时间谈谈描述是很重要的;有一些可用。

文件 _Flickr8k.token.txt_ 包含图像标识符列表（用于图像文件名）和分词描述。每个图像都有多个描述。

以下是文件中的描述示例，显示了单个图像的 5 种不同描述。

```py
1305564994_00513f9a5b.jpg#0 A man in street racer armor be examine the tire of another racer 's motorbike .
1305564994_00513f9a5b.jpg#1 Two racer drive a white bike down a road .
1305564994_00513f9a5b.jpg#2 Two motorist be ride along on their vehicle that be oddly design and color .
1305564994_00513f9a5b.jpg#3 Two person be in a small race car drive by a green hill .
1305564994_00513f9a5b.jpg#4 Two person in race uniform in a street car .
```

文件 _ExpertAnnotations.txt_ 表示每个图像的哪些描述是由“_ 专家 _”编写的，这些描述是由众包工作者写的，要求描述图像。

最后，文件 _CrowdFlowerAnnotations.txt_ 提供群众工作者的频率，指示字幕是否适合每个图像。可以概率地解释这些频率。

该论文的作者描述了注释如下：

> ......要求注释者写出描述描绘的场景，情境，事件和实体（人，动物，其他物体）的句子。我们为每个图像收集了多个字幕，因为可以描述许多图像的方式存在相当大的差异。

- [框架图像描述作为排名任务：数据，模型和评估指标](https://www.jair.org/media/3994/live-3994-7274-jair.pdf)，2013。

还有训练/测试拆分中使用的照片标识符列表，以便您可以比较报告中报告的结果。

第一步是决定使用哪些字幕。最简单的方法是对每张照片使用第一个描述。

首先，我们需要一个函数将整个注释文件（' _Flickr8k.token.txt_ '）加载到内存中。下面是一个执行此操作的函数，称为 _load_doc（）_，给定文件名，将以字符串形式返回文档。

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
```

我们可以从上面的文件示例中看到，我们只需要用空格分割每一行，并将第一个元素作为图像标识符，其余元素作为图像描述。例如：

```py
# split line by white space
tokens = line.split()
# take the first token as the image id, the rest as the description
image_id, image_desc = tokens[0], tokens[1:]
```

然后我们可以通过删除文件扩展名和描述号来清理图像标识符。

```py
# remove filename from image id
image_id = image_id.split('.')[0]
```

我们还可以将描述标记重新组合成一个字符串，以便以后处理。

```py
# convert description tokens back to string
image_desc = ' '.join(image_desc)
```

我们可以把所有这些放在一个函数中。

下面定义`load_descriptions()`函数，它将获取加载的文件，逐行处理，并将图像标识符字典返回到它们的第一个描述。

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

filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
```

运行该示例将打印已加载的图像描述的数量。

```py
Loaded: 8092
```

还有其他方法可以加载可能对数据更准确的描述。

使用上面的示例作为起点，让我知道你提出了什么。
在下面的评论中发布您的方法。

## 准备说明文字

描述是分词的;这意味着每个标记由用空格分隔的单词组成。

它还意味着标点符号被分隔为它们自己的标记，例如句点（'。'）和单词复数（'s）的撇号。

在模型中使用之前清理描述文本是个好主意。我们可以形成一些数据清理的想法包括：

*   将所有标记的大小写归一化为小写。
*   从标记中删除所有标点符号。
*   删除包含一个或多个字符的所有标记（删除标点符号后），例如'a'和挂's'字符。

我们可以在一个函数中实现这些简单的清理操作，该函数清除上一节中加载的字典中的每个描述。下面定义了`clean_descriptions()`函数，它将清理每个加载的描述。

```py
# clean description text
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
```

然后我们可以将干净的文本保存到文件中以供我们的模型稍后使用。

每行将包含图像标识符，后跟干净描述。下面定义了`save_doc()`函数，用于将已清理的描述保存到文件中。

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
```

将这一切与上一节中的描述加载在一起，下面列出了完整的示例。

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

# clean description text
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

运行该示例首先加载 8,092 个描述，清除它们，汇总 4,484 个唯一单词的词汇表，然后将它们保存到名为“ _descriptionss.txt_ ”的新文件中。

```py
Loaded: 8092
Vocabulary Size: 4484
```

在文本编辑器中打开新文件' _descriptionss.txt_ '并查看内容。您应该看到准备好进行建模的照片的可读描述。

```py
...
3139118874_599b30b116 two girls pose for picture at christmastime
2065875490_a46b58c12b person is walking on sidewalk and skeleton is on the left inside of fence
2682382530_f9f8fd1e89 man in black shorts is stretching out his leg
3484019369_354e0b88c0 hockey team in red and white on the side of the ice rink
505955292_026f1489f2 boy rides horse
```

词汇量仍然相对较大。为了使建模更容易，特别是第一次，我建议通过删除仅在所有描述中出现一次或两次的单词来进一步减少词汇量。

## 整个描述序列模型

有很多方法可以模拟字幕生成问题。

一种朴素的方式是创建一个模型，以一次性方式输出整个文本描述。

这是一个朴素的模型，因为它给模型带来了沉重的负担，既可以解释照片的含义，也可以生成单词，然后将这些单词排列成正确的顺序。

这与编码器 - 解码器循环神经网络中使用的语言翻译问题不同，其中在给定输入序列的编码的情况下，整个翻译的句子一次输出一个字。在这里，我们将使用图像的编码来生成输出句子。

可以使用用于图像分类的预训练模型对图像进行编码，例如在上述 ImageNet 模型上训练的 VGG。

模型的输出将是词汇表中每个单词的概率分布。序列与最长的照片描述一样长。

因此，描述需要首先进行整数编码，其中词汇表中的每个单词被赋予唯一的整数，并且单词序列将被整数序列替换。然后，整数序列需要是一个热编码，以表示序列中每个单词的词汇表的理想化概率分布。

我们可以使用 Keras 中的工具来准备此类模型的描述。

第一步是将图像标识符的映射加载到存储在' _descriptionss.txt_ '中的干净描述中。

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

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
```

运行此片段将 8,092 张照片描述加载到以图像标识符为中心的字典中。然后，可以使用这些标识符将每个照片文件加载到模型的相应输入。

```py
Loaded 8092
```

接下来，我们需要提取所有描述文本，以便我们对其进行编码。

```py
# extract all text
desc_text = list(descriptions.values())
```

我们可以使用 Keras`Tokenizer`类将词汇表中的每个单词一致地映射为整数。首先，创建对象，然后将其放在描述文本上。稍后可以将拟合标记器保存到文件中，以便将预测一致地解码回词汇单词。

```py
from keras.preprocessing.text import Tokenizer
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
```

接下来，我们可以使用 fit tokenizer 将照片描述编码为整数序列。

```py
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
```

该模型将要求所有输出序列具有相同的训练长度。我们可以通过填充所有编码序列以使其具有与最长编码序列相同的长度来实现这一点。我们可以在单词列表之后用 0 值填充序列。 Keras 提供`pad_sequences()`函数来填充序列。

```py
from keras.preprocessing.sequence import pad_sequences
# pad all sequences to a fixed length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
```

最后，我们可以对填充序列进行热编码，以便为序列中的每个字提供一个稀疏向量。 Keras 提供`to_categorical()`函数来执行此操作。

```py
from keras.utils import to_categorical
# one hot encode
y = to_categorical(padded, num_classes=vocab_size)
```

编码后，我们可以确保序列输出数据具有正确的模型形状。

```py
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)
```

将所有这些放在一起，下面列出了完整的示例。

```py
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# pad all sequences to a fixed length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
# one hot encode
y = to_categorical(padded, num_classes=vocab_size)
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)
```

运行该示例首先打印加载的图像描述的数量（8,092 张照片），数据集词汇量大小（4,485 个单词），最长描述的长度（28 个单词），然后最终打印用于拟合预测模型的数据的形状。形式 _[样品，序列长度，特征]_ 。

```py
Loaded 8092
Vocabulary Size: 4485
Description Length: 28
(8092, 28, 4485)
```

如上所述，输出整个序列对于模型可能是具有挑战性的。

我们将在下一节中讨论一个更简单的模型。

## 逐字模型

用于生成照片标题的更简单的模型是在给定图像作为输入和生成的最后一个单词的情况下生成一个单词。

然后必须递归地调用该模型以生成描述中的每个单词，其中先前的预测作为输入。

使用单词作为输入，为模型提供强制上下文，以预测序列中的下一个单词。

这是以前研究中使用的模型，例如：

*   [Show and Tell：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。

字嵌入层可用于表示输入字。与照片的特征提取模型一样，这也可以在大型语料库或所有描述的数据集上进行预训练。

该模型将完整的单词序列作为输入;序列的长度将是数据集中描述的最大长度。

该模型必须以某种方式开始。一种方法是用特殊标签围绕每个照片描述以指示描述的开始和结束，例如“STARTDESC”和“ENDDESC”。

例如，描述：

```py
boy rides horse
```

会成为：

```py
STARTDESC boy rides horse ENDDESC
```

并且将被输入到具有相同图像输入的模型，以产生以下输入 - 输出字序列对：

```py
Input (X), 						Output (y)
STARTDESC, 						boy
STARTDESC, boy,					rides
STARTDESC, boy, rides, 			horse
STARTDESC, boy, rides, horse	ENDDESC
```

数据准备工作将与上一节中描述的大致相同。

每个描述必须是整数编码。在编码之后，序列被分成多个输入和输出对，并且只有输出字（y）是一个热编码的。这是因为该模型仅需要一次预测一个单词的概率分布。

代码是相同的，直到我们计算序列的最大长度。

```py
...
descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# determine the maximum sequence length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
```

接下来，我们将每个整数编码序列分成输入和输出对。

让我们在序列中的第 i 个单词处逐步执行称为 seq 的单个序列，其中 i&gt; = 1。

首先，我们将第一个 i-1 个字作为输入序列，将第 i 个字作为输出字。

```py
# split into input and output pair
in_seq, out_seq = seq[:i], seq[i]
```

接下来，将输入序列填充到输入序列的最大长度。使用预填充（默认值），以便在序列的末尾显示新单词，而不是输入开头。

使用预填充（默认值），以便在序列的末尾显示新单词，而不是输入的开头。

```py
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
```

输出字是一个热编码，与上一节非常相似。

```py
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
```

我们可以将所有这些放在一个完整的例子中，为逐字模型准备描述数据。

```py
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# determine the maximum sequence length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)

X, y = list(), list()
for img_no, seq in enumerate(sequences):
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# split into input and output pair
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		X.append(in_seq)
		y.append(out_seq)

# convert to numpy arrays
X, y = array(X), array(y)
print(X.shape)
print(y.shape)
```

运行该示例将打印相同的统计信息，但会打印生成的编码输入和输出序列的大小。

请注意，图像的输入必须遵循完全相同的顺序，其中针对从单个描述中绘制的每个示例显示相同的照片。实现此目的的一种方法是加载照片并将其存储为从单个描述准备的每个示例。

```py
Loaded 8092
Vocabulary Size: 4485
Description Length: 28
(66456, 28)
(66456, 4485)
```

## 渐进式加载

如果你有大量的 RAM（例如 8 千兆字节或更多），并且大多数现代系统都有，那么照片和描述的 Flicr8K 数据集可以放入 RAM 中。

如果您想使用 CPU 适合深度学习模型，这很好。

或者，如果您想使用 GPU 调整模型，那么您将无法将数据放入普通 GPU 视频卡的内存中。

一种解决方案是根据模型逐步加载照片和描述。

Keras 通过在模型上使用`fit_generator()`函数来支持逐步加载的数据集。生成器是用于描述用于返回模型进行训练的批量样本的函数的术语。这可以像独立函数一样简单，其名称在拟合模型时传递给`fit_generator()`函数。

作为提醒，模型适用于多个时期，其中一个时期是一个遍历整个训练数据集的时期，例如所有照片。一个时期由多批示例组成，其中模型权重在每批结束时更新。

生成器必须创建并生成一批示例。例如，数据集中的平均句子长度为 11 个字;这意味着每张照片将产生 11 个用于拟合模型的示例，而两张照片将产生平均约 22 个示例。现代硬件的良好默认批量大小可能是 32 个示例，因此这是大约 2-3 张照片的示例。

我们可以编写一个自定义生成器来加载一些照片并将样本作为一个批次返回。

让我们假设我们正在使用上一节中描述的逐字模型，该模型期望一系列单词和准备好的图像作为输入并预测单个单词。

让我们设计一个数据生成器，给出一个加载的图像标识符字典来清理描述，一个训练好的标记器，最大序列长度将为每个批次加载一个图像的例子。

生成器必须永远循环并产生每批样品。如果生成器和产量是新概念，请考虑阅读本文：

*   [Python 生成器](https://wiki.python.org/moin/Generators)

我们可以使用 while 循环永远循环，并在其中循环遍历图像目录中的每个图像。对于每个图像文件名，我们可以加载图像并从图像的描述中创建所有输入 - 输出序列对。

以下是数据生成器功能。

```py
def data_generator(mapping, tokenizer, max_length):
	# loop for ever over images
	directory = 'Flicker8k_Dataset'
	while 1:
		for name in listdir(directory):
			# load an image from file
			filename = directory + '/' + name
			image, image_id = load_image(filename)
			# create word sequences
			desc = mapping[image_id]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
			yield [[in_img, in_seq], out_word]
```

您可以扩展它以将数据集目录的名称作为参数。

生成器返回一个包含模型输入（X）和输出（y）的数组。输入包括一个数组，其中包含两个输入图像和编码单词序列的项目。输出是一个热编码的单词。

你可以看到它调用一个名为`load_photo()`的函数来加载一张照片并返回像素和图像标识符。这是本教程开头开发的照片加载功能的简化版本。

```py
# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)[0]
	# get image id
	image_id = filename.split('/')[-1].split('.')[0]
	return image, image_id
```

调用名为`create_sequences()`的另一个函数来创建图像序列，输入单词序列和输出单词，然后我们将其输出给调用者。这是一个功能，包括上一节中讨论的所有内容，还可以创建图像像素的副本，每个输入 - 输出对都是根据照片的描述创建的。

```py
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, images):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	for j in range(len(descriptions)):
		seq = descriptions[j]
		image = images[j]
		# integer encode
		seq = tokenizer.texts_to_sequences([seq])[0]
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
	Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return Ximages, XSeq, y
```

在准备使用数据生成器的模型之前，我们必须加载干净的描述，准备标记生成器，并计算最大序列长度。必须将所有 3 个作为参数传递给 _data_generator（）_。

我们使用先前开发的相同`load_clean_descriptions()`函数和新的`create_tokenizer()`函数来简化标记生成器的创建。

将所有这些结合在一起，下面列出了完整的数据生成器，随时可用于训练模型。

```py
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)[0]
	# get image id
	image_id = filename.split('/')[-1].split('.')[0]
	return image, image_id

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc, image):
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
	Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, tokenizer, max_length):
	# loop for ever over images
	directory = 'Flicker8k_Dataset'
	while 1:
		for name in listdir(directory):
			# load an image from file
			filename = directory + '/' + name
			image, image_id = load_photo(filename)
			# create word sequences
			desc = descriptions[image_id]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
			yield [[in_img, in_seq], out_word]

# load mapping of ids to descriptions
descriptions = load_clean_descriptions('descriptions.txt')
# integer encode sequences of words
tokenizer = create_tokenizer(descriptions)
# pad to fixed length
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)

# test the data generator
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
```

可以通过调用 [next（）](https://docs.python.org/3/library/functions.html#next)函数来测试数据生成器。

我们可以按如下方式测试发电机。

```py
# test the data generator
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
```

运行该示例打印单个批量的输入和输出示例的形状（例如，13 个输入 - 输出对）：

```py
(13, 224, 224, 3)
(13, 28)
(13, 4485)
```

通过调用模型上的 fit_generator（）函数（而不是 _fit（）_）并传入生成器，可以使用生成器来拟合模型。

我们还必须指定每个时期的步数或批次数。我们可以将此估计为（10 x 训练数据集大小），如果使用 7,000 个图像进行训练，则可能估计为 70,000。

```py
# define model
# ...
# fit model
model.fit_generator(data_generator(descriptions, tokenizer, max_length), steps_per_epoch=70000, ...)
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### Flickr8K 数据集

*   [将图像描述框架化为排名任务：数据，模型和评估指标](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)（主页）
*   [框架图像描述作为排名任务：数据，模型和评估指标](https://www.jair.org/media/3994/live-3994-7274-jair.pdf)，（PDF）2013。
*   [数据集申请表](https://illinois.edu/fb/sec/1713398)
*   [Old Flicrk8K 主页](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)

### API

*   [Python 生成器](https://wiki.python.org/moin/Generators)
*   [Keras Model API](https://keras.io/models/model/)
*   [Keras pad_sequences（）API](https://keras.io/preprocessing/sequence/#pad_sequences)
*   [Keras Tokenizer API](https://keras.io/preprocessing/text/#tokenizer)
*   [Keras VGG16 API](https://keras.io/applications/#vgg16)

## 摘要

在本教程中，您了解了如何准备照片和文本描述，以便开发自动照片标题生成模型。

具体来说，你学到了：

*   关于 Flickr8K 数据集，包含 8,000 多张照片和每张照片最多 5 个字幕。
*   如何为深度学习建模一般加载和准备照片和文本数据。
*   如何在 Keras 中为两种不同类型的深度学习模型专门编码数据。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。