# 如何开发一种深度学习的词袋模型来预测电影评论情感

> 原文： [https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)

电影评论可以被分类为有利或无。

电影评论文本的评估是一种通常称为情感分析的分类问题。用于开发情感分析模型的流行技术是使用词袋模型，其将文档转换为向量，其中文档中的每个单词被分配分数。

在本教程中，您将了解如何使用词袋表示形成电影评论情感分类来开发深度学习预测模型。

完成本教程后，您将了解：

*   如何准备评论文本数据以便使用受限词汇表进行建模。
*   如何使用词袋模型来准备训练和测试数据。
*   如何开发多层 Perceptron 词袋模型并使用它来预测新的评论文本数据。

让我们开始吧。

*   **2017 年 10 月更新**：修正了加载和命名正面和负面评论时的小错字（感谢 Arthur）。

![How to Develop a Deep Learning Bag-of-Words Model for Predicting Sentiment in Movie Reviews](img/b6d93ac7970686bc3488e5204a5e6459.jpg)

如何开发一种用于预测电影评论情感的深度学习词袋模型
[jai Mansson](https://www.flickr.com/photos/75348994@N00/302260108/) 的照片，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  电影评论数据集
2.  数据准备
3.  词袋表示
4.  情感分析模型

## 电影评论数据集

电影评论数据是 Bo Pang 和 Lillian Lee 在 21 世纪初从 imdb.com 网站上检索到的电影评论的集合。收集的评论作为他们自然语言处理研究的一部分。

评论最初于 2002 年发布，但更新和清理版本于 2004 年发布，称为“v2.0”。

该数据集包含 1,000 个正面和 1,000 个负面电影评论，这些评论来自 [imdb.com](http://reviews.imdb.com/Reviews) 上托管的 rec.arts.movi​​es.reviews 新闻组的存档。作者将此数据集称为“极性数据集”。

> 我们的数据包含 2000 年之前写的 1000 份正面和 1000 份负面评论，每位作者的评论上限为 20（每位作者共 312 位）。我们将此语料库称为极性数据集。

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

数据已经有所清理，例如：

*   数据集仅包含英语评论。
*   所有文本都已转换为小写。
*   标点符号周围有空格，如句号，逗号和括号。
*   文本每行被分成一个句子。

该数据已用于一些相关的自然语言处理任务。对于分类，经典模型（例如支持向量机）对数据的表现在高 70％至低 80％（例如 78％-82％）的范围内。

更复杂的数据准备可以看到高达 86％的结果，交叉验证 10 倍。如果我们想在现代方法的实验中使用这个数据集，这给了我们 80 年代中期的球场。

> ...根据下游极性分类器的选择，我们可以实现高度统计上的显着改善（从 82.8％到 86.4％）

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

您可以从此处下载数据集：

*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （review_polarity.tar.gz，3MB）

解压缩文件后，您将有一个名为“txt_sen`t`oken”的目录，其中包含两个子目录，其中包含文本“`neg`”和“`pos`”消极和积极的评论。对于每个 neg 和 pos，每个文件存储一个评论约定`cv000`到`cv999`。

接下来，我们来看看加载和准备文本数据。

## 数据准备

在本节中，我们将看看 3 件事：

1.  将数据分成训练和测试集。
2.  加载和清理数据以删除标点符号和数字。
3.  定义首选词汇的词汇。

### 分为训练和测试装置

我们假装我们正在开发一种系统，可以预测文本电影评论的情感是积极的还是消极的。

这意味着在开发模型之后，我们需要对新的文本评论做出预测。这将要求对这些新评论执行所有相同的数据准备，就像对模型的训练数据执行一样。

我们将通过在任何数据准备之前拆分训练和测试数据集来确保将此约束纳入我们模型的评估中。这意味着在数据准备和模型训练期间，测试集中可以帮助我们更好地准备数据（例如使用的单词）的任何知识都是不可用的。

话虽如此，我们将使用最近 100 次正面评论和最后 100 次负面评论作为测试集（100 条评论），其余 1,800 条评论作为训练数据集。

这是 90％的训练，10％的数据分割。

通过使用评论的文件名可以轻松实现拆分，其中评论为 000 至 899 的评论用于训练数据，而评论为 900 以上的评论用于测试模型。

### 装载和清洁评论

文本数据已经相当干净，因此不需要太多准备工作。

在不了解细节的情况下，我们将使用以下方法准备数据：

*   在白色空间的分裂标记。
*   从单词中删除所有标点符号。
*   删除所有不完全由字母字符组成的单词。
*   删除所有已知停用词的单词。
*   删除长度为＆lt; = 1 个字符的所有单词。

我们可以将所有这些步骤放入一个名为 clean_doc（）的函数中，该函数将从文件加载的原始文本作为参数，并返回已清理的标记列表。我们还可以定义一个函数 load_doc（），它从文件中加载文档，以便与 clean_doc（）函数一起使用。

下面列出了清理第一次正面评价的示例。

```py
from nltk.corpus import stopwords
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

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
```

运行该示例会打印一长串清洁令牌。

我们可能想要探索更多的清洁步骤，并将其作为进一步的练习。我很想知道你能想出什么。

```py
...
'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']
```

### 定义词汇表

在使用词袋模型时，定义已知单词的词汇表很重要。

单词越多，文档的表示越大，因此将单词限制为仅被认为具有预测性的单词是很重要的。这很难事先知道，并且通常重要的是测试关于如何构建有用词汇的不同假设。

我们已经看到了如何从上一节中的词汇表中删除标点符号和数字。我们可以对所有文档重复此操作，并构建一组所有已知单词。

我们可以开发一个词汇作为 _ 计数器 _，这是一个词典及其计数的字典映射，可以让我们轻松更新和查询。

每个文档都可以添加到计数器（一个名为`add_doc_to_vocab()`的新函数），我们可以跳过负目录中的所有评论，然后是肯定目录（一个名为 _process_docs 的新函数） （）_）。

下面列出了完整的示例。

```py
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
```

运行该示例表明我们的词汇量为 43,476 个单词。

我们还可以看到电影评论中前 50 个最常用单词的样本。

请注意，此词汇表仅基于训练数据集中的那些评论构建。

```py
44276
[('film', 7983), ('one', 4946), ('movie', 4826), ('like', 3201), ('even', 2262), ('good', 2080), ('time', 2041), ('story', 1907), ('films', 1873), ('would', 1844), ('much', 1824), ('also', 1757), ('characters', 1735), ('get', 1724), ('character', 1703), ('two', 1643), ('first', 1588), ('see', 1557), ('way', 1515), ('well', 1511), ('make', 1418), ('really', 1407), ('little', 1351), ('life', 1334), ('plot', 1288), ('people', 1269), ('could', 1248), ('bad', 1248), ('scene', 1241), ('movies', 1238), ('never', 1201), ('best', 1179), ('new', 1140), ('scenes', 1135), ('man', 1131), ('many', 1130), ('doesnt', 1118), ('know', 1092), ('dont', 1086), ('hes', 1024), ('great', 1014), ('another', 992), ('action', 985), ('love', 977), ('us', 967), ('go', 952), ('director', 948), ('end', 946), ('something', 945), ('still', 936)]
```

我们可以逐步浏览词汇表并删除所有发生率较低的单词，例如仅在所有评论中使用一次或两次。

例如，以下代码段将仅检索在所有评论中出现 2 次或更多次的代币。

```py
# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
```

使用此添加运行上面的示例表明，词汇量大小略大于其大小的一半，从 43,476 到 25,767 个单词。

```py
25767
```

最后，可以将词汇表保存到名为 vocab.txt 的新文件中，以后我们可以加载并使用它来过滤电影评论，然后再对其进行编码以进行建模。我们定义了一个名为 save_list（）的新函数，它将词汇表保存到文件中，每个文件只有一个单词。

例如：

```py
# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

在词汇表上运行最小出现过滤器并将其保存到文件，您现在应该有一个名为`vocab.txt`的新文件，其中只包含我们感兴趣的词。

文件中的单词顺序会有所不同，但应如下所示：

```py
aberdeen
dupe
burt
libido
hamlet
arlene
available
corners
web
columbia
...
```

我们现在已准备好从准备建模的评论中提取特征。

## 词袋表示

在本节中，我们将了解如何将每个评论转换为我们可以为多层感知器模型提供的表示。

词袋模型是一种从文本中提取特征的方法，因此文本输入可以与神经网络等机器学习算法一起使用。

每个文档（在这种情况下是评论）被转换为向量表示。表示文档的向量中的项目数对应于词汇表中的单词数。词汇量越大，向量表示越长，因此在前一部分中对较小词汇表的偏好。

对文档中的单词进行评分，并将分数放在表示中的相应位置。我们将在下一节中介绍不同的单词评分方法。

在本节中，我们关注的是将评论转换为准备用于训练第一神经网络模型的向量。

本节分为两个步骤：

1.  将评论转换为代币行。
2.  使用词袋模型表示编码评论。

### 对令牌行的评论

在我们将评论转换为向量进行建模之前，我们必须首先清理它们。

这涉及加载它们，执行上面开发的清洁操作，过滤掉不在所选词汇表中的单词，并将剩余的标记转换成准备编码的单个字符串或行。

首先，我们需要一个函数来准备一个文档。下面列出了函数 _doc_to_line（）_，它将加载文档，清理它，过滤掉不在词汇表中的标记，然后将文档作为一串空白分隔的标记返回。

```py
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)
```

接下来，我们需要一个函数来处理目录中的所有文档（例如'`pos`'和'`neg`'）将文档转换为行。

下面列出了`process_docs()`函数，该函数执行此操作，期望将目录名称和词汇表设置为输入参数并返回已处理文档的列表。

```py
# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines
```

最后，我们需要加载词汇表并将其转换为用于清理评论的集合。

```py
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

我们可以将所有这些放在一起，重复使用前面部分中开发的加载和清理功能。

下面列出了完整的示例，演示了如何从训练数据集准备正面和负面评论。

```py
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
negative_lines = process_docs('txt_sentoken/neg', vocab)
# summarize what we have
print(len(positive_lines), len(negative_lines))
```

### 电影评论到词袋向量

我们将使用 Keras API 将评论转换为编码的文档向量。

Keras 提供 [Tokenize 类](https://keras.io/preprocessing/text/#tokenizer)，它可以执行我们在上一节中处理的一些清理和词汇定义任务。

最好自己做这件事，以确切知道做了什么以及为什么做。然而，Tokenizer 类很方便，很容易将文档转换为编码向量。

首先，必须创建 Tokenizer，然后适合训练数据集中的文本文档。

在这种情况下，这些是前一节中开发的`positive_lines`和`negative_lines`数组的聚合。

```py
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = positive_lines + negative_lines
tokenizer.fit_on_texts(docs)
```

此过程确定将词汇表转换为具有 25,768 个元素的固定长度向量的一致方式，这是词汇表文件`vocab.txt`中的单词总数。

接下来，可以使用 Tokenizer 通过调用`texts_to_matrix()`对文档进行编码。该函数接受要编码的文档列表和编码模式，这是用于对文档中的单词进行评分的方法。在这里，我们指定'`freq`'根据文档中的频率对单词进行评分。

这可用于编码训练数据，例如：

```py
# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)
```

这将对训练数据集中的所有正面和负面评论进行编码，并将所得矩阵的形状打印为 1,800 个文档，每个文档的长度为 25,768 个元素。它可以用作模型的训练数据。

```py
(1800, 25768)
```

我们可以用类似的方式对测试数据进行编码。

首先，需要修改上一节中的`process_docs()`函数，以仅处理测试数据集中的评论，而不是训练数据集。

我们通过添加`is_trian`参数并使用它来决定要跳过哪些评论文件名来支持加载训练和测试数据集。

```py
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines
```

接下来，我们可以像在训练集中一样，在测试集中加载和编码正面和负面评论。

```py
...
# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtest.shape)
```

我们可以将所有这些放在一个例子中。

```py
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtest.shape)
```

运行该示例分别打印编码的训练数据集和测试数据集的形状，分别具有 1,800 和 200 个文档，每个文档具有相同大小的编码词汇表（向量长度）。

```py
(1800, 25768)
(200, 25768)
```

## 情感分析模型

在本节中，我们将开发多层感知器（MLP）模型，将编码文档分类为正面或负面。

模型将是简单的前馈网络模型，在 Keras 深度学习库中具有称为`Dense`的完全连接层。

本节分为 3 个部分：

1.  第一个情感分析模型
2.  比较单词评分模式
3.  预测新的评论

### 第一情感分析模型

我们可以开发一个简单的 MLP 模型来预测编码评论的情感。

模型将具有一个输入层，该输入层等于词汇表中的单词数，进而是输入文档的长度。

我们可以将它存储在一个名为`n_words`的新变量中，如下所示：

```py
n_words = Xtest.shape[1]
```

我们还需要所有训练和测试审核数据的类标签。我们确定性地加载并编码了这些评论（否定，然后是正面），因此我们可以直接指定标签，如下所示：

```py
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
```

我们现在可以定义网络。

发现所有模型配置的试验和错误非常少，不应该考虑针对此问题进行调整。

我们将使用具有 50 个神经元和整流线性激活函数的单个隐藏层。输出层是具有 S 形激活函数的单个神经元，用于预测 0 为阴性，1 为阳性评价。

将使用梯度下降的有效 [Adam 实现](http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)和二元交叉熵损失函数来训练网络，适合于二元分类问题。在训练和评估模型时，我们将跟踪准确率。

```py
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

接下来，我们可以将模型拟合到训练数据上;在这种情况下，该模型很小，很容易适应 50 个时代。

```py
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
```

最后，一旦训练了模型，我们就可以通过在测试数据集中做出预测并打印精度来评估其表现。

```py
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

下面列出了完整的示例。

```py
from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)
# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

运行该示例，我们可以看到该模型很容易适应 50 个时期内的训练数据，实现 100％的准确率。

在测试数据集上评估模型，我们可以看到模型表现良好，达到 90％以上的准确率，完全在原始论文中看到的 80-80 年代中期。

虽然，重要的是要注意这不是一个苹果对苹果的比较，因为原始论文使用 10 倍交叉验证来估计模型技能而不是单个训练/测试分裂。

```py
...
Epoch 46/50
0s - loss: 0.0167 - acc: 1.0000
Epoch 47/50
0s - loss: 0.0157 - acc: 1.0000
Epoch 48/50
0s - loss: 0.0148 - acc: 1.0000
Epoch 49/50
0s - loss: 0.0140 - acc: 1.0000
Epoch 50/50
0s - loss: 0.0132 - acc: 1.0000

Test Accuracy: 91.000000
```

接下来，让我们看看为词袋模型测试不同的单词评分方法。

### 比较单词评分方法

Keras API 中 Tokenizer 的`texts_to_matrix()`函数提供了 4 种不同的评分方法;他们是：

*   “_ 二元 _”其中单词被标记为存在（1）或不存在（0）。
*   “`count`”将每个单词的出现次数标记为整数。
*   “`tfidf`”每个单词根据其频率进行评分，其中所有文档中共同的单词都会受到惩罚。
*   “`freq`”根据文档中出现的频率对单词进行评分。

我们可以使用 4 种支持的单词评分模式中的每一种来评估上一节中开发的模型的技能。

这首先涉及开发一种函数，以基于所选择的评分模型来创建所加载文档的编码。该函数创建标记器，将其拟合到训练文档上，然后使用所选模型创建训练和测试编码。函数`prepare_data()`在给定训练和测试文档列表的情况下实现此行为。

```py
# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest
```

我们还需要一个函数来评估给定特定数据编码的 MLP。

因为神经网络是随机的，当相同的模型适合相同的数据时，它们可以产生不同的结果。这主要是因为随机初始权重和小批量梯度下降期间的模式混洗。这意味着任何一个模型评分都是不可靠的，我们应该根据多次运行的平均值来估计模型技能。

下面的函数名为 _evaluate_mode（）_，它通过在训练上训练它并在测试集上估计技能 30 次来获取编码文档并评估 MLP，并返回所有这些精度得分的列表。运行。

```py
# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 30
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(50, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=50, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores
```

我们现在准备评估 4 种不同单词评分方法的表现。

将所有这些结合在一起，下面列出了完整的示例。

```py
from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from matplotlib import pyplot

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 30
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(50, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=50, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores

# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_lines + positive_lines
# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_lines + positive_lines
# prepare labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

modes = ['binary', 'count', 'tfidf', 'freq']
results = DataFrame()
for mode in modes:
	# prepare data for mode
	Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
	# evaluate model on data for mode
	results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
# summarize results
print(results.describe())
# plot results
results.boxplot()
pyplot.show()
```

运行该示例可能需要一段时间（在具有 CPU 的现代硬件上大约一个小时，而不是 GPU）。

在运行结束时，提供了每个单词评分方法的摘要统计，总结了每个模式 30 次运行中每个模型技能得分的分布。

我们可以看到'`freq`'和'_ 二元 _'方法的平均得分似乎优于'_ 计数 _'和'`tfidf`'。

```py
          binary     count      tfidf       freq
count  30.000000  30.00000  30.000000  30.000000
mean    0.915833   0.88900   0.856333   0.908167
std     0.009010   0.01012   0.013126   0.002451
min     0.900000   0.86500   0.830000   0.905000
25%     0.906250   0.88500   0.850000   0.905000
50%     0.915000   0.89000   0.857500   0.910000
75%     0.920000   0.89500   0.865000   0.910000
max     0.935000   0.90500   0.885000   0.910000
```

还给出了结果的盒子和须状图，总结了每种配置的准确度分布。

我们可以看到'freq'配置的分布是紧张的，这是令人鼓舞的，因为它也表现良好。此外，我们可以看到“二元”通过适度的传播实现了最佳结果，可能是此数据集的首选方法。

![Box and Whisker Plot for Model Accuracy with Different Word Scoring Methods](img/41c1d5a636ede307ab02540acd449347.jpg)

不同单词评分方法的模型准确率框和晶须图

## 预测新评论

最后，我们可以使用最终模型对新的文本评论做出预测。

这就是我们首先想要模型的原因。

预测新评论的情感涉及遵循用于准备测试数据的相同步骤。具体来说，加载文本，清理文档，通过所选词汇过滤标记，将剩余标记转换为线，使用 Tokenizer 对其进行编码，以及做出预测。

我们可以通过调用`predict()`直接使用拟合模型预测类值，该值将返回一个值，该值可以舍入为 0 的整数表示负面评论，1 表示正面评论。

所有这些步骤都可以放入一个名为`predict_sentiment()`的新函数中，该函数需要复习文本，词汇表，标记符和拟合模型，如下所示：

```py
# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab, tokenizer, model):
	# clean
	tokens = clean_doc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])
```

我们现在可以对新的评论文本做出预测。

以下是使用频率词评分模式使用上面开发的简单 MLP 进行明确肯定和明显否定评论的示例。

```py
# test positive text
text = 'Best movie ever!'
print(predict_sentiment(text, vocab, tokenizer, model))
# test negative text
text = 'This is a bad movie.'
print(predict_sentiment(text, vocab, tokenizer, model))
```

正确运行示例会对这些评论进行分类。

```py
1
0
```

理想情况下，我们将模型放在所有可用数据（训练和测试）上以创建[最终模型](http://machinelearningmastery.com/train-final-machine-learning-model/)并将模型和标记器保存到文件中，以便可以在新软件中加载和使用它们。

## 扩展

如果您希望从本教程中获得更多信息，本节列出了一些扩展。

*   **管理词汇**。使用更大或更小的词汇进行探索。也许你可以通过一组较小的单词获得更好的表现。
*   **调整网络拓扑**。探索其他网络拓扑，例如更深或更广的网络。也许您可以通过更适合的网络获得更好的表现。
*   **使用正则化**。探索正规化技术的使用，例如丢失。也许您可以延迟模型的收敛并实现更好的测试集表现。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 数据集

*   [电影评论数据](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
*   [一种感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。
*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （。tgz）
*   数据集自述文件 [v2.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt) 和 [v1.1](http://www.cs.cornell.edu/people/pabo/movie-review-data/README.1.1) 。

### 蜜蜂

*   [nltk.tokenize 包 API](http://www.nltk.org/api/nltk.tokenize.html)
*   [第 2 章，访问文本语料库和词汇资源](http://www.nltk.org/book/ch02.html)
*   [os API 其他操作系统接口](https://docs.python.org/3/library/os.html)
*   [集合 API - 容器数据类型](https://docs.python.org/3/library/collections.html)
*   [Tokenizer Keras API](https://keras.io/preprocessing/text/#tokenizer)

## 摘要

在本教程中，您了解了如何开发一个词袋模型来预测电影评论的情感。

具体来说，你学到了：

*   如何准备评论文本数据以便使用受限词汇表进行建模。
*   如何使用词袋模型来准备训练和测试数据。
*   如何开发多层 Perceptron 词袋模型并使用它来预测新的评论文本数据。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。