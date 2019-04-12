# 如何开发一种预测电影评论情感的词嵌入模型

> 原文： [https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)

#### 开发一个深度学习模型，使用 Keras 自动将电影评论
分类为 Python 中的正面或负面，一步一步。

单词嵌入是用于表示文本的技术，其中具有相似含义的不同单词具有类似的实值向量表示。

它们是一项重大突破，它使神经网络模型在一系列具有挑战性的自然语言处理问题上表现出色。

在本教程中，您将了解如何为神经网络开发单词嵌入模型以对电影评论进行分类。

完成本教程后，您将了解：

*   如何使用深度学习方法准备电影评论文本数据进行分类。
*   如何学习嵌入词作为拟合深度学习模型的一部分。
*   如何学习独立的单词嵌入以及如何在神经网络模型中使用预先训练的嵌入。

让我们开始吧。

**注**：摘录自：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。
看一下，如果你想要更多的分步教程，在使用文本数据时充分利用深度学习方法。

![How to Develop a Word Embedding Model for Predicting Movie Review Sentiment](img/eb22780eed5923389f03e367a305ea48.jpg)

如何开发用于预测电影评论情绪的词嵌入模型
照片由 [Katrina Br *？＃*！@ nd](https://www.flickr.com/photos/fuzzyblue/6351564408/) ，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  电影评论数据集
2.  数据准备
3.  火车嵌入层
4.  训练 word2vec 嵌入
5.  使用预先训练的嵌入

### Python 环境

本教程假设您安装了 Python SciPy 环境，理想情况下使用 Python 3。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.2 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅本教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

本教程不需要 GPU，但您可以在 Amazon Web Services 上以低成本方式访问 GPU。在本教程中学习如何：

*   [如何设置 Amazon AWS EC2 GPU 以培训 Keras 深度学习模型（循序渐进）](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

让我们潜入。

## 1.电影评论数据集

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

该数据已用于一些相关的自然语言处理任务。对于分类，机器学习模型（例如支持向量机）对数据的性能在高 70％到低 80％（例如 78％-82％）的范围内。

更复杂的数据准备可以看到高达 86％的结果，交叉验证 10 倍。如果我们想在现代方法的实验中使用这个数据集，这给了我们 80 年代中期的球场。

> ...根据下游极性分类器的选择，我们可以实现高度统计上的显着改善（从 82.8％到 86.4％）

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

您可以从此处下载数据集：

*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （review_polarity.tar.gz，3MB）

解压缩文件后，您将有一个名为“ _txt_sentoken_ ”的目录，其中包含两个子目录，其中包含文本“ _neg_ ”和“ _pos_ ”的负数和积极的评论。对于每个 neg 和 pos，每个文件存储一个评论，命名约定为 cv000 到 cv999。

接下来，我们来看看加载和准备文本数据。

## 2.数据准备

在本节中，我们将看看 3 件事：

1.  将数据分成训练和测试集。
2.  加载和清理数据以删除标点符号和数字。
3.  定义首选词汇的词汇。

### 分为火车和测试装置

我们假装我们正在开发一种系统，可以预测文本电影评论的情绪是积极的还是消极的。

这意味着在开发模型之后，我们需要对新的文本评论进行预测。这将要求对这些新评论执行所有相同的数据准备，就像对模型的训练数据执行一样。

我们将通过在任何数据准备之前拆分训练和测试数据集来确保将此约束纳入我们模型的评估中。这意味着在准备用于训练模型的数据时，测试集中的数据中的任何知识可以帮助我们更好地准备数据（例如，所使用的单词）。

话虽如此，我们将使用最近 100 次正面评论和最后 100 次负面评论作为测试集（100 条评论），其余 1,800 条评论作为训练数据集。

这是 90％的列车，10％的数据分割。

通过使用评论的文件名可以轻松实现拆分，其中评论为 000 至 899 的评论用于培训数据，而评论为 900 以上的评论用于测试。

### 装载和清洁评论

文本数据已经非常干净;没有太多准备工作。

如果您不熟悉清理文本数据，请参阅此帖子：

*   [如何使用 Python 清理机器学习文本](https://machinelearningmastery.com/clean-text-machine-learning-python/)

如果没有太多陷入细节，我们将使用以下方式准备数据：

*   在白色空间的分裂标记。
*   从单词中删除所有标点符号。
*   删除所有不完全由字母字符组成的单词。
*   删除所有已知停用词的单词。
*   删除长度为＆lt; = 1 个字符的所有单词。

我们可以将所有这些步骤放入一个名为 _clean_doc（）_ 的函数中，该函数将从文件加载的原始文本作为参数，并返回已清理的标记列表。我们还可以定义一个函数 _load_doc（）_，它从文件中加载文件，以便与 _clean_doc（）_ 函数一起使用。

下面列出了清理第一次正面评价的示例。

```
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

我们可能想要探索更多清洁步骤，并将其留作进一步练习。

我很想知道你能想出什么。
最后在评论中发布您的方法和结果。

```
...
'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']
```

### 定义词汇表

在使用词袋或嵌入模型时，定义已知单词的词汇表很重要。

单词越多，文档的表示越大，因此将单词限制为仅被认为具有预测性的单词是很重要的。这很难事先知道，并且通常重要的是测试关于如何构建有用词汇的不同假设。

我们已经看到了如何从上一节中的词汇表中删除标点符号和数字。我们可以对所有文档重复此操作，并构建一组所有已知单词。

我们可以开发一个词汇表作为计数器，这是一个单词及其计数的字典映射，允许我们轻松更新和查询。

每个文档都可以添加到计数器（一个名为 _add_doc_to_vocab（）_ 的新函数），我们可以跳过负目录中的所有评论，然后是肯定目录（一个名为 _process_docs 的新函数） （）_）。

下面列出了完整的示例。

```
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
def process_docs(directory, vocab, is_trian):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
```

运行该示例表明我们的词汇量为 43,476 个单词。

我们还可以看到电影评论中前 50 个最常用单词的样本。

请注意，此词汇表仅基于训练数据集中的那些评论构建。

```
44276
[('film', 7983), ('one', 4946), ('movie', 4826), ('like', 3201), ('even', 2262), ('good', 2080), ('time', 2041), ('story', 1907), ('films', 1873), ('would', 1844), ('much', 1824), ('also', 1757), ('characters', 1735), ('get', 1724), ('character', 1703), ('two', 1643), ('first', 1588), ('see', 1557), ('way', 1515), ('well', 1511), ('make', 1418), ('really', 1407), ('little', 1351), ('life', 1334), ('plot', 1288), ('people', 1269), ('could', 1248), ('bad', 1248), ('scene', 1241), ('movies', 1238), ('never', 1201), ('best', 1179), ('new', 1140), ('scenes', 1135), ('man', 1131), ('many', 1130), ('doesnt', 1118), ('know', 1092), ('dont', 1086), ('hes', 1024), ('great', 1014), ('another', 992), ('action', 985), ('love', 977), ('us', 967), ('go', 952), ('director', 948), ('end', 946), ('something', 945), ('still', 936)]
```

我们可以逐步浏览词汇表并删除所有发生率较低的单词，例如仅在所有评论中使用一次或两次。

例如，以下代码段将仅检索在所有评论中出现 2 次或更多次的令牌。

```
# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
```

使用此添加运行上面的示例表明，词汇量大小略大于其大小的一半（从 43,476 到 25,767 个单词）。

```
25767
```

最后，词汇表可以保存到一个名为 _vocab.txt_ 的新文件中，以后我们可以加载并使用它来过滤电影评论，然后再编码进行建模。我们定义了一个名为 _save_list（）_ 的新函数，它将词汇表保存到文件中，每个文件只有一个单词。

例如：

```
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

在词汇表上运行最小出现过滤器并将其保存到文件，您现在应该有一个名为 _vocab.txt_ 的新文件，其中只包含我们感兴趣的词。

文件中的单词顺序会有所不同，但应如下所示：

```
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

我们现在准备从评论中查看学习功能。

## 3.训练嵌入层

在本节中，我们将在分类问题上训练神经网络时学习嵌入一词。

单词嵌入是表示文本的一种方式，其中词汇表中的每个单词由高维空间中的实值向量表示。以这样的方式学习向量：具有相似含义的单词在向量空间中具有相似的表示（在向量空间中接近）。对于文本而言，这是一种更具表现力的表达，而不是像词袋这样的经典方法，其中单词或标记之间的关系被忽略，或者在 bigram 和 trigram 方法中被强制使用。

在训练神经网络时可以学习单词的实值向量表示。我们可以使用[嵌入层](https://keras.io/layers/embeddings/)在 Keras 深度学习库中完成此操作。

如果您不熟悉单词嵌入，请参阅帖子：

*   [什么是 Word 嵌入文本？](https://machinelearningmastery.com/what-are-word-embeddings/)

如果您不熟悉 Keras 中的字嵌入图层，请参阅帖子：

*   [如何使用 Keras 深入学习使用 Word 嵌入层](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

第一步是加载词汇表。我们将用它来过滤我们不感兴趣的电影评论中的单词。

如果你已完成上一节，你应该有一个名为' _vocab.txt_ '的本地文件，每行一个单词。我们可以加载该文件并构建一个词汇表作为检查令牌有效性的集合。

```
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

接下来，我们需要加载所有培训数据电影评论。为此，我们可以调整上一节中的 _process_docs（）_ 来加载文档，清理它们，并将它们作为字符串列表返回，每个字符串有一个文档。我们希望每个文档都是一个字符串，以便以后简单编码为整数序列。

清理文档涉及根据空白区域拆分每个评论，删除标点符号，然后过滤掉不在词汇表中的所有标记。

更新的 _clean_doc（）_ 功能如下所示。

```
# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens
```

更新的 _process_docs（）_ 然后可以为' _pos_ '和' _neg_ '目录中的每个文档调用 _clean_doc（）_ 在我们的训练数据集中。

```
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs
```

下一步是将每个文档编码为整数序列。

Keras 嵌入层需要整数输入，其中每个整数映射到单个标记，该标记在嵌入中具有特定的实值向量表示。这些向量在训练开始时是随机的，但在训练期间对网络有意义。

我们可以使用 Keras API 中的 [Tokenizer](https://keras.io/preprocessing/text/#tokenizer) 类将训练文档编码为整数序列。

首先，我们必须构造一个类的实例，然后在训练数据集中的所有文档上训练它。在这种情况下，它开发了训练数据集中所有标记的词汇表，并开发了从词汇表中的单词到唯一整数的一致映射。我们可以使用我们的词汇表文件轻松地开发此映射。

```
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
```

现在已经准备好了单词到整数的映射，我们可以使用它来对训练数据集中的评论进行编码。我们可以通过调用 Tokenizer 上的 _texts_to_sequences（）_ 函数来实现。

```
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
```

我们还需要确保所有文档具有相同的长度。

这是 Keras 对高效计算的要求。我们可以将评论截断为最小尺寸或零填充（具有值'0'的填充）评论到最大长度，或者某些混合。在这种情况下，我们将所有评论填充到训练数据集中最长评论的长度。

首先，我们可以使用训练数据集上的 _max（）_ 函数找到最长的评论并获取其长度。然后，我们可以调用 Keras 函数 _pad_sequences（）_，通过在末尾添加 0 值将序列填充到最大长度。

```
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
```

最后，我们可以定义训练数据集的类标签，以适应监督的神经网络模型来预测评论的情绪。

```
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
```

然后我们可以对测试数据集进行编码和填充，稍后需要在我们训练之后评估模型。

```
# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
```

我们现在准备定义我们的神经网络模型。

该模型将使用嵌入层作为第一个隐藏层。嵌入需要规范词汇量大小，实值向量空间的大小以及输入文档的最大长度。

词汇量大小是我们词汇表中的单词总数，加上一个未知单词。这可以是用于对文档进行整数编码的标记器内的词汇集长度或词汇大小，例如：

```
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
```

我们将使用 100 维向量空间，但您可以尝试其他值，例如 50 或 150.最后，最大文档长度在填充期间使用的 _max_length_ 变量中计算。

下面列出了完整的模型定义，包括嵌入层。

我们使用卷积神经网络（CNN），因为它们已经证明在文档分类问题上是成功的。保守的 CNN 配置与 32 个滤波器（用于处理字的并行字段）和具有整流线性（'relu'）激活功能的 8 的内核大小一起使用。接下来是一个池化层，它将卷积层的输出减少一半。

接下来，将来自模型的 CNN 部分的 2D 输出展平为一个长 2D 矢量，以表示由 CNN 提取的“特征”。模型的后端是标准的多层感知器层，用于解释 CNN 功能。输出层使用 sigmoid 激活函数为评论中的消极和积极情绪输出介于 0 和 1 之间的值。

有关文本分类的有效深度学习模型配置的更多建议，请参阅帖子：

*   [深度学习文档分类的最佳实践](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

```
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

仅运行此片段可提供已定义网络的摘要。

我们可以看到嵌入层需要长度为 442 个单词的文档作为输入，并将文档中的每个单词编码为 100 个元素向量。

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 442, 100)          2576800
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 435, 32)           25632
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 217, 32)           0
_________________________________________________________________
flatten_1 (Flatten)          (None, 6944)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                69450
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,671,893
Trainable params: 2,671,893
Non-trainable params: 0
_________________________________________________________________
```

接下来，我们使网络适应训练数据。

我们使用二元交叉熵损失函数，因为我们正在学习的问题是二元分类问题。使用随机梯度下降的高效 Adam 实现，除了训练期间的损失之外，我们还跟踪准确性。该模型训练 10 个时期，或 10 次通过训练数据。

通过一些试验和错误找到了网络配置和培训计划，但对于此问题并不是最佳选择。如果您可以使用其他配置获得更好的结果，请告诉我们。

```
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
```

在拟合模型之后，在测试数据集上对其进行评估。此数据集包含我们以前从未见过的单词和在培训期间未看到的评论。

```
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

我们可以将所有这些结合在一起。

完整的代码清单如下。

```
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

运行该示例在每个训练时期结束时打印损失和准确性。我们可以看到该模型很快就能在训练数据集上实现 100％的准确性。

在运行结束时，模型在测试数据集上达到了 84.5％的准确度，这是一个很好的分数。

鉴于神经网络的随机性，您的具体结果会有所不同。考虑运行几次示例并将平均分数作为模型的技能。

```
...
Epoch 6/10
2s - loss: 0.0013 - acc: 1.0000
Epoch 7/10
2s - loss: 8.4573e-04 - acc: 1.0000
Epoch 8/10
2s - loss: 5.8323e-04 - acc: 1.0000
Epoch 9/10
2s - loss: 4.3155e-04 - acc: 1.0000
Epoch 10/10
2s - loss: 3.3083e-04 - acc: 1.0000
Test Accuracy: 84.500000
```

我们刚刚看到一个例子，说明我们如何学习嵌入字作为拟合神经网络模型的一部分。

接下来，让我们看看如何有效地学习我们以后可以在神经网络中使用的独立嵌入。

## 4.训练 word2vec 嵌入

在本节中，我们将了解如何使用名为 word2vec 的高效算法学习独立的单词嵌入。

学习单词嵌入作为网络一部分的缺点是它可能非常慢，特别是对于非常大的文本数据集。

word2vec 算法是一种以独立方式从文本语料库中学习单词嵌入的方法。该方法的好处是它可以在空间和时间复杂性方面非常有效地产生高质量的字嵌入。

第一步是准备好文档以便学习嵌入。

这涉及与前一节相同的数据清理步骤，即通过空白区域分割文档，删除标点符号，以及过滤掉不在词汇表中的标记。

word2vec 算法逐句处理文档。这意味着我们将在清洁期间保留基于句子的结构。

我们开始像以前一样加载词汇表。

```
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

接下来，我们定义一个名为 _doc_to_clean_lines（）_ 的函数来逐行清理已加载的文档并返回已清理行的列表。

```
# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines
```

接下来，我们调整 process_docs（）函数来加载和清理文件夹中的所有文档，并返回所有文档行的列表。

该函数的结果将是 word2vec 模型的训练数据。

```
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
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines
```

然后我们可以加载所有训练数据并将其转换为一长串的“句子”（令牌列表），以便为 word2vec 模型拟合。

```
# load training data
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))
```

我们将使用 Gensim Python 库中提供的 word2vec 实现。具体是 [Word2Vec 类](https://radimrehurek.com/gensim/models/word2vec.html)。

有关使用 Gensim 培训独立单词的更多信息，请参阅帖子：

*   [如何使用 Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/) 在 Python 中开发 Word 嵌入

在构造类时，该模型是合适的。我们从训练数据中传入干净的句子列表，然后指定嵌入向量空间的大小（我们再次使用 100），在学习如何在训练句子中嵌入每个单词时要查看的相邻单词的数量（我们使用 5 个邻居），在拟合模型时使用的线程数（我们使用 8，但是如果你有更多或更少的 CPU 核心则更改它），以及词汇表中要考虑的单词的最小出现次数（我们将其设置为 1 因为我们已经准备好了词汇表）。

在模型拟合之后，我们打印学习词汇的大小，这应该与我们在 25,767 个令牌的 vocab.txt 中的词汇量相匹配。

```
# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
```

最后，我们使用模型的' _wv_ '（字向量）属性上的 [save_word2vec_format（）](https://radimrehurek.com/gensim/models/keyedvectors.html)将学习的嵌入向量保存到文件中。嵌入以 ASCII 格式保存，每行一个字和矢量。

下面列出了完整的示例。

```
from string import punctuation
from os import listdir
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

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

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
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))

# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
```

运行该示例从训练数据中加载 58,109 个句子，并为 25,767 个单词的词汇表创建嵌入。

您现在应该在当前工作目录中有一个带有学习向量的文件'embedding_word2vec.txt'。

```
Total training sentences: 58109
Vocabulary size: 25767
```

接下来，让我们看看在我们的模型中使用这些学习过的向量。

## 5.使用预先训练的嵌入

在本节中，我们将使用在非常大的文本语料库上准备的预训练的单词嵌入。

我们可以使用前一节中开发的预训练单词嵌入和之前部分开发的 CNN 模型。

第一步是将单词嵌入作为单词目录加载到向量。单词嵌入保存在包含标题行的所谓' _word2vec_ '格式中。加载嵌入时我们将跳过此标题行。

下面名为 _load_embedding（）_ 的函数加载嵌入并返回映射到 NumPy 格式的向量的单词目录。

```
# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding
```

现在我们已经在内存中拥有了所有向量，我们可以按照匹配 Keras Tokenizer 准备的整数编码的方式对它们进行排序。

回想一下，我们在将审阅文档传递给嵌入层之前对它们进行整数编码。整数映射到嵌入层中特定向量的索引。因此，重要的是我们将向量放置在嵌入层中，使得编码的单词映射到正确的向量。

下面定义了一个函数 _get_weight_matrix（）_，它将加载的嵌入和 tokenizer.word_index 词汇表作为参数，并返回一个矩阵，其中的单词 vector 位于正确的位置。

```
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix
```

现在我们可以使用这些函数为我们的模型创建新的嵌入层。

```
...
# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
```

请注意，准备好的权重矩阵 _embedding_vectors_ 作为参数传递给新的嵌入层，并且我们将'_ 可训练 _'参数设置为' _False_ '以确保网络不会尝试将预先学习的向量作为训练网络的一部分。

我们现在可以将此图层添加到我们的模型中。我们还有一个稍微不同的模型配置，在 CNN 模型中有更多的过滤器（128），以及在开发 word2vec 嵌入时匹配用作邻居的 5 个单词的内核。最后，简化了模型的后端。

```
# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

通过一些试验和错误发现了这些变化。

完整的代码清单如下。

```
from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

运行该示例显示性能未得到改善。

事实上，表现差得多。结果表明训练数据集是成功学习的，但对测试数据集的评估非常差，准确度仅略高于 50％。

测试性能差的原因可能是因为选择了 word2vec 配置或选择的神经网络配置。

```
...
Epoch 6/10
2s - loss: 0.3306 - acc: 0.8778
Epoch 7/10
2s - loss: 0.2888 - acc: 0.8917
Epoch 8/10
2s - loss: 0.1878 - acc: 0.9439
Epoch 9/10
2s - loss: 0.1255 - acc: 0.9750
Epoch 10/10
2s - loss: 0.0812 - acc: 0.9928
Test Accuracy: 53.000000
```

嵌入层中的权重可以用作网络的起始点，并且在网络训练期间进行调整。我们可以通过在创建嵌入层时设置' _trainable = True_ '（默认值）来实现。

使用此更改重复实验显示略微更好的结果，但仍然很差。

我鼓励您探索嵌入和网络的备用配置，看看您是否可以做得更好。让我知道你是怎么做的。

```
...
Epoch 6/10
4s - loss: 0.0950 - acc: 0.9917
Epoch 7/10
4s - loss: 0.0355 - acc: 0.9983
Epoch 8/10
4s - loss: 0.0158 - acc: 1.0000
Epoch 9/10
4s - loss: 0.0080 - acc: 1.0000
Epoch 10/10
4s - loss: 0.0050 - acc: 1.0000
Test Accuracy: 57.500000
```

可以使用在非常大的文本数据集上准备的预训练的单词向量。

例如，Google 和 Stanford 都提供了可以下载的预训练单词向量，分别使用高效的 word2vec 和 GloVe 方法进行训练。

让我们尝试在我们的模型中使用预先训练的矢量。

您可以从斯坦福网页下载[预训练的 GloVe 载体](https://nlp.stanford.edu/projects/glove/)。具体来说，培训维基百科数据的矢量：

*   [手套.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip) （822 兆字节下载）

解压缩文件，您将找到各种不同尺寸的预先培训嵌入。我们将在文件' _glove.6B.100d.txt_ '中加载 100 维版本

Glove 文件不包含头文件，因此在将嵌入加载到内存时我们不需要跳过第一行。下面列出了更新的 _load_embedding（）_ 功能。

```
# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding
```

加载的嵌入可能不包含我们选择的词汇表中的所有单词。因此，在创建嵌入权重矩阵时，我们需要跳过在加载的 GloVe 数据中没有相应向量的单词。以下是 _get_weight_matrix（）_ 功能的更新，更具防御性的版本。

```
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix
```

我们现在可以像以前一样加载 GloVe 嵌入并创建嵌入层。

```
# load embedding from file
raw_embedding = load_embedding('glove.6B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
```

我们将使用与以前相同的模型。

下面列出了完整的示例。

```
from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('glove.6B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

运行该示例显示了更好的性能。

同样，训练数据集很容易学习，模型在测试数据集上达到 76％的准确度。这很好，但不如使用学习的嵌入层。

这可能是由于在更多数据上训练的更高质量的矢量和/或使用稍微不同的训练过程的原因。

鉴于神经网络的随机性，您的具体结果可能会有所不同。尝试运行几次示例。

```
...
Epoch 6/10
2s - loss: 0.0278 - acc: 1.0000
Epoch 7/10
2s - loss: 0.0174 - acc: 1.0000
Epoch 8/10
2s - loss: 0.0117 - acc: 1.0000
Epoch 9/10
2s - loss: 0.0086 - acc: 1.0000
Epoch 10/10
2s - loss: 0.0068 - acc: 1.0000
Test Accuracy: 76.000000
```

在这种情况下，似乎学习嵌入作为学习任务的一部分可能是比使用专门训练的嵌入或更一般的预训练嵌入更好的方向。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 数据集

*   [电影评论数据](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
*   [一种感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。
*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （。tgz）
*   数据集自述文件 [v2.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt) 和 [v1.1](http://www.cs.cornell.edu/people/pabo/movie-review-data/README.1.1) 。

### 蜜蜂

*   [集合 API - 容器数据类型](https://docs.python.org/3/library/collections.html)
*   [Tokenizer Keras API](https://keras.io/preprocessing/text/#tokenizer)
*   [嵌入 Keras API](https://keras.io/layers/embeddings/)
*   [Gensim Word2Vec API](https://radimrehurek.com/gensim/models/word2vec.html)
*   [Gensim WordVector API](https://radimrehurek.com/gensim/models/keyedvectors.html)

### 嵌入方法

*   Google 代码上的 [word2vec](https://code.google.com/archive/p/word2vec/)
*   [斯坦福大学](https://nlp.stanford.edu/projects/glove/)

### 相关文章

*   [在 Keras 模型中使用预训练的字嵌入](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)，2016。
*   [在 TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) ，2015 年实施 CNN 进行文本分类。

## 摘要

在本教程中，您了解了如何为电影评论的分类开发单词嵌入。

具体来说，你学到了：

*   如何使用深度学习方法准备电影评论文本数据进行分类。
*   如何学习嵌入词作为拟合深度学习模型的一部分。
*   如何学习独立的单词嵌入以及如何在神经网络模型中使用预先训练的嵌入。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。

**注**：这篇文章摘录自：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。看一下，如果您想要在使用文本数据时获得有关深入学习方法的更多分步教程。