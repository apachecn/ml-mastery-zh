# 如何开发用于情感分析的 N-gram 多通道卷积神经网络

> 原文： [https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/)

用于文本分类和情感分析的标准深度学习模型使用单词嵌入层和一维卷积神经网络。

可以通过使用多个并行卷积神经网络来扩展模型，该网络使用不同的内核大小读取源文档。实际上，这为文本创建了一个多通道卷积神经网络，用于读取具有不同 n-gram 大小（单词组）的文本。

在本教程中，您将了解如何开发一个多通道卷积神经网络，用于文本电影评论数据的情感预测。

完成本教程后，您将了解：

*   如何准备电影评论文本数据进行建模。
*   如何为 Keras 中的文本开发多通道卷积神经网络。
*   如何评估看不见的电影评论数据的拟合模型。

让我们开始吧。

*   **2018 年 2 月更新**：小代码更改以反映 Keras 2.1.3 API 中的更改。

![How to Develop an N-gram Multichannel Convolutional Neural Network for Sentiment Analysis](img/d19e490e849082fca1f80af6ce6a80e3.jpg)

如何开发用于情感分析的 N-gram 多通道卷积神经网络
[Ed Dunens](https://www.flickr.com/photos/blachswan/32732882104/) 的照片，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  电影评论数据集
2.  数据准备
3.  开发多渠道模型
4.  评估模型

### Python 环境

本教程假定您已安装 Python 3 SciPy 环境。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 电影评论数据集

电影评论数据是 Bo Pang 和 Lillian Lee 在 21 世纪初从 imdb.com 网站上检索到的电影评论的集合。收集的评论作为他们自然语言处理研究的一部分。

评论最初于 2002 年发布，但更新和清理版本于 2004 年发布，称为“v2.0”。

该数据集包含 1,000 个正面和 1,000 个负面电影评论，这些评论来自 imdb.com 上托管的 rec.arts.movi​​es.reviews 新闻组的存档。作者将此数据集称为“极性数据集”。

> 我们的数据包含 2000 年之前写的 1000 份正面和 1000 份负面评论，每位作者的评论上限为 20（每位作者共 312 位）。我们将此语料库称为极性数据集。

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

数据已经有所清理;例如：

*   数据集仅包含英语评论。
*   所有文本都已转换为小写。
*   标点符号周围有空格，如句号，逗号和括号。
*   文本每行被分成一个句子。

该数据已用于一些相关的自然语言处理任务。对于分类，机器学习模型（例如支持向量机）对数据的表现在高 70％到低 80％（例如 78％-82％）的范围内。

更复杂的数据准备可以看到高达 86％的结果，交叉验证 10 倍。如果我们想在现代方法的实验中使用这个数据集，这给了我们 80 年代中期的球场。

> ...根据下游极性分类器的选择，我们可以实现高度统计上的显着改善（从 82.8％到 86.4％）

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

您可以从此处下载数据集：

*   [电影评论 Polarity Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （review_polarity.tar.gz，3MB）

解压缩文件后，您将有一个名为“ _txt_sentoken_ ”的目录，其中包含两个子目录，其中包含文本“ _neg_ ”和“ _pos_ ”的负数和积极的评论。对于每个 neg 和 pos，每个文件存储一个评论约定 _cv000_ 到 _cv999_ 。

接下来，我们来看看加载和准备文本数据。

## 数据准备

在本节中，我们将看看 3 件事：

1.  将数据分成训练和测试集。
2.  加载和清理数据以删除标点符号和数字。
3.  准备所有评论并保存到文件。

### 分为训练和测试装置

我们假装我们正在开发一种系统，可以预测文本电影评论的情感是积极的还是消极的。

这意味着在开发模型之后，我们需要对新的文本评论进行预测。这将要求对这些新评论执行所有相同的数据准备，就像对模型的训练数据执行一样。

我们将通过在任何数据准备之前拆分训练和测试数据集来确保将此约束纳入我们模型的评估中。这意味着测试集中的数据中的任何知识可以帮助我们更好地准备数据（例如，所使用的单词）在用于训练模型的数据的准备中是不可用的。

话虽如此，我们将使用最近 100 次正面评论和最后 100 次负面评论作为测试集（100 条评论），其余 1,800 条评论作为训练数据集。

这是 90％的训练，10％的数据分割。

通过使用评论的文件名可以轻松实现拆分，其中评论为 000 至 899 的评论用于训练数据，而评论为 900 以上的评论用于测试。

### 装载和清洁评论

文本数据已经非常干净;没有太多准备工作。

不会因细节问题而陷入困境，我们将按以下方式准备数据：

*   在白色空间的分裂标记。
*   从单词中删除所有标点符号。
*   删除所有不完全由字母字符组成的单词。
*   删除所有已知停用词的单词。
*   删除长度为＆lt; = 1 个字符的所有单词。

我们可以将所有这些步骤放入一个名为 _clean_doc（）_ 的函数中，该函数将从文件加载的原始文本作为参数，并返回已清理的标记列表。我们还可以定义一个函数 _load_doc（）_，它从文件中加载文件，以便与 _clean_doc（）_ 函数一起使用。下面列出了清理第一次正面评价的示例。

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

运行该示例加载并清除一个电影评论。

打印清洁评论中的标记以供审阅。

```py
...
'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']
```

### 清除所有评论并保存

我们现在可以使用该功能来清理评论并将其应用于所有评论。

为此，我们将在下面开发一个名为 _process_docs（）_ 的新函数，它将遍历目录中的所有评论，清理它们并将它们作为列表返回。

我们还将为函数添加一个参数，以指示函数是处理序列还是测试评论，这样可以过滤文件名（如上所述），并且只清理和返回所请求的那些训练或测试评论。

完整功能如下所列。

```py
# load all docs in a directory
def process_docs(directory, is_trian):
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
		tokens = clean_doc(doc)
		# add to list
		documents.append(tokens)
	return documents
```

我们可以将此功能称为负面训练评论，如下所示：

```py
negative_docs = process_docs('txt_sentoken/neg', True)
```

接下来，我们需要训练和测试文件的标签。我们知道我们有 900 份训练文件和 100 份测试文件。我们可以使用 Python 列表推导为训练和测试集的负（0）和正（1）评论创建标签。

```py
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
testY = [0 for _ in range(100)] + [1 for _ in range(100)]
```

最后，我们希望将准备好的训练和测试集保存到文件中，以便我们以后可以加载它们进行建模和模型评估。

下面命名为 _save_dataset（）_ 的函数将使用 pickle API 将给定的准备数据集（X 和 y 元素）保存到文件中。

```py
# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)
```

### 完整的例子

我们可以将所有这些数据准备步骤结合在一起。

下面列出了完整的示例。

```py
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
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
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, is_trian):
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
		tokens = clean_doc(doc)
		# add to list
		documents.append(tokens)
	return documents

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load all training reviews
negative_docs = process_docs('txt_sentoken/neg', True)
positive_docs = process_docs('txt_sentoken/pos', True)
trainX = negative_docs + positive_docs
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
save_dataset([trainX,trainy], 'train.pkl')

# load all test reviews
negative_docs = process_docs('txt_sentoken/neg', False)
positive_docs = process_docs('txt_sentoken/pos', False)
testX = negative_docs + positive_docs
testY = [0 for _ in range(100)] + [1 for _ in range(100)]
save_dataset([testX,testY], 'test.pkl')
```

运行该示例分别清除文本电影评论文档，创建标签，并分别为 _train.pkl_ 和 _test.pkl_ 中的训练和测试数据集保存准备好的数据。

现在我们准备开发我们的模型了。

## 开发多渠道模型

在本节中，我们将开发一个用于情感分析预测问题的多通道卷积神经网络。

本节分为 3 部分：

1.  编码数据
2.  定义模型。
3.  完整的例子。

### 编码数据

第一步是加载已清理的训练数据集。

可以调用以下名为 _load_dataset（）_ 的函数来加载 pickle 训练数据集。

```py
# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

trainLines, trainLabels = load_dataset('train.pkl')
```

接下来，我们必须在训练数据集上安装 Keras Tokenizer。我们将使用此标记器来定义嵌入层的词汇表，并将审阅文档编码为整数。

下面的函数 _create_tokenizer（）_ 将创建一个给定文档列表的 Tokenizer。

```py
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

我们还需要知道输入序列的最大长度作为模型的输入并将所有序列填充到固定长度。

下面的函数 _max_length（）_ 将计算训练数据集中所有评论的最大长度（单词数）。

```py
# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])
```

我们还需要知道嵌入层的词汇量大小。

这可以从准备好的 Tokenizer 计算，如下：

```py
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
```

最后，我们可以整数编码并填充干净的电影评论文本。

名为 _encode_text（）_ 的以下函数将编码和填充文本数据到最大查看长度。

```py
# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded
```

### 定义模型

文档分类的标准模型是使用嵌入层作为输入，然后是一维卷积神经网络，池化层，然后是预测输出层。

卷积层中的内核大小定义了卷积在输入文本文档中传递时要考虑的单词数，从而提供分组参数。

用于文档分类的多通道卷积神经网络涉及使用具有不同大小的内核的标准模型的多个版本。这允许一次以不同的分辨率或不同的 n-gram（单词组）处理文档，同时模型学习如何最好地整合这些解释。

Yoon Kim 在他的 2014 年题为“[用于句子分类的卷积神经网络](https://arxiv.org/abs/1408.5882)”的论文中首次描述了这种方法。

在本文中，Kim 尝试了静态和动态（更新）嵌入层，我们可以简化方法，而只关注使用不同的内核大小。

使用 Kim 的论文中的图表可以最好地理解这种方法：

![Depiction of the multiple-channel convolutional neural network for text](img/1d7144ec7b965e35ec9366cc83c40995.jpg)

描述文本的多通道卷积神经网络。
取自“用于句子分类的卷积神经网络”。

在 Keras 中，可以使用[功能 API](https://keras.io/getting-started/functional-api-guide/) 定义多输入模型。

我们将定义一个带有三个输入通道的模型，用于处理 4 克，6 克和 8 克的电影评论文本。

每个频道由以下元素组成：

*   输入层，用于定义输入序列的长度。
*   嵌入层设置为词汇表的大小和 100 维实值表示。
*   一维卷积层，具有 32 个滤波器，内核大小设置为一次读取的字数。
*   Max Pooling 层用于合并卷积层的输出。
*   展平层以将三维输出减少为二维以进行连接。

三个通道的输出连接成一个向量，并由 Dense 层和输出层处理。

下面的函数定义并返回模型。作为定义模型的一部分，将打印已定义模型的摘要，并创建模型图的图并将其保存到文件中。

```py
# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model
```

### 完整的例子

将所有这些结合在一起，下面列出了完整的示例。

```py
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)

# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=10, batch_size=16)
# save the model
model.save('model.h5')
```

首先运行该示例将打印准备好的训练数据集的摘要。

```py
Max document length: 1380
Vocabulary size: 44277
(1800, 1380)
```

接下来，打印已定义模型的摘要。

```py
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 1380)          0
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 1380)          0
____________________________________________________________________________________________________
input_3 (InputLayer)             (None, 1380)          0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 1380, 100)     4427700     input_1[0][0]
____________________________________________________________________________________________________
embedding_2 (Embedding)          (None, 1380, 100)     4427700     input_2[0][0]
____________________________________________________________________________________________________
embedding_3 (Embedding)          (None, 1380, 100)     4427700     input_3[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 1377, 32)      12832       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 1375, 32)      19232       embedding_2[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 1373, 32)      25632       embedding_3[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1377, 32)      0           conv1d_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1375, 32)      0           conv1d_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 1373, 32)      0           conv1d_3[0][0]
____________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)   (None, 688, 32)       0           dropout_1[0][0]
____________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)   (None, 687, 32)       0           dropout_2[0][0]
____________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)   (None, 686, 32)       0           dropout_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 22016)         0           max_pooling1d_1[0][0]
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 21984)         0           max_pooling1d_2[0][0]
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 21952)         0           max_pooling1d_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 65952)         0           flatten_1[0][0]
                                                                   flatten_2[0][0]
                                                                   flatten_3[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            659530      concatenate_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             11          dense_1[0][0]
====================================================================================================
Total params: 14,000,337
Trainable params: 14,000,337
Non-trainable params: 0
____________________________________________________________________________________________________
```

该模型相对较快，并且似乎在训练数据集上表现出良好的技能。

```py
...
Epoch 6/10
1800/1800 [==============================] - 30s - loss: 9.9093e-04 - acc: 1.0000
Epoch 7/10
1800/1800 [==============================] - 29s - loss: 5.1899e-04 - acc: 1.0000
Epoch 8/10
1800/1800 [==============================] - 28s - loss: 3.7958e-04 - acc: 1.0000
Epoch 9/10
1800/1800 [==============================] - 29s - loss: 3.0534e-04 - acc: 1.0000
Epoch 10/10
1800/1800 [==============================] - 29s - loss: 2.6234e-04 - acc: 1.0000
```

定义模型的图表将保存到文件中，清楚地显示模型的三个输入通道。

![Plot of the Multichannel Convolutional Neural Network For Text](img/bf3bb1f5fdb70a9f04e09e0cd212c5af.jpg)

文本多通道卷积神经网络图

该模型适用于多个时期并保存到文件 _model.h5_ 以供以后评估。

## 评估模型

在本节中，我们可以通过预测未见测试数据集中所有评论的情感来评估拟合模型。

使用上一节中开发的数据加载函数，我们可以加载和编码训练和测试数据集。

```py
# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')

# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)
```

我们可以加载保存的模型并在训练和测试数据集上进行评估。

下面列出了完整的示例。

```py
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')

# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)

# load the model
model = load_model('model.h5')

# evaluate model on training dataset
loss, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %f' % (acc*100))

# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX,testX,testX],array(testLabels), verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

运行该示例将在训练和测试数据集上打印模型的技能。

```py
Max document length: 1380
Vocabulary size: 44277
(1800, 1380) (200, 1380)

Train Accuracy: 100.000000
Test Accuracy: 87.500000
```

我们可以看到，正如预期的那样，训练数据集的技能非常出色，这里的准确率为 100％。

我们还可以看到模型在看不见的测试数据集上的技能也非常令人印象深刻，达到了 87.5％，这高于 2014 年论文中报告的模型的技能（尽管不是直接的苹果对苹果的比较）。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **不同的 n-gram** 。通过更改模型中通道使用的内核大小（n-gram 的数量）来探索模型，以了解它如何影响模型技能。
*   **更多或更少的频道**。探索在模型中使用更多或更少的渠道，并了解它如何影响模型技能。
*   **深层网络**。卷积神经网络在更深层时在计算机视觉中表现更好。在这里探索使用更深层的模型，看看它如何影响模型技能。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [用于句子分类的卷积神经网络](https://arxiv.org/abs/1408.5882)，2014。
*   [用于句子分类的卷积神经网络（代码）](https://github.com/yoonkim/CNN_sentence)。
*   [Keras 功能 API](https://keras.io/getting-started/functional-api-guide/)

## 摘要

在本教程中，您了解了如何为文本电影评论数据开发多通道卷积神经网络以进行情感预测。

具体来说，你学到了：

*   如何准备电影评论文本数据进行建模。
*   如何为 Keras 中的文本开发多通道卷积神经网络。
*   如何评估看不见的电影评论数据的拟合模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。