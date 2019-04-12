# 如何使用 Keras 将 Word 嵌入图层用于深度学习

> 原文： [https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

Word 嵌入提供了单词及其相对含义的密集表示。

它们是对简单包含的单词模型表示中使用的稀疏表示的改进。

可以从文本数据中学习 Word 嵌入，并在项目中重用。它们也可以作为在文本数据上拟合神经网络的一部分来学习。

在本教程中，您将了解如何使用 Word 嵌入在 Python 中使用 Keras 进行深度学习。

完成本教程后，您将了解：

*   关于字嵌入和 Keras 通过嵌入层支持字嵌入。
*   如何在拟合神经网络时学习单词嵌入。
*   如何在神经网络中使用预先训练的单词嵌入。

让我们开始吧。

*   **2018 年 2 月更新**：修复了由于底层 API 发生变化而导致的错误。

![How to Use Word Embedding Layers for Deep Learning with Keras](img/2218ca2cd0b8ce367f990aafb1cebca5.jpg)

如何使用 Keras
使用 Word 嵌入图层深度学习照片由 [thisguy](https://www.flickr.com/photos/davebloggs007/36375879215/) 拍摄，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  单词嵌入
2.  Keras 嵌入层
3.  学习嵌入的示例
4.  使用预训练 GloVe 嵌入的示例

## 1.单词嵌入

单词嵌入是使用密集向量表示来表示单词和文档的一类方法。

它是对传统的词袋模型编码方案的改进，其中使用大的稀疏矢量来表示每个单词或者对矢量中的每个单词进行评分以表示整个词汇表。这些表示是稀疏的，因为词汇量很大，并且给定的单词或文档将由主要由零值组成的大向量表示。

相反，在嵌入中，单词由密集向量表示，其中向量表示单词到连续向量空间的投影。

向量空间中的单词的位置是从文本中学习的，并且基于在使用单词时围绕单词的单词。

学习向量空间中的单词的位置被称为其嵌入。

从文本中学习单词嵌入的两种流行方法示例包括：

*   Word2Vec。
*   手套。

除了这些精心设计的方法之外，还可以学习单词嵌入作为深度学习模型的一部分。这可能是一种较慢的方法，但可以将模型定制为特定的训练数据集。

## 2\. Keras 嵌入层

Keras 提供[嵌入](https://keras.io/layers/embeddings/#embedding)层，可用于文本数据上的神经网络。

它要求输入数据是整数编码的，以便每个单词由唯一的整数表示。该数据准备步骤可以使用 Keras 提供的 [Tokenizer API](https://keras.io/preprocessing/text/#tokenizer) 来执行。

使用随机权重初始化嵌入层，并将学习训练数据集中所有单词的嵌入。

它是一个灵活的层，可以以多种方式使用，例如：

*   它可以单独用于学习可以保存并在以后用于其他模型的单词嵌入。
*   它可以用作深度学习模型的一部分，其中嵌入与模型本身一起被学习。
*   它可以用于加载预训练的单词嵌入模型，一种转移学习。

嵌入层被定义为网络的第一个隐藏层。它必须指定 3 个参数：

它必须指定 3 个参数：

*   **input_dim** ：这是文本数据中词汇表的大小。例如，如果您的数据整数编码为 0-10 之间的值，那么词汇表的大小将为 11 个单词。
*   **output_dim** ：这是将嵌入单词的向量空间的大小。它为每个单词定义了该层的输出向量的大小。例如，它可以是 32 或 100 甚至更大。测试问题的不同值。
*   **input_length** ：这是输入序列的长度，正如您为 Keras 模型的任何输入层定义的那样。例如，如果所有输入文档都包含 1000 个单词，则为 1000。

例如，下面我们定义具有 200 的词汇表的嵌入层（例如，从 0 到 199 的整数编码的单词），其中将嵌入单词的 32 维的向量空间，以及每个具有 50 个单词的输入文档。

```
e = Embedding(200, 32, input_length=50)
```

嵌入层具有学习的权重。如果将模型保存到文件，则将包括嵌入图层的权重。

_ 嵌入 _ 层的输出是 2D 矢量，在输入的单词序列（输入文档）中为每个单词嵌入一个。

如果您希望将 _Dense_ 层直接连接到嵌入层，则必须先使用 _Flatten_ 图层将 2D 输出矩阵展平为 1D 向量。

现在，让我们看看我们如何在实践中使用嵌入层。

## 3.学习嵌入的示例

在本节中，我们将看看如何在将神经网络拟合到文本分类问题时学习单词嵌入。

我们将定义一个小问题，其中我们有 10 个文本文档，每个文档都有一个学生提交的工作评论。每个文本文档被分类为正“1”或负“0”。这是一个简单的情绪分析问题。

首先，我们将定义文档及其类标签。

```
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
```

接下来，我们可以整数编码每个文档。这意味着作为输入，嵌入层将具有整数序列。我们可以尝试其他更复杂的单词模型编码，如计数或 TF-IDF。

Keras 提供 [one_hot（）函数](https://keras.io/preprocessing/text/#one_hot)，它将每个单词的散列创建为有效的整数编码。我们将估计 50 的词汇量，这比减少哈希函数碰撞的概率要大得多。

```
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
```

序列具有不同的长度，Keras 更喜欢输入以进行矢量化，并且所有输入具有相同的长度。我们将填充所有输入序列的长度为 4.再次，我们可以使用内置的 Keras 函数，在这种情况下 [pad_sequences（）函数](https://keras.io/preprocessing/sequence/#pad_sequences)。

```
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
```

我们现在准备将我们的 _ 嵌入 _ 层定义为我们的神经网络模型的一部分。

_ 嵌入 _ 的词汇量为 50，输入长度为 4.我们将选择 8 维的小嵌入空间。

该模型是一个简单的二元分类模型。重要的是，_ 嵌入 _ 层的输出将是 4 个向量，每个维度为 8 维，每个单词一个。我们将其展平为一个 32 元素向量，以传递给 _Dense_ 输出层。

```
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
```

最后，我们可以拟合和评估分类模型。

```
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

完整的代码清单如下。

```
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

首先运行该示例打印整数编码的文档。

```
[[6, 16], [42, 24], [2, 17], [42, 24], [18], [17], [22, 17], [27, 42], [22, 24], [49, 46, 16, 34]]
```

然后打印每个文档的填充版本，使它们均匀长度。

```
[[ 6 16  0  0]
 [42 24  0  0]
 [ 2 17  0  0]
 [42 24  0  0]
 [18  0  0  0]
 [17  0  0  0]
 [22 17  0  0]
 [27 42  0  0]
 [22 24  0  0]
 [49 46 16 34]]
```

定义网络后，将打印层的摘要。我们可以看到，正如预期的那样，嵌入层的输出是一个 4×8 矩阵，并且由 Flatten 层压缩为 32 个元素的矢量。

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 4, 8)              400
_________________________________________________________________
flatten_1 (Flatten)          (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33
=================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0
_________________________________________________________________
```

最后，打印训练模型的准确性，表明它完美地学习了训练数据集（这并不奇怪）。

```
Accuracy: 100.000000
```

您可以将已学习的权重从嵌入层保存到文件中，以便以后在其他模型中使用。

您通常也可以使用此模型对在测试数据集中看到的具有相同类型词汇的其他文档进行分类。

接下来，让我们看看在 Keras 中加载预先训练好的单词嵌入。

## 4.使用预训练 GloVe 嵌入的示例

Keras 嵌入层还可以使用在其他地方学习的单词嵌入。

在自然语言处理领域中常见的是学习，保存和免费提供单词嵌入。

例如，GloVe 方法背后的研究人员在其公共领域许可下发布的网站上提供了一套预先训练过的单词嵌入。看到：

*   [GloVe：用于词表示的全局向量](https://nlp.stanford.edu/projects/glove/)

最小的嵌入包是 822Mb，称为“_ 手套.6B.zip_ ”。它是在 10 亿个令牌（单词）的数据集上训练的，词汇量为 40 万字。有一些不同的嵌入矢量大小，包括 50,100,200 和 300 维度。

您可以下载这个嵌入集合，我们可以使用训练数据集中单词的训练前嵌入的权重对 Keras _ 嵌入 _ 图层进行播种。

这个例子的灵感来自 Keras 项目中的一个例子： [pretrained_word_embeddings.py](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py) 。

下载并解压缩后，您将看到一些文件，其中一个是“ _glove.6B.100d.txt_ ”，其中包含 100 维版本的嵌入。

如果您查看文件内部，您将看到一个标记（单词），后面跟着每行的权重（100 个数字）。例如，下面是嵌入 ASCII 文本文件的第一行，显示“”的嵌入。

```
the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459 0.28787 -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336 0.1464 -0.37304 0.34577 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107 -0.039722 0.28277 0.14393 0.23464 -0.31021 0.086173 0.20397 0.52624 0.17164 -0.082378 -0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 -0.33477 -0.36559 -0.54857 -0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 0.72201 -0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044 0.057162 -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624 0.70565 0.49744 -0.42063 0.26148 -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217 0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 -0.1459 0.8278 0.27062
```

与前一节一样，第一步是定义示例，将它们编码为整数，然后将序列填充为相同的长度。

在这种情况下，我们需要能够将单词映射到整数以及将整数映射到单词。

Keras 提供了一个 [Tokenizer](https://keras.io/preprocessing/text/#tokenizer) 类，它可以适应训练数据，可以通过调用 _Tokenizer_ 类上的 _texts_to_sequences（）_ 方法将文本转换为序列，并提供对 _word_index_ 属性中单词到整数的字典映射的访问。

```
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
```

接下来，我们需要将整个 GloVe 字嵌入文件作为嵌入数组的字典加载到内存中。

```
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
```

这很慢。最好过滤训练数据中唯一单词的嵌入。

接下来，我们需要为训练数据集中的每个单词创建一个嵌入矩阵。我们可以通过枚举 _Tokenizer.word_index_ 中的所有唯一单词并从加载的 GloVe 嵌入中定位嵌入权重向量来实现。

结果是仅在我们将在训练期间看到的单词的权重矩阵。

```
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
```

现在我们可以像以前一样定义我们的模型，拟合并评估它。

关键的区别在于嵌入层可以使用 GloVe 字嵌入权重进行播种。我们选择了 100 维版本，因此必须在 _output_dim_ 设置为 100 的情况下定义嵌入层。最后，我们不想更新此模型中的学习单词权重，因此我们将设置 _]模型的可训练 _ 属性为 _False_ 。

```
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
```

下面列出了完整的工作示例。

```
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('../glove_data/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

运行示例可能需要更长的时间，但后来证明它能够适应这个简单的问题。

```
[[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]

[[ 6  2  0  0]
 [ 3  1  0  0]
 [ 7  4  0  0]
 [ 8  1  0  0]
 [ 9  0  0  0]
 [10  0  0  0]
 [ 5  4  0  0]
 [11  3  0  0]
 [ 5  1  0  0]
 [12 13  2 14]]

Loaded 400000 word vectors.

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 4, 100)            1500
_________________________________________________________________
flatten_1 (Flatten)          (None, 400)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 401
=================================================================
Total params: 1,901
Trainable params: 401
Non-trainable params: 1,500
_________________________________________________________________

Accuracy: 100.000000
```

在实践中，我鼓励您尝试使用经过预先训练的嵌入来学习单词嵌入，该嵌入是固定的并且尝试在预训练嵌入之上进行学习。

了解哪种方法最适合您的具体问题。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [Word 嵌入维基百科](https://en.wikipedia.org/wiki/Word_embedding)
*   [Keras 嵌入层 API](https://keras.io/layers/embeddings/#embedding)
*   [在 Keras 模型中使用预训练的字嵌入](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)，2016
*   [在 Keras](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py) 中使用预先训练的 GloVe 嵌入的示例
*   [GloVe 嵌入](https://nlp.stanford.edu/projects/glove/)
*   [词汇嵌入概述及其与分布式语义模型的联系](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/)，2016
*   [Deep Learning，NLP 和 Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) ，2014

## 摘要

在本教程中，您了解了如何使用 Word 嵌入在 Python 中使用 Keras 进行深度学习。

具体来说，你学到了：

*   关于字嵌入和 Keras 通过嵌入层支持字嵌入。
*   如何在拟合神经网络时学习单词嵌入。
*   如何在神经网络中使用预先训练的单词嵌入。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。