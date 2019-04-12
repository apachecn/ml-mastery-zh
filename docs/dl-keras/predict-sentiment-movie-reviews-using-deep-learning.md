# 用深度学习预测电影评论的情感

> 原文： [https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/](https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)

[情感分析](https://en.wikipedia.org/wiki/Sentiment_analysis)是一种自然语言处理问题，其中理解文本并预测潜在意图。

在这篇文章中，您将了解如何使用 Keras 深度学习库在 Python 中预测电影评论的积极或消极情感。

阅读这篇文章后你会知道：

*   关于自然语言处理的 IMDB 情感分析问题以及如何在 Keras 中加载它。
*   如何在 Keras 中使用单词嵌入来解决自然语言问题。
*   如何开发和评估 IMDB 问题的多层感知器模型。
*   如何为 IMDB 问题开发一维卷积神经网络模型。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 TensorFlow 0.10.0 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![Predict Sentiment From Movie Reviews Using Deep Learning](img/5c59c425196cb23cb43afa5a74345b1e.png)

使用深度学习预测电影评论的情感
[SparkCBC](https://www.flickr.com/photos/25031050@N06/3407720762/) 的照片，保留一些权利。

## IMDB 电影评论情感问题描述

数据集是[大型电影评论数据集](http://ai.stanford.edu/~amaas/data/sentiment/)，通常称为 IMDB 数据集。

大型电影评论数据集（通常称为 IMDB 数据集）包含 25,000 个用于训练的高极移动评论（好的或坏的）以及用于测试的相同数量。问题是确定给定的移动评论是否具有正面或负面情感。

这些数据由斯坦福大学的研究人员收集并用于 [2011 年论文](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) [PDF]，其中 50/50 的数据用于训练和测试。准确度达到 88.89％。

该数据还被用作 2014 年底至 2015 年初名为“[袋子爆米花袋](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)”的 Kaggle 比赛的基础。获胜者达到 99％时，准确率达到 97％以上。

## 使用 Keras 加载 IMDB 数据集

Keras 可以访问内置的 [IMDB 数据集。](http://keras.io/datasets/)

keras.datasets.imdb.load_data（）允许您以可用于神经网络和深度学习模型的格式加载数据集。

单词已被整数替换，表示数据集中单词的绝对流行度。因此，每个评论中的句子由一系列整数组成。

第一次调用 imdb.load_data（）会将 IMDB 数据集下载到您的计算机并将其作为 32 兆字节文件存储在〜/ .keras / datasets / imdb.pkl 下的主目录中。

有用的是，imdb.load_data（）提供了额外的参数，包括要加载的顶部字的数量（其中具有较低整数的字在返回的数据中标记为零），要跳过的顶部字的数量（以避免“该”的）以及支持的最大评论期限。

让我们加载数据集并计算它的一些属性。我们将首先加载一些库并将整个 IMDB 数据集作为训练数据集加载。

```py
import numpy
from keras.datasets import imdb
from matplotlib import pyplot
# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
```

接下来，我们可以显示训练数据集的形状。

```py
# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)
```

运行此代码段，我们可以看到有 50,000 条记录。

```py
Training data:
(50000,)
(50000,)
```

我们还可以打印唯一的类值。

```py
# Summarize number of classes
print("Classes: ")
print(numpy.unique(y))
```

我们可以看到，这是一个二元分类问题，在评论中有好的和坏的情感。

```py
Classes:
[0 1]
```

接下来，我们可以了解数据集中唯一单词的总数。

```py
# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))
```

有趣的是，我们可以看到整个数据集中只有不到 100,000 个单词。

```py
Number of words:
88585
```

最后，我们可以了解平均审核长度。

```py
# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length
pyplot.boxplot(result)
pyplot.show()
```

我们可以看到，平均评论不到 300 字，标准差超过 200 字。

```py
Review length:
Mean 234.76 words (172.911495)
```

我们可以看到一个指数分布的盒子和胡须图，我们可以看到一个指数分布，我们可以覆盖分布的质量，剪切长度为 400 到 500 字。

![Review Length in Words for IMDB Dataset](img/f73add12895695350fadc1995b8d77e9.png)

查看 IMDB 数据集的单词长度

## Word 嵌入

自然语言处理领域的最新突破称为[字嵌入](https://en.wikipedia.org/wiki/Word_embedding)。

这是一种在高维空间中将单词编码为实值向量的技术，其中单词之间的意义相似性转换为向量空间中的接近度。

离散词被映射到连续数的向量。当使用神经网络处理自然语言问题时这很有用，深度学习模型我们需要数字作为输入。

Keras 提供了一种方便的方法，可以将单词的正整数表示转换为[嵌入层](http://keras.io/layers/embeddings/)的单词嵌入。

该层采用定义映射的参数，包括也称为词汇量大小的预期词的最大数量（例如，将被视为整数的最大整数值）。该层还允许您为每个单词向量指定维度，称为输出维度。

我们想为 IMDB 数据集使用单词嵌入表示。

假设我们只对数据集中前 5,000 个最常用的单词感兴趣。因此我们的词汇量将为 5,000。我们可以选择使用 32 维向量来表示每个单词。最后，我们可以选择将最大审核长度限制为 500 字，将评论截断时间缩短，并将填充评论缩短为 0 值。

我们将加载 IMDB 数据集，如下所示：

```py
imdb.load_data(nb_words=5000)
```

然后，我们将使用 Keras 实用程序使用 sequence.pad_sequences（）函数将数据集截断或填充到每个观察的长度 500。

```py
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)
```

最后，稍后，我们模型的第一层将是使用 Embedding 类创建的单词嵌入层，如下所示：

```py
Embedding(5000, 32, input_length=500)
```

对于给定的复习训练或整数格式的测试模式，该第一层的输出将是大小为 32×500 的矩阵。

既然我们知道如何在 Keras 中加载 IMDB 数据集以及如何为它使用单词嵌入表示，那么让我们开发并评估一些模型。

## 用于 IMDB 数据集的简单多层感知器模型

我们可以从开发一个具有单个隐藏层的简单多层感知器模型开始。

嵌入表示这个词是一个真正的创新，我们将通过一个相对简单的神经网络展示 2011 年被认为是世界级的结果。

让我们首先导入此模型所需的类和函数，并将随机数生成器初始化为常量值，以确保我们可以轻松地重现结果。

```py
# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

接下来，我们将加载 IMDB 数据集。我们将简化数据集，正如单词嵌入一节中所讨论的那样。只会加载前 5,000 个单词。

我们还将使用 50％/ 50％的数据集拆分进行训练和测试。这是一种很好的标准拆分方法。

```py
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

我们将以 500 字的方式绑定评论，截断更长的评论和零填充更短的评论。

```py
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
```

现在我们可以创建我们的模型。我们将使用嵌入层作为输入层，将词汇表设置为 5,000，将字向量大小设置为 32 维，将 input_length 设置为 500.第一层的输出将是 32×500 大小的矩阵，如上一节所述。

我们将嵌入层输出展平为一维，然后使用一个 250 单位的密集隐藏层和整流器激活功能。输出层有一个神经元，并将使用 sigmoid 激活输出 0 和 1 的值作为预测。

该模型使用对数损失，并使用有效的 ADAM 优化程序进行优化。

```py
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

我们可以适应模型并在训练时使用测试集作为验证。这个模型很快就会过度使用，因此我们将使用很少的训练时期，在这种情况下只需 2 个。

有很多数据，所以我们将使用 128 的批量大小。在训练模型后，我们评估其在测试数据集上的准确性。

```py
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行此示例适合模型并总结估计的表现。我们可以看到，这个非常简单的模型获得了近 86.94％的分数，这是在原始论文的附近，只需很少的努力。

```py
Train on 25000 samples, validate on 25000 samples
Epoch 1/2
39s - loss: 0.5160 - acc: 0.7040 - val_loss: 0.2982 - val_acc: 0.8716
Epoch 2/2
37s - loss: 0.1897 - acc: 0.9266 - val_loss: 0.3143 - val_acc: 0.8694
Accuracy: 86.94%
```

我确信如果我们训练这个网络，我们可以做得更好，也许使用更大的嵌入并添加更多的隐藏层。让我们尝试不同的网络类型。

## IMDB 数据集的一维卷积神经网络模型

设计卷积神经网络以尊重图像数据中的空间结构，同时对场景中学习对象的位置和方向具有鲁棒性。

该相同原理可用于序列，例如电影评论中的一维单词序列。使 CNN 模型对于学习识别图像中的对象具有吸引力的相同属性可以帮助学习单词段落中的结构，即对于特征的特定位置的技术不变性。

Keras 分别支持 Conv1D 和 MaxPooling1D 类的一维卷积和池化。

再次，让我们导入此示例所需的类和函数，并将随机数生成器初始化为常量值，以便我们可以轻松地重现结果。

```py
# CNN for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

我们也可以像以前一样加载和准备我们的 IMDB 数据集。

```py
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
```

我们现在可以定义我们的卷积神经网络模型。这次，在嵌入输入层之后，我们插入一个 Conv1D 层。该卷积层具有 32 个特征映射，并且一次读取嵌入的单词表示 3 个向量元素的嵌入单词嵌入。

卷积层之后是 1D max pooling layer，其长度和步幅为 2，使卷积层的特征映射的大小减半。网络的其余部分与上面的神经网络相同。

```py
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

我们也像以前一样适应网络。

```py
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行该示例，我们首先会看到网络结构的摘要。我们可以看到我们的卷积层保留了 32 维输入的嵌入输入层的维度，最多 500 个字。池化层通过将其减半来压缩该表示。

运行该示例对上述神经网络模型提供了一个小但令人欢迎的改进，准确率接近 87.79％。

```py
Train on 25000 samples, validate on 25000 samples
Epoch 1/2
38s - loss: 0.4451 - acc: 0.7640 - val_loss: 0.3107 - val_acc: 0.8660
Epoch 2/2
39s - loss: 0.2373 - acc: 0.9064 - val_loss: 0.2909 - val_acc: 0.8779
Accuracy: 87.79%
```

同样，存在很多进一步优化的机会，例如使用更深和/或更大的卷积层。一个有趣的想法是将最大池化层设置为使用 500 的输入长度。这会将每个要素图压缩为单个 32 长度向量，并可以提高表现。

## 摘要

在这篇文章中，您发现了用于自然语言处理的 IMDB 情感分析数据集。

您学习了如何为情感分析开发深度学习模型，包括：

*   如何加载和查看 Keras 中的 IMDB 数据集。
*   如何开发一个用于情感分析的大型神经网络模型。
*   如何开发一种用于情感分析的一维卷积神经网络模型。

您对情感分析或此帖有任何疑问吗？在评论中提出您的问题，我会尽力回答。