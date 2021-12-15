# 使用 Python 和 Keras 的 LSTM 循环神经网络的序列分类

> 原文： [https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

序列分类是一种预测建模问题，您可以在空间或时间上获得一些输入序列，任务是预测序列的类别。

使这个问题困难的原因是序列的长度可以变化，由输入符号的非常大的词汇表组成，并且可能要求模型学习输入序列中的符号之间的长期上下文或依赖性。

在这篇文章中，您将了解如何使用 Keras 深度学习库为 Python 中的序列分类问题开发 LSTM 循环神经网络模型。

阅读这篇文章后你会知道：

*   如何为序列分类问题开发 LSTM 模型。
*   如何通过使用压差来减少 LSTM 模型中的过拟合。
*   如何将 LSTM 模型与擅长学习空间关系的卷积神经网络相结合。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 TensorFlow 0.10.0 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 May / 2018** ：更新了使用最新 Keras API 的代码，感谢 jeremy rutman。

![Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras](img/0f56cceecbb4dedba0833021799047d2.jpg)

用 Keras
用 Python 中的 LSTM 循环神经网络进行序列分类 [photophilde](https://www.flickr.com/photos/photophilde/4875287879/) ，保留一些权利。

## 问题描述

我们将在本教程中用于演示序列学习的问题是 [IMDB 电影评论情感分类问题](http://ai.stanford.edu/~amaas/data/sentiment/)。每个电影评论是一个可变的单词序列，每个电影评论的情感必须分类。

大型电影评论数据集（通常称为 IMDB 数据集）包含 25,000 个用于训练的高极电影评论（好或坏），并且再次用于测试。问题是确定给定的电影评论是否具有积极或消极的情感。

这些数据由[斯坦福研究人员收集并用于 2011 年的论文](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)，其中 50-50 的数据被用于训练和测试。准确度达到 88.89％。

Keras 提供对内置 IMDB 数据集的访问。 **imdb.load_data（）**函数允许您以准备好在神经网络和深度学习模型中使用的格式加载数据集。

单词已被整数替换，这些整数表示数据集中每个单词的有序频率。因此，每个评论中的句子由一系列整数组成。

### 单词嵌入

我们将每个电影评论映射到一个真正的向量域，这是一种处理文字的流行技术，称为文字嵌入。这是一种在高维空间中将单词编码为实值向量的技术，其中单词之间的意义相似性转换为向量空间中的接近度。

Keras 提供了一种方便的方法，可以将单词的正整数表示转换为嵌入层的单词嵌入。

我们将每个单词映射到 32 长度的实值向量上。我们还将对建模感兴趣的单词总数限制为 5000 个最常用的单词，其余为零。最后，每个评论中的序列长度（单词数量）各不相同，因此我们将每个评论限制为 500 个单词，截断长评论并用零值填充较短的评论。

现在我们已经定义了我们的问题以及如何准备和建模数据，我们已经准备好开发 LSTM 模型来对电影评论的情感进行分类。

## 用于序列分类的简单 LSTM

我们可以快速开发用于 IMDB 问题的小型 LSTM 并获得良好的准确率。

让我们首先导入此模型所需的类和函数，并将随机数生成器初始化为常量值，以确保我们可以轻松地重现结果。

```py
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
```

我们需要加载 IMDB 数据集。我们将数据集限制在前 5,000 个单词中。我们还将数据集拆分为 train（50％）和 test（50％）集。

```py
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

接下来，我们需要截断并填充输入序列，以便它们具有相同的建模长度。模型将学习零值不携带信息，因此实际上序列在内容方面不是相同的长度，但是在 Keras 中执行计算需要相同的长度向量。

```py
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

我们现在可以定义，编译和拟合我们的 LSTM 模型。

第一层是嵌入层，它使用 32 个长度向量来表示每个单词。下一层是具有 100 个内存单元（智能神经元）的 LSTM 层。最后，因为这是一个分类问题，我们使用具有单个神经元和 S 形激活函数的密集输出层来对问题中的两个类（好的和坏的）进行 0 或 1 个预测。

因为它是二分类问题，所以使用对数丢失作为损失函数（Keras 中的 **binary_crossentropy** ）。使用有效的 ADAM 优化算法。该模型仅适用于 2 个时期，因为它很快就能解决问题。 64 个评论的大批量用于分隔重量更新。

```py
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
```

一旦适合，我们估计模型在看不见的评论上的表现。

```py
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

为了完整性，以下是 IMDB 数据集上此 LSTM 网络的完整代码清单。

```py
# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行此示例将生成以下输出。

请注意，如果您使用的是 TensorFlow 后端，您可能会看到一些与“PoolAllocator”相关的警告消息，您现在可以忽略这些消息。

```py
Epoch 1/3
16750/16750 [==============================] - 107s - loss: 0.5570 - acc: 0.7149
Epoch 2/3
16750/16750 [==============================] - 107s - loss: 0.3530 - acc: 0.8577
Epoch 3/3
16750/16750 [==============================] - 107s - loss: 0.2559 - acc: 0.9019
Accuracy: 86.79%
```

您可以看到，这种简单的 LSTM 几乎没有调整，可以在 IMDB 问题上获得最接近的最新结果。重要的是，这是一个模板，您可以使用该模板将 LSTM 网络应用于您自己的序列分类问题。

现在，让我们看一下这个简单模型的一些扩展，您可能也希望将这些扩展带给您自己的问题。

## LSTM 用于带有 Dropout 的序列分类

像 LSTM 这样的循环神经网络通常具有过拟合的问题。

可以使用 Dropout Keras 层在层之间应用 Dropout。我们可以通过在 Embedding 和 LSTM 层以及 LSTM 和 Dense 输出层之间添加新的 Dropout 层来轻松完成此操作。例如：

```py
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
```

上面添加了 Dropout 层的完整代码列表示例如下：

```py
# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行此示例提供以下输出。

```py
Epoch 1/3
16750/16750 [==============================] - 108s - loss: 0.5802 - acc: 0.6898
Epoch 2/3
16750/16750 [==============================] - 108s - loss: 0.4112 - acc: 0.8232
Epoch 3/3
16750/16750 [==============================] - 108s - loss: 0.3825 - acc: 0.8365
Accuracy: 85.56%
```

我们可以看到dropout对训练产生了预期的影响，收敛趋势略微缓慢，在这种情况下，最终的准确率较低。该模型可能会使用更多的训练时代，并可能获得更高的技能（试试看）。

或者，可以使用 LSTM 精确地和单独地将压差应用于存储器单元的输入和循环连接。

Keras 通过 LSTM 层上的参数提供此功能，**丢失**用于配置输入丢失， **recurrent_dropout** 用于配置重复丢失。例如，我们可以修改第一个示例，将 dropout 添加到输入和循环连接，如下所示：

```py
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```

下面列出了具有更精确 LSTM 丢失的完整代码清单，以确保完整性。

```py
# LSTM with dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行此示例提供以下输出。

```py
Epoch 1/3
16750/16750 [==============================] - 112s - loss: 0.6623 - acc: 0.5935
Epoch 2/3
16750/16750 [==============================] - 113s - loss: 0.5159 - acc: 0.7484
Epoch 3/3
16750/16750 [==============================] - 113s - loss: 0.4502 - acc: 0.7981
Accuracy: 82.82%
```

我们可以看到，LSTM 特定的丢失对网络的收敛具有比逐层丢失更明显的影响。如上所述，时期的数量保持不变并且可以增加以查看模型的技能是否可以进一步提升。

Dropout 是一种强大的技术，用于对抗 LSTM 模型中的过拟合，并且尝试这两种方法是个好主意，但是您可以使用 Keras 中提供的特定于门的丢失来获得更好的结果。

## 用于序列分类的 LSTM 和卷积神经网络

卷积神经网络在学习输入数据的空间结构方面表现出色。

IMDB 评论数据确实在评论中的单词序列中具有一维空间结构，并且 CNN 可能能够针对良好和不良情感挑选不变特征。然后，可以通过 LSTM 层将该学习的空间特征学习为序列。

我们可以在嵌入层之后轻松添加一维 CNN 和最大池池，然后将合并的特征提供给 LSTM。我们可以使用一小组 32 个特征，滤波器长度为 3 小。池化层可以使用标准长度 2 来将特征映射大小减半。

例如，我们将按如下方式创建模型：

```py
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
```

下面列出了具有 CNN 和 LSTM 层的完整代码清单，以确保完整性。

```py
# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

运行此示例提供以下输出。

```py
Epoch 1/3
16750/16750 [==============================] - 58s - loss: 0.5186 - acc: 0.7263
Epoch 2/3
16750/16750 [==============================] - 58s - loss: 0.2946 - acc: 0.8825
Epoch 3/3
16750/16750 [==============================] - 58s - loss: 0.2291 - acc: 0.9126
Accuracy: 86.36%
```

我们可以看到，我们获得了与第一个示例类似的结果，尽管权重更小，训练时间更短。

如果将此示例进一步扩展为使用 dropout，我希望可以实现更好的结果。

## 资源

如果您有兴趣深入了解序列预测或这个具体的例子，下面是一些资源。

*   [LSTM 的 Theano 教程应用于 IMDB 数据集](http://deeplearning.net/tutorial/lstm.html)
*   在 IMDB 数据集上使用 [LSTM 和 CNN](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py) 和 [LSTM 的 Keras 代码示例。](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py)
*   [监督序列标记与循环神经网络](http://www.amazon.com/dp/3642247962?tag=inspiredalgor-20)，2012 年由 Alex Graves（[和 PDF 预打印](https://www.cs.toronto.edu/~graves/preprint.pdf)）出版。

## 摘要

在这篇文章中，您了解了如何为序列分类预测建模问题开发 LSTM 网络模型。

具体来说，你学到了：

*   如何为 IMDB 电影评论情感分类问题开发一个简单的单层 LSTM 模型。
*   如何使用分层和 LSTM 特定的压差扩展 LSTM 模型以减少过拟合。
*   如何将卷积神经网络的空间结构学习特性与 LSTM 的序列学习相结合。

您对 LSTM 或此帖的序列分类有任何疑问吗？在评论中提出您的问题，我会尽力回答。