# 堆叠长短期内存网络

> 原文： [https://machinelearningmastery.com/stacked-long-short-term-memory-networks/](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)

### 使用Python中的示例代码轻松介绍Stacked LSTM
。

原始LSTM模型由单个隐藏的LSTM层和后面的标准前馈输出层组成。

Stacked LSTM是此模型的扩展，具有多个隐藏的LSTM层，其中每个层包含多个存储器单元。

在这篇文章中，您将发现Stacked LSTM模型架构。

完成本教程后，您将了解：

*   深度神经网络架构的好处。
*   Stacked LSTM循环神经网络架构。
*   如何使用Keras在Python中实现堆叠LSTM。

让我们开始吧。

![Gentle Introduction to Stacked Long Short-Term Memory Networks](img/1fe10e9feb242d8947c00a2647e704f4.jpg)

堆叠长短期记忆网络
的照片由 [Joost Markerink](https://www.flickr.com/photos/joostmarkerink/17303551006/) 拍摄，保留一些权利。

## 概观

这篇文章分为3个部分，它们是：

1.  为什么要增加深度？
2.  堆叠式LSTM架构
3.  在Keras中实现堆叠LSTM

## 为什么要增加深度？

堆叠LSTM隐藏层使得模型更深入，更准确地将描述作为深度学习技术获得。

神经网络的深度通常归因于该方法在广泛的挑战性预测问题上的成功。

> [深度神经网络的成功]通常归因于由于多个层而引入的层次结构。每个层处理我们希望解决的任务的某些部分，并将其传递给下一个。从这个意义上讲，DNN可以看作是一个处理流水线，其中每个层在将任务传递给下一个任务之前解决了部分任务，直到最后一层提供输出。

- [训练和分析深度循环神经网络](https://papers.nips.cc/paper/5166-training-and-analysing-deep-recurrent-neural-networks)，2013

可以将其他隐藏层添加到多层感知器神经网络中以使其更深。附加隐藏层被理解为重新组合来自先前层的学习表示并在高抽象级别创建新表示。例如，从线到形状到对象。

足够大的单个隐藏层多层感知器可用于近似大多数功能。增加网络的深度提供了另一种解决方案，需要更少的神经元和更快的训练。最终，添加深度是一种代表性优化。

> 深度学习是围绕一个假设建立的，即深层次分层模型在表示某些函数时可以指数级更高效，而不是浅层函数。

- [如何构建深度循环神经网络](https://arxiv.org/abs/1312.6026)，2013。

## 堆叠式LSTM架构

LSTM可以利用相同的好处。

鉴于LSTM对序列数据进行操作，这意味着层的添加增加了输入观察随时间的抽象级别。实际上，随着时间的推移进行分块观察或在不同时间尺度上表示问题。

> ...通过将多个重复隐藏状态堆叠在一起来构建深RNN。该方法可能允许每个级别的隐藏状态在不同的时间尺度上操作

- [如何构建深度循环神经网络](https://arxiv.org/abs/1312.6026)，2013

Graves等人介绍了堆叠的LSTM或深LSTM。在将LSTM应用于语音识别方面，打破了具有挑战性的标准问题的基准。

> RNN本身就具有深度，因为它们的隐藏状态是所有先前隐藏状态的函数。启发本文的问题是RNN是否也能从太空深度中获益;这就是将多个重复隐藏层堆叠在一起，就像在传统的深层网络中堆叠前馈层一样。

- [语音识别与深度循环神经网络](https://arxiv.org/abs/1303.5778)，2013

在同样的工作中，他们发现网络的深度比给定层中的存储器单元的数量更重要。

堆叠式LSTM现在是用于挑战序列预测问题的稳定技术。堆叠LSTM架构可以定义为由多个LSTM层组成的LSTM模型。上面的LSTM层提供序列输出而不是单个值输出到下面的LSTM层。具体地说，每个输入时间步长一个输出，而不是所有输入时间步长的一个输出时间步长。

![Stacked Long Short-Term Memory Archiecture](img/18919d3d5e8e3ef675e8f630308fa156.jpg)

堆叠长短期内存架构

## 在Keras中实现堆叠LSTM

我们可以在Keras Python深度学习库中轻松创建Stacked LSTM模型

每个LSTM存储器单元都需要3D输入。当LSTM处理一个输入时间步长序列时，每个存储器单元将输出整个序列的单个值作为2D阵列。

我们可以使用具有单个隐藏LSTM层的模型来演示以下内容，该LSTM层也是输出层。

```py
# Example of one output for whole sequence
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
# input time steps
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

输入序列有3个值。运行该示例将输入序列的单个值输出为2D数组。

```py
[[ 0.00031043]]
```

要堆叠LSTM层，我们需要更改先前LSTM层的配置，以输出3D数组作为后续层的输入。

我们可以通过将层上的return_sequences参数设置为True（默认为False）来完成此操作。这将为每个输入时间步返回一个输出并提供3D数组。
以下是与return_sequences = True相同的例子。

```py
# Example of one output for each input time step
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
# input time steps
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

运行该示例为输入序列中的每个时间步输出单个值。

```py
[[[-0.02115841]
[-0.05322712]
[-0.08976141]]]
```

下面是定义两个隐藏层Stacked LSTM的示例：

```py
model = Sequential()
model.add(LSTM(..., return_sequences=True, input_shape=(...)))
model.add(LSTM(...))
model.add(Dense(...))
```

只要先前的LSTM层提供3D输出作为后续层的输入，我们就可以继续添加隐藏的LSTM层。例如，下面是一个有4个隐藏层的Stacked LSTM。

```py
model = Sequential()
model.add(LSTM(..., return_sequences=True, input_shape=(...)))
model.add(LSTM(..., return_sequences=True))
model.add(LSTM(..., return_sequences=True))
model.add(LSTM(...))
model.add(Dense(...))
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [如何构建深度循环神经网络](https://arxiv.org/abs/1312.6026)，2013。
*   [深度循环神经网络的训练和分析](https://papers.nips.cc/paper/5166-training-and-analysing-deep-recurrent-neural-networks)，2013。
*   [语音识别与深度循环神经网络](https://arxiv.org/abs/1303.5778)，2013。
*   [使用循环神经网络生成序列](https://arxiv.org/abs/1308.0850)，2014年。

## 摘要

在这篇文章中，您发现了Stacked Long Short-Term Memory网络架构。

具体来说，你学到了：

*   深度神经网络架构的好处。
*   Stacked LSTM循环神经网络架构。
*   如何使用Keras在Python中实现堆叠LSTM。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。