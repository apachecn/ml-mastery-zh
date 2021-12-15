# 如何使用编解码器 LSTM 来打印随机整数序列

> 原文： [https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/](https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/)

长短期记忆（LSTM）循环神经网络的一个强大功能是它们可以记住长序列间隔的观察结果。

这可以通过设计简单的序列回波问题来证明，其中输入序列的整个输入序列或部分连续块作为输出序列被回应。

开发 LSTM 循环神经网络以解决序列回波问题既是 LSTM 功能的良好证明，也可用于演示最先进的循环神经网络架构。

在这篇文章中，您将了解如何开发 LSTM 以使用 Keras 深度学习库解决 Python 中的完整和部分序列打印问题。

完成本教程后，您将了解：

*   如何生成随机的整数序列，使用单热编码表示它们，并将序列构建为具有输入和输出对的监督学习问题。
*   如何开发序列到序列 LSTM 以将整个输入序列作为输出进行打印。
*   如何开发编解码器 LSTM 以打印长度与输入序列不同的部分序列。

让我们开始吧。

![How to use an Encoder-Decoder LSTM to Echo Sequences of Random Integers](img/2f2624d74546ef2b5dcc7b1ecf597328.jpg)

如何使用编解码器 LSTM 来回放随机整数序列
照片来自 [hammy24601](https://www.flickr.com/photos/jdjtc/27740211344/) ，保留一些权利。

## 概观

本教程分为 3 个部分;他们是：

1.  序列回波问题
2.  打印整个序列（序列到序列模型）
3.  打印部分序列（编解码器模型）

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

本教程假设您安装了 TensorFlow 或 Theano 后端的 Keras v2.0 或更高版本。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 序列回波问题

回波序列问题涉及将 LSTM 暴露于一系列观察中，一次一个，然后要求网络打印观察到的部分或完整的连续观察列表。

这迫使网络记住连续观测的块，并且是 LSTM 循环神经网络学习能力的一个很好的证明。

第一步是编写一些代码来生成随机的整数序列，并为网络编码。

这涉及 3 个步骤：

1.  生成随机序列
2.  单热编码随机序列
3.  用于学习的帧编码序列

### 生成随机序列

我们可以使用 [randint（）函数](https://docs.python.org/3/library/random.html#random.randint)在 Python 中生成随机整数，该函数接受两个参数，指示从中绘制值的整数范围。

在本教程中，我们将问题定义为具有 0 到 99 之间的整数值以及 100 个唯一值。

```py
randint(0, 99)
```

我们可以将它放在一个名为 generate_sequence（）的函数中，该函数将生成所需长度的随机整数序列，默认长度设置为 25 个元素。

此功能如下所列。

```py
# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]
```

### 单热编码随机序列

一旦我们生成了随机整数序列，我们需要将它们转换为适合训练 LSTM 网络的格式。

一种选择是将整数重新缩放到范围[0,1]。这可行，并要求将问题表述为回归。

我有兴趣预测正确的数字，而不是接近预期值的数字。这意味着我更倾向于将问题框架化为分类而不是回归，其中预期输出是一个类，并且有 100 个可能的类值。

在这种情况下，我们可以使用整数值的单热编码，其中每个值由 100 个元素的二进制向量表示，除了整数的索引（标记为 1）之外，该二进制向量都是“0”值。

下面的函数 one_hot_encode（）定义了如何迭代整数序列并为每个整数创建二进制向量表示，并将结果作为二维数组返回。

```py
# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

我们还需要解码编码值，以便我们可以使用预测，在这种情况下，只需查看它们。

可以通过使用 [argmax（）NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)来反转单热编码，该函数返回具有最大值的向量中的值的索引。

下面的函数名为 one_hot_decode（），将对编码序列进行解码，并可用于稍后解码来自我们网络的预测。

```py
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

### 用于学习的帧编码序列

一旦生成并编码了序列，就必须将它们组织成适合学习的框架。

这涉及将线性序列组织成输入（X）和输出（y）对。

例如，序列[1,2,3,4,5]可以被构造为具有 2 个输入（t 和 t-1）和 1 个输出（t-1）的序列预测问题，如下所示：

```py
X,			y
NaN, 1,		NaN
1, 2,		1
2, 3,		2
3, 4,		3
4, 5,		4
5, NaN,		5
```

请注意，缺少数据标有 NaN 值。这些行可以用特殊字符填充并屏蔽。或者，更简单地说，可以从数据集中移除这些行，代价是从要学习的序列中提供更少的示例。后一种方法将是本例中使用的方法。

我们将使用 [Pandas shift（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)来创建编码序列的移位版本。使用 [dropna（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)删除缺少数据的行。

然后，我们可以从移位的数据帧中指定输入和输出数据。数据必须是 3 维的，以用于序列预测 LSTM。输入和输出都需要维度[样本，时间步长，特征]，其中样本是行数，时间步长是从中学习预测的滞后观察数，而特征是单独值的数量（例如 100 单热门编码）。

下面的代码定义了实现此功能的 to_supervised（）函数。它需要两个参数来指定用作输入和输出的编码整数的数量。输出整数的数量必须小于或等于输入的数量，并从最早的观察计算。

例如，如果在序列[1,2,3,4,5]上为输入和输出指定了 5 和 1，那么该函数将返回 X 中的单行[1,2,3,4,5]和[1]中的 y 一行。

```py
# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y
```

有关将序列转换为监督学习问题的更多信息，请参阅帖子：

*   [时间序列预测作为监督学习](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
*   [如何将时间序列转换为 Python 中的监督学习问题](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

### 完整的例子

我们可以将所有这些结合在一起。

下面是完整的代码清单，用于生成 25 个随机整数的序列，并将每个整数编码为二进制向量，将它们分成 X，y 对进行学习，然后打印解码对以供查看。

```py
from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y

# generate random sequence
sequence = generate_sequence()
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence)
# convert to X,y pairs
X,y = to_supervised(encoded, 5, 3)
# decode all pairs
for i in range(len(X)):
	print(one_hot_decode(X[i]), '=>', one_hot_decode(y[i]))
```

运行示例可能会为您生成不同的特定数字。

将序列分成 5 个数字的输入序列，并输出来自输入序列的 3 个最老观察结果的序列。

```py
[86, 81, 88, 1, 23, 78, 64, 7, 99, 23, 2, 36, 73, 26, 27, 33, 24, 51, 73, 64, 13, 13, 53, 40, 64]

[86, 81, 88, 1, 23] => [86, 81, 88]
[81, 88, 1, 23, 78] => [81, 88, 1]
[88, 1, 23, 78, 64] => [88, 1, 23]
[1, 23, 78, 64, 7] => [1, 23, 78]
[23, 78, 64, 7, 99] => [23, 78, 64]
[78, 64, 7, 99, 23] => [78, 64, 7]
[64, 7, 99, 23, 2] => [64, 7, 99]
[7, 99, 23, 2, 36] => [7, 99, 23]
[99, 23, 2, 36, 73] => [99, 23, 2]
[23, 2, 36, 73, 26] => [23, 2, 36]
[2, 36, 73, 26, 27] => [2, 36, 73]
[36, 73, 26, 27, 33] => [36, 73, 26]
[73, 26, 27, 33, 24] => [73, 26, 27]
[26, 27, 33, 24, 51] => [26, 27, 33]
[27, 33, 24, 51, 73] => [27, 33, 24]
[33, 24, 51, 73, 64] => [33, 24, 51]
[24, 51, 73, 64, 13] => [24, 51, 73]
[51, 73, 64, 13, 13] => [51, 73, 64]
[73, 64, 13, 13, 53] => [73, 64, 13]
[64, 13, 13, 53, 40] => [64, 13, 13]
[13, 13, 53, 40, 64] => [13, 13, 53]
```

现在我们知道如何准备和表示整数的随机序列，我们可以看一下使用 LSTM 来学习它们。

## 打印全序列
（_ 序列到序列模型 _）

在本节中，我们将开发一个 LSTM，用于简单地解决问题，即预测或再现整个输入序列。

也就是说，给定固定的输入序列，例如 5 个随机整数，输出相同的序列。这可能听起来不像问题的简单框架，但这是因为解决它所需的网络架构很简单。

我们将生成 25 个整数的随机序列，并将它们构造为 5 个值的输入 - 输出对。我们将创建一个名为 get_data（）的便捷函数，我们将使用它来创建编码的 X，y 对随机整数，使用上一节中准备的所有功能。此功能如下所列。

```py
# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y
```

将使用参数 5 和 5 调用此函数，以创建 21 个样本，其中 5 个步骤的 100 个特征作为输入，并且与输出相同（21 个而不是 25 个，因为删除了由于序列移位而导致缺失值的某些行）。

我们现在可以为此问题开发 LSTM。我们将使用有状态 LSTM 并在训练结束时为每个生成的样本显式重置内部状态。由于学习所需的上下文将作为时间步长提供，因此可能需要或可能不需要跨序列中的样本维持网络内的内部状态;尽管如此，这个额外的状态可能会有所帮助。

让我们首先将输入数据的预期尺寸定义为 100 个特征的 5 个时间步长。因为我们使用有状态 LSTM，所以将使用 batch_input_shape 参数而不是 input_shape 指定。 LSTM 隐藏层将使用 20 个内存单元，这应该足以学习这个问题。

批量大小为 7。批量大小必须是训练样本数量的因子（在这种情况下为 21），并定义数量样本，之后更新 LSTM 中的权重。这意味着对于训练网络的每个随机序列，权重将更新 3 次。

```py
model = Sequential()
model.add(LSTM(20, batch_input_shape=(7, 5, 100), return_sequences=True, stateful=True))
```

我们希望输出层一次输出一个整数，每个输出观察一个。

我们将输出层定义为完全连接的层（Dense），其中 100 个神经元用于单热编码中的 100 个可能的整数值中的每一个。因为我们使用单热编码并将问题框架化为多分类，所以我们可以在 Dense 层中使用 softmax 激活函数。

```py
Dense(100, activation='softmax')
```

我们需要将此输出层包装在 TimeDistributed 层中。这是为了确保我们可以使用输出层为输入序列中的每个项目预测一个整数。这是关键，因此我们正在实现真正的多对多模型（例如序列到序列），而不是多对一模型，其中基于内部状态和值创建一次性向量输出输入序列中最后一次观察的结果（例如，输出层一次输出 5 * 100 个值）。

这要求先前的 LSTM 层通过设置 return_sequences = True 来返回序列（例如，输入序列中的每个观察的一个输出而不是整个输入序列的一个输出）。 TimeDistributed 层执行将来自 LSTM 层的序列的每个切片应用为包装的 Dense 层的输入的技巧，以便一次可以预测一个整数。

```py
model.add(TimeDistributed(Dense(100, activation='softmax')))
```

我们将使用适用于多分类问题的日志丢失函数（categorical_crossentropy）和具有默认超参数的高效 ADAM 优化算法。

除了报告每个时期的日志损失之外，我们还将报告分类准确率，以了解我们的模型是如何训练的。

```py
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

为了确保网络不记住问题，并且网络学习为所有可能的输入序列推广解决方案，我们将在每个训练时期生成新的随机序列。 LSTM 隐藏层内的内部状态将在每个时期结束时重置。我们将为 500 个训练时期拟合模型。

```py
# train LSTM
for epoch in range(500):
	# generate new random sequence
	X,y = get_data(5, 5)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=7, verbose=2, shuffle=False)
	model.reset_states()
```

一旦拟合，我们将通过对一个新的整数随机序列做出预测来评估模型，并将解码的预期输出序列与预测的序列进行比较。

```py
# evaluate LSTM
X,y = get_data(5, 5)
yhat = model.predict(X, batch_size=7, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))
```

综合这些，下面提供了完整的代码清单。

```py
from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y

# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y

# define LSTM
n_in = 5
n_out = 5
encoded_length = 100
batch_size = 7
model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
	# generate new random sequence
	X,y = get_data(n_in, n_out)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# evaluate LSTM
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))
```

运行该示例会在每个时期打印日志丢失和准确率。通过生成新的随机序列并将预期序列与预测序列进行比较来结束运行。

根据所选择的配置，每次运行时，网络几乎都会收敛到 100％的准确度。请注意，您可能会看到不同的最终序列和最终的日志丢失。

```py
...
Epoch 1/1
0s - loss: 1.7310 - acc: 1.0000
Epoch 1/1
0s - loss: 1.5712 - acc: 1.0000
Epoch 1/1
0s - loss: 1.7447 - acc: 1.0000
Epoch 1/1
0s - loss: 1.5704 - acc: 1.0000
Epoch 1/1
0s - loss: 1.6124 - acc: 1.0000
Expected: [98, 30, 98, 11, 49] Predicted [98, 30, 98, 11, 49]
Expected: [30, 98, 11, 49, 1] Predicted [30, 98, 11, 49, 1]
Expected: [98, 11, 49, 1, 77] Predicted [98, 11, 49, 1, 77]
Expected: [11, 49, 1, 77, 80] Predicted [11, 49, 1, 77, 80]
Expected: [49, 1, 77, 80, 23] Predicted [49, 1, 77, 80, 23]
Expected: [1, 77, 80, 23, 32] Predicted [1, 77, 80, 23, 32]
Expected: [77, 80, 23, 32, 27] Predicted [77, 80, 23, 32, 27]
Expected: [80, 23, 32, 27, 66] Predicted [80, 23, 32, 27, 66]
Expected: [23, 32, 27, 66, 96] Predicted [23, 32, 27, 66, 96]
Expected: [32, 27, 66, 96, 76] Predicted [32, 27, 66, 96, 76]
Expected: [27, 66, 96, 76, 10] Predicted [27, 66, 96, 76, 10]
Expected: [66, 96, 76, 10, 39] Predicted [66, 96, 76, 10, 39]
Expected: [96, 76, 10, 39, 44] Predicted [96, 76, 10, 39, 44]
Expected: [76, 10, 39, 44, 57] Predicted [76, 10, 39, 44, 57]
Expected: [10, 39, 44, 57, 11] Predicted [10, 39, 44, 57, 11]
Expected: [39, 44, 57, 11, 48] Predicted [39, 44, 57, 11, 48]
Expected: [44, 57, 11, 48, 39] Predicted [44, 57, 11, 48, 39]
Expected: [57, 11, 48, 39, 28] Predicted [57, 11, 48, 39, 28]
Expected: [11, 48, 39, 28, 15] Predicted [11, 48, 39, 28, 15]
Expected: [48, 39, 28, 15, 49] Predicted [48, 39, 28, 15, 49]
Expected: [39, 28, 15, 49, 76] Predicted [39, 28, 15, 49, 76]
```

## 打印部分序列
（_ 编解码器模型 _）

到目前为止，这么好，但是如果我们希望输出序列的长度与输入序列的长度不同呢？

也就是说，我们希望从 5 个观测值的输入序列中打印前 2 个观测值：

```py
[1, 2, 3, 4, 5] => [1, 2]
```

这仍然是序列到序列预测问题，但需要更改网络架构。

一种方法是使输出序列具有相同的长度，并使用填充将输出序列填充到相同的长度。

或者，我们可以使用更优雅的解决方案。我们可以实现允许可变长度输出的编解码器网络，其中编码器学习输入序列的内部表示，并且解码器读取内部表示并学习如何创建相同或不同长度的输出序列。

对于网络来说，这是一个更具挑战性的问题，并且需要额外的容量（更多的内存单元）和更长的训练（更多的时期）。

网络的输入，第一个隐藏的 LSTM 层和 TimeDistributed 密集输出层保持不变，除了我们将内存单元的数量从 20 增加到 150.我们还将批量大小从 7 增加到 21，以便重量更新是在随机序列的所有样本的末尾执行。经过一些实验，发现这可以使这个网络的学习更快。

```py
model.add(LSTM(150, batch_input_shape=(21, 5, 100), stateful=True))
...
model.add(TimeDistributed(Dense(100, activation='softmax')))
```

第一个隐藏层是编码器。

我们必须添加一个额外的隐藏 LSTM 层来充当解码器。同样，我们将在此层中使用 150 个内存单元，并且与前面的示例一样，TimeDistributed 层之前的层将返回序列而不是单个值。

```py
model.add(LSTM(150, return_sequences=True, stateful=True))
```

这两层不能整齐地配合在一起。编码器层将输出 2D 数组（21,150），并且解码器期望 3D 数组作为输入（21，α，150）。

我们通过在编码器和解码器之间添加 RepeatVector（）层来解决这个问题，并确保编码器的输出重复适当的次数以匹配输出序列的长度。在这种情况下，输出序列中两个时间步长为 2 次。

```py
model.add(RepeatVector(2))
```

因此，LSTM 网络定义为：

```py
model = Sequential()
model.add(LSTM(150, batch_input_shape=(21, 5, 100), stateful=True))
model.add(RepeatVector(2))
model.add(LSTM(150, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(100, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

训练时期的数量从 500 增加到 5,000，以说明网络的额外容量。

示例的其余部分是相同的。

将这些结合在一起，下面提供了完整的代码清单。

```py
from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y

# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y

# define LSTM
n_in = 5
n_out = 2
encoded_length = 100
batch_size = 21
model = Sequential()
model.add(LSTM(150, batch_input_shape=(batch_size, n_in, encoded_length), stateful=True))
model.add(RepeatVector(n_out))
model.add(LSTM(150, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(5000):
	# generate new random sequence
	X,y = get_data(n_in, n_out)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# evaluate LSTM
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))
```

运行该示例将显示每个训练时期随机生成的序列的日志丢失和准确率。

选择的配置意味着模型将收敛到 100％的分类准确度。

生成最终随机生成的序列，并比较预期序列与预测序列。运行示例时生成的特定序列可能不同。

```py
...
Epoch 1/1
0s - loss: 0.0248 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0399 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0285 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0410 - acc: 0.9762
Epoch 1/1
0s - loss: 0.0236 - acc: 1.0000
Expected: [6, 52] Predicted [6, 52]
Expected: [52, 96] Predicted [52, 96]
Expected: [96, 45] Predicted [96, 45]
Expected: [45, 69] Predicted [45, 69]
Expected: [69, 52] Predicted [69, 52]
Expected: [52, 96] Predicted [52, 96]
Expected: [96, 11] Predicted [96, 96]
Expected: [11, 96] Predicted [11, 96]
Expected: [96, 54] Predicted [96, 54]
Expected: [54, 27] Predicted [54, 27]
Expected: [27, 48] Predicted [27, 48]
Expected: [48, 9] Predicted [48, 9]
Expected: [9, 32] Predicted [9, 32]
Expected: [32, 62] Predicted [32, 62]
Expected: [62, 41] Predicted [62, 41]
Expected: [41, 54] Predicted [41, 54]
Expected: [54, 20] Predicted [54, 20]
Expected: [20, 80] Predicted [20, 80]
Expected: [80, 63] Predicted [80, 63]
Expected: [63, 69] Predicted [63, 69]
Expected: [69, 36] Predicted [69, 36]
```

## 扩展

本节列出了您可能希望探索的教程的一些可能的扩展。

*   **线性表示**。使用了问题的分类（单热编码）框架，显着增加了模拟该问题所需的权重数量（每个随机整数 100 个）。使用线性表示（将整数缩放到 0-1 之间的值）进行探索，并将问题建模为回归。了解这将如何影响系统技能，所需的网络规模（内存单元）和训练时间（时期）。
*   **掩码缺失值**。当序列数据被构造为监督学习问题时，删除了具有缺失数据的行。使用屏蔽层或特殊值（例如-1）进行探索，以允许网络忽略或学习忽略这些值。
*   **Echo Longer Sequences** 。学会回应的部分子序列只有 2 个项目。使用编解码器网络探索打印更长的序列。请注意，您可能需要更长时间（更多迭代）训练更大的隐藏层（更多内存单元）。
*   **忽略状态**。注意仅在每个随机整数序列的所有样品的末尾清除状态而不是在序列内洗涤样品。这可能不是必需的。使用批量大小为 1 的无状态 LSTM 探索和对比模型表现（在每个序列的每个样本之后进行重量更新和状态重置）。我希望几乎没有变化。
*   **备用网络拓扑**。注意使用 TimeDistributed 包装器作为输出层，以确保使用多对多网络来模拟问题。通过一对一配置（作为功能的时间步长）或多对一（作为功能向量的输出）探索序列打印问题，并了解它如何影响所需的网络大小（内存单元）和训练时间（时代）。我希望它需要更大的网络，并需要更长的时间。

你有没有探索过这些扩展？
在下面的评论中分享您的发现。

## 摘要

在本教程中，您了解了如何开发 LSTM 循环神经网络以从随机生成的整数列表中打印序列和部分序列。

具体来说，你学到了：

*   如何生成随机的整数序列，使用单热编码表示它们，并将问题构建为监督学习问题。
*   如何开发基于序列到序列的 LSTM 网络以打印整个输入序列。
*   如何开发基于编解码器的 LSTM 网络，以打印长度与输入序列长度不同的部分输入序列。

你有任何问题吗？
在评论中发表您的问题，我会尽力回答。