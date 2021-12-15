# 如何使用长短期记忆循环神经网络来打印随机整数

> 原文： [https://machinelearningmastery.com/learn-echo-random-integers-long-short-term-memory-recurrent-neural-networks/](https://machinelearningmastery.com/learn-echo-random-integers-long-short-term-memory-recurrent-neural-networks/)

长短期记忆（LSTM）循环神经网络能够学习长序列数据中的顺序依赖性。

它们是用于一系列最先进结果的基本技术，例如图像字幕和机器翻译。

它们也很难理解，特别是如何构建问题以充分利用这种类型的网络。

在本教程中，您将了解如何开发一个简单的 LSTM 循环神经网络，以学习如何在随机整数的特殊序列中打印数字。虽然这是一个微不足道的问题，但开发这个网络将提供将 LSTM 应用于一系列序列预测问题所需的技能。

完成本教程后，您将了解：

*   如何为打印任何给定输入的简单问题开发 LSTM。
*   如何在应用 LSTM 来排序问题时避免初学者的错误，例如打印整数。
*   如何开发一个强大的 LSTM 来回应随机整数的特殊序列的最后一次观察。

让我们开始吧。

![How to Learn to Echo Random Integers with Long Short-Term Memory Recurrent Neural Networks](img/f7e4c5f43c327b02a63ca4cfbe163428.jpg)

如何学习使用长短期记忆循环神经网络回应随机整数
照片由 [Franck Michel](https://www.flickr.com/photos/franckmichel/14942703299/) ，保留一些权利。

## 概观

本教程分为 4 个部分;他们是：

1.  生成和编码随机序列
2.  打印电流观察
3.  没有语境的打印滞后观察（初学者错误）
4.  打印滞后观察

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

本教程假设您安装了 TensorFlow 或 Theano 后端的 Keras v2.0 或更高版本。本教程不需要 GPU，所有代码都可以在 CPU 中轻松运行。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 生成和编码随机序列

第一步是编写一些代码来生成随机的整数序列，并为网络编码。

### 生成随机序列

我们可以使用 randint（）函数在 Python 中生成随机整数，该函数接受两个参数，指示从中绘制值的整数范围。

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

通过使用 argmax（）NumPy 函数可以反转单热编码，该函数返回具有最大值的向量中的值的索引。

下面的函数名为 one_hot_decode（），将对编码序列进行解码，并可用于稍后解码来自我们网络的预测。

```py
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

### 完整的例子

我们可以将所有这些结合在一起。

下面是生成 25 个随机整数序列并将每个整数编码为二进制向量的完整代码清单。

```py
from random import randint
from numpy import array
from numpy import argmax

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

# generate random sequence
sequence = generate_sequence()
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence)
print(encoded)
# one hot decode
decoded = one_hot_decode(encoded)
print(decoded)
```

运行该示例首先打印 25 个随机整数的列表，然后是序列中所有整数的二进制表示的截断视图，每行一个向量，然后再次解码序列。

您可能会得到不同的结果，因为每次运行代码时都会生成不同的随机整数。

```py
[37, 99, 40, 98, 44, 27, 99, 18, 52, 97, 46, 39, 60, 13, 66, 29, 26, 4, 65, 85, 29, 88, 8, 23, 61]
[[0 0 0 ..., 0 0 0]
[0 0 0 ..., 0 0 1]
[0 0 0 ..., 0 0 0]
...,
[0 0 0 ..., 0 0 0]
[0 0 0 ..., 0 0 0]
[0 0 0 ..., 0 0 0]]
[37, 99, 40, 98, 44, 27, 99, 18, 52, 97, 46, 39, 60, 13, 66, 29, 26, 4, 65, 85, 29, 88, 8, 23, 61]
```

现在我们知道如何准备和表示整数的随机序列，我们可以看一下使用 LSTM 来学习它们。

## 打印电流观察

让我们从一个更简单的打印问题开始。

在本节中，我们将开发一个 LSTM 来回应当前的观察结果。给出一个随机整数作为输入，返回与输出相同的整数。

或者稍微更正式地说：

```py
yhat(t) = f(X(t))
```

也就是说，该模型是将当前时间的值（yhat（t））预测为当前时间（X（t））的观测值的函数（f（））。

这是一个简单的问题，因为不需要内存，只是将输入映射到相同输出的函数。

这是一个微不足道的问题，并将展示一些有用的东西：

*   如何使用上面的问题表示机制。
*   如何在 Keras 中使用 LSTM。
*   LSTM 的能力需要学习这样一个微不足道的问题。

这将为接下来的滞后观察的打印奠定基础。

首先，我们将开发一个函数来准备随机序列，以便训练或评估 LSTM。此函数必须首先生成随机的整数序列，使用单热编码，然后将输入数据转换为三维数组。

LSTM 需要包含尺寸[样本，时间步长，特征]的 3D 输入。我们的问题将包括每个序列 25 个示例，1 个时间步长和 100 个用于单热编码的特征。

下面列出了此函数，名为 generate_data（）。

```py
# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to 3d for input
	X = encoded.reshape(encoded.shape[0], 1, encoded.shape[1])
	return X, encoded
```

接下来，我们可以定义我们的 LSTM 模型。

模型必须指定输入数据的预期维度。在这种情况下，就时间步长（1）和特征（100）而言。我们将使用具有 15 个内存单元的单个隐藏层 LSTM。

输出层是一个完全连接的层（密集），有 100 个神经元，可以输出 100 个可能的整数。在输出层上使用 softmax 激活功能，以允许网络在可能的输出值上学习和输出分布。

网络将在训练时使用对数丢失函数，适用于多分类问题，以及高效的 ADAM 优化算法。每个训练时期都会报告准确度量，以便了解除损失之外的模型技能。

```py
# define model
model = Sequential()
model.add(LSTM(15, input_shape=(1, 100)))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

我们将通过手动运行每个迭代与新生成的序列手动拟合模型。该模型将适合 500 个时期，或以另一种方式说明，训练 500 个随机生成的序列。

这将鼓励网络学习重现实际输入而不是记忆固定的训练数据集。

```py
# fit model
for i in range(500):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
```

一旦模型拟合，我们将对新序列做出预测，并将预测输出与预期输出进行比较。

```py
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

下面列出了完整的示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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

# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to 3d for input
	X = encoded.reshape(encoded.shape[0], 1, encoded.shape[1])
	return X, encoded

# define model
model = Sequential()
model.add(LSTM(15, input_shape=(1, 100)))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model
for i in range(500):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

运行该示例会在每个时期打印日志丢失和准确率。

网络有点过度规划，拥有比这个简单问题所需的更多的内存单元和训练时期，你可以通过网络快速达到 100％的准确率来看到这一点。

在运行结束时，将预测的序列与随机生成的序列进行比较，并且两者看起来相同。

您的具体结果可能会有所不同，但您的网络应该达到 100％的准确率，因为网络规模较大，训练时间长于问题所需的时间。

```py
...
0s - loss: 0.0895 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0785 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0789 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0832 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0927 - acc: 1.0000
Expected: [18, 41, 49, 56, 86, 25, 96, 3, 75, 24, 57, 95, 81, 44, 2, 22, 76, 34, 41, 4, 69, 47, 1, 97, 57]
Predicted: [18, 41, 49, 56, 86, 25, 96, 3, 75, 24, 57, 95, 81, 44, 2, 22, 76, 34, 41, 4, 69, 47, 1, 97, 57]
```

现在我们知道如何使用工具来创建和表示随机序列并使 LSTM 适合学习回应当前序列，让我们看看如何使用 LSTM 来学习如何回应过去的观察。

## 无背景的打印滞后观察
（_ 初学者错误 _）

预测滞后观察的问题可以更正式地定义如下：

```py
yhat(t) = f(X(t-n))
```

其中当前时间步长 yhat（t）的预期输出被定义为特定先前观察（X（t-n））的函数（f（））。

LSTM 的承诺表明，您可以一次向网络显示一个示例，并且网络将使用内部状态来学习并充分记住先前的观察以解决此问题。

我们来试试吧。

首先，我们必须更新 generate_data（）函数并重新定义问题。

我们将使用编码序列的移位版本作为输入，并使用截断版本的编码序列作为输出，而不是使用相同的序列进行输入和输出。

需要进行这些更改才能获取一系列数字，例如[1,2,3,4]，并将它们转换为带有输入（X）和输出（y）组件的监督学习问题，例如：

```py
X y
1, NaN
2, 1
3, 2
4, 3
NaN, 4
```

在此示例中，您可以看到第一行和最后一行不包含足够的数据供网络学习。这可以被标记为“无数据”值并被屏蔽，但更简单的解决方案是简单地从数据集中删除它。

更新的 generate_data（）函数如下所示：

```py
# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# drop first value from X
	X = encoded[1:, :]
	# convert to 3d for input
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# drop last value from y
	y = encoded[:-1, :]
	return X, y
```

我们必须测试这个更新的数据表示，以确认它符合我们的预期。为此，我们可以生成序列并查看序列上的解码 X 和 y 值。

```py
X, y = generate_data()
for i in range(len(X)):
	a, b = argmax(X[i,0]), argmax(y[i])
	print(a, b)
```

下面提供了此完整性检查的完整代码清单。

```py
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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

# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# drop first value from X
	X = encoded[1:, :]
	# convert to 3d for input
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# drop last value from y
	y = encoded[:-1, :]
	return X, y

# test data generator
X, y = generate_data()
for i in range(len(X)):
	a, b = argmax(X[i,0]), argmax(y[i])
	print(a, b)
```

运行该示例将打印问题框架的 X 和 y 组件。

我们可以看到，在冷启动的情况下，网络学习的第一个模式很难（不可能）。我们可以通过数据看到 yhat（t）== X（t-1）的预期模式。

```py
78 65
7 78
16 7
11 16
23 11
99 23
39 99
53 39
82 53
6 82
18 6
17 18
49 17
4 49
34 4
77 34
46 77
22 46
40 22
76 40
85 76
87 85
17 87
75 17
```

网络设计类似，但只有一个小的变化。

一次一个地向网络显示观察并且执行重量更新。因为我们期望观察之间的状态携带学习先前观察所需的信息，所以我们需要确保在每批之后不重置该状态（在这种情况下，一批是一次训练观察）。我们可以通过使 LSTM 层有状态并在状态重置时手动管理来实现。

这涉及在 LSTM 层上将有状态参数设置为 True，并使用包含尺寸[batchsize，timesteps，features]的 batch_input_shape 参数定义输入形状。

对于给定的随机序列，存在 24 个 X，y 对，因此使用 6 的批量大小（4 批 6 个样品= 24 个样品）。请记住，序列被分解为样本，并且在执行网络权重更新之前，样本可以批量显示给网络。使用 50 个内存单元的大型网络，再次过度规定问题所需的容量。

```py
model.add(LSTM(50, batch_input_shape=(6, 1, 100), stateful=True))
```

接下来，在每个时期（随机生成的序列的一次迭代）之后，可以手动重置网络的内部状态。该模型适用于 2,000 个训练时期，并且注意不要使序列内的样本混洗。

```py
# fit model
for i in range(2000):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=6, verbose=2, shuffle=False)
	model.reset_states()
```

综合这些，下面列出了完整的例子。

```py
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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

# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# drop first value from X
	X = encoded[1:, :]
	# convert to 3d for input
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# drop last value from y
	y = encoded[:-1, :]
	return X, y

# define model
model = Sequential()
model.add(LSTM(50, batch_input_shape=(6, 1, 100), stateful=True))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model
for i in range(2000):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=6, verbose=2, shuffle=False)
	model.reset_states()
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X, batch_size=6)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

运行该示例会产生令人惊讶的结果。

无法学习问题，训练以 0％精确度回归序列中最后一次观察的模型结束。

```py
...
Epoch 1/1
0s - loss: 4.6042 - acc: 0.0417
Epoch 1/1
0s - loss: 4.6215 - acc: 0.0000e+00
Epoch 1/1
0s - loss: 4.5802 - acc: 0.0000e+00
Epoch 1/1
0s - loss: 4.6023 - acc: 0.0000e+00
Epoch 1/1
0s - loss: 4.6071 - acc: 0.0000e+00
Expected: [71, 44, 6, 11, 91, 23, 55, 37, 53, 4, 42, 15, 81, 6, 57, 97, 49, 69, 56, 86, 70, 12, 61, 48]
Predicted: [49, 49, 49, 87, 49, 96, 96, 96, 96, 96, 85, 96, 96, 96, 96, 96, 96, 96, 49, 49, 87, 96, 49, 49]
```

怎么会这样？

### 初学者的错误

这是初学者常犯的错误，如果您使用 RNN 或 LSTM 进行了阻塞，那么您可能会发现上述错误。

具体而言，LSTM 的力量确实来自所保持的学习内部状态，但是如果它被训练为过去观察的函数，则该状态仅是强大的。

换句话说，您必须为网络提供预测的上下文（例如，可能包含时间依赖性的观察）作为输入上的时间步长。

上述公式训练网络将输出学习为仅当前输入值的函数，如第一个示例所示：

```py
yhat(t) = f(X(t))
```

根据我们的要求，不是最后 n 个观测值的函数，甚至不是之前的观测值：

```py
yhat(t) = f(X(t-1))
```

LSTM 一次只需要一个输入来学习这种未知的时间依赖性，但它必须对序列进行反向传播才能学习这种依赖性。您必须提供序列的过去观察结果作为上下文。

您没有定义窗口（如多层感知机的情况，其中每个过去的观察是加权输入）;相反，您正在定义历史观测的范围，LSTM 将尝试从中学习时间依赖性（f（X（t-1），... X（t-n）））。

需要明确的是，这是在 Keras 中使用 LSTM 时的初学者的错误，而不一定是一般的。

## 打印滞后观察

现在我们已经为初学者找到了常见的陷阱，我们可以开发一个 LSTM 来回应之前的观察。

第一步是再次重新定义问题的定义。

我们知道网络只需要最后一次观察作为输入，以便做出正确的预测。但是我们希望网络能够了解哪些过去的观察结果是为了正确解决这个问题。因此，我们将提供最后 5 个观察的子序列作为上下文。

具体来说，如果我们的序列包含：[1,2,3,4,5,6,7,8,9,10]，则 X，y 对看起来如下：

```py
X, 							y
NaN, NaN, NaN, NaN, NaN, 	NaN
NaN, NaN, NaN, NaN, 1, 		NaN
NaN, NaN, NaN, 1, 2, 		1
NaN, NaN, 1, 2, 3, 			2
NaN, 1, 2, 3, 4, 			3
1, 2, 3, 4, 5, 				4
2, 3, 4, 5, 6, 				5
3, 4, 5, 6, 7, 				6
4, 5, 6, 7, 8, 				7
5, 6, 7, 8, 9, 				8
6, 7, 8, 9, 10, 			9
7, 8, 9, 10, NaN, 			10
```

在这种情况下，您可以看到前 5 行和后 1 行不包含足够的数据，因此在这种情况下，我们将删除它们。

我们将使用 Pandas shift（）函数创建序列的移位版本，并使用 Pandas concat（）函数将移位序列重新组合在一起。然后，我们将手动排除不可行的行。

下面列出了更新的 generate_data（）函数。

```py
# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# create lag inputs
	df = DataFrame(encoded)
	df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
	# remove non-viable rows
	values = df.values
	values = values[5:,:]
	# convert to 3d for input
	X = values.reshape(len(values), 5, 100)
	# drop last value from y
	y = encoded[4:-1,:]
	print(X.shape, y.shape)
	return X, y
```

同样，我们可以通过生成序列并比较解码的 X，y 对来理智地检查这个更新的函数。下面列出了完整的示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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

# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# create lag inputs
	df = DataFrame(encoded)
	df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
	# remove non-viable rows
	values = df.values
	values = values[5:,:]
	# convert to 3d for input
	X = values.reshape(len(values), 5, 100)
	# drop last value from y
	y = encoded[4:-1,:]
	return X, y

# test data generator
X, y = generate_data()
for i in range(len(X)):
	a, b, c, d, e, f = argmax(X[i,0]), argmax(X[i,1]), argmax(X[i,2]), argmax(X[i,3]), argmax(X[i,4]), argmax(y[i])
	print(a, b, c, d, e, f)
```

运行该示例将最后 5 个值的上下文显示为输入，将最后一个先前观察（X（t-1））显示为输出。

```py
57 96 99 77 44 77
96 99 77 44 45 44
99 77 44 45 28 45
77 44 45 28 70 28
44 45 28 70 73 70
45 28 70 73 74 73
28 70 73 74 73 74
70 73 74 73 64 73
73 74 73 64 29 64
74 73 64 29 15 29
73 64 29 15 94 15
64 29 15 94 98 94
29 15 94 98 89 98
15 94 98 89 52 89
94 98 89 52 96 52
98 89 52 96 46 96
89 52 96 46 46 46
52 96 46 46 85 46
96 46 46 85 49 85
46 46 85 49 59 49
```

我们现在可以开发 LSTM 来学习这个问题。

给定序列有 20 个 X，y 对;因此，选择批量大小为 5（4 批 5 个实例= 20 个样品）。

使用相同的结构，LSTM 隐藏层中有 50 个内存单元，输出层中有 100 个神经元。该网络适合 2000 个时期，每个时期后内部状态重置。

完整的代码清单如下。

```py
from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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

# generate data for the lstm
def generate_data():
	# generate sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# create lag inputs
	df = DataFrame(encoded)
	df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
	# remove non-viable rows
	values = df.values
	values = values[5:,:]
	# convert to 3d for input
	X = values.reshape(len(values), 5, 100)
	# drop last value from y
	y = encoded[4:-1,:]
	return X, y

# define model
model = Sequential()
model.add(LSTM(50, batch_input_shape=(5, 5, 100), stateful=True))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# fit model
for i in range(2000):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=5, verbose=2, shuffle=False)
	model.reset_states()
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X, batch_size=5)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

运行该示例表明网络可以适应问题并且正确地学习在先前 5 次观察的上下文中将 X（t-1）观测作为预测返回。

示例输出如下;根据不同的随机序列，您的具体输出可能不同

```py
...
Epoch 1/1
0s - loss: 0.1763 - acc: 1.0000
Epoch 1/1
0s - loss: 0.2393 - acc: 0.9500
Epoch 1/1
0s - loss: 0.1674 - acc: 1.0000
Epoch 1/1
0s - loss: 0.1256 - acc: 1.0000
Epoch 1/1
0s - loss: 0.1539 - acc: 1.0000
Expected: [24, 49, 86, 73, 51, 6, 6, 52, 34, 32, 0, 14, 83, 16, 37, 75, 41, 40, 80, 33]
Predicted: [24, 49, 86, 73, 51, 6, 6, 52, 34, 32, 0, 14, 83, 16, 37, 75, 41, 40, 80, 33]
```

## 扩展

本节列出了本教程中实验的一些扩展。

*   **忽略内部状态**。注意通过在时期结束时手动重置状态来保持序列内样本的 LSTM 的内部状态。我们知道网络已经具有通过时间步长在每个样本中所需的所有上下文和状态。探索额外的跨样本状态是否为模型的技能带来任何好处。
*   **掩码丢失数据**。在数据准备期间，删除了缺少数据的行。探索使用特殊值（例如-1）标记缺失值并查看 LSTM 是否可以从这些示例中学习。还要探索使用蒙版层作为输入并探索掩盖缺失值。
*   **整个序列为时间步**。仅提供了最后 5 个观测值的背景作为学习打印的上下文。探索使用整个随机序列作为每个样本的上下文，随着序列的展开而构建。这可能需要填充甚至屏蔽缺失值以满足对 LSTM 的固定大小输入的期望。
*   **打印不同滞后值**。在回波示例中使用特定滞后值（t-1）。探索在打印中使用不同的滞后值以及它如何影响属性，例如模型技能，训练时间和 LSTM 层大小。我希望每个滞后都可以使用相同的模型结构来学习。
*   **回波延迟序列**。训练该网络以回应特定的滞后观察。探索打印滞后序列的变体。这可能需要在网络输出上使用 TimeDistributed 层来实现序列到序列预测。

你有没有探索过这些扩展？
在下面的评论中分享您的发现。

## 摘要

在本教程中，您了解了如何开发 LSTM 以解决从随机整数序列打印滞后观察的问题。

具体来说，你学到了：

*   如何为问题生成和编码测试数据。
*   在尝试解决 LSTM 的这个问题和类似问题时，如何避免初学者的错误。
*   如何开发一个强大的 LSTM 来回应随机整数的特殊序列中的整数。

你有任何问题吗？
在评论中提出您的问题，我会尽力回答。