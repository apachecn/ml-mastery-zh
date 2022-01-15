# 如何学习在 Keras 中用 LSTMs 回显随机整数

> 原文:[https://machinelearning master . com/learn-echo-random-integs-long-short-memory-recurrent-neural-networks/](https://machinelearningmastery.com/learn-echo-random-integers-long-short-term-memory-recurrent-neural-networks/)

最后更新于 2020 年 8 月 27 日

长短期记忆(LSTM)递归神经网络能够学习长序列数据中的顺序依赖性。

它们是一种基本技术，用于一系列最先进的结果，如图像字幕和机器翻译。

它们也很难理解，特别是如何设计一个问题来充分利用这种网络。

在本教程中，您将发现如何开发一个简单的 LSTM 递归神经网络，以学习如何在随机整数序列中回显数字。虽然这是一个微不足道的问题，但开发这个网络将提供将 LSTM 应用于一系列序列预测问题所需的技能。

完成本教程后，您将知道:

*   如何开发一个 LSTM，用于任何给定输入的简单回显问题。
*   如何避免初学者在将 LSTMs 应用于类似回显整数的数列问题时出现的错误？
*   如何开发一个健壮的 LSTM 来回应随机整数序列的最后观察。

**用我的新书[Python 的长短期记忆网络](https://machinelearningmastery.com/lstms-with-python/)启动你的项目**，包括*循序渐进教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:更新了 Keras 2.3 和 TensorFlow 2.0 的 API。

![How to Learn to Echo Random Integers with Long Short-Term Memory Recurrent Neural Networks](img/1a5775c99f7165a4ed8300a4f6fe9b77.png)

如何学习用长短期记忆递归神经网络回应随机整数
图片由[弗兰克·米歇尔](https://www.flickr.com/photos/franckmichel/14942703299/)提供，版权所有。

## 概观

本教程分为 4 个部分；它们是:

1.  生成和编码随机序列
2.  回声电流观测
3.  无语境回声滞后观察(初学者错误)
4.  回声滞后观测

### 环境

本教程假设您安装了 Python SciPy 环境。这个例子可以使用 Python 2 或 3。

本教程假设您已经安装了 Keras v2.0 或更高版本的 TensorFlow 或 Anano 后端。本教程不需要图形处理器，所有代码都可以在中央处理器中轻松运行。

本教程还假设您已经安装了 scikit-learn、Pandas、NumPy 和 Matplotlib。

如果您需要帮助来设置您的 Python 环境，请查看这篇文章:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习和深度学习的 Python 环境

## 生成和编码随机序列

第一步是编写一些代码来生成整数的随机序列，并为网络编码。

### 生成随机序列

我们可以在 Python 中使用 randint()函数生成随机整数，该函数接受两个参数，指示从中得出值的整数范围。

在本教程中，我们将把问题定义为 0 到 99 之间有 100 个唯一值的整数值。

```
randint(0, 99)
```

我们可以将它放在一个名为 generate_sequence()的函数中，该函数将生成一个所需长度的随机整数序列，默认长度设置为 25 个元素。

下面列出了该功能。

```
# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]
```

### 一种热编码随机序列

一旦我们生成了随机整数序列，我们就需要将它们转换成适合训练 LSTM 网络的格式。

一种选择是将整数重新缩放到范围[0，1]。这是可行的，需要将问题表述为回归。

我感兴趣的是预测正确的数字，而不是接近期望值的数字。这意味着我更愿意将问题框架为分类而不是回归，其中预期输出是一个类，并且有 100 个可能的类值。

在这种情况下，我们可以使用整数值的一种热编码，其中每个值由一个 100 个元素的二进制向量表示，该向量是除了标记为 1 的整数索引之外的所有“0”值。

下面称为 one_hot_encode()的函数定义了如何迭代整数序列，并为每个整数创建二进制向量表示，并将结果作为二维数组返回。

```
# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

我们还需要解码编码值，以便我们可以利用预测，在这种情况下，只需回顾它们。

一个热编码可以通过使用 argmax() NumPy 函数来反转，该函数返回向量中具有最大值的值的索引。

下面这个名为 one_hot_decode()的函数将对一个编码序列进行解码，并可用于以后解码来自我们网络的预测。

```
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

### 完整示例

我们可以把这一切联系起来。

下面是完整的代码清单，用于生成 25 个随机整数的序列，并将每个整数编码为二进制向量。

```
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

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先打印 25 个随机整数的列表，然后是序列中所有整数的二进制表示的截断视图，每行一个向量，然后是解码后的序列。

```
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

现在我们知道了如何准备和表示整数的随机序列，我们可以看看如何使用 LSTMs 来学习它们。

## 回声电流观测

让我们从一个更简单的回声问题开始。

在本节中，我们将开发一个 LSTM 来回应当前的观察。即给定一个随机整数作为输入，返回与输出相同的整数。

或者更正式的说法是:

```
yhat(t) = f(X(t))
```

也就是说，该模型是将当前时间的值(yhat(t))预测为当前时间的观测值(X(t))的函数(f())。

这是一个简单的问题，因为不需要内存，只需要一个将输入映射到相同输出的函数。

这是一个微不足道的问题，将展示一些有用的东西:

*   如何使用上面的问题表征机制？
*   如何在 Keras 中使用 LSTMs？
*   一个 LSTM 人学习如此琐碎问题的能力。

这将为后面的滞后观测回波奠定基础。

首先，我们将开发一个函数来准备一个随机序列，准备训练或评估一个 LSTM。这个函数必须首先生成一个随机的整数序列，使用一个热编码，然后将输入数据转换成一个三维数组。

LSTMs 需要由维度[样本、时间步长、特征]组成的 3D 输入。我们的问题将由每个序列 25 个例子、1 个时间步长和一个热编码的 100 个特征组成。

下面列出了这个名为 generate_data()的函数。

```
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

模型必须指定输入数据的预期维度。在这种情况下，根据时间步长(1)和特征(100)。我们将使用具有 15 个存储单元的单个隐藏层 LSTM。

输出层是一个完全连接的层(密集层)，对于可能输出的 100 个可能的整数，有 100 个神经元。输出层使用 softmax 激活功能，允许网络学习并输出可能输出值的分布。

该网络在训练时将使用对数损失函数，适用于多类分类问题，以及高效的 ADAM 优化算法。精度度量将在每个训练时期报告，以便除了损失之外，还能了解模型的技能。

```
# define model
model = Sequential()
model.add(LSTM(15, input_shape=(1, 100)))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

我们将通过用新生成的序列手工运行每个纪元来手动调整模型。该模型将适合 500 个时代，或者换句话说，在 500 个随机生成的序列上训练。

这将鼓励网络学习再现实际输入，而不是记忆固定的训练数据集。

```
# fit model
for i in range(500):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
```

一旦模型被拟合，我们将对一个新的序列进行预测，并将预测的输出与预期的输出进行比较。

```
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

下面列出了完整的示例。

```
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

运行该示例会打印每个时期的日志丢失和准确性。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

网络有点过了，内存单元和训练时期比这么简单的问题需要的多，你可以从网络很快达到 100%的准确率这一事实看出这一点。

在运行结束时，将预测的序列与随机生成的序列进行比较，两者看起来完全相同。

```
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

既然我们知道了如何使用工具来创建和表示随机序列，以及如何拟合 LSTM 来学习呼应当前序列，那么让我们看看如何使用 LSTMs 来学习如何呼应过去的观察。

## 无语境回声滞后观察
( *初学者的错误*)

预测滞后观测值的问题可以更正式地定义如下:

```
yhat(t) = f(X(t-n))
```

其中当前时间步长 yhat(t)的预期输出被定义为特定先前观测值(X(t-n))的函数(f())。

LSTMs 的承诺表明，你可以一次向网络展示一个例子，网络将使用内部状态来学习和充分记住先前的观察，以解决这个问题。

让我们试试这个。

首先，我们必须更新 generate_data()函数并重新定义问题。

我们将使用编码序列的移位版本作为输入，使用编码序列的截断版本作为输出，而不是使用相同的序列作为输入和输出。

这些变化是必需的，以便取一系列数字，如[1，2，3，4]，并将它们转化为具有输入(X)和输出(y)分量的监督学习问题，如:

```
X y
1, NaN
2, 1
3, 2
4, 3
NaN, 4
```

在本例中，您可以看到第一行和最后一行没有包含足够的数据供网络学习。这可能会被标记为“无数据”值并被屏蔽，但更简单的解决方案是将其从数据集中移除。

更新后的 generate_data()函数如下所示:

```
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

我们必须测试数据的更新表示，以确认它符合我们的预期。为此，我们可以生成一个序列，并检查该序列上的解码 X 和 y 值。

```
X, y = generate_data()
for i in range(len(X)):
	a, b = argmax(X[i,0]), argmax(y[i])
	print(a, b)
```

下面提供了这种健全性检查的完整代码清单。

```
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

运行该示例会打印问题框架的 X 和 y 部分。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到，考虑到冷启动，第一种模式对于网络来说很难(不可能)学会。我们可以通过数据看到 yhat(t)= X(t-1)的预期模式。

```
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

网络设计相似，但有一个小变化。

每次向网络显示一个观察值，并执行权重更新。因为我们期望观察之间的状态携带学习先前观察所需的信息，所以我们需要确保在每个批次之后该状态不被重置(在这种情况下，一个批次是一个训练观察)。我们可以通过使 LSTM 层有状态并在状态重置时手动管理来做到这一点。

这包括在 LSTM 图层上将有状态参数设置为真，并使用包含维度[batchsize，timesteps，features]的 batch_input_shape 参数定义输入形状。

对于给定的随机序列，有 24 个 X，y 对，因此使用 6 的批次大小(4 批次 6 个样本= 24 个样本)。请记住，序列被分解为样本，样本可以在网络权重更新之前分批显示给网络。使用 50 个内存单元的大网络大小，再次超出了解决问题所需的容量。

```
model.add(LSTM(50, batch_input_shape=(6, 1, 100), stateful=True))
```

接下来，在每个时期(随机生成的序列的一次迭代)之后，可以手动重置网络的内部状态。该模型适用于 2000 个训练时期，注意不要在一个序列中打乱样本。

```
# fit model
for i in range(2000):
	X, y = generate_data()
	model.fit(X, y, epochs=1, batch_size=6, verbose=2, shuffle=False)
	model.reset_states()
```

综上所述，下面列出了完整的示例。

```
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

运行这个例子给出了一个令人惊讶的结果。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

该问题无法学习，训练以模型结束，该模型以 0%的精度回应序列中的最后一个观察。

```
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

这怎么可能呢？

### 初学者的错误

这是初学者经常犯的错误，如果你曾经接触过 RNNs 或 LSTMs，那么你会发现上面的这个错误。

具体来说，LSTMs 的力量确实来自于所保持的习得的内部状态，但这种状态只有在被训练为过去观察的函数时才是强大的。

换句话说，您必须向网络提供预测的上下文(例如，可能包含时间相关性的观察)作为输入的时间步长。

上面的公式训练网络只学习当前输入值的输出函数，如第一个例子:

```
yhat(t) = f(X(t))
```

不是作为最后 n 次观察的函数，或者甚至只是我们要求的前一次观察:

```
yhat(t) = f(X(t-1))
```

为了学习这种未知的时间依赖性，LSTM 每次只需要一个输入，但是为了学习这种依赖性，它必须在序列上执行反向传播。您必须提供序列的过去观察作为上下文。

您没有定义窗口(如多层感知器的情况，其中每个过去的观察都是加权输入)；相反，你正在定义一个历史观测的范围，LSTM 将试图从中学习时间相关性(f(X(t-1)，… X(t-n))。

需要明确的是，这是初学者在 Keras 中使用 LSTMs 时的错误，不一定是一般的错误。

## 回声滞后观测

现在，我们已经绕过了初学者的一个常见陷阱，我们可以开发一个 LSTM 来重复前面的观察。

第一步是重新定义问题。

我们知道，为了做出正确的预测，网络只需要最后一次观察作为输入。但是，我们希望网络了解哪些过去的观察结果可以得到回应，以便正确解决这个问题。因此，我们将提供最后 5 个观察的子序列作为上下文。

具体来说，如果我们的序列包含:[1，2，3，4，5，6，7，8，9，10]，则 X，y 对如下所示:

```
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

在这种情况下，您可以看到前 5 行和最后 1 行没有包含足够的数据，因此在这种情况下，我们将删除它们。

我们将使用 Pandas shift()函数创建序列的移位版本，使用 Pandas concat()函数将移位的序列重新组合在一起。然后，我们将手动排除不可行的行。

下面列出了更新后的 generate_data()函数。

```
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

同样，我们可以通过生成一个序列并比较解码的 X，y 对来检查这个更新的函数。下面列出了完整的示例。

```
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

运行该示例显示最后 5 个值的上下文作为输入，最后一个先前的观察(X(t-1))作为输出。

```
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

我们现在可以开发一个 LSTM 来学习这个问题。

给定序列有 20 个 X，y 对；因此，选择批量为 5(4 批 5 个实施例= 20 个样品)。

同样的结构用于 LSTM 隐藏层的 50 个存储单元和输出层的 100 个神经元。该网络适用于 2000 个纪元，每个纪元后内部状态都会重置。

完整的代码列表如下。

```
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

运行实例表明，该网络能够拟合问题，并正确地学习在 5 个先验观测值的背景下将 X(t-1)观测值作为预测值返回。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

输出示例如下。

```
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

## 扩展ˌ扩张

本节列出了本教程中实验的一些扩展。

*   **忽略内部状态**。通过在纪元结束时手动重置状态，注意保留序列内所有样本的 LSTMs 内部状态。我们知道，网络已经通过时间步长获得了每个样本中所需的所有上下文和状态。探索额外的交叉样本状态是否增加了模型的技能。
*   **屏蔽缺失数据**。在数据准备期间，删除了缺少数据的行。探索用特殊值(例如-1)标记缺失值的用法，看看 LSTM 是否可以从这些例子中学习。此外，探索如何使用掩蔽层作为输入，并探索如何掩蔽丢失的值。
*   **作为时间步长的整个序列**。仅提供了最后 5 个观察的上下文，作为学习呼应的上下文。探索使用整个随机序列作为每个样本的上下文，随着序列的展开而构建。这可能需要填充甚至屏蔽缺失值，以满足 LSTM 的固定大小输入的期望。
*   **回声不同滞后值**。在回声示例中使用了特定的滞后值(t-1)。探索在回声中使用不同的滞后值，以及这如何影响模型技能、训练时间和 LSTM 层大小等属性。我希望可以使用相同的模型结构来学习每个滞后。
*   **回波滞后序列**。这个网络被训练来回应一个特定的滞后观测。探索滞后序列的变化。这可能需要在网络的输出端使用时间分布层来实现顺序到顺序的预测。

你探索过这些扩展吗？
在下面的评论中分享你的发现。

## 摘要

在本教程中，您发现了如何开发一个 LSTM 来解决从随机整数序列中回显滞后观测值的问题。

具体来说，您了解到:

*   如何为问题生成和编码测试数据？
*   如何避免初学者在尝试用 LSTMs 解决这个和类似问题时犯的错误。
*   如何开发一个健壮的 LSTM 来回应随机整数序列中的整数。

你有什么问题吗？
在评论中提问，我会尽力回答。