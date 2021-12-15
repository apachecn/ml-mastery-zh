# 如何在使用LSTM进行训练和预测时使用不同的批量大小

> 原文： [https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/)

Keras使用快速符号数学库作为后端，例如TensorFlow和Theano。

使用这些库的缺点是，无论您是在训练网络还是做出预测，数据的形状和大小都必须预先定义并保持不变。

在序列预测问题上，可能需要在训练网络时使用大批量大小，并且在做出预测时使用批量大小为1以便预测序列中的下一步骤。

在本教程中，您将了解如何解决此问题，甚至在训练和预测期间使用不同的批量大小。

完成本教程后，您将了解：

*   如何设计简单的序列预测问题并开发LSTM来学习它。
*   如何改变在线和基于批量的学习和预测的LSTM配置。
*   如何改变用于训练的批量大小与用于预测的批量大小。

让我们开始吧。

![How to use Different Batch Sizes for Training and Predicting in Python with Keras](img/fe6017057826b04e06e50385d7cc375e.jpg)

如何使用不同的批量大小进行Python的训练和预测与Keras
照片由 [steveandtwyla](https://www.flickr.com/photos/25303648@N08/5751682588/) ，保留一些权利。

## 教程概述

本教程分为6个部分，如下所示：

1.  批量大小
2.  序列预测问题描述
3.  LSTM模型和变化的批量大小
4.  解决方案1：在线学习（批量大小= 1）
5.  解决方案2：批量预测（批量大小= N）
6.  解决方案3：复制权重

### 教程环境

假设Python 2或3环境已安装并正常工作。

这包括SciPy与NumPy和Pandas。必须使用TensorFlow或Keras后端安装Keras 2.0或更高版本。

有关设置Python环境的帮助，请参阅帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 批量大小

使用Keras的一个好处是它建立在象征性数学库之上，如TensorFlow和Theano，可实现快速高效的计算。大型神经网络需要这样做。

使用这些高效库的一个缺点是您必须预先定义数据的范围。具体来说，批量大小。

批量大小限制在可以执行权重更新之前要显示给网络的样本数。当使用拟合模型做出预测时，会施加同样的限制。

具体而言，在拟合模型时使用的批量大小控制着您一次必须进行多少次预测。

当您希望一次进行与训练期间使用的批量大小相同的数字预测时，这通常不是问题。

当您希望进行的预测少于批量大小时，这确实会成为一个问题。例如，您可以获得批量较大的最佳结果，但需要在时间序列或序列问题等方面对一次观察做出预测。

这就是为什么在将网络拟合到训练数据时可能需要具有与在对测试数据或新输入数据做出预测时不同的批量大小的原因。

在本教程中，我们将探索解决此问题的不同方法。

## 序列预测问题描述

我们将使用简单的序列预测问题作为上下文来演示在训练和预测之间改变批量大小的解决方案。

序列预测问题为不同的批量大小提供了一个很好的案例，因为您可能希望批量大小等于训练期间的训练数据集大小（批量学习），并且在对一步输出做出预测时批量大小为1。

序列预测问题涉及学习预测以下10步序列中的下一步：

```py
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

我们可以在Python中创建这个序列，如下所示：

```py
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
```

运行该示例打印我们的序列：

```py
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

我们必须将序列转换为监督学习问题。这意味着当0.0显示为输入模式时，网络必须学会将下一步预测为0.1。

我们可以使用Pandas _shift（）_函数在Python中执行此操作，如下所示：

```py
from pandas import concat
from pandas import DataFrame
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
print(df)
```

运行该示例显示所有输入和输出对。

```py
1  0.1  0.0
2  0.2  0.1
3  0.3  0.2
4  0.4  0.3
5  0.5  0.4
6  0.6  0.5
7  0.7  0.6
8  0.8  0.7
9  0.9  0.8
```

我们将使用称为长短期记忆网络的循环神经网络来学习序列。因此，我们必须将输入模式从2D数组（1列9行）转换为由[_行，时间步长，列_]组成的3D数组，其中时间步长为1，因为我们只有一个时间步长观察每一行。

我们可以使用NumPy函数 _reshape（）_执行此操作，如下所示：

```py
from pandas import concat
from pandas import DataFrame
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
print(X.shape, y.shape)
```

运行该示例创建`X`和`y`阵列，准备与LSTM一起使用并打印其形状。

```py
(9, 1, 1) (9,)
```

## LSTM模型和变化的批量大小

在本节中，我们将针对该问题设计LSTM网络。

训练批量大小将覆盖整个训练数据集（批量学习），并且将一次一个地做出预测（一步预测）。我们将证明虽然模型能够解决问题，但一步预测会导致错误。

我们将使用适合1000个时期的LSTM网络。

权重将在每个训练时期结束时更新（批量学习），这意味着批量大小将等于训练观察的数量（9）。

对于这些实验，我们将需要对LSTM的内部状态何时更新进行细粒度控制。通常LSTM状态在Keras的每个批次结束时被清除，但是我们可以通过使LSTM有状态并且调用 _model.reset_state（）_来手动管理该状态来控制它。这将在后面的章节中提到。

网络有一个输入，一个隐藏层有10个单元，一个输出层有1个单元。默认的tanh激活函数用于LSTM单元，而线性激活函数用于输出层。

使用有效的ADAM优化算法将均方误差优化函数用于该回归问题。

以下示例配置并创建网络。

```py
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

我们将网络适合每个时期的所有示例，并在每个时期结束时手动重置网络状态。

```py
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
```

最后，我们将一次预测序列中的每个步骤。

这需要批量大小为1，这与用于适合网络的批量大小9不同，并且在运行示例时将导致错误。

```py
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = model.predict(testX, batch_size=1)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
```

下面是完整的代码示例。

```py
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = model.predict(testX, batch_size=1)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
```

运行该示例可以很好地匹配模型，并在做出预测时导致错误。

报告的错误如下：

```py
ValueError: Cannot feed value of shape (1, 1, 1) for Tensor 'lstm_1_input:0', which has shape '(9, 1, 1)'
```

## 解决方案1：在线学习（批量大小= 1）

该问题的一个解决方案是使用在线学习来拟合模型。

这是批量大小设置为值1并且在每个训练示例之后更新网络权重的位置。

这可以具有更快学习的效果，但也会增加学习过程的不稳定性，因为权重随着每批次而变化很大。

尽管如此，这将使我们能够对问题进行一步预测。唯一需要做的更改是将`n_batch`设置为1，如下所示：

```py
n_batch = 1
```

完整的代码清单如下。

```py
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
n_batch = 1
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = model.predict(testX, batch_size=1)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
```

运行该示例将打印9个预期结果和正确的预测。

```py
>Expected=0.0, Predicted=0.0
>Expected=0.1, Predicted=0.1
>Expected=0.2, Predicted=0.2
>Expected=0.3, Predicted=0.3
>Expected=0.4, Predicted=0.4
>Expected=0.5, Predicted=0.5
>Expected=0.6, Predicted=0.6
>Expected=0.7, Predicted=0.7
>Expected=0.8, Predicted=0.8
```

## 解决方案2：批量预测（批量大小= N）

另一种解决方案是批量生产所有预测。

这意味着我们在模型使用方式上可能非常有限。

我们必须立即使用所有预测，或者只保留第一个预测并丢弃其余的预测。

我们可以通过预测批量大小等于训练批量大小来调整批量预测的示例，然后枚举预测批次，如下所示：

```py
# batch forecast
yhat = model.predict(X, batch_size=n_batch)
for i in range(len(y)):
	print('>Expected=%.1f, Predicted=%.1f' % (y[i], yhat[i]))
```

下面列出了完整的示例。

```py
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
# batch forecast
yhat = model.predict(X, batch_size=n_batch)
for i in range(len(y)):
	print('>Expected=%.1f, Predicted=%.1f' % (y[i], yhat[i]))
```

运行该示例将打印预期和正确的预测值。

```py
>Expected=0.0, Predicted=0.0
>Expected=0.1, Predicted=0.1
>Expected=0.2, Predicted=0.2
>Expected=0.3, Predicted=0.3
>Expected=0.4, Predicted=0.4
>Expected=0.5, Predicted=0.5
>Expected=0.6, Predicted=0.6
>Expected=0.7, Predicted=0.7
>Expected=0.8, Predicted=0.8
```

## 解决方案3：复制权重

更好的解决方案是使用不同的批量大小进行训练和预测。

执行此操作的方法是从拟合网络复制权重，并使用预先训练的权重创建新网络。

我们可以使用Keras API中的 _get_weights（）_和 _set_weights（）_函数轻松完成此操作，如下所示：

```py
# re-define the batch size
n_batch = 1
# re-define model
new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
new_model.add(Dense(1))
# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)
```

这将创建一个以批量大小为1编译的新模型。然后，我们可以使用此新模型进行一步预测：

```py
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = new_model.predict(testX, batch_size=n_batch)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
```

The complete example is listed below.

```py
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
n_batch = 3
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
# re-define the batch size
n_batch = 1
# re-define model
new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
new_model.add(Dense(1))
# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)
# compile model
new_model.compile(loss='mean_squared_error', optimizer='adam')
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = new_model.predict(testX, batch_size=n_batch)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
```

运行该示例将打印预期的值，并再次正确预测值。

```py
>Expected=0.0, Predicted=0.0
>Expected=0.1, Predicted=0.1
>Expected=0.2, Predicted=0.2
>Expected=0.3, Predicted=0.3
>Expected=0.4, Predicted=0.4
>Expected=0.5, Predicted=0.5
>Expected=0.6, Predicted=0.6
>Expected=0.7, Predicted=0.7
>Expected=0.8, Predicted=0.8
```

## 摘要

在本教程中，您了解了如何通过相同的网络来改变用于训练和预测的批量大小的需求。

具体来说，你学到了：

*   如何设计简单的序列预测问题并开发LSTM来学习它。
*   如何改变在线和基于批量的学习和预测的LSTM配置。
*   如何改变用于训练的批量大小与用于预测的批量大小。

您对批量大小有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。