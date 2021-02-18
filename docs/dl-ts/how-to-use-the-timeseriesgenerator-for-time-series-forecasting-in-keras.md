# 如何使用 TimeseriesGenerator 进行 Keras 中的时间序列预测

> 原文： [https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/)

必须将时间序列数据转换为具有输入和输出组件的样本结构，然后才能用于拟合监督学习模型。

如果您必须手动执行此转换，这可能具有挑战性。 Keras 深度学习库提供了 TimeseriesGenerator，可以将单变量和多变量时间序列数据自动转换为样本，随时可以训练深度学习模型。

在本教程中，您将了解如何使用 Keras TimeseriesGenerator 准备时间序列数据，以便使用深度学习方法进行建模。

完成本教程后，您将了解：

*   如何定义 TimeseriesGenerator 生成器并将其用于适合深度学习模型。
*   如何为单变量时间序列准备发电机并适合 MLP 和 LSTM 模型。
*   如何为多变量时间序列准备生成器并适合 LSTM 模型。

让我们开始吧。

![How to Use the TimeseriesGenerator for Time Series Forecasting in Keras](img/cf21fe77ad8f90bbad3a4061cfddf266.jpg)

如何在 Keras 中使用 TimeseriesGenerator 进行时间序列预测
照片由 [Chris Fithall](https://www.flickr.com/photos/chrisfithall/13933989150/) 拍摄，保留一些权利。

## 教程概述

本教程分为六个部分;他们是：

1.  监督学习的时间序列问题
2.  如何使用 TimeseriesGenerator
3.  单变量时间序列示例
4.  多变量时间序列示例
5.  多变量输入和相关系列示例
6.  多步预测示例

**注**：本教程假设您使用的是 **Keras v2.2.4** 或更高版本。

## 监督学习的时间序列问题

时间序列数据需要准备才能用于训练监督学习模型，例如深度学习模型。

例如，单变量时间序列表示为观察向量：

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

监督学习算法要求数据作为样本集合提供，其中每个样本具有输入分量（`X`）和输出分量（`y`）。

```py
X,					y
example input, 		example output
example input, 		example output
example input, 		example output
...
```

该模型将学习如何将输入映射到所提供示例的输出。

```py
y = f(X)
```

必须将时间序列转换为具有输入和输出组件的样本。该变换既可以告知模型将要学习什么，也可以在进行预测时告知将来如何使用该模型，例如：做出预测需要什么（`X`）和做出预测（`y`）。

对于对一步预测感兴趣的单变量时间序列，在先前时间步骤的观察，即所谓的滞后观察，被用作输入，输出是在当前时间步骤的观察。

例如，上述 10 步单变量系列可以表示为监督学习问题，其中输入的三个时间步长和输出的一个步骤如下：

```py
X,			y
[1, 2, 3],	[4]
[2, 3, 4],	[5]
[3, 4, 5],	[6]
...
```

您可以编写代码来自行执行此转换;例如，看帖子：

*   [如何将时间序列转换为 Python 中的监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

或者，当您对使用 Keras 训练神经网络模型感兴趣时，可以使用 TimeseriesGenerator 类。

## 如何使用 TimeseriesGenerator

Keras 提供 [TimeseriesGenerator](https://keras.io/preprocessing/sequence/) ，可用于将单变量或多变量时间序列数据集自动转换为监督学习问题。

使用 TimeseriesGenerator 有两个部分：定义它并使用它来训练模型。

### 定义 TimeseriesGenerator

您可以创建类的实例并指定时间序列问题的输入和输出方面，它将提供[序列类](https://keras.io/utils/#sequence)的实例，然后可以使用它来迭代输入和输出系列。

在大多数时间序列预测问题中，输入和输出系列将是同一系列。

例如：

```py
# load data
inputs = ...
outputs = ...
# define generator
generator = TimeseriesGenerator(inputs, outputs, ...)
# iterator generator
for i in range(len(generator)):
	...
```

从技术上讲，该类不是一个生成器，因为它不是 [Python 生成器](https://wiki.python.org/moin/Generators)，你不能在它上面使用 _next（）_ 函数。

除了指定时间序列问题的输入和输出方面外，还应该配置一些其他参数;例如：

*   **长度**：在每个样本的输入部分中使用的滞后观察的数量（例如 3）。
*   **batch_size** ：每次迭代时返回的样本数（例如 32）。

您必须根据设计的问题框架定义长度参数。这是用作输入的所需滞后观察数。

您还必须在训练期间将批量大小定义为模型的批量大小。如果数据集中的样本数小于批量大小，则可以通过计算其长度，将生成器和模型中的批量大小设置为生成器中的样本总数;例如：

```py
print(len(generator))
```

还有其他参数，例如定义数据的开始和结束偏移，采样率，步幅等。您不太可能使用这些功能，但您可以查看[完整 API](https://keras.io/preprocessing/sequence/) 以获取更多详细信息。

默认情况下，样本不会被洗牌。这对于像 LSTM 这样的一些循环神经网络很有用，这些神经网络在一批中的样本之间保持状态。

它可以使其他神经网络（例如 CNN 和 MLP）在训练时对样本进行混洗。可以通过将'`shuffle`'参数设置为 True 来启用随机播放。这将具有为每个批次返回的洗样样本的效果。

在撰写本文时，TimeseriesGenerator 仅限于一步输出。不支持多步时间序列预测。

### 使用 TimeseriesGenerator 训练模型

一旦定义了 TimeseriesGenerator 实例，它就可以用于训练神经网络模型。

可以使用 TimeseriesGenerator 作为数据生成器来训练模型。这可以通过使用 _fit_generator（）_ 函数拟合定义的模型来实现。

此函数将生成器作为参数。它还需要`steps_per_epoch`参数来定义每个时期中使用的样本数。可以将此值设置为 TimeseriesGenerator 实例的长度，以使用生成器中的所有样本。

例如：

```py
# define generator
generator = TimeseriesGenerator(...)
# define model
model = ...
# fit model
model.fit_generator(generator, steps_per_epoch=len(generator), ...)
```

类似地，生成器可用于通过调用 _evaluate_generator（）_ 函数来评估拟合模型，并使用拟合模型使用 _predict_generator（）_ 函数对新数据进行预测。

与数据生成器匹配的模型不必使用 evaluate 和 predict 函数的生成器版本。只有在您希望数据生成器为模型准备数据时，才能使用它们。

## 单变量时间序列示例

我们可以使用一个小的设计单变量时间序列数据集的工作示例使 TimeseriesGenerator 具体化。

首先，让我们定义我们的数据集。

```py
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

我们将选择框架问题，其中最后两个滞后观察将用于预测序列中的下一个值。例如：

```py
X,			y
[1, 2]		3
```

现在，我们将使用批量大小为 1，以便我们可以探索生成器中的数据。

```py
# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)
```

接下来，我们可以看到数据生成器将为此时间序列准备多少样本。

```py
# number of samples
print('Samples: %d' % len(generator))
```

最后，我们可以打印每个样本的输入和输出组件，以确认数据是按照我们的预期准备的。

```py
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

下面列出了完整的示例。

```py
# univariate one step problem
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)
# number of samples
print('Samples: %d' % len(generator))
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

首先运行该示例打印生成器中的样本总数，即 8。

然后我们可以看到每个输入数组的形状为[1,2]，每个输出的形状为[1，]。

观察结果按照我们的预期进行准备，其中两个滞后观察结果将用作输入，序列中的后续值作为输出。

```py
Samples: 8

[[1\. 2.]] => [3.]
[[2\. 3.]] => [4.]
[[3\. 4.]] => [5.]
[[4\. 5.]] => [6.]
[[5\. 6.]] => [7.]
[[6\. 7.]] => [8.]
[[7\. 8.]] => [9.]
[[8\. 9.]] => [10.]
```

现在我们可以在这个数据上拟合模型，并学习将输入序列映射到输出序列。

我们将从一个简单的多层感知器或 MLP 模型开始。

将定义发生器，以便在给定少量样品的情况下，将在每批中使用所有样品。

```py
# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=8)
```

我们可以定义一个简单的模型，其中一个隐藏层有 50 个节点，一个输出层将进行预测。

```py
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

然后我们可以使用 _fit_generator（）_ 函数将模型与生成器拟合。我们在生成器中只有一批数据，因此我们将`steps_per_epoch`设置为 1.该模型适用于 200 个时期。

```py
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=200, verbose=0)
```

一旦适合，我们将进行样本预测。

给定输入[9,10]，我们将进行预测并期望模型预测[11]或接近它。该模型没有调整;这只是如何使用发电机的一个例子。

```py
# make a one step prediction out of sample
x_input = array([9, 10]).reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
```

下面列出了完整的示例。

```py
# univariate one step problem with mlp
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=8)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=200, verbose=0)
# make a one step prediction out of sample
x_input = array([9, 10]).reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例准备生成器，拟合模型，并进行样本预测，正确预测接近 11 的值。

```py
[[11.510406]]
```

我们还可以使用生成器来适应循环神经网络，例如长短期内存网络或 LSTM。

LSTM 期望数据输入具有[_ 样本，时间步长，特征 _]的形状，而到目前为止描述的生成器提供滞后观察作为特征或形状[_ 样本，特征 _]。

我们可以在从[10，]到[10,1]准备发电机之前重新设计单变量时间序列 10 个时间步长和 1 个特征;例如：

```py
# reshape to [10, 1]
n_features = 1
series = series.reshape((len(series), n_features))
```

然后，TimeseriesGenerator 将系列分为样品，其形状为[ _batch，n_input，1_ ]或[8,2,1]，用于生成器中的所有八个样品，两个滞后观察用作时间步长。

下面列出了完整的示例。

```py
# univariate one step problem with lstm
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# reshape to [10, 1]
n_features = 1
series = series.reshape((len(series), n_features))
# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=8)
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=500, verbose=0)
# make a one step prediction out of sample
x_input = array([9, 10]).reshape((1, n_input, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

再次，运行该示例准备数据，拟合模型，并预测序列中的下一个样本外值。

```py
[[11.092189]]
```

## 多变量时间序列示例

TimeseriesGenerator 还支持多变量时间序列问题。

这些是您有多个并行系列的问题，每个系列中的观察步骤同时进行。

我们可以用一个例子来证明这一点。

首先，我们可以设计两个并行系列的数据集。

```py
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
```

具有多变量时间序列格式的标准结构使得每个时间序列是单独的列，行是每个时间步的观察值。

我们定义的系列是向量，但我们可以将它们转换为列。我们可以将每个系列重塑为具有 10 个时间步长和 1 个特征的形状[10,1]的数组。

```py
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
```

我们现在可以通过调用 _hstack（）_ NumPy 函数将列水平堆叠到数据集中。

```py
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
```

我们现在可以直接将此数据集提供给 TimeseriesGenerator。我们将使用每个系列的前两个观测值作为输入，并将每个序列的下一个观测值作为输出。

```py
# define generator
n_input = 2
generator = TimeseriesGenerator(dataset, dataset, length=n_input, batch_size=1)
```

然后，对于 1 个样本，2 个时间步长和 2 个特征或平行序列，每个样本将是[1,2,2]的三维阵列。对于 1 个样本和 2 个特征，输出将是[1,2]的二维系列。第一个样本将是：

```py
X, 							y
[[10, 15], [20, 25]]		[[30, 35]]
```

下面列出了完整的示例。

```py
# multivariate one step problem
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
print(dataset)
# define generator
n_input = 2
generator = TimeseriesGenerator(dataset, dataset, length=n_input, batch_size=1)
# number of samples
print('Samples: %d' % len(generator))
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

运行该示例将首先打印准备好的数据集，然后打印数据集中的总样本数。

接下来，打印每个样品的输入和输出部分，确认我们的预期结构。

```py
[[ 10  15]
 [ 20  25]
 [ 30  35]
 [ 40  45]
 [ 50  55]
 [ 60  65]
 [ 70  75]
 [ 80  85]
 [ 90  95]
 [100 105]]

Samples: 8

[[[10\. 15.]
  [20\. 25.]]] => [[30\. 35.]]
[[[20\. 25.]
  [30\. 35.]]] => [[40\. 45.]]
[[[30\. 35.]
  [40\. 45.]]] => [[50\. 55.]]
[[[40\. 45.]
  [50\. 55.]]] => [[60\. 65.]]
[[[50\. 55.]
  [60\. 65.]]] => [[70\. 75.]]
[[[60\. 65.]
  [70\. 75.]]] => [[80\. 85.]]
[[[70\. 75.]
  [80\. 85.]]] => [[90\. 95.]]
[[[80\. 85.]
  [90\. 95.]]] => [[100\. 105.]]
```

样本的三维结构意味着生成器不能直接用于像 MLP 这样的简单模型。

这可以通过首先将时间序列数据集展平为一维向量，然后将其提供给 TimeseriesGenerator 并将长度设置为用作输入的步数乘以系列中的列数（ _n_steps 来实现。 * n_features_ ）。

这种方法的局限性在于生成器只允许您预测一个变量。几乎可以肯定，编写自己的函数来为 MLP 准备多变量时间序列比使用 TimeseriesGenerator 更好。

样品的三维结构可以由 CNN 和 LSTM 模型直接使用。下面列出了使用 TimeseriesGenerator 进行多变量时间序列预测的完整示例。

```py
# multivariate one step problem with lstm
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# define generator
n_features = dataset.shape[1]
n_input = 2
generator = TimeseriesGenerator(dataset, dataset, length=n_input, batch_size=8)
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=500, verbose=0)
# make a one step prediction out of sample
x_input = array([[90, 95], [100, 105]]).reshape((1, n_input, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例准备数据，拟合模型，并预测每个输入时间序列中的下一个值，我们期望[110,115]。

```py
[[111.03207 116.58153]]
```

## 多变量输入和相关系列示例

存在多变量时间序列问题，其中存在一个或多个输入序列和要预测的单独输出序列，其取决于输入序列。

为了使这个具体，我们可以设计一个带有两个输入时间序列和一个输出系列的例子，它是输入系列的总和。

```py
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
```

其中输出序列中的值是输入时间序列中同一时间步长的值的总和。

```py
10 + 15 = 25
```

这与先前的示例不同，在给定输入的情况下，我们希望预测下一时间步的目标时间序列中的值，而不是与输入相同的时间步长。

例如，我们想要样本：

```py
X, 			y
[10, 15],	25
[20, 25],	45
[30, 35],	65
...
```

我们不希望样品如下：

```py
X, 			y
[10, 15],	45
[20, 25],	65
[30, 35],	85
...
```

尽管如此，TimeseriesGenerator 类假定我们正在预测下一个时间步骤，并将提供数据，如上面的第二种情况。

例如：

```py
# multivariate one step problem
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# define generator
n_input = 1
generator = TimeseriesGenerator(dataset, out_seq, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

运行该示例打印样本的输入和输出部分，其具有下一时间步的输出值而不是当前时间步，因为我们可能期望这种类型的问题。

```py
[[[10\. 15.]]] => [[45.]]
[[[20\. 25.]]] => [[65.]]
[[[30\. 35.]]] => [[85.]]
[[[40\. 45.]]] => [[105.]]
[[[50\. 55.]]] => [[125.]]
[[[60\. 65.]]] => [[145.]]
[[[70\. 75.]]] => [[165.]]
[[[80\. 85.]]] => [[185.]]
[[[90\. 95.]]] => [[205.]]
```

因此，我们可以修改目标序列（`out_seq`）并在开头插入一个额外的值，以便将所有观察值向下推一步。

这种人为的转变将允许优选的问题框架。

```py
# shift the target sample by one step
out_seq = insert(out_seq, 0, 0)
```

下面提供了这种转变的完整示例。

```py
# multivariate one step problem
from numpy import array
from numpy import hstack
from numpy import insert
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# shift the target sample by one step
out_seq = insert(out_seq, 0, 0)
# define generator
n_input = 1
generator = TimeseriesGenerator(dataset, out_seq, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

运行该示例显示问题的首选框架。

无论输入样本的长度如何，此方法都将起作用。

```py
[[[10\. 15.]]] => [25.]
[[[20\. 25.]]] => [45.]
[[[30\. 35.]]] => [65.]
[[[40\. 45.]]] => [85.]
[[[50\. 55.]]] => [105.]
[[[60\. 65.]]] => [125.]
[[[70\. 75.]]] => [145.]
[[[80\. 85.]]] => [165.]
[[[90\. 95.]]] => [185.]
```

## 多步预测示例

与许多其他类型的经典和机器学习模型相比，神经网络模型的一个好处是它们可以进行多步预测。

也就是说，模型可以学习将一个或多个特征的输入模式映射到多于一个特征的输出模式。这可用于时间序列预测，以直接预测多个未来时间步骤。

这可以通过直接从模型输出向量，通过将所需数量的输出指定为输出层中的节点数来实现，或者可以通过诸如编码器 - 解码器模型的专用序列预测模型来实现。

TimeseriesGenerator 的一个限制是它不直接支持多步输出。具体而言，它不会创建目标序列中可能需要的多个步骤。

然而，如果您准备目标序列有多个步骤，它将尊重并使用它们作为每个样本的输出部分。这意味着你有责任为每个时间步骤准备预期的输出。

我们可以通过一个简单的单变量时间序列来证明这一点，在输出序列中有两个时间步长。

您可以看到目标序列中的行数必须与输入序列中的行数相同。在这种情况下，我们必须知道输入序列中的值之外的值，或者将输入序列修剪为目标序列的长度。

```py
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
target = array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
```

下面列出了完整的示例。

```py
# univariate multi-step problem
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
target = array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
# define generator
n_input = 2
generator = TimeseriesGenerator(series, target, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
```

运行该示例打印样本的输入和输出部分，显示两个滞后观察值作为输入，两个步骤作为多步骤预测问题的输出。

```py
[[1\. 2.]] => [[3\. 4.]]
[[2\. 3.]] => [[4\. 5.]]
[[3\. 4.]] => [[5\. 6.]]
[[4\. 5.]] => [[6\. 7.]]
[[5\. 6.]] => [[7\. 8.]]
[[6\. 7.]] => [[8\. 9.]]
[[7\. 8.]] => [[ 9\. 10.]]
[[8\. 9.]] => [[10\. 11.]]
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [如何将时间序列转换为 Python 中的监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [TimeseriesGenerator Keras API](https://keras.io/preprocessing/sequence/)
*   [序列 Keras API](https://keras.io/utils/#sequence)
*   [顺序模型 Keras API](https://keras.io/models/sequential/)
*   [Python Generator](https://wiki.python.org/moin/Generators)

## 摘要

在本教程中，您了解了如何使用 Keras TimeseriesGenerator 准备时间序列数据，以便使用深度学习方法进行建模。

具体来说，你学到了：

*   如何定义 TimeseriesGenerator 生成器并将其用于适合深度学习模型。
*   如何为单变量时间序列准备发电机并适合 MLP 和 LSTM 模型。
*   如何为多变量时间序列准备生成器并适合 LSTM 模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。