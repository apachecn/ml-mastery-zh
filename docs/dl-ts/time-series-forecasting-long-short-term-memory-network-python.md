# Python 中长短期记忆网络的时间序列预测

> 原文： [https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

长期短期记忆复发神经网络有望学习长序列的观察。

它似乎是时间序列预测的完美匹配，事实上，它可能是。

在本教程中，您将了解如何为单步单变量时间序列预测问题开发 LSTM 预测模型。

完成本教程后，您将了解：

*   如何为预测问题制定绩效基准。
*   如何为一步时间序列预测设计一个强大的测试工具。
*   如何为时间序列预测准备数据，开发和评估 LSTM 循环神经网络。

让我们开始吧。

*   **2017 年 5 月更新**：修正了 invert_scale（）函数中的错误，谢谢 Max。

![Time Series Forecasting with the Long Short-Term Memory Network in Python](img/42fc13496734cacdffd9f8460c8637b1.jpg)

使用 Python 中的长短期记忆网络进行时间序列预测
照片由 [Matt MacGillivray](https://www.flickr.com/photos/qmnonic/179791867/) 拍摄，保留一些权利。

## 教程概述

这是一个很大的话题，我们将涉及很多方面。带子。

本教程分为 9 个部分;他们是：

1.  洗发水销售数据集
2.  测试设置
3.  持久性模型预测
4.  LSTM 数据准备
5.  LSTM 模型开发
6.  LSTM 预测
7.  完整的 LSTM 示例
8.  制定稳健的结果
9.  教程扩展

### Python 环境

本教程假定您已安装 Python SciPy 环境。您可以在本教程中使用 Python 2 或 3。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 洗发水销售数据集

该数据集描述了 3 年期间每月洗发水的销售数量。

单位是销售计数，有 36 个观察。原始数据集归功于 Makridakis，Wheelwright 和 Hyndman（1998）。

[您可以在此处下载并了解有关数据集的更多信息](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)。

**更新**：这是数据集的直接链接，可以使用： [shampoo.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv)

将数据集下载到当前工作目录，名称为“ _shampoo-sales.csv_ ”。请注意，您可能需要删除 DataMarket 添加的页脚信息。

下面的示例加载并创建已加载数据集的图。

```py
# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()
```

运行该示例将数据集作为 Pandas Series 加载并打印前 5 行。

```py
Month
1901-01-01 266.0
1901-02-01 145.9
1901-03-01 183.1
1901-04-01 119.3
1901-05-01 180.3
Name: Sales, dtype: float64
```

然后创建该系列的线图，显示明显的增加趋势。

![Line Plot of Monthly Shampoo Sales Dataset](img/b49b84d78ed18007a6d0edaa1b71ce11.jpg)

每月洗发水销售数据集的线图

## 实验测试设置

我们将 Shampoo Sales 数据集分为两部分：训练和测试集。

前两年的数据将用于训练数据集，剩余的一年数据将用于测试集。

例如：

```py
# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
```

将使用训练数据集开发模型，并对测试数据集进行预测。

将使用滚动预测场景，也称为前进模型验证。

测试数据集的每个时间步骤将一次一个地走。将使用模型对时间步长进行预测，然后将获取测试集的实际预期值，并使其可用于下一时间步的预测模型。

例如：

```py
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction...
```

这模仿了一个真实世界的场景，每个月都会有新的洗发水销售观察结果，并用于下个月的预测。

最后，将收集测试数据集的所有预测，并计算错误分数以总结模型的技能。将使用均方根误差（RMSE），因为它会对大错误进行处罚，并产生与预测数据相同的分数，即每月洗发水销售额。

例如：

```py
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

## 持久性模型预测

具有线性增长趋势的时间序列的良好基线预测是持久性预测。

持续性预测是使用先前时间步骤（t-1）的观察来预测当前时间步骤（t）的观察。

我们可以通过从训练数据和前进验证累积的历史记录中进行最后一次观察并使用它来预测当前时间步长来实现这一点。

例如：

```py
# make prediction
yhat = history[-1]
```

我们将在数组中累积所有预测，以便可以直接将它们与测试数据集进行比较。

下面列出了 Shampoo Sales 数据集上的持久性预测模型的完整示例。

```py
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
```

运行该示例将打印出大约 136 个月洗发水销售额的 RMSE，用于测试数据集的预测。

```py
RMSE: 136.761
```

还创建了测试数据集（蓝色）与预测值（橙色）的线图，显示了上下文中的持久性模型预测。

![Persistence Forecast of Observed vs Predicted for Shampoo Sales Dataset](img/5865df70cf90107cfdab9a23b81198e5.jpg)

洗发水销售数据集的观察与预测持续性预测

有关时间序列预测的持久性模型的更多信息，请参阅此帖子：

*   [如何使用 Python 进行时间序列预测的基线预测](http://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)

现在我们已经在数据集上有了表现基准，我们可以开始为数据开发 LSTM 模型。

## LSTM 数据准备

在我们将 LSTM 模型拟合到数据集之前，我们必须转换数据。

本节分为三个步骤：

1.  将时间序列转换为监督学习问题
2.  转换时间序列数据，使其静止不动。
3.  将观察结果转换为具有特定比例。

### 将时间序列转换为监督学习

Keras 中的 LSTM 模型假设您的数据分为输入（X）和输出（y）组件。

对于时间序列问题，我们可以通过使用从最后时间步骤（t-1）作为输入的观察和在当前时间步骤（t）的观察作为输出来实现这一点。

我们可以使用 Pandas 中的 [shift（）](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)函数来实现这一点，它会将一系列中的所有值按下指定的数字位置。我们需要移位 1 个位置，这将成为输入变量。现在的时间序列将是输出变量。

然后，我们可以将这两个系列连接在一起，为监督学习创建一个 DataFrame。推下的系列将在顶部有一个新的位置，没有任何价值。在此位置将使用 NaN（非数字）值。我们将这些 NaN 值替换为 0 值，LSTM 模型必须将其作为“系列的开头”或“我这里没有数据”来学习，因为没有观察到该数据集零销售的月份。

下面的代码定义了一个名为`timeseries_to_supervised()`的辅助函数。它需要一个原始时间序列数据的 NumPy 数组和一个滞后或移位序列的数量来创建和用作输入。

```py
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
```

我们可以使用我们加载的 Shampoo Sales 数据集测试此功能，并将其转换为监督学习问题。

```py
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# transform to supervised learning
X = series.values
supervised = timeseries_to_supervised(X, 1)
print(supervised.head())
```

运行该示例将打印新监督学习问题的前 5 行。

```py
            0           0
0    0.000000  266.000000
1  266.000000  145.899994
2  145.899994  183.100006
3  183.100006  119.300003
4  119.300003  180.300003
```

有关将时间序列问题转换为监督学习问题的更多信息，请参阅帖子：

*   [时间序列预测作为监督学习](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

### 将时间序列转换为静止

Shampoo Sales 数据集不是固定的。

这意味着数据中的结构取决于时间。具体而言，数据呈上升趋势。

固定数据更易于建模，很可能会产生更加熟练的预测。

可以从观察中移除趋势，然后将其添加回预测以将预测返回到原始比例并计算可比较的误差分数。

消除趋势的标准方法是区分数据。即，从当前观察（t）中减去前一时间步骤（t-1）的观察结果。这消除了趋势，我们留下了差异序列，或者从一个时间步到下一个步骤的观察结果的变化。

我们可以使用 pandas 中的 [diff（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html)自动实现。或者，我们可以获得更精细的控制并编写我们自己的函数来执行此操作，这在这种情况下是灵活的首选。

下面是一个名为`difference()`的函数，用于计算差分序列。请注意，系列中的第一个观察值被跳过，因为没有先前的观察值来计算差值。

```py
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
```

我们还需要反转此过程，以便将差异系列的预测恢复到原始比例。

以下函数称为 _inverse_difference（）_，反转此操作。

```py
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

我们可以通过对整个系列进行差分来测试这些函数，然后将其返回到原始比例，如下所示：

```py
from pandas import read_csv
from pandas import datetime
from pandas import Series

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
# transform to be stationary
differenced = difference(series, 1)
print(differenced.head())
# invert transform
inverted = list()
for i in range(len(differenced)):
	value = inverse_difference(series, differenced[i], len(series)-i)
	inverted.append(value)
inverted = Series(inverted)
print(inverted.head())
```

运行该示例打印加载数据的前 5 行，然后打印差异系列的前 5 行，最后打印差异操作的前 5 行。

请注意，原始数据集中的第一个观察值已从反向差异数据中删除。除此之外，最后一组数据与第一组数据匹配。

```py
Month
1901-01-01    266.0
1901-02-01    145.9
1901-03-01    183.1
1901-04-01    119.3
1901-05-01    180.3

Name: Sales, dtype: float64
0   -120.1
1     37.2
2    -63.8
3     61.0
4    -11.8
dtype: float64

0    145.9
1    183.1
2    119.3
3    180.3
4    168.5
dtype: float64
```

有关使时间序列固定和差分的更多信息，请参阅帖子：

*   [如何使用 Python 检查时间序列数据是否固定](http://machinelearningmastery.com/time-series-data-stationary-python/)
*   [如何区分时间序列数据集与 Python](http://machinelearningmastery.com/difference-time-series-dataset-python/)

### 将时间序列转换为比例

与其他神经网络一样，LSTM 期望数据在网络使用的激活函数的范围内。

LSTM 的默认激活函数是双曲正切（`tanh`），它输出介于-1 和 1 之间的值。这是时间序列数据的首选范围。

为了使实验公平，必须在训练数据集上计算缩放系数（最小和最大）值，并应用于缩放测试数据集和任何预测。这是为了避免使用测试数据集中的知识污染实验，这可能会给模型带来小的优势。

我们可以使用 [MinMaxScaler 类](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)将数据集转换为[-1,1]范围。与其他 scikit-learn 变换类一样，它需要以行和列的矩阵格式提供数据。因此，我们必须在转换之前重塑我们的 NumPy 数组。

例如：

```py
# transform scale
X = series.values
X = X.reshape(len(X), 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)
```

同样，我们必须反转预测的比例，以将值返回到原始比例，以便可以解释结果并计算可比较的误差分数。

```py
# invert transform
inverted_X = scaler.inverse_transform(scaled_X)
```

将所有这些放在一起，下面的例子改变了 Shampoo Sales 数据的规模。

```py
from pandas import read_csv
from pandas import datetime
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
# transform scale
X = series.values
X = X.reshape(len(X), 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)
scaled_series = Series(scaled_X[:, 0])
print(scaled_series.head())
# invert transform
inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
print(inverted_series.head())
```

运行该示例首先打印加载数据的前 5 行，然后打印缩放数据的前 5 行，然后是缩放比例变换的前 5 行，与原始数据匹配。

```py
Month
1901-01-01    266.0
1901-02-01    145.9
1901-03-01    183.1
1901-04-01    119.3
1901-05-01    180.3

Name: Sales, dtype: float64
0   -0.478585
1   -0.905456
2   -0.773236
3   -1.000000
4   -0.783188
dtype: float64

0    266.0
1    145.9
2    183.1
3    119.3
4    180.3
dtype: float64
```

现在我们知道如何为 LSTM 网络准备数据，我们可以开始开发我们的模型了。

## LSTM 模型开发

长短期记忆网络（LSTM）是一种循环神经网络（RNN）。

这种类型的网络的一个好处是它可以学习和记住长序列，并且不依赖于预先指定的窗口滞后观察作为输入。

在 Keras 中，这被称为有状态，并且涉及在定义 LSTM 层时将“_ 有状态 _”参数设置为“`True`”。

默认情况下，Keras 中的 LSTM 层维护一批数据之间的状态。一批数据是来自训练数据集的固定大小的行数，用于定义在更新网络权重之前要处理的模式数。默认情况下，批次之间的 LSTM 层中的状态被清除，因此我们必须使 LSTM 成为有状态。通过调用`reset_states()`函数，这可以精确控制 LSTM 层的状态何时被清除。

LSTM 层期望输入位于具有以下尺寸的矩阵中：[_ 样本，时间步长，特征 _]。

*   **样本**：这些是来自域的独立观察，通常是数据行。
*   **时间步长**：这些是给定观察的给定变量的单独时间步长。
*   **特征**：这些是在观察时观察到的单独测量。

我们在如何为网络构建 Shampoo Sales 数据集方面具有一定的灵活性。我们将保持简单并构建问题，因为原始序列中的每个步骤都是一个单独的样本，具有一个时间步长和一个特征。

鉴于训练数据集定义为 X 输入和 y 输出，必须将其重新整形为 Samples / TimeSteps / Features 格式，例如：

```py
X, y = train[:, 0:-1], train[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])
```

必须在 LSTM 层中使用“`batch_input_shape`”参数指定输入数据的形状作为元组，该元组指定读取每批的预期观察数，时间步数和数量特征。

批量大小通常远小于样本总数。它与时期的数量一起定义了网络学习数据的速度（权重更新的频率）。

定义 LSTM 层的最后一个导入参数是神经元的数量，也称为内存单元或块的数量。这是一个相当简单的问题，1 到 5 之间的数字就足够了。

下面的行创建了一个 LSTM 隐藏层，它还通过“`batch_input_shape`”参数指定输入层的期望值。

```py
layer = LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True)
```

网络需要输出层中的单个神经元具有线性激活，以预测下一时间步骤的洗发水销售数量。

一旦指定了网络，就必须使用后端数学库（例如 TensorFlow 或 Theano）将其编译成有效的符号表示。

在编译网络时，我们必须指定一个损失函数和优化算法。我们将使用“`mean_squared_error`”作为损失函数，因为它与我们感兴趣的 RMSE 紧密匹配，以及有效的 ADAM 优化算法。

使用 Sequential Keras API 定义网络，下面的代码片段创建并编译网络。

```py
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

编译后，它可以适合训练数据。由于网络是有状态的，我们必须控制何时重置内部状态。因此，我们必须在所需数量的时期内一次手动管理一个时期的训练过程。

默认情况下，一个迭代内的样本在暴露给网络之前进行混洗。同样，这对于 LSTM 来说是不合需要的，因为我们希望网络在整个观察序列中学习时建立状态。我们可以通过将“`shuffle`”设置为“`False`”来禁用样本的混洗。

此外，默认情况下，网络会在每个时代结束时报告有关模型学习进度和技能的大量调试信息。我们可以通过将“`verbose`”参数设置为“`0`”的级别来禁用它。

然后我们可以在训练时期结束时重置内部状态，为下一次训练迭代做好准备。

下面是一个手动使网络适合训练数据的循环。

```py
for i in range(nb_epoch):
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
	model.reset_states()
```

综上所述，我们可以定义一个名为`fit_lstm()`的函数来训练并返回一个 LSTM 模型。作为参数，它将训练数据集置于监督学习格式，批量大小，多个时期和许多神经元中。

```py
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
```

batch_size 必须设置为 1.这是因为它必须是训练和测试数据集大小的因子。

模型上的`predict()`函数也受批量大小的限制;它必须设置为 1，因为我们有兴趣对测试数据进行一步预测。

我们不会在本教程中调整网络参数;相反，我们将使用以下配置，通过一些试验和错误找到：

*   批量大小：1
*   时代：3000
*   神经元：4

作为本教程的扩展，您可能希望探索不同的模型参数，看看是否可以提高表现。

*   **更新**：考虑尝试 1500 个迭代和 1 个神经元，表现可能更好！

接下来，我们将了解如何使用适合的 LSTM 模型进行一步预测。

## LSTM 预测

一旦 LSTM 模型适合训练数据，它就可用于进行预测。

我们再次具有一定的灵活性。我们可以决定在所有训练数据上拟合一次模型，然后从测试数据中一次预测每个新时间步骤（我们称之为固定方法），或者我们可以重新拟合模型或更新模型的每一步作为测试数据的新观察结果的模型都可用（我们称之为动态方法）。

在本教程中，我们将采用固定方法来实现其简单性，但我们希望动态方法能够带来更好的模型技能。

为了进行预测，我们可以在模型上调用`predict()`函数。这需要 3D NumPy 数组输入作为参数。在这种情况下，它将是一个值的数组，即前一时间步的观察值。

`predict()`函数返回一个预测数组，每个输入行一个。因为我们提供单个输入，所以输出将是具有一个值的 2D NumPy 数组。

我们可以在下面列出的名为`forecast()`的函数中捕获此行为。给定拟合模型，在拟合模型时使用的批量大小（例如 1）和测试数据中的行，该函数将从测试行中分离输入数据，重新整形，并将预测作为单个返回浮点值。

```py
def forecast(model, batch_size, row):
	X = row[0:-1]
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
```

在训练期间，内部状态在每个时期之后被重置。在预测时，我们不希望重置预测之间的内部状态。事实上，我们希望模型在我们预测测试数据集中的每个时间步骤时建立状态。

这就提出了一个问题，即在预测测试数据集之前，网络的初始状态是什么。

在本教程中，我们将通过对训练数据集中的所有样本进行预测来播种状态。从理论上讲，应该建立内部状态，以便预测下一个时间步骤。

我们现在拥有适合 Shampoo Sales 数据集的 LSTM 网络模型并评估其表现的所有部分。

在下一节中，我们将把所有这些部分放在一起。

## 完整的 LSTM 示例

在本节中，我们将 LSTM 适用于 Shampoo Sales 数据集并评估模型。

这将涉及将前面部分中的所有元素汇总在一起。其中有很多，所以让我们回顾一下：

1.  从 CSV 文件加载数据集。
2.  转换数据集以使其适用于 LSTM 模型，包括：
    1.  将数据转换为监督学习问题。
    2.  将数据转换为静止。
    3.  转换数据使其具有-1 到 1 的比例。
3.  将有状态 LSTM 网络模型拟合到训练数据。
4.  评估测试数据上的静态 LSTM 模型。
5.  报告预测的表现。

有关示例的一些注意事项：

*   为简洁起见，缩放和反缩放行为已移至函数`scale()`和 _invert_scale（）_。
*   使用缩放器在训练数据上的拟合来缩放测试数据，这是确保测试数据的最小值/最大值不影响模型所需的。
*   调整数据变换的顺序以便于首先使数据静止，然后是监督学习问题，然后进行缩放。
*   为了方便起见，在分成训练和测试集之前对整个数据集执行差分。我们可以在前进验证过程中轻松收集观察结果，并在我们前进时区分它们。为了便于阅读，我决定反对它。

下面列出了完整的示例。

```py
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()
```

运行该示例将在测试数据集中打印 12 个月中每个月的预期值和预测值。

该示例还打印所有预测的 RMSE。该模型显示每月洗发水销售量为 71.721 的 RMSE，优于达到 136.761 洗发水销售 RMSE 的持久性模型。

随机数用于播种 LSTM，因此，您可能会从模型的单次运行中获得不同的结果。我们将在下一节进一步讨论这个问题。

```py
Month=1, Predicted=351.582196, Expected=339.700000
Month=2, Predicted=432.169667, Expected=440.400000
Month=3, Predicted=378.064505, Expected=315.900000
Month=4, Predicted=441.370077, Expected=439.300000
Month=5, Predicted=446.872627, Expected=401.300000
Month=6, Predicted=514.021244, Expected=437.400000
Month=7, Predicted=525.608903, Expected=575.500000
Month=8, Predicted=473.072365, Expected=407.600000
Month=9, Predicted=523.126979, Expected=682.000000
Month=10, Predicted=592.274106, Expected=475.300000
Month=11, Predicted=589.299863, Expected=581.300000
Month=12, Predicted=584.149152, Expected=646.900000
Test RMSE: 71.721
```

还创建了测试数据（蓝色）与预测值（橙色）的线图，为模型技能提供了上下文。

![Line Plot of LSTM Forecast vs Expected Values](img/23ca281ff10f02aae3919d3aa204108f.jpg)

LSTM 预测和预期值的线图

作为 afternote，您可以进行快速实验，以建立您对测试工具以及所有变换和逆变换的信任。

在前进验证中注释掉适合 LSTM 模型的行：

```py
yhat = forecast_lstm(lstm_model, 1, X)
```

并将其替换为以下内容：

```py
yhat = y
```

这应该产生具有完美技能的模型（例如，将预期结果预测为模型输出的模型）。

结果应如下所示，表明如果 LSTM 模型能够完美地预测该系列，则逆变换和误差计算将正确显示。

```py
Month=1, Predicted=339.700000, Expected=339.700000
Month=2, Predicted=440.400000, Expected=440.400000
Month=3, Predicted=315.900000, Expected=315.900000
Month=4, Predicted=439.300000, Expected=439.300000
Month=5, Predicted=401.300000, Expected=401.300000
Month=6, Predicted=437.400000, Expected=437.400000
Month=7, Predicted=575.500000, Expected=575.500000
Month=8, Predicted=407.600000, Expected=407.600000
Month=9, Predicted=682.000000, Expected=682.000000
Month=10, Predicted=475.300000, Expected=475.300000
Month=11, Predicted=581.300000, Expected=581.300000
Month=12, Predicted=646.900000, Expected=646.900000
Test RMSE: 0.000
```

## 制定稳健的结果

神经网络的一个难点是它们在不同的起始条件下给出不同的结果。

一种方法可能是修复 Keras 使用的随机数种子，以确保结果可重复。另一种方法是使用不同的实验设置来控制随机初始条件。

有关机器学习中随机性的更多信息，请参阅帖子：

*   [在机器学习中拥抱随机性](http://machinelearningmastery.com/randomness-in-machine-learning/)

我们可以多次重复上一节中的实验，然后将平均 RMSE 作为预期配置在平均看不见的数据上的表现的指示。

这通常称为多次重复或多次重启。

我们可以在固定数量的重复循环中包装模型拟合和前进验证。每次迭代都可以记录运行的 RMSE。然后我们可以总结 RMSE 分数的分布。

```py
# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)
```

数据准备与以前相同。

我们将使用 30 次重复，因为这足以提供 RMSE 分数的良好分布。

下面列出了完整的示例。

```py
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()
```

运行该示例每次重复打印 RMSE 分数。运行结束提供收集的 RMSE 分数的摘要统计。

我们可以看到平均值和标准差 RMSE 分数分别为 138.491905 和 46.313783 月洗发水销售额。

这是一个非常有用的结果，因为它表明上面报告的结果可能是一个统计上的侥幸。实验表明，该模型可能与持久性模型一样好（136.761），如果不是稍差的话。

这表明，至少需要进一步的模型调整。

```py
1) Test RMSE: 136.191
2) Test RMSE: 169.693
3) Test RMSE: 176.553
4) Test RMSE: 198.954
5) Test RMSE: 148.960
6) Test RMSE: 103.744
7) Test RMSE: 164.344
8) Test RMSE: 108.829
9) Test RMSE: 232.282
10) Test RMSE: 110.824
11) Test RMSE: 163.741
12) Test RMSE: 111.535
13) Test RMSE: 118.324
14) Test RMSE: 107.486
15) Test RMSE: 97.719
16) Test RMSE: 87.817
17) Test RMSE: 92.920
18) Test RMSE: 112.528
19) Test RMSE: 131.687
20) Test RMSE: 92.343
21) Test RMSE: 173.249
22) Test RMSE: 182.336
23) Test RMSE: 101.477
24) Test RMSE: 108.171
25) Test RMSE: 135.880
26) Test RMSE: 254.507
27) Test RMSE: 87.198
28) Test RMSE: 122.588
29) Test RMSE: 228.449
30) Test RMSE: 94.427
             rmse
count   30.000000
mean   138.491905
std     46.313783
min     87.198493
25%    104.679391
50%    120.456233
75%    168.356040
max    254.507272
```

根据下面显示的分布创建一个盒子和胡须图。这将捕获数据的中间以及范围和异常值结果。

![LSTM Repeated Experiment Box and Whisker Plot](img/5bff80c488cfe96f08b491d070af89d9.jpg)

LSTM 重复实验箱和晶须图

这是一个实验设置，可用于比较 LSTM 模型的一个配置或设置到另一个配置。

## 教程扩展

我们可能会考虑本教程的许多扩展。

也许您可以自己探索其中一些，并在下面的评论中发布您的发现。

*   **多步预测**。可以改变实验设置以预测下一个`n`- 时间步骤而不是下一个单一时间步骤。这也将允许更大的批量和更快的训练。请注意，我们基本上在本教程中执行了一种 12 步一步预测，因为模型未更新，尽管有新的观察结果并且可用作输入变量。
*   **调谐 LSTM 模型**。该模型没有调整;相反，配置被发现有一些快速的试验和错误。我相信通过至少调整神经元的数量和训练时期的数量可以获得更好的结果。我还认为通过回调提前停止可能在训练期间很有用。
*   **种子状态实验**。目前尚不清楚在预测之前通过预测所有训练数据来播种系统是否有益。理论上这似乎是一个好主意，但这需要得到证明。此外，也许在预测之前播种模型的其他方法也是有益的。
*   **更新模型**。可以在前进验证的每个时间步骤中更新模型。需要进行实验以确定从头开始重新修改模型是否更好，或者使用包括新样本在内的更多训练时期更新权重。
*   **输入时间步**。 LSTM 输入支持样本的多个时间步长。需要进行实验以查看是否包括滞后观察作为时间步骤提供任何益处。
*   **输入延迟功能**。可以包括滞后观察作为输入特征。需要进行实验以查看包含滞后特征是否提供任何益处，与 AR（k）线性模型不同。
*   **输入错误系列**。可以构造误差序列（来自持久性模型的预测误差）并且用作附加输入特征，与 MA（k）线性模型不同。需要进行实验以确定这是否会带来任何好处。
*   **学习非固定**。 LSTM 网络可能能够了解数据中的趋势并做出合理的预测。需要进行实验以了解 LSTM 是否可以学习和有效预测数据中剩余的时间依赖结构，如趋势和季节性。
*   **对比无状态**。本教程中使用了有状态 LSTM。应将结果与无状态 LSTM 配置进行比较。
*   **统计学意义**。可以进一步扩展多重复实验方案以包括统计显着性检验，以证明具有不同构型的 RMSE 结果群体之间的差异是否具有统计学显着性。

## 摘要

在本教程中，您了解了如何为时间序列预测开发 LSTM 模型。

具体来说，你学到了：

*   如何准备用于开发 LSTM 模型的时间序列数据。
*   如何开发 LSTM 模型进行时间序列预测。
*   如何使用强大的测试工具评估 LSTM 模型。

你能得到更好的结果吗？
在下面的评论中分享您的发现。