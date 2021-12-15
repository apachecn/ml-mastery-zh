# 如何在 Python 和 LSTM 中为时间序列预测播种状态

> 原文： [https://machinelearningmastery.com/seed-state-lstms-time-series-forecasting-python/](https://machinelearningmastery.com/seed-state-lstms-time-series-forecasting-python/)

长短期记忆网络（LSTM）是一种强大的循环神经网络，能够学习长序列观察。

LSTM 的承诺是它们可能在时间序列预测中有效，尽管已知该方法难以配置和用于这些目的。

LSTM 的一个关键特征是它们保持内部状态，有助于预测。这提出了在做出预测之前如何最好地使拟合 LSTM 模型的状态播种的问题。

在本教程中，您将了解如何设计，执行和解释实验的结果，以探索是否更好地从训练数据集中获取拟合 LSTM 的状态或不使用先前状态。

完成本教程后，您将了解：

*   关于如何最好地初始化适合 LSTM 状态以做出预测的未决问题。
*   如何开发一个强大的测试工具来评估 LSTM 模型的单变量时间序列预测问题。
*   如何确定在预测之前是否播种 LSTM 的状态对于您的时间序列预测问题是一个好主意。

让我们开始吧。

![How to Seed State for LSTMs for Time Series Forecasting in Python](img/05031ee46bb81b6491c1604cc4553aea.jpg)

如何为 LSTM 制作状态用于 Python 中的时间序列预测
照片由 [Tony Hisgett](https://www.flickr.com/photos/hisgett/6940877193/) ，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  播种 LSTM 国家
2.  洗发水销售数据集
3.  LSTM 模型和测试线束
4.  代码清单
5.  实验结果

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 播种 LSTM 国家

在 Keras 中使用无状态 LSTM 时，您可以在清除模型的内部状态时进行细粒度控制。

这是使用`model.reset_states()`函数实现的。

在训练有状态 LSTM 时，重要的是清除训练时期之间的模型状态。这使得在时期训练期间建立的国家与时代中的观察序列相称。

鉴于我们有这种细粒度控制，在做出预测之前，是否以及如何初始化 LSTM 的状态存在一个问题。

选项是：

*   在预测之前重置状态。
*   在预测之前使用训练数据集初始化状态。

假设使用训练数据初始化模型的状态将是优越的，但这需要通过实验来确认。

此外，可能有多种方法来种植这种状态;例如：

*   完成训练时期，包括体重更新。例如，不要在上一个训练时期结束时重置。
*   完成训练数据的预测。

通常，认为这两种方法在某种程度上是等同的。后者预测训练数据集是首选，因为它不需要对网络权重进行任何修改，并且可以是保存到文件的不可变网络的可重复过程。

在本教程中，我们将考虑以下两者之间的区别：

*   使用没有状态的拟合 LSTM 预测测试数据集（例如，在重置之后）。
*   在预测了训练数据集之后，使用状态 LSTM 预测测试数据集。

接下来，让我们看一下我们将在本实验中使用的标准时间序列数据集。

## 洗发水销售数据集

该数据集描述了 3 年期间每月洗发水的销售数量。

单位是销售计数，有 36 个观察。原始数据集归功于 Makridakis，Wheelwright 和 Hyndman（1998）。

[您可以在此处下载并了解有关数据集的更多信息](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)。

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

![Line Plot of Shampoo Sales Dataset](img/ab56d09b72803271b91fa5bd0ccc3f0f.jpg)

洗发水销售数据集的线图

接下来，我们将了解实验中使用的 LSTM 配置和测试工具。

## LSTM 模型和测试线束

### 数据拆分

我们将 Shampoo Sales 数据集分为两部分：训练和测试集。

前两年的数据将用于训练数据集，剩余的一年数据将用于测试集。

将使用训练数据集开发模型，并对测试数据集做出预测。

### 模型评估

将使用滚动预测场景，也称为前进模型验证。

测试数据集的每个时间步骤将一次一个地走。将使用模型对时间步长做出预测，然后将获取测试集的实际预期值，并使其可用于下一时间步的预测模型。

这模仿了一个真实世界的场景，每个月都会有新的洗发水销售观察结果，并用于下个月的预测。

这将通过训练和测试数据集的结构进行模拟。我们将以一次性方法进行所有预测。

将收集关于测试数据集的所有预测，并计算错误分数以总结模型的技能。将使用均方根误差（RMSE），因为它会对大错误进行处罚，并产生与预测数据相同的分数，即每月洗发水销售额。

### 数据准备

在我们将 LSTM 模型拟合到数据集之前，我们必须转换数据。

在拟合模型和做出预测之前，对数据集执行以下三个数据变换。

1.  **转换时间序列数据，使其静止**。具体地，_ 滞后= 1_ 差异以消除数据中的增加趋势。
2.  **将时间序列转换为监督学习问题**。具体而言，将数据组织成输入和输出模式，其中前一时间步骤的观察被用作在当前时间步长预测观察的输入。
3.  **将观察结果转换为具有特定比例**。具体而言，要将数据重缩放为-1 到 1 之间的值，以满足 LSTM 模型的默认双曲正切激活函数。

### LSTM 模型

将使用熟练但未调整的 LSTM 模型配置。

这意味着模型将适合数据并且能够进行有意义的预测，但不会是数据集的最佳模型。

网络拓扑由 1 个输入，4 个单元的隐藏层和 1 个输出值的输出层组成。

该模型适用于批量大小为 4 的 3,000 个时期。数据准备后，训练数据集将减少到 20 个观测值。这样，批量大小均匀地分为训练数据集和测试数据集（需求）。

### 实验运行

每个方案将运行 30 次。

这意味着将为每个方案创建和评估 30 个模型。将收集每次运行的 RMSE，提供一组结果，可使用描述性统计量（如均值和标准差）进行汇总。

这是必需的，因为像 LSTM 这样的神经网络受其初始条件（例如它们的初始随机权重）的影响。

每个场景的平均结果将允许我们解释每个场景的平均行为以及它们的比较方式。

让我们深入研究结果。

## 代码清单

关键模块化行为被分为可读性和可测试性功能，以备您重复使用此实验设置。

_ 实验（）_ 功能描述了这些场景的细节。

完整的代码清单如下。

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
import matplotlib
# be able to save images on server
matplotlib.use('Agg')
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
	df = df.drop(0)
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
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
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

# run a repeated experiment
def experiment(repeats, series, seed):
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
	# run experiment
	error_scores = list()
	for r in range(repeats):
		# fit the model
		batch_size = 4
		train_trimmed = train_scaled[2:, :]
		lstm_model = fit_lstm(train_trimmed, batch_size, 3000, 4)
		# forecast the entire training dataset to build up state for forecasting
		if seed:
			train_reshaped = train_trimmed[:, 0].reshape(len(train_trimmed), 1, 1)
			lstm_model.predict(train_reshaped, batch_size=batch_size)
		# forecast test dataset
		test_reshaped = test_scaled[:,0:-1]
		test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
		output = lstm_model.predict(test_reshaped, batch_size=batch_size)
		predictions = list()
		for i in range(len(output)):
			yhat = output[i,0]
			X = test_scaled[i, 0:-1]
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
	return error_scores

# load dataset
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# experiment
repeats = 30
results = DataFrame()
# with seeding
with_seed = experiment(repeats, series, True)
results['with-seed'] = with_seed
# without seeding
without_seed = experiment(repeats, series, False)
results['without-seed'] = without_seed
# summarize results
print(results.describe())
# save boxplot
results.boxplot()
pyplot.savefig('boxplot.png')
```

## 实验结果

运行实验需要一些时间或 CPU 或 GPU 硬件。

打印每次运行的 RMSE 以了解进度。

在运行结束时，将为每个方案计算和打印摘要统计量，包括平均值和标准差。

完整输出如下所示。

```py
1) Test RMSE: 86.566
2) Test RMSE: 300.874
3) Test RMSE: 169.237
4) Test RMSE: 167.939
5) Test RMSE: 135.416
6) Test RMSE: 291.746
7) Test RMSE: 220.729
8) Test RMSE: 222.815
9) Test RMSE: 236.043
10) Test RMSE: 134.183
11) Test RMSE: 145.320
12) Test RMSE: 142.771
13) Test RMSE: 239.289
14) Test RMSE: 218.629
15) Test RMSE: 208.855
16) Test RMSE: 187.421
17) Test RMSE: 141.151
18) Test RMSE: 174.379
19) Test RMSE: 241.310
20) Test RMSE: 226.963
21) Test RMSE: 126.777
22) Test RMSE: 197.340
23) Test RMSE: 149.662
24) Test RMSE: 235.681
25) Test RMSE: 200.320
26) Test RMSE: 92.396
27) Test RMSE: 169.573
28) Test RMSE: 219.894
29) Test RMSE: 168.048
30) Test RMSE: 141.638
1) Test RMSE: 85.470
2) Test RMSE: 151.935
3) Test RMSE: 102.314
4) Test RMSE: 215.588
5) Test RMSE: 172.948
6) Test RMSE: 114.746
7) Test RMSE: 205.958
8) Test RMSE: 89.335
9) Test RMSE: 183.635
10) Test RMSE: 173.400
11) Test RMSE: 116.645
12) Test RMSE: 133.473
13) Test RMSE: 155.044
14) Test RMSE: 153.582
15) Test RMSE: 146.693
16) Test RMSE: 95.455
17) Test RMSE: 104.970
18) Test RMSE: 127.700
19) Test RMSE: 189.728
20) Test RMSE: 127.756
21) Test RMSE: 102.795
22) Test RMSE: 189.742
23) Test RMSE: 144.621
24) Test RMSE: 132.053
25) Test RMSE: 238.034
26) Test RMSE: 139.800
27) Test RMSE: 202.881
28) Test RMSE: 172.278
29) Test RMSE: 125.565
30) Test RMSE: 103.868
        with-seed  without-seed
count   30.000000     30.000000
mean   186.432143    146.600505
std     52.559598     40.554595
min     86.565993     85.469737
25%    143.408162    115.221000
50%    180.899814    142.210265
75%    222.293194    173.287017
max    300.873841    238.034137
```

还会创建一个框和胡须图并将其保存到文件中，如下所示。

![Box and Whisker Plot of LSTM with and Without Seed of State](img/a6ea8e6072ec5a377562eae3a0b4a337.jpg)

含有和不含种子的 LSTM 的盒子和晶须图

结果令人惊讶。

他们建议通过在预测测试数据集之前不播种 LSTM 的状态来获得更好的结果。

这可以通过每月洗发水销售额 146.600505 的平均误差较低来看出，而播种的平均误差为 186.432143。分布的盒子和须状图更加清晰。

也许所选择的模型配置导致模型太小而不依赖于序列和内部状态以在预测之前受益于播种。也许需要更大的实验。

### 扩展

令人惊讶的结果为进一步的实验打开了大门。

*   评估在最后一个训练时期结束后清除与不清除状态的影响。
*   评估一次预测训练和测试集的效果，一次评估一个时间步。
*   评估在每个迭代结束时重置和不重置 LSTM 状态的效果。

你尝试过其中一种扩展吗？在下面的评论中分享您的发现。

## 摘要

在本教程中，您了解了如何通过实验确定在单变量时间序列预测问题上为 LSTM 模型的状态设定种子的最佳方法。

具体来说，你学到了：

*   关于在预测之前播种 LSTM 状态的问题及解决方法。
*   如何开发一个强大的测试工具来评估 LSTM 模型的时间序列预测。
*   如何在预测之前确定是否使用训练数据对 LSTM 模型的状态进行种子化。

您是否运行了实验或运行了实验的修改版本？
在评论中分享您的结果;我很想见到他们。

你对这篇文章有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。