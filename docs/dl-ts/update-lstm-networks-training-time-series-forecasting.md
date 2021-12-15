# 如何在时间序列预测训练期间更新 LSTM 网络

> 原文： [https://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/](https://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/)

使用神经网络模型进行时间序列预测的好处是可以在新数据可用时更新权重。

在本教程中，您将了解如何使用新数据更新长期短期记忆（LSTM）循环神经网络以进行时间序列预测。

完成本教程后，您将了解：

*   如何用新数据更新 LSTM 神经网络。
*   如何开发测试工具来评估不同的更新方案。
*   如何解释使用新数据更新 LSTM 网络的结果。

让我们开始吧。

*   **2017 年 4 月更新**：添加了缺少的 update_model（）函数。

![How to Update LSTM Networks During Training for Time Series Forecasting](img/f48debb1f6b1b8f3f780f8dcb7ef230c.jpg)

如何在时间序列预测训练期间更新 LSTM 网络
照片由 [Esteban Alvarez](https://www.flickr.com/photos/alvaretz/8427810143/) ，保留一些权利。

## 教程概述

本教程分为 9 个部分。他们是：

1.  洗发水销售数据集
2.  实验测试线束
3.  实验：没有更新
4.  实验：2 更新时期
5.  实验：5 更新时期
6.  实验：10 个更新时期
7.  实验：20 个更新时期
8.  实验：50 更新时期
9.  结果比较

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

本教程假设您安装了 TensorFlow 或 Theano 后端的 Keras v2.0 或更高版本。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

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

![Line Plot of Shampoo Sales Dataset](img/646e3de8684355414799cd9964ad1d4f.jpg)

洗发水销售数据集的线图

接下来，我们将了解实验中使用的 LSTM 配置和测试工具。

## 实验测试线束

本节介绍本教程中使用的测试工具。

### 数据拆分

我们将 Shampoo Sales 数据集分为两部分：训练和测试集。

前两年的数据将用于训练数据集，剩余的一年数据将用于测试集。

将使用训练数据集开发模型，并对测试数据集做出预测。

测试数据集的持久性预测（朴素预测）实现了每月洗发水销售 136.761 的错误。这为测试集提供了可接受的表现下限。

### 模型评估

将使用滚动预测场景，也称为前进模型验证。

测试数据集的每个时间步骤将一次一个地走。将使用模型对时间步长做出预测，然后将获取测试集的实际预期值，并使其可用于下一时间步的预测模型。

这模仿了一个真实世界的场景，每个月都会有新的洗发水销售观察结果，并用于下个月的预测。

这将通过训练和测试数据集的结构进行模拟。

将收集关于测试数据集的所有预测，并计算错误分数以总结模型的技能。将使用均方根误差（RMSE），因为它会对大错误进行处罚，并产生与预测数据相同的分数，即每月洗发水销售额。

### 数据准备

在我们将 LSTM 模型拟合到数据集之前，我们必须转换数据。

在拟合模型和做出预测之前，对数据集执行以下三个数据变换。

1.  **转换时间序列数据，使其静止**。具体而言，滞后= 1 差分以消除数据中的增加趋势。
2.  **将时间序列转换为监督学习问题**。具体而言，将数据组织成输入和输出模式，其中前一时间步的观察被用作预测当前时间步的观察的输入
3.  **将观察结果转换为具有特定比例**。具体而言，要将数据重缩放为-1 到 1 之间的值，以满足 LSTM 模型的默认双曲正切激活函数。

在计算错误分数之前，这些变换在预测中反转以将它们返回到其原始比例。

### LSTM 模型

我们将使用 LSTM 模型，其中 1 个神经元适合 500 个时期。

批量大小为 1 是必需的，因为我们将使用前向验证并对最后 12 个月的每个数据进行一步预测。

批量大小为 1 意味着该模型将使用在线训练（而不是批量训练或小批量训练）。因此，预计模型拟合将具有一些变化。

理想情况下，将使用更多的训练时期（例如 1000 或 1500），但这被截断为 500 以保持运行时间合理。

使用有效的 ADAM 优化算法和均方误差损失函数来拟合模型。

### 实验运行

每个实验场景将运行 10 次。

其原因在于，每次训练给定配置时，LSTM 网络的随机初始条件可导致非常不同的表现。

让我们深入研究实验。

## 实验：没有更新

在第一个实验中，我们将评估一次训练的 LSTM 并重复使用以对每个时间步做出预测。

我们将其称为'_ 无更新模型 _'或'_ 固定模型 _'，因为一旦模型首次适合训练数据，将不会进行更新。这提供了一个表现基准，我们期望实验能够对模型进行适度更新，从而超越表现。

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
import numpy
from numpy import concatenate

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
def experiment(repeats, series):
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
		# fit the base model
		lstm_model = fit_lstm(train_scaled, 1, 500, 1)
		# forecast test dataset
		predictions = list()
		for i in range(len(test_scaled)):
			# predict
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
	return error_scores

# execute the experiment
def run():
	# load dataset
	series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
	# experiment
	repeats = 10
	results = DataFrame()
	# run experiment
	results['results'] = experiment(repeats, series)
	# summarize results
	print(results.describe())
	# save results
	results.to_csv('experiment_fixed.csv', index=False)

 # entry point
run()
```

运行该示例使用前向验证存储在测试数据集上计算的 RMSE 分数。它们存储在名为`experiment_fixed.csv`的文件中，以供以后分析。打印分数摘要，如下所示。

结果表明，平均表现优于持久性模型，显示测试 RMSE 为 109.565465，而持久性的洗发水销售额为 136.761。

```py
          results
count   10.000000
mean   109.565465
std     14.329646
min     95.357198
25%     99.870983
50%    104.864387
75%    113.553952
max    138.261929
```

接下来，我们将开始查看在前向验证期间对模型进行更新的配置。

## 实验：2 更新时期

在本实验中，我们将模型拟合到所有训练数据上，然后在前向验证期间的每个预测之后更新模型。

然后，将用于在测试数据集中引出预测的每个测试模式添加到训练数据集中，并更新模型。

在这种情况下，该模型在进行下一次预测之前适合额外的 2 个训练时期。

使用与第一个实验中使用的相同的代码清单。代码清单的更改如下所示。

```py
# Update LSTM model
def update_model(model, train, batch_size, updates):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	for i in range(updates):
		model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()

# run a repeated experiment
def experiment(repeats, series, updates):
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
		# fit the base model
		lstm_model = fit_lstm(train_scaled, 1, 500, 1)
		# forecast test dataset
		train_copy = numpy.copy(train_scaled)
		predictions = list()
		for i in range(len(test_scaled)):
			# update model
			if i > 0:
				update_model(lstm_model, train_copy, 1, updates)
			# predict
			X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
			yhat = forecast_lstm(lstm_model, 1, X)
			# invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
			# store forecast
			predictions.append(yhat)
			# add to training set
			train_copy = concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
		# report performance
		rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
		print('%d) Test RMSE: %.3f' % (r+1, rmse))
		error_scores.append(rmse)
	return error_scores

# execute the experiment
def run():
	# load dataset
	series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
	# experiment
	repeats = 10
	results = DataFrame()
	# run experiment
	updates = 2
	results['results'] = experiment(repeats, series, updates)
	# summarize results
	print(results.describe())
	# save results
	results.to_csv('experiment_update_2.csv', index=False)

 # entry point
run()
```

运行实验将最终测试 RMSE 分数保存在“`experiment_update_2.csv`”中，并打印结果的摘要统计量，如下所示。

```py
          results
count   10.000000
mean    99.566270
std     10.511337
min     87.771671
25%     93.925243
50%     97.903038
75%    101.213058
max    124.748746
```

## 实验：5 更新时期

该实验重复上述更新实验，并在将每个测试模式添加到训练数据集之后将模型训练另外 5 个时期。

运行实验将最终测试 RMSE 分数保存在“`experiment_update_5.csv`”中，并打印结果的摘要统计量，如下所示。

```py
          results
count   10.000000
mean   101.094469
std      9.422711
min     91.642706
25%     94.593701
50%     98.954743
75%    104.998420
max    123.651985
```

## 实验：10 个更新时期

该实验重复上述更新实验，并在将每个测试模式添加到训练数据集之后将模型训练另外 10 个时期。

运行实验将最终测试 RMSE 分数保存在“`experiment_update_10.csv`”中，并打印结果的摘要统计量，如下所示。

```py
          results
count   10.000000
mean   108.806418
std     21.707665
min     92.161703
25%     94.872009
50%     99.652295
75%    112.607260
max    159.921749
```

## 实验：20 个更新时期

该实验重复上述更新实验，并在将每个测试模式添加到训练数据集之后将模型训练另外 20 个时期。

运行实验将最终测试 RMSE 分数保存在“`experiment_update_20.csv`”中，并打印结果的摘要统计量，如下所示。

```py
          results
count   10.000000
mean   112.070895
std     16.631902
min     96.822760
25%    101.790705
50%    103.380896
75%    119.479211
max    140.828410
```

## 实验：50 更新时期

该实验重复上述更新实验，并在将每个测试模式添加到训练数据集之后将模型训练另外 50 个时期。

运行实验将最终测试 RMSE 分数保存在“`experiment_update_50.csv`”中，并打印结果的摘要统计量，如下所示。

```py
          results
count   10.000000
mean   110.721971
std     22.788192
min     93.362982
25%     96.833140
50%     98.411940
75%    123.793652
max    161.463289
```

## 结果比较

在本节中，我们比较了之前实验中保存的结果。

我们加载每个保存的结果，使用描述性统计量汇总结果，并使用 box 和 whisker 图比较结果。

完整的代码清单如下。

```py
from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot
# load results into a dataframe
filenames = ['experiment_fixed.csv',
	'experiment_update_2.csv', 'experiment_update_5.csv',
	'experiment_update_10.csv', 'experiment_update_20.csv',
	'experiment_update_50.csv']
results = DataFrame()
for name in filenames:
	results[name[11:-4]] = read_csv(name, header=0)
# describe all results
print(results.describe())
# box and whisker plot
results.boxplot()
pyplot.show()
```

首先运行该示例计算并打印每个实验结果的描述性统计量。

如果我们看一下平均表现，我们可以看到固定模型提供了良好的表现基线，但我们发现适度数量的更新时期（20 和 50）平均会产生更差的测试集 RMSE。

我们看到少数更新时期导致更好的整体测试集表现，特别是 2 个时期，接着是 5 个时期。这是令人鼓舞的。

```py
            fixed    update_2    update_5   update_10   update_20   update_50
count   10.000000   10.000000   10.000000   10.000000   10.000000   10.000000
mean   109.565465   99.566270  101.094469  108.806418  112.070895  110.721971
std     14.329646   10.511337    9.422711   21.707665   16.631902   22.788192
min     95.357198   87.771671   91.642706   92.161703   96.822760   93.362982
25%     99.870983   93.925243   94.593701   94.872009  101.790705   96.833140
50%    104.864387   97.903038   98.954743   99.652295  103.380896   98.411940
75%    113.553952  101.213058  104.998420  112.607260  119.479211  123.793652
max    138.261929  124.748746  123.651985  159.921749  140.828410  161.463289
```

还创建了一个盒子和须状图，用于比较每个实验的测试 RMSE 结果的分布。

该图突出显示了每个实验的中位数（绿线）以及中间 50％的数据（框）。该图描述了与平均表现相同的故事，表明少数训练时期（2 或 5 个时期）导致最佳的整体测试 RMSE。

该图显示测试 RMSE 上升，因为更新次数增加到 20 个时期，然后再次下降 50 个时期。这可能是改进模型（11 * 50 时期）或少量重复（10）的人工制品的重要进一步训练的标志。

![Box and Whisker Plots Comparing the Number of Update Epochs](img/a2c6d8510072e11227a234fa5344bab9.jpg)

Box 和 Whisker Plots 比较更新时期的数量

重要的是要指出这些结果特定于模型配置和此数据集。

虽然这些实验确实为您自己的预测性建模问题执行类似的实验提供了框架，但很难将这些结果概括为超出此具体示例。

### 扩展

本节列出了本节中有关实验扩展的建议。

*   **统计显着性检验**。我们可以计算成对统计显着性检验，例如学生 t 检验，以查看结果群体中均值之间的差异是否具有统计学意义。
*   **更多重复**。我们可以将重复次数从 10 增加到 30,100 或更多，以使结果更加稳健。
*   **更多时代**。基础 LSTM 模型仅适用于 500 个具有在线训练的时期，并且相信额外的训练时期将导致更准确的基线模型。减少了时期的数量以减少实验运行时间。
*   **与更多 Epochs** 比较。更新模型的实验结果应直接与固定模型的实验进行比较，固定模型使用相同数量的总体时期来查看是否将额外的测试模式添加到训练数据集中会产生明显的差异。例如，可以将每个测试模式的 2 个更新时期与针对 500 +（12-1）* 2）或 522 个时期训练的固定模型进行比较，更新模型 5 与适合 500 +（12-1）的固定模型进行比较）* 5）或 555 个时代，依此类推。
*   **全新模型**。在将每个测试模式添加到训练数据集后，添加一个适合新模型的实验。这是尝试过的，但延长的运行时间阻止了在完成本教程之前收集结果。预计这将提供与更新和固定模型的有趣比较点。

你有没有探索过这些扩展？
在评论中报告您的结果;我很想听听你发现了什么。

## 摘要

在本教程中，您了解了如何更新 LSTM 网络，因为新数据可用于 Python 中的时间序列预测。

具体来说，你学到了：

*   如何设计一套系统的实验来探索更新 LSTM 模型的效果。
*   如何在新数据可用时更新 LSTM 模型。
*   对 LSTM 模型的更新可以产生更有效的预测模型，但是需要仔细校准预测问题。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。