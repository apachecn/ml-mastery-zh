# 用于时间序列预测的随机森林

> 原文：<https://machinelearningmastery.com/random-forest-for-time-series-forecasting/>

**随机森林**是一种流行且有效的集成机器学习算法。

它广泛用于结构化(表格)数据集的分类和回归预测建模问题，例如电子表格或数据库表中的数据。

随机森林也可以用于**时间序列预测**，尽管它需要先将时间序列数据集转化为有监督的学习问题。它还需要使用一种专门的技术来评估模型，称为向前验证，因为使用 k-fold 交叉验证来评估模型会导致乐观偏差的结果。

在本教程中，您将发现如何开发用于时间序列预测的随机森林模型。

完成本教程后，您将知道:

*   随机森林是决策树算法的集合，可用于分类和回归预测建模。
*   可以使用滑动窗口表示将时间序列数据集转换为监督学习。
*   如何使用随机森林回归模型拟合、评估和预测时间序列预测。

我们开始吧。

![Random Forest for Time Series Forecasting](img/79df98dbda016e207fa2ade66b1f1d01.png)

时间序列预测随机森林
图片由 [IvyMike](https://www.flickr.com/photos/ivymike/3894514618/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  随机森林集合
2.  时间序列数据准备
3.  时间序列的随机森林

## 随机森林集合

[随机森林](https://machinelearningmastery.com/random-forest-ensemble-in-python/)是决策树算法的集成。

它是决策树[自举聚合(bagging)](https://machinelearningmastery.com/bagging-ensemble-with-python/) 的扩展，可用于分类和回归问题。

在打包过程中，会生成许多决策树，其中每个树都是从训练数据集的不同引导样本中创建的。[引导样本](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)是训练数据集的样本，其中一个示例可能在样本中出现不止一次。这被称为“带替换的*采样*”。

Bagging 是一种有效的集成算法，因为每个决策树都适合于稍微不同的训练数据集，并且反过来具有稍微不同的表现。与普通的决策树模型(如分类和回归树(CART))不同，集成中使用的树是未标记的，这使得它们略微超出了训练数据集。这是可取的，因为它有助于使每棵树更加不同，并且具有较少的相关预测或预测误差。

来自这些树的预测在所有决策树中被平均，导致比模型中的任何单个树更好的表现。

回归问题的预测是集合中所有树的预测平均值。对分类问题的预测是对集合中所有树的类别标签的多数投票。

*   **回归**:预测是整个决策树的平均预测。
*   **分类**:预测是跨决策树预测的多数票类标签。

随机森林包括从训练数据集中的自举样本构建大量决策树，如 bagging。

与打包不同，随机森林还包括在构建树的每个分割点选择输入特征(列或变量)的子集。通常，构建决策树包括评估数据中每个输入变量的值，以便选择分割点。通过将特征简化为可以在每个分割点考虑的随机子集，它迫使集合中的每个决策树更加不同。

结果是，由系综中的每棵树做出的预测，进而预测误差，或多或少是不同的或相关的。当对来自这些相关性较低的树的预测进行平均以做出预测时，它通常比袋装决策树产生更好的表现。

有关随机森林算法的更多信息，请参见教程:

*   [如何用 Python 开发随机森林集成](https://machinelearningmastery.com/random-forest-ensemble-in-python/)

## 时间序列数据准备

时间序列数据可以被称为监督学习。

给定一个时间序列数据集的数字序列，我们可以重构数据，使其看起来像一个有监督的学习问题。我们可以通过使用以前的时间步长作为输入变量，并使用下一个时间步长作为输出变量来实现这一点。

让我们用一个例子来具体说明。假设我们有一个时间序列如下:

```py
time, measure
1, 100
2, 110
3, 108
4, 115
5, 120
```

我们可以通过使用前一时间步的值来预测下一时间步的值，从而将这个时间序列数据集重构为一个有监督的学习问题。

以这种方式重新组织时间序列数据集，数据将如下所示:

```py
X, y
?, 100
100, 110
110, 108
108, 115
115, 120
120, ?
```

请注意，时间列被删除，一些数据行不可用于训练模型，例如第一行和最后一行。

这种表示被称为滑动窗口，因为输入和预期输出的窗口随着时间向前移动，为监督学习模型创建新的“*样本*”。

有关准备时间序列预测数据的滑动窗口方法的更多信息，请参见教程:

*   [作为监督学习的时间序列预测](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

我们可以使用 Pandas 中的 [shift()函数](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)在给定输入和输出序列的期望长度的情况下，自动创建时间序列问题的新框架。

这将是一个有用的工具，因为它将允许我们用机器学习算法探索时间序列问题的不同框架，看看哪一个可能导致更好的模型。

下面的函数将把一个时间序列作为一个具有一列或多列的 NumPy 数组时间序列，并将其转换为具有指定数量的输入和输出的监督学习问题。

```py
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
```

我们可以使用这个函数为随机森林准备一个时间序列数据集。

有关该函数逐步开发的更多信息，请参见教程:

*   [如何在 Python 中将时间序列转换为监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

一旦准备好数据集，我们必须小心如何使用它来拟合和评估模型。

例如，用未来的数据拟合模型并让它预测过去是无效的。模型必须基于过去进行训练，并预测未来。

这意味着在评估期间随机化数据集的方法，如 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)，不能使用。相反，我们必须使用一种叫做向前验证的技术。

在向前行走验证中，首先通过选择切割点将数据集分割成训练集和测试集，例如，除了最后 12 个月之外的所有数据都用于训练，最后 12 个月用于测试。

如果我们有兴趣进行一步预测，例如一个月，那么我们可以通过在训练数据集上训练和预测测试数据集中的第一步来评估模型。然后，我们可以将测试集中的真实观察值添加到训练数据集中，重新调整模型，然后让模型预测测试数据集中的第二步。

对整个测试数据集重复这一过程将给出对整个测试数据集的一步预测，由此可以计算误差度量来评估模型的技能。

有关向前验证的更多信息，请参见教程:

*   [如何回测时间序列预测的机器学习模型](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

下面的函数执行向前行走验证。

它将时间序列数据集的整个监督学习版本和用作测试集的行数作为参数。

然后它遍历测试集，调用 *random_forest_forecast()* 函数进行一步预测。计算误差度量，并返回详细信息进行分析。

```py
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, 1], predictions
```

调用 *train_test_split()* 函数将数据集拆分为训练集和测试集。

我们可以在下面定义这个函数。

```py
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
```

我们可以使用[randomforestreversor](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)类进行一步预测。

下面的 *random_forest_forecast()* 函数实现了这一点，以训练数据集和测试输入行为输入，拟合一个模型，进行一步预测。

```py
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
```

现在我们知道了如何准备时间序列数据来预测和评估随机森林模型，接下来我们可以看看在真实数据集上使用随机森林。

## 时间序列的随机森林

在本节中，我们将探讨如何使用随机森林回归器进行时间序列预测。

我们将使用标准的单变量时间序列数据集，目的是使用该模型进行一步预测。

您可以将本节中的代码用作自己项目的起点，并轻松地将其用于多元输入、多元预测和多步预测。

我们将使用每日女性出生数据，即三年间的每月出生数据。

您可以从这里下载数据集，并将其放在当前工作目录中，文件名为“*每日女性出生总数. csv* ”。

*   [数据集(每日总女性分娩数)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)
*   [描述(每日出生总数-女性姓名)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.names)

数据集的前几行如下所示:

```py
"Date","Births"
"1959-01-01",35
"1959-01-02",32
"1959-01-03",30
"1959-01-04",31
"1959-01-05",44
...
```

首先，让我们加载并绘制数据集。

下面列出了完整的示例。

```py
# load and plot the time series dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
values = series.values
# plot dataset
pyplot.plot(values)
pyplot.show()
```

运行该示例会创建数据集的线图。

我们可以看到没有明显的趋势或季节性。

![Line Plot of Monthly Births Time Series Dataset](img/f43fdd1f847ab2ecc322d1387332f35b.png)

月出生时间序列数据集的线图

当预测最后 12 个月时，持久性模型可以实现大约 6.7 个出生的 MAE。这提供了一个表现基线，在这个基线之上，模型可以被认为是熟练的。

接下来，当对过去 12 个月的数据进行一步预测时，我们可以在数据集上评估随机森林模型。

我们将只使用前面的六个时间步长作为模型和默认模型超参数的输入，除了我们将在集合中使用 1000 棵树(以避免欠学习)。

下面列出了完整的示例。

```py
# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# load the dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
values = series.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=6)
# evaluate
mae, y, yhat = walk_forward_validation(data, 12)
print('MAE: %.3f' % mae)
# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()
```

运行该示例会报告测试集中每个步骤的预期值和预测值，然后是所有预测值的 MAE。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到，该模型比持久性模型表现得更好，实现了大约 5.9 次出生的 MAE，而不是 6.7 次出生。

**你能做得更好吗？**

您可以测试不同的随机森林超参数和时间步长数作为输入，看看是否可以获得更好的表现。在下面的评论中分享你的结果。

```py
>expected=42.0, predicted=45.0
>expected=53.0, predicted=43.7
>expected=39.0, predicted=41.4
>expected=40.0, predicted=38.1
>expected=38.0, predicted=42.6
>expected=44.0, predicted=48.7
>expected=34.0, predicted=42.7
>expected=37.0, predicted=37.0
>expected=52.0, predicted=38.4
>expected=48.0, predicted=41.4
>expected=55.0, predicted=43.7
>expected=50.0, predicted=45.3
MAE: 5.905
```

将数据集最后 12 个月的一系列预期值和预测值进行比较，创建一个折线图。

这给出了模型在测试集上表现如何的几何解释。

![Line Plot of Expected vs. Births Predicted Using Random Forest](img/ffe5a37d37dfa4df4632c127654018cc.png)

使用随机森林预测的预期与出生的线图

一旦选择了最终的随机森林模型配置，就可以最终确定模型，并用于对新数据进行预测。

这称为样本外预测，例如超出训练数据集的预测。这与在模型评估过程中进行预测是一样的，因为我们总是希望使用我们期望在模型用于对新数据进行预测时使用的相同过程来评估模型。

下面的示例演示了在所有可用数据上拟合最终的随机森林模型，并在数据集结束后进行一步预测。

```py
# finalize model and make a prediction for monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# load the dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
values = series.values
# transform the time series data into supervised learning
train = series_to_supervised(values, n_in=6)
# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
model = RandomForestRegressor(n_estimators=1000)
model.fit(trainX, trainy)
# construct an input for a new prediction
row = values[-6:].flatten()
# make a one-step prediction
yhat = model.predict(asarray([row]))
print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
```

运行该示例适合所有可用数据的随机森林模型。

使用过去六个月的已知数据准备新的输入行，并预测数据集结束后的下一个月。

```py
Input: [34 37 52 48 55 50], Predicted: 43.053
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何用 Python 开发随机森林集成](https://machinelearningmastery.com/random-forest-ensemble-in-python/)
*   [作为监督学习的时间序列预测](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
*   [如何在 Python 中将时间序列转换为监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [如何回溯测试用于时间序列预测的机器学习模型](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

### 蜜蜂

*   [硬化。一起。随机应变回归 API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 。

## 摘要

在本教程中，您发现了如何开发用于时间序列预测的随机森林模型。

具体来说，您了解到:

*   随机森林是决策树算法的集合，可用于分类和回归预测建模。
*   可以使用滑动窗口表示将时间序列数据集转换为监督学习。
*   如何使用随机森林回归模型拟合、评估和预测时间序列预测。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。