# 如何为单变量时间序列预测网格搜索朴素方法

> 原文： [https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/](https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/)

简单的预测方法包括朴素地使用最后一个观测值作为预测或先前观测值的平均值。

在使用更复杂的方法之前评估简单预测方法对单变量时间序列预测问题的表现非常重要，因为它们的表现提供了一个下限和比较点，可用于确定模型是否具有给定技能的技能问题。

尽管简单，诸如幼稚和平均预测策略之类的方法可以根据选择哪个先前观察持续存在或者先前观察到多少平均值来调整到特定问题。通常，调整这些简单策略的超参数可以为模型表现提供更强大和可防御的下限，以及可以为更复杂方法的选择和配置提供信息的令人惊讶的结果。

在本教程中，您将了解如何从零开始构建一个框架，用于网格搜索简单朴素和平均策略，用于使用单变量数据进行时间序列预测。

完成本教程后，您将了解：

*   如何使用前向验证从零开始开发网格搜索简单模型的框架。
*   如何为出生日常时间序列数据网格搜索简单模型超参数。
*   如何为洗发水销售，汽车销售和温度的月度时间序列数据网格搜索简单模型超参数。

让我们开始吧。

![How to Grid Search Naive Methods for Univariate Time Series Forecasting](img/18f93f0283af63bcca8a5887473b3e99.jpg)

如何网格搜索单变量时间序列预测的朴素方法
照片由 [Rob 和 Stephanie Levy](https://www.flickr.com/photos/robandstephanielevy/526862866/) ，保留一些权利。

## 教程概述

本教程分为六个部分;他们是：

1.  简单的预测策略
2.  开发网格搜索框架
3.  案例研究 1：没有趋势或季节性
4.  案例研究 2：趋势
5.  案例研究 3：季节性
6.  案例研究 4：趋势和季节性

## 简单的预测策略

在测试更复杂的模型之前测试简单的预测策略是非常重要和有用的。

简单的预测策略是那些对预测问题的性质很少或根本没有假设的策略，并且可以快速实现和计算。

结果可用作表现的基线，并用作比较点。如果模型的表现优于简单预测策略的表现，则可以说它具有技巧性。

简单预测策略有两个主题;他们是：

*   **朴素**，或直接使用观察值。
*   **平均**，或使用先前观察计算的统计量。

让我们仔细看看这两种策略。

### 朴素的预测策略

朴素的预测涉及直接使用先前的观察作为预测而没有任何改变。

它通常被称为持久性预测，因为之前的观察是持久的。

对于季节性数据，可以稍微调整这种简单的方法。在这种情况下，可以保持前一周期中的同时观察。

这可以进一步推广到将历史数据中的每个可能偏移量测试，以用于保持预测值。

例如，给定系列：

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

我们可以将最后一次观察（相对指数-1）保持为值 9，或者将第二次最后一次观察（相对指数-2）保持为 8，依此类推。

### 平均预测策略

朴素预测之上的一步是平均先前值的策略。

所有先前的观察结果均使用均值或中位数进行收集和平均，而不对数据进行其他处理。

在某些情况下，我们可能希望将平均计算中使用的历史缩短到最后几个观察结果。

我们可以将这一点推广到测试每个可能的 n 个先验观察的集合以包括在平均计算中的情况。

例如，给定系列：

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

我们可以平均最后一个观察（9），最后两个观察（8,9），依此类推。

在季节性数据的情况下，我们可能希望将周期中的最后 n 次先前观测值与预测时间进行平均。

例如，给定具有 3 步循环的系列：

```py
[1, 2, 3, 1, 2, 3, 1, 2, 3]
```

我们可以使用窗口大小为 3 并平均最后一个观察值（-3 或 1），最后两个观察值（-3 或 1 和 - （3 * 2）或 1），依此类推。

## 开发网格搜索框架

在本节中，我们将开发一个网格搜索框架，用于搜索前一节中描述的两个简单预测策略，即朴素和平均策略。

我们可以从实现一个朴素的预测策略开始。

对于给定的历史观测数据集，我们可以在该历史中保留任何值，即从索引-1 处的先前观察到历史上的第一次观察 - （len（data））。

下面的`naive_forecast()`函数实现了从 1 到数据集长度的给定偏移的朴素预测策略。

```py
# one-step naive forecast
def naive_forecast(history, n):
	return history[-n]
```

我们可以在一个小的设计数据集上测试这个功能。

```py
# one-step naive forecast
def naive_forecast(history, n):
	return history[-n]

# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# test naive forecast
for i in range(1, len(data)+1):
	print(naive_forecast(data, i))
```

首先运行示例打印设计的数据集，然后打印历史数据集中每个偏移的朴素预测。

```py
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
100.0
90.0
80.0
70.0
60.0
50.0
40.0
30.0
20.0
10.0
```

我们现在可以考虑为平均预测策略开发一个函数。

平均最后 n 次观测是直截了当的;例如：

```py
from numpy import mean
result = mean(history[-n:])
```

我们可能还想测试观察分布是非高斯分布的情况下的中位数。

```py
from numpy import median
result = median(history[-n:])
```

下面的`average_forecast()`函数实现了这一过程，它将历史数据和一个配置数组或元组指定为平均值作为整数的先前值的数量，以及一个描述计算平均值的方法的字符串（' _ 表示 _'或'_ 中值 _'）。

```py
# one-step average forecast
def average_forecast(history, config):
	n, avg_type = config
	# mean of last n values
	if avg_type is 'mean':
		return mean(history[-n:])
	# median of last n values
	return median(history[-n:])
```

下面列出了一个小型人为数据集的完整示例。

```py
from numpy import mean
from numpy import median

# one-step average forecast
def average_forecast(history, config):
	n, avg_type = config
	# mean of last n values
	if avg_type is 'mean':
		return mean(history[-n:])
	# median of last n values
	return median(history[-n:])

# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# test naive forecast
for i in range(1, len(data)+1):
	print(average_forecast(data, (i, 'mean')))
```

运行该示例将系列中的下一个值预测为先前观察的连续子集的平均值，从-1 到-10，包括在内。

```py
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
100.0
95.0
90.0
85.0
80.0
75.0
70.0
65.0
60.0
55.0
```

我们可以更新函数以支持季节性数据的平均值，并考虑季节性偏移。

可以向函数添加偏移量参数，当未设置为 1 时，将在收集要包括在平均值中的值之前确定先前观察的数量向后计数。

例如，如果 n = 1 且 offset = 3，则从 n * offset 或 1 * 3 = -3 处的单个值计算平均值。如果 n = 2 且 offset = 3，则从 1 * 3 或-3 和 2 * 3 或-6 的值计算平均值。

当季节性配置（n *偏移）超出历史观察结束时，我们还可以添加一些保护来引发异常。

更新的功能如下所示。

```py
# one-step average forecast
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# mean of last n values
	if avg_type is 'mean':
		return mean(values)
	# median of last n values
	return median(values)
```

我们可以在季节性循环的小型人为数据集上测试这个函数。

下面列出了完整的示例。

```py
from numpy import mean
from numpy import median

# one-step average forecast
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# mean of last n values
	if avg_type is 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# define dataset
data = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
print(data)
# test naive forecast
for i in [1, 2, 3]:
	print(average_forecast(data, (i, 3, 'mean')))
```

运行该示例计算[10]，[10,10]和[10,10,10]的平均值。

```py
[10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
10.0
10.0
10.0
```

可以将朴素预测策略和平均预测策略结合在一起，形成相同的功能。

这些方法之间存在一些重叠，特别是 _n-_ 偏移到历史记录中，用于持久化值或确定要平均的值的数量。

让一个函数支持这两种策略是有帮助的，这样我们就可以同时测试这两种策略的一套配置，作为简单模型的更广泛网格搜索的一部分。

下面的`simple_forecast()`函数将两种策略组合成一个函数。

```py
# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)
```

接下来，我们需要建立一些函数，通过前向验证重复拟合和评估模型，包括将数据集拆分为训练集和测试集以及评估一步预测。

我们可以使用给定指定大小的分割的切片来分割列表或 NumPy 数据数组，例如，从测试集中的数据中使用的时间步数。

下面的`train_test_split()`函数为提供的数据集和要在测试集中使用的指定数量的时间步骤实现此功能。

```py
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

在对测试数据集中的每个步骤做出预测之后，需要将它们与测试集进行比较以计算错误分数。

时间序列预测有许多流行的错误分数。在这种情况下，我们将使用均方根误差（RMSE），但您可以将其更改为您的首选度量，例如 MAPE，MAE 等

下面的`measure_rmse()`函数将根据实际（测试集）和预测值列表计算 RMSE。

```py
# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

我们现在可以实现[前进验证方案](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)。这是评估尊重观测时间顺序的时间序列预测模型的标准方法。

首先，使用 _train_test_split（_ _）_ 函数将提供的单变量时间序列数据集分成训练集和测试集。然后枚举测试集中的观察数。对于每一个我们都适合所有历史的模型，并进行一步预测。然后将对时间步骤的真实观察添加到历史中，并重复该过程。调用`simple_forecast`_（）_ 函数以适合模型并做出预测。最后，通过调用`measure_rmse()`函数，将所有一步预测与实际测试集进行比较，计算错误分数。

下面的`walk_forward_validation()`函数实现了这一点，采用单变量时间序列，在测试集中使用的一些时间步骤，以及模型配置数组。

```py
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error
```

如果您对进行多步预测感兴趣，可以在`simple_forecast`_（）_ 函数中更改 _ 预测（_ _）_ 的调用并且还改变`measure_rmse()`函数中的误差计算。

我们可以使用不同的模型配置列表重复调用 _walk_forward_validation（）_。

一个可能的问题是，可能不会为模型调用某些模型配置组合，并会抛出异常。

我们可以在网格搜索期间捕获异常并忽略警告，方法是将所有调用包含在 _walk_forward_validation（_ _）_ 中，并使用 try-except 和 block 来忽略警告。我们还可以添加调试支持来禁用这些保护，以防我们想要查看实际情况。最后，如果确实发生错误，我们可以返回 _ 无 _ 结果;否则，我们可以打印一些关于评估的每个模型的技能的信息。当评估大量模型时，这很有用。

下面的`score_model()`函数实现了这个并返回（键和结果）的元组，其中键是测试模型配置的字符串版本。

```py
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
```

接下来，我们需要一个循环来测试不同模型配置的列表。

这是驱动网格搜索过程的主要功能，并将为每个模型配置调用`score_model()`函数。

通过并行评估模型配置，我们可以大大加快网格搜索过程。一种方法是使用 [Joblib 库](https://pythonhosted.org/joblib/)。

我们可以定义一个 Parallel 对象，其中包含要使用的核心数，并将其设置为硬件中检测到的分数。

```py
executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
```

然后我们可以创建一个并行执行的任务列表，这将是我们每个模型配置对 score_model（）函数的一次调用。

```py
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
```

最后，我们可以使用`Parallel`对象并行执行任务列表。

```py
scores = executor(tasks)
```

而已。

我们还可以提供评估所有模型配置的非并行版本，以防我们想要调试某些内容。

```py
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
```

评估配置列表的结果将是元组列表，每个元组的名称总结了特定的模型配置，并且使用该配置评估的模型的错误为 RMSE 或 _ 无 _（如果有）一个错误。

我们可以过滤掉所有设置为 _ 无 _ 的分数。

```py
scores = [r for r in scores if r[1] != None]
```

然后我们可以按照升序排列列表中的所有元组（最好是第一个），然后返回此分数列表以供审阅。

给定单变量时间序列数据集，模型配置列表（列表列表）以及在测试集中使用的时间步数，下面的`grid_search()`函数实现此行为。可选的并行参数允许对所有内核的模型进行开启或关闭调整，默认情况下处于打开状态。

```py
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores
```

我们差不多完成了。

剩下要做的唯一事情是定义模型配置列表以尝试数据集。

我们可以一般地定义它。我们可能想要指定的唯一参数是系列中季节性组件的周期性（偏移量）（如果存在）。默认情况下，我们假设没有季节性组件。

下面的`simple_configs()`函数将创建要评估的模型配置列表。

该函数仅需要历史数据的最大长度作为参数，并且可选地需要任何季节性组件的周期性，其默认为 1（无季节性组件）。

```py
# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs
```

我们现在有一个网格搜索简单模型超参数的框架，通过一步前进验证。

它是通用的，适用于作为列表或 NumPy 数组提供的任何内存中单变量时间序列。

我们可以通过在人为设计的 10 步数据集上进行测试来确保所有部分协同工作。

下面列出了完整的示例。

```py
# grid search simple forecasts
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

if`_name_`== '__main__':
	# define dataset
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	print(data)
	# data split
	n_test = 4
	# model configs
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

首先运行该示例打印设计的时间序列数据集。

接下来，在评估模型配置及其错误时报告它们。

最后，报告前三种配置的配置和错误。

我们可以看到，配置为 1 的持久性模型（例如，持续最后一次观察）实现了所测试的简单模型的最佳表现，如预期的那样。

```py
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

> Model[[1, 1, 'persist']] 10.000
> Model[[2, 1, 'persist']] 20.000
> Model[[2, 1, 'mean']] 15.000
> Model[[2, 1, 'median']] 15.000
> Model[[3, 1, 'persist']] 30.000
> Model[[4, 1, 'persist']] 40.000
> Model[[5, 1, 'persist']] 50.000
> Model[[5, 1, 'mean']] 30.000
> Model[[3, 1, 'mean']] 20.000
> Model[[4, 1, 'median']] 25.000
> Model[[6, 1, 'persist']] 60.000
> Model[[4, 1, 'mean']] 25.000
> Model[[3, 1, 'median']] 20.000
> Model[[6, 1, 'mean']] 35.000
> Model[[5, 1, 'median']] 30.000
> Model[[6, 1, 'median']] 35.000
done

[1, 1, 'persist'] 10.0
[2, 1, 'mean'] 15.0
[2, 1, 'median'] 15.0
```

现在我们有一个强大的网格搜索简单模型超参数框架，让我们在一套标准的单变量时间序列数据集上进行测试。

每个数据集上显示的结果提供了一个表现基准，可用于比较更复杂的方法，如 SARIMA，ETS，甚至机器学习方法。

## 案例研究 1：没有趋势或季节性

“每日女性分娩”数据集总结了 1959 年美国加利福尼亚州每日女性总分娩数。

数据集没有明显的趋势或季节性成分。

![Line Plot of the Daily Female Births Dataset](img/dabb24b02643324fb9d69586be439808.jpg)

每日女性出生数据集的线图

您可以从 [DataMarket](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959#!ds=235k&display=line) 了解有关数据集的更多信息。

直接从这里下载数据集：

*   [每日总数 - 女性分娩.sv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在当前工作目录中使用文件名“ _daily-total-female-births.csv_ ”保存文件。

我们可以使用函数`read_csv()`将此数据集作为 Pandas 系列加载。

```py
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
```

数据集有一年或 365 个观测值。我们将使用前 200 个进行训练，将剩余的 165 个作为测试集。

下面列出了搜索每日女性单变量时间序列预测问题的完整示例网格。

```py
# grid search simple forecast for daily female births
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

if`_name_`== '__main__':
	# define dataset
	series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	print(data)
	# data split
	n_test = 165
	# model configs
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

运行该示例将打印模型配置，并在评估模型时打印 RMSE。

在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是出生率约为 6.93 的 RMSE，具有以下配置：

*   **策略**：平均值
*   **n** ：22
*   **函数**：mean（）

这是令人惊讶的，因为缺乏趋势或季节性，我会预期持续-1 或整个历史数据集的平均值，以产生最佳表现。

```py
...
> Model[[186, 1, 'mean']] 7.523
> Model[[200, 1, 'median']] 7.681
> Model[[186, 1, 'median']] 7.691
> Model[[187, 1, 'persist']] 11.137
> Model[[187, 1, 'mean']] 7.527
done

[22, 1, 'mean'] 6.930411499775709
[23, 1, 'mean'] 6.932293117115201
[21, 1, 'mean'] 6.951918385845375
```

## 案例研究 2：趋势

“洗发水”数据集总结了三年内洗发水的月销售额。

数据集包含明显的趋势，但没有明显的季节性成分。

![Line Plot of the Monthly Shampoo Sales Dataset](img/334f0b617304a565e1d93a31bdd1c50d.jpg)

月度洗发水销售数据集的线图

您可以从 [DataMarket](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds=22r0&display=line) 了解有关数据集的更多信息。

直接从这里下载数据集：

*   [shampoo.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv)

在当前工作目录中使用文件名“`shampoo.csv`”保存文件。

我们可以使用函数`read_csv()`将此数据集作为 Pandas 系列加载。

```py
# parse dates
def custom_parser(x):
	return datetime.strptime('195'+x, '%Y-%m')

# load dataset
series = read_csv('shampoo.csv', header=0, index_col=0, date_parser=custom_parser)
```

数据集有三年，或 36 个观测值。我们将使用前 24 个用于训练，其余 12 个用作测试集。

下面列出了搜索洗发水销售单变量时间序列预测问题的完整示例网格。

```py
# grid search simple forecast for monthly shampoo sales
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

# parse dates
def custom_parser(x):
	return datetime.strptime('195'+x, '%Y-%m')

if`_name_`== '__main__':
	# load dataset
	series = read_csv('shampoo.csv', header=0, index_col=0, date_parser=custom_parser)
	data = series.values
	print(data.shape)
	# data split
	n_test = 12
	# model configs
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

运行该示例将打印配置，并在评估模型时打印 RMSE。

在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是 RMSE 约 95.69 销售，具有以下配置：

*   **策略**：坚持
*   **n** ：2

这是令人惊讶的，因为数据的趋势结构表明持久的前一个值（-1）将是最好的方法，而不是持久的倒数第二个值。

```py
...
> Model[[23, 1, 'mean']] 209.782
> Model[[23, 1, 'median']] 221.863
> Model[[24, 1, 'persist']] 305.635
> Model[[24, 1, 'mean']] 213.466
> Model[[24, 1, 'median']] 226.061
done

[2, 1, 'persist'] 95.69454007413378
[2, 1, 'mean'] 96.01140340258198
[2, 1, 'median'] 96.01140340258198
```

## 案例研究 3：季节性

“月平均温度”数据集总结了 1920 至 1939 年华氏诺丁汉城堡的月平均气温，以华氏度为单位。

数据集具有明显的季节性成分，没有明显的趋势。

![Line Plot of the Monthly Mean Temperatures Dataset](img/72a4721b5d8d493844936208df537995.jpg)

月平均气温数据集的线图

您可以从 [DataMarket](https://datamarket.com/data/set/22li/mean-monthly-air-temperature-deg-f-nottingham-castle-1920-1939#!ds=22li&display=line) 了解有关数据集的更多信息。

直接从这里下载数据集：

*   [monthly-mean-temp.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-mean-temp.csv)

在当前工作目录中使用文件名“ _monthly-mean-temp.csv_ ”保存文件。

我们可以使用函数`read_csv()`将此数据集作为 Pandas 系列加载。

```py
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
```

数据集有 20 年，或 240 个观测值。我们将数据集修剪为过去五年的数据（60 个观测值），以加快模型评估过程并使用去年或 12 个观测值进行测试集。

```py
# trim dataset to 5 years
data = data[-(5*12):]
```

季节性成分的周期约为一年，或 12 个观测值。在准备模型配置时，我们将此作为调用`simple_configs()`函数的季节性时段。

```py
# model configs
cfg_list = simple_configs(seasonal=[0, 12])
```

下面列出了搜索月平均温度时间序列预测问题的完整示例网格。

```py
# grid search simple forecast for monthly mean temperature
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

if`_name_`== '__main__':
	# define dataset
	series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	print(data)
	# data split
	n_test = 12
	# model configs
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length, offsets=[1,12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

运行该示例将打印模型配置，并在评估模型时打印 RMSE。

在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是大约 1.501 度的 RMSE，具有以下配置：

*   **策略**：平均值
*   **n** ：4
*   **偏移**：12
*   **函数**：mean（）

这个发现并不太令人惊讶。鉴于数据的季节性结构，我们预计年度周期中先前点的最后几个观测值的函数是有效的。

```py
...
> Model[[227, 12, 'persist']] 5.365
> Model[[228, 1, 'persist']] 2.818
> Model[[228, 1, 'mean']] 8.258
> Model[[228, 1, 'median']] 8.361
> Model[[228, 12, 'persist']] 2.818
done
[4, 12, 'mean'] 1.5015616870445234
[8, 12, 'mean'] 1.5794579766489512
[13, 12, 'mean'] 1.586186052546763
```

## 案例研究 4：趋势和季节性

“月度汽车销售”数据集总结了 1960 年至 1968 年间加拿大魁北克省的月度汽车销量。

数据集具有明显的趋势和季节性成分。

![Line Plot of the Monthly Car Sales Dataset](img/6ff80f643e5ca7c00df5ce9e8b51cbc5.jpg)

月度汽车销售数据集的线图

您可以从 [DataMarket](https://datamarket.com/data/set/22n4/monthly-car-sales-in-quebec-1960-1968#!ds=22n4&display=line) 了解有关数据集的更多信息。

直接从这里下载数据集：

*   [month-car-sales.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv)

在当前工作目录中使用文件名“ _monthly-car-sales.csv_ ”保存文件。

我们可以使用函数`read_csv()`将此数据集作为 Pandas 系列加载。

```py
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
```

数据集有 9 年或 108 个观测值。我们将使用去年或 12 个观测值作为测试集。

季节性成分的期限可能是六个月或 12 个月。在准备模型配置时，我们将尝试将两者作为调用`simple_configs()`函数的季节性时段。

```py
# model configs
cfg_list = simple_configs(seasonal=[0,6,12])
```

下面列出了搜索月度汽车销售时间序列预测问题的完整示例网格。

```py
# grid search simple forecast for monthly car sales
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

if`_name_`== '__main__':
	# define dataset
	series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	print(data)
	# data split
	n_test = 12
	# model configs
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length, offsets=[1,12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

运行该示例将打印模型配置，并在评估模型时打印 RMSE。

在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是大约 1841.155 销售的 RMSE，具有以下配置：

*   **策略**：平均值
*   **n** ：3
*   **偏移**：12
*   **功能**：中位数（）

所选模型是先前周期中同一点的最后几次观察的函数并不奇怪，尽管使用中位数而不是均值可能不会立即明显，结果比平均值好得多。

```py
...
> Model[[79, 1, 'median']] 5124.113
> Model[[91, 12, 'persist']] 9580.149
> Model[[79, 12, 'persist']] 8641.529
> Model[[92, 1, 'persist']] 9830.921
> Model[[92, 1, 'mean']] 5148.126
done
[3, 12, 'median'] 1841.1559321976688
[3, 12, 'mean'] 2115.198495632485
[4, 12, 'median'] 2184.37708988932
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **地块预测**。更新框架以重新拟合具有最佳配置的模型并预测整个测试数据集，然后将预测与测试集中的实际观察值进行比较。
*   **漂移方法**。实现简单预测的漂移方法，并将结果与​​平均和朴素的方法进行比较。
*   **另一个数据集**。将开发的框架应用于另外的单变量时间序列问题（例如，来自[时间序列数据集库](https://datamarket.com/data/list/?q=provider:tsdl)）。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [预测，维基百科](https://en.wikipedia.org/wiki/Forecasting)
*   [Joblib：运行 Python 函数作为管道作业](https://pythonhosted.org/joblib/)
*   [时间序列数据集库](https://datamarket.com/data/list/?q=provider:tsdl)，DataMarket。

## 摘要

在本教程中，您了解了如何从零开始构建一个框架，用于网格搜索简单的朴素和平均策略，用于使用单变量数据进行时间序列预测。

具体来说，你学到了：

*   如何使用前向验证从零开始开发网格搜索简单模型的框架。
*   如何为出生日常时间序列数据网格搜索简单模型超参数。
*   如何为洗发水销售，汽车销售和温度的月度时间序列数据网格搜索简单模型超参数。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。