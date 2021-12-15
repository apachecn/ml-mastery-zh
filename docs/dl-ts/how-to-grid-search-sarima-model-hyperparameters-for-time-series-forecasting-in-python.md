# 如何在 Python 中为时间序列预测搜索 SARIMA 模型超参数

> 原文： [https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/](https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/)

季节性自回归整合移动平均线（SARIMA）模型是一种对可能包含趋势和季节性成分的单变量时间序列数据进行建模的方法。

它是时间序列预测的有效方法，但需要仔细分析和领域专业知识才能配置七个或更多模型超参数。

配置模型的另一种方法是利用快速和并行的现代硬件来网格搜索一套超参数配置，以便发现最有效的方法。通常，此过程可以揭示非直观模型配置，这些配置导致预测误差低于通过仔细分析指定的配置。

在本教程中，您将了解如何开发网格搜索所有 SARIMA 模型超参数的框架，以进行单变量时间序列预测。

完成本教程后，您将了解：

*   如何使用前向验证从零开始开发网格搜索 SARIMA 模型的框架。
*   如何为出生日常时间序列数据网格搜索 SARIMA 模型超参数。
*   如何对洗发水销售，汽车销售和温度的月度时间序列数据进行网格搜索 SARIMA 模型超参数。

让我们开始吧。

![How to Grid Search SARIMA Model Hyperparameters for Time Series Forecasting in Python](img/708ea207c23e8bbb98e0546431cbfbd8.jpg)

如何在 Python 中搜索用于时间序列预测的 SARIMA 模型超参数
[Thomas](https://www.flickr.com/photos/photommo/17832992898/) 的照片，保留一些权利。

## 教程概述

本教程分为六个部分;他们是：

1.  SARIMA 用于时间序列预测
2.  开发网格搜索框架
3.  案例研究 1：没有趋势或季节性
4.  案例研究 2：趋势
5.  案例研究 3：季节性
6.  案例研究 4：趋势和季节性

## SARIMA 用于时间序列预测

季节性自回归整合移动平均线，SARIMA 或季节性 ARIMA，是 ARIMA 的扩展，明确支持具有季节性成分的单变量时间序列数据。

它增加了三个新的超参数来指定系列季节性成分的自回归（AR），差分（I）和移动平均（MA），以及季节性周期的附加参数。

> 通过在 ARIMA 中包含额外的季节性术语来形成季节性 ARIMA 模型[...]模型的季节性部分由与模型的非季节性组成非常相似的术语组成，但它们涉及季节性时段的后移。

- 第 242 页，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。

配置 SARIMA 需要为系列的趋势和季节性元素选择超参数。

有三个趋势元素需要配置。

它们与 ARIMA 模型相同;特别：

*   **p** ：趋势自动回归顺序。
*   **d** ：趋势差异顺序。
*   **q** ：趋势均线。

有四个不属于 ARIMA 的季节性元素必须配置;他们是：

*   **P** ：季节性自回归顺序。
*   **D** ：季节性差异顺序。
*   **Q** ：季节性移动平均线。
*   **m** ：单个季节性时段的时间步数。

同时，SARIMA 模型的表示法指定为：

```py
SARIMA(p,d,q)(P,D,Q)m
```

SARIMA 模型可以通过模型配置参数包含 ARIMA，ARMA，AR 和 MA 模型。

可以通过分析自相关和部分自相关图来配置模型的趋势和季节性超参数，这可能需要一些专业知识。

另一种方法是对一组模型配置进行网格搜索，并发现哪些配置最适合特定的单变量时间序列。

> 季节性 ARIMA 模型可能具有大量参数和术语组合。因此，在拟合数据时尝试各种模型并使用适当的标准选择最佳拟合模型是合适的...

- 第 143-144 页，[介绍时间序列与 R](https://amzn.to/2smB9LR) ，2009 年。

这种方法在现代计算机上比分析过程更快，并且可以揭示可能不明显的令人惊讶的结果并导致较低的预测误差。

## 开发网格搜索框架

在本节中，我们将针对给定的单变量时间序列预测问题开发网格搜索 SARIMA 模型超参数的框架。

我们将使用 statsmodels 库提供的 [SARIMA](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) 的实现。

该模型具有超参数，可控制为系列，趋势和季节性执行的模型的性质，具体为：

*   **order** ：用于趋势建模的元组 p，d 和 q 参数。
*   **sesonal_order** ：用于建模季节性的 P，D，Q 和 m 参数元组
*   **趋势**：用于将确定性趋势模型控制为“n”，“c”，“t”，“ct”之一的参数，无趋势，常数，线性和常数，线性趋势，分别。

如果您对问题了解得足以指定其中一个或多个参数，则应指定它们。如果没有，您可以尝试网格搜索这些参数。

我们可以通过定义一个适合具有给定配置的模型的函数来开始，并进行一步预测。

下面的`sarima_forecast()`实现了这种行为。

该函数采用连续先前观察的数组或列表以及用于配置模型的配置参数列表，特别是两个元组和趋势顺序，季节性顺序趋势和参数的字符串。

我们还尝试通过放宽约束来使模型健壮，例如数据必须是静止的并且 MA 变换是可逆的。

```py
# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
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

我们现在可以实现前向验证方案。这是评估尊重观测时间顺序的时间序列预测模型的标准方法。

首先，使用`train_test_split()`函数将提供的单变量时间序列数据集分成训练集和测试集。然后枚举测试集中的观察数。对于每一个我们都适合所有历史的模型，并进行一步预测。然后将对时间步骤的真实观察添加到历史中并重复该过程。调用`sarima_forecast()`函数以适合模型并做出预测。最后，通过调用`measure_rmse()`函数，将所有一步预测与实际测试集进行比较，计算错误分数。

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
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error
```

如果您对进行多步预测感兴趣，可以在`sarima_forecast()`函数中更改`predict()`的调用，并更改 _ 中的错误计算 measure_rmse（）_ 功能。

我们可以使用不同的模型配置列表重复调用 _walk_forward_validation（）_。

一个可能的问题是，可能不会为模型调用模型配置的某些组合，并且会抛出异常，例如，指定数据中季节性结构的一些但不是所有方面。

此外，某些型号还可能会对某些数据发出警告，例如：来自 statsmodels 库调用的线性代数库。

我们可以在网格搜索期间捕获异常并忽略警告，方法是将所有调用包含在`walk_forward_validation()`中，并使用 try-except 和 block 来忽略警告。我们还可以添加调试支持来禁用这些保护，以防我们想要查看实际情况。最后，如果确实发生了错误，我们可以返回 None 结果，否则我们可以打印一些有关每个模型评估技能的信息。当评估大量模型时，这很有用。

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

然后我们可以创建一个并行执行的任务列表，这将是对我们拥有的每个模型配置的`score_model()`函数的一次调用。

```py
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
```

最后，我们可以使用 Parallel 对象并行执行任务列表。

```py
scores = executor(tasks)
```

而已。

我们还可以提供评估所有模型配置的非并行版本，以防我们想要调试某些内容。

```py
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
```

评估配置列表的结果将是元组列表，每个元组都有一个名称，该名称总结了特定的模型配置，并且使用该配置评估的模型的错误为 RMSE，如果出现错误则为 None。

我们可以使用“无”过滤掉所有分数。

```py
scores = [r for r in scores if r[1] != None]
```

然后我们可以按照升序排列列表中的所有元组（最好是第一个），然后返回此分数列表以供审阅。

给定单变量时间序列数据集，模型配置列表（列表列表）以及在测试集中使用的时间步数，下面的`grid_search()`函数实现此行为。可选的 _ 并行 _ 参数允许对所有内核的模型进行开启或关闭调整，默认情况下处于打开状态。

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

我们可以一般地定义它。我们可能想要指定的唯一参数是系列中季节性组件的周期性（如果存在）。默认情况下，我们假设没有季节性组件。

下面的`sarima_configs()`函数将创建要评估的模型配置列表。

这些配置假设趋势和季节性的每个 AR，MA 和 I 分量都是低阶的，例如，关（0）或[1,2]。如果您认为订单可能更高，则可能需要扩展这些范围。可以指定季节性时段的可选列表，您甚至可以更改该功能以指定您可能了解的有关时间序列的其他元素。

理论上，有 1,296 种可能的模型配置需要评估，但在实践中，许多模型配置无效并会导致我们将陷入和忽略的错误。

```py
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models
```

我们现在有一个网格搜索 SARIMA 模型超参数的框架，通过一步前进验证。

它是通用的，适用于作为列表或 NumPy 数组提供的任何内存中单变量时间序列。

我们可以通过在人为设计的 10 步数据集上进行测试来确保所有部分协同工作。

下面列出了完整的示例。

```py
# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = sarima_forecast(history, cfg)
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if`_name_`== '__main__':
	# define dataset
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	print(data)
	# data split
	n_test = 4
	# model configs
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

首先运行该示例打印设计的时间序列数据集。

接下来，在评估模型配置及其错误时报告模型配置及其错误，为简洁起见，将其截断。

最后，报告前三种配置的配置和错误。我们可以看到，许多模型在这个简单的线性增长的时间序列问题上实现了完美的表现。

```py
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

...
 > Model[[(2, 0, 0), (2, 0, 0, 0), 'ct']] 0.001
 > Model[[(2, 0, 0), (2, 0, 1, 0), 'ct']] 0.000
 > Model[[(2, 0, 1), (0, 0, 0, 0), 'n']] 0.000
 > Model[[(2, 0, 1), (0, 0, 1, 0), 'n']] 0.000
done

[(2, 1, 0), (1, 0, 0, 0), 'n'] 0.0
[(2, 1, 0), (2, 0, 0, 0), 'n'] 0.0
[(2, 1, 1), (1, 0, 1, 0), 'n'] 0.0
```

现在我们有一个强大的网格搜索 SARIMA 模型超参数框架，让我们在一套标准的单变量时间序列数据集上进行测试。

选择数据集用于演示目的;我并不是说 SARIMA 模型是每个数据集的最佳方法;在某些情况下，或许 ETS 或其他更合适的东西。

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
# grid search sarima hyperparameters for daily female dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = sarima_forecast(history, cfg)
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if`_name_`== '__main__':
	# load dataset
	series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	print(data.shape)
	# data split
	n_test = 165
	# model configs
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

在现代硬件上运行该示例可能需要几分钟。

在评估模型时打印模型配置和 RMSE 在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是大约 6.77 个出生的 RMSE，具有以下配置：

*   **订单** :( 1,0,2）
*   **季节性命令** :( 1,0,1,0）
*   **趋势参数**：'t'表示线性趋势

令人惊讶的是，具有一些季节性元素的配置导致最低的错误。我不会猜到这种配置，可能会坚持使用 ARIMA 模型。

```py
...
> Model[[(2, 1, 2), (1, 0, 1, 0), 'ct']] 6.905
> Model[[(2, 1, 2), (2, 0, 0, 0), 'ct']] 7.031
> Model[[(2, 1, 2), (2, 0, 1, 0), 'ct']] 6.985
> Model[[(2, 1, 2), (1, 0, 2, 0), 'ct']] 6.941
> Model[[(2, 1, 2), (2, 0, 2, 0), 'ct']] 7.056
done

[(1, 0, 2), (1, 0, 1, 0), 't'] 6.770349800255089
[(0, 1, 2), (1, 0, 2, 0), 'ct'] 6.773217122759515
[(2, 1, 1), (2, 0, 2, 0), 'ct'] 6.886633191752254
```

## 案例研究 2：趋势

“洗发水”数据集总结了三年内洗发水的月销售额。

数据集包含明显的趋势，但没有明显的季节性成分。

![Line Plot of the Monthly Shampoo Sales Dataset](img/334f0b617304a565e1d93a31bdd1c50d.jpg)

月度洗发水销售数据集的线图

您可以从 [DataMarket](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds=22r0&display=line) 了解有关数据集的更多信息。

直接从这里下载数据集：

*   [shampoo.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv)

在当前工作目录中使用文件名“shampoo.csv”保存文件。

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
# grid search sarima hyperparameters for monthly shampoo sales dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = sarima_forecast(history, cfg)
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

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
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

在现代硬件上运行该示例可能需要几分钟。

在评估模型时打印模型配置和 RMSE 在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是 RMSE 约为 54.76，具有以下配置：

*   **趋势订单** :( 0,1,2）
*   **季节性命令** :( 2,0,2,0）
*   **趋势参数**：'t'（线性趋势）

```py
...
> Model[[(2, 1, 2), (1, 0, 1, 0), 'ct']] 68.891
> Model[[(2, 1, 2), (2, 0, 0, 0), 'ct']] 75.406
> Model[[(2, 1, 2), (1, 0, 2, 0), 'ct']] 80.908
> Model[[(2, 1, 2), (2, 0, 1, 0), 'ct']] 78.734
> Model[[(2, 1, 2), (2, 0, 2, 0), 'ct']] 82.958
done
[(0, 1, 2), (2, 0, 2, 0), 't'] 54.767582003072874
[(0, 1, 1), (2, 0, 2, 0), 'ct'] 58.69987083057107
[(1, 1, 2), (0, 0, 1, 0), 't'] 58.709089340600094
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

数据集有 20 年，或 240 个观测值。我们将数据集修剪为过去五年的数据（60 个观测值），以加快模型评估过程，并使用去年或 12 个观测值来测试集。

```py
# trim dataset to 5 years
data = data[-(5*12):]
```

季节性成分的周期约为一年，或 12 个观测值。在准备模型配置时，我们将此作为调用`sarima_configs()`函数的季节性时段。

```py
# model configs
cfg_list = sarima_configs(seasonal=[0, 12])
```

下面列出了搜索月平均温度时间序列预测问题的完整示例网格。

```py
# grid search sarima hyperparameters for monthly mean temp dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = sarima_forecast(history, cfg)
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if`_name_`== '__main__':
	# load dataset
	series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	# trim dataset to 5 years
	data = data[-(5*12):]
	# data split
	n_test = 12
	# model configs
	cfg_list = sarima_configs(seasonal=[0, 12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

在现代硬件上运行该示例可能需要几分钟。

在评估模型时打印模型配置和 RMSE 在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是大约 1.5 度的 RMSE，具有以下配置：

*   **趋势订单** :( 0,0,0）
*   **季节性命令** :( 1,0,1,12）
*   **趋势参数**：'n'（无趋势）

正如我们所料，该模型没有趋势组件和 12 个月的季节性 ARMA 组件。

```py
...
> Model[[(2, 1, 2), (2, 1, 0, 12), 't']] 4.599
> Model[[(2, 1, 2), (1, 1, 0, 12), 'ct']] 2.477
> Model[[(2, 1, 2), (2, 0, 0, 12), 'ct']] 2.548
> Model[[(2, 1, 2), (2, 0, 1, 12), 'ct']] 2.893
> Model[[(2, 1, 2), (2, 1, 0, 12), 'ct']] 5.404
done

[(0, 0, 0), (1, 0, 1, 12), 'n'] 1.5577613610905712
[(0, 0, 0), (1, 1, 0, 12), 'n'] 1.6469530713847962
[(0, 0, 0), (2, 0, 0, 12), 'n'] 1.7314448163607488
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

季节性成分的期限可能是六个月或 12 个月。在准备模型配置时，我们将尝试将两者作为调用`sarima_configs()`函数的季节性时段。

```py
# model configs
cfg_list = sarima_configs(seasonal=[0,6,12])
```

下面列出了搜索月度汽车销售时间序列预测问题的完整示例网格。

```py
# grid search sarima hyperparameters for monthly car sales dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = sarima_forecast(history, cfg)
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if`_name_`== '__main__':
	# load dataset
	series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	print(data.shape)
	# data split
	n_test = 12
	# model configs
	cfg_list = sarima_configs(seasonal=[0,6,12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

在现代硬件上运行该示例可能需要几分钟。

在评估模型时打印模型配置和 RMSE 在运行结束时报告前三个模型配置及其错误。

我们可以看到最好的结果是 RMSE 大约 1,551 销售，具有以下配置：

*   **趋势订单** :( 0,0,0）
*   **季节性命令** :( 1,1,0,12）
*   **趋势参数**：'t'（线性趋势）

```py
> Model[[(2, 1, 2), (2, 1, 1, 6), 'ct']] 2246.248
> Model[[(2, 1, 2), (2, 0, 2, 12), 'ct']] 10710.462
> Model[[(2, 1, 2), (2, 1, 2, 6), 'ct']] 2183.568
> Model[[(2, 1, 2), (2, 1, 0, 12), 'ct']] 2105.800
> Model[[(2, 1, 2), (2, 1, 1, 12), 'ct']] 2330.361
> Model[[(2, 1, 2), (2, 1, 2, 12), 'ct']] 31580326686.803
done
[(0, 0, 0), (1, 1, 0, 12), 't'] 1551.8423920342414
[(0, 0, 0), (2, 1, 1, 12), 'c'] 1557.334614575545
[(0, 0, 0), (1, 1, 0, 12), 'c'] 1559.3276311282675
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **数据转换**。更新框架以支持可配置的数据转换，例如规范化和标准化。
*   **地块预测**。更新框架以重新拟合具有最佳配置的模型并预测整个测试数据集，然后将预测与测试集中的实际观察值进行比较。
*   **调整历史数量**。更新框架以调整用于拟合模型的历史数据量（例如，在 10 年最高温度数据的情况下）。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何使用 Python 创建用于时间序列预测的 ARIMA 模型](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
*   [如何使用 Python 网格搜索 ARIMA 模型超参数](https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)
*   [自相关和部分自相关的温和介绍](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/)

### 图书

*   第 8 章 ARIMA 模型，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。
*   第 7 章，非平稳模型， [R](https://amzn.to/2smB9LR) 的入门时间序列，2009。

### API

*   [Statsmodels 状态空间方法的时间序列分析](http://www.statsmodels.org/dev/statespace.html)
*   [statsmodels.tsa.statespace.sarimax.SARIMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
*   [statsmodels.tsa.statespace.sarimax.SARIMAXResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html)
*   [Statsmodels SARIMAX 笔记本](http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html)
*   [Joblib：运行 Python 函数作为管道作业](https://pythonhosted.org/joblib/)

### 用品

*   [维基百科上的自回归综合移动平均线](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

## 摘要

在本教程中，您了解了如何开发网格搜索所有 SARIMA 模型超参数的框架，以进行单变量时间序列预测。

具体来说，你学到了：

*   如何使用前向验证从零开始开发网格搜索 SARIMA 模型的框架。
*   如何为出生日常时间序列数据网格搜索 SARIMA 模型超参数。
*   如何针对洗发水销售，汽车销售和温度的月度时间序列数据网格搜索 SARIMA 模型超参数。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。