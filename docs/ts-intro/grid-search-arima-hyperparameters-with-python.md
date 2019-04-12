# 如何使用 Python 网格搜索 ARIMA 模型超参数

> 原文： [https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/](https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)

用于时间序列分析和预测的 ARIMA 模型可能很难配置。

有 3 个参数需要通过迭代试验和错误来评估诊断图和使用 40 年的启发式规则。

我们可以使用网格搜索程序自动化为 ARIMA 模型评估大量超参数的过程。

在本教程中，您将了解如何使用 Python 中的超参数网格搜索来调整 ARIMA 模型。

完成本教程后，您将了解：

*   可用于调整 ARIMA 超参数以进行滚动一步预测的一般过程。
*   如何在标准单变量时间序列数据集上应用 ARIMA 超参数优化。
*   扩展程序的想法，以更精细和强大的模型。

让我们开始吧。

![How to Grid Search ARIMA Model Hyperparameters with Python](img/9059a2b70d6e079454dccb4ea1b13ae2.jpg)

如何使用 Python 网格搜索 ARIMA 模型超参数
照片由 [Alpha](https://www.flickr.com/photos/avlxyz/3825340087/) ，保留一些权利。

## 网格搜索方法

时间序列的诊断图可以与启发式规则一起使用，以确定 ARIMA 模型的超参数。

这些在大多数情况下都很好，但也许并非所有情况都很好。

我们可以在模型超参数的不同组合上自动化训练和评估 ARIMA 模型的过程。在机器学习中，这称为网格搜索或模型调整。

在本教程中，我们将开发一种网格搜索 ARIMA 超参数的方法，用于一步滚动预测。

该方法分为两部分：

1.  评估 ARIMA 模型。
2.  评估 ARIMA 参数集。

本教程中的代码使用了 scikit-learn，Pandas 和 statsmodels Python 库。

### 1.评估 ARIMA 模型

我们可以通过在训练数据集上准备 ARIMA 模型并评估测试数据集的预测来评估 ARIMA 模型。

此方法包括以下步骤：

1.  将数据集拆分为训练和测试集。
2.  遍历测试数据集中的时间步长。
    1.  训练 ARIMA 模型。
    2.  进行一步预测。
    3.  商店预测;获取并存储实际观察结果。
3.  计算预测的误差分数与预期值的比较。

我们可以在 Python 中实现这个作为一个名为 _evaluate_arima_model（）_ 的新独立函数，它将时间序列数据集作为输入，以及 _p_ ， _d_ 的元组]和 _q_ 参数用于评估模型。

数据集分为两部分：初始训练数据集为 66％，测试数据集为剩余的 34％。

迭代测试集的每个时间步。只需一次迭代就可以提供一个模型，您可以使用该模型对新数据进行预测。迭代方法允许每个步骤训练新的 ARIMA 模型。

每次迭代都进行预测并存储在列表中。这样，在测试集的末尾，可以将所有预测与预期值列表进行比较，并计算出错误分数。在这种情况下，计算并返回均方误差分数。

完整的功能如下所列。

```py
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
```

现在我们知道如何评估一组 ARIMA 超参数，让我们看看我们如何重复调用此函数以获得要评估的参数网格。

### 2.迭代 ARIMA 参数

评估一组参数相对简单。

用户必须指定 _p_ ， _d_ 和 _q_ ARIMA 参数的网格进行迭代。通过调用上一节中描述的 _evaluate_arima_model（）_ 函数，为每个参数创建模型并评估其表现。

该函数必须跟踪观察到的最低错误分数和导致它的配置。这可以在函数的末尾进行总结，并打印出标准输出。

我们可以将这个名为 _evaluate_models（）_ 的函数实现为一系列的四个循环。

还有两个注意事项。第一个是确保输入数据是浮点值（而不是整数或字符串），因为这可能导致 ARIMA 过程失败。

其次，statsmodels ARIMA 程序在内部使用数值优化程序来为模型找到一组系数。这些过程可能会失败，从而可能会引发异常。我们必须捕获这些异常并跳过导致问题的配置。这种情况经常发生，你会想到。

此外，建议忽略此代码的警告，以避免运行该过程产生大量噪音。这可以按如下方式完成：

```py
import warnings
warnings.filterwarnings("ignore")
```

最后，即使有了所有这些保护，底层的 C 和 Fortran 库仍然可以向标准输出报告警告，例如：

```py
** On entry to DLASCL, parameter number 4 had an illegal value
```

为简洁起见，这些内容已从本教程中报告的结果中删除。

下面列出了评估 ARIMA 超参数网格的完整过程。

```py
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
```

现在我们有一个网格搜索 ARIMA 超参数的过程，让我们测试两个单变量时间序列问题的过程。

我们将从 Shampoo Sales 数据集开始。

## 洗发水销售案例研究

Shampoo Sales 数据集描述了 3 年期间每月洗发水的销售数量。

单位是销售计数，有 36 个观察。原始数据集归功于 Makridakis，Wheelwright 和 Hyndman（1998）。

[从这里了解有关数据集的更多信息](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)。

下载数据集并将其放入当前工作目录，文件名为“ _shampoo-sales.csv_ ”。

时间系列中的时间戳不包含绝对年份组件。从 1900 年开始加载数据和基线时，我们可以使用自定义日期解析功能，如下所示：

```py
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
```

加载后，我们可以指定 _p_ ， _d_ 和 _q_ 值的位点进行搜索并将它们传递给 _evaluate_models（）_ 功能。

我们将尝试一组滞后值（ _p_ ）和一些差异迭代（ _d_ ）和残余误差滞后值（ _q_ ）。

```py
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

将这一切与上一节中定义的通用程序结合起来，我们可以在 Shampoo Sales 数据集中搜索 ARIMA 超参数。

完整的代码示例如下所示。

```py
import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

运行该示例将为每个成功完成的评估打印 ARIMA 参数和 MSE。

ARIMA（4,2,1）的最佳参数在运行结束时报告，均方误差为 4,694.873。

```py
ARIMA(0, 0, 0) MSE=52425.268
ARIMA(0, 0, 1) MSE=38145.167
ARIMA(0, 0, 2) MSE=23989.567
ARIMA(0, 1, 0) MSE=18003.173
ARIMA(0, 1, 1) MSE=9558.410
ARIMA(0, 2, 0) MSE=67339.808
ARIMA(0, 2, 1) MSE=18323.163
ARIMA(1, 0, 0) MSE=23112.958
ARIMA(1, 1, 0) MSE=7121.373
ARIMA(1, 1, 1) MSE=7003.683
ARIMA(1, 2, 0) MSE=18607.980
ARIMA(2, 1, 0) MSE=5689.932
ARIMA(2, 1, 1) MSE=7759.707
ARIMA(2, 2, 0) MSE=9860.948
ARIMA(4, 1, 0) MSE=6649.594
ARIMA(4, 1, 1) MSE=6796.279
ARIMA(4, 2, 0) MSE=7596.332
ARIMA(4, 2, 1) MSE=4694.873
ARIMA(6, 1, 0) MSE=6810.080
ARIMA(6, 2, 0) MSE=6261.107
ARIMA(8, 0, 0) MSE=7256.028
ARIMA(8, 1, 0) MSE=6579.403
Best ARIMA(4, 2, 1) MSE=4694.873
```

## 每日女性出生案例研究

每日女性出生数据集描述了 1959 年加利福尼亚州每日女性出生人数。

单位是计数，有 365 个观测值。数据集的来源归功于 Newton（1988）。

[在此处了解有关数据集的更多信息](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959)。

下载数据集并将其放在当前工作目录中，文件名为“ _daily-total-female-births.csv_ ”。

此数据集可以直接作为 Pandas 系列轻松加载。

```py
# load dataset
series = Series.from_csv('daily-total-female-births.csv', header=0)
```

为了简单起见，我们将探索与上一节中相同的 ARIMA 超参数网格。

```py
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

综上所述，我们可以在 Daily Female Births 数据集上搜索 ARIMA 参数。完整的代码清单如下。

```py
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
series = Series.from_csv('daily-total-female-births.csv', header=0)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

运行该示例将为成功评估的每个配置打印 ARIMA 参数和均方误差。

最佳平均参数报告为 ARIMA（6,1,0），均方误差为 53.187。

```py
ARIMA(0, 0, 0) MSE=67.063
ARIMA(0, 0, 1) MSE=62.165
ARIMA(0, 0, 2) MSE=60.386
ARIMA(0, 1, 0) MSE=84.038
ARIMA(0, 1, 1) MSE=56.653
ARIMA(0, 1, 2) MSE=55.272
ARIMA(0, 2, 0) MSE=246.414
ARIMA(0, 2, 1) MSE=84.659
ARIMA(1, 0, 0) MSE=60.876
ARIMA(1, 1, 0) MSE=65.928
ARIMA(1, 1, 1) MSE=55.129
ARIMA(1, 1, 2) MSE=55.197
ARIMA(1, 2, 0) MSE=143.755
ARIMA(2, 0, 0) MSE=59.251
ARIMA(2, 1, 0) MSE=59.487
ARIMA(2, 1, 1) MSE=55.013
ARIMA(2, 2, 0) MSE=107.600
ARIMA(4, 0, 0) MSE=59.189
ARIMA(4, 1, 0) MSE=57.428
ARIMA(4, 1, 1) MSE=55.862
ARIMA(4, 2, 0) MSE=80.207
ARIMA(6, 0, 0) MSE=58.773
ARIMA(6, 1, 0) MSE=53.187
ARIMA(6, 1, 1) MSE=57.055
ARIMA(6, 2, 0) MSE=69.753
ARIMA(8, 0, 0) MSE=56.984
ARIMA(8, 1, 0) MSE=57.290
ARIMA(8, 2, 0) MSE=66.034
ARIMA(8, 2, 1) MSE=57.884
ARIMA(10, 0, 0) MSE=57.470
ARIMA(10, 1, 0) MSE=57.359
ARIMA(10, 2, 0) MSE=65.503
ARIMA(10, 2, 1) MSE=57.878
ARIMA(10, 2, 2) MSE=58.309
Best ARIMA(6, 1, 0) MSE=53.187
```

## 扩展

本教程中使用的网格搜索方法很简单，可以轻松扩展。

本节列出了一些扩展您可能希望探索的方法的想法。

*   **种子网格**。 ACF 和 PACF 图的经典诊断工具仍然可以与用于搜索 ARIMA 参数网格的结果一起使用。
*   **替代措施**。搜索旨在优化样本外均方误差。这可以更改为另一个样本外统计数据，样本内统计数据，如 AIC 或 BIC，或两者的某种组合。您可以选择对项目最有意义的指标。
*   **剩余诊断**。可以自动计算残差预测误差的统计数据，以提供拟合质量的附加指示。例子包括残差分布是否为高斯分布以及残差中是否存在自相关的统计检验。
*   **更新模型**。 ARIMA 模型是从头开始为每个一步预测创建的。通过仔细检查 API，可以使用新的观察更新模型的内部数据，而不是从头开始重新创建。
*   **前提条件**。 ARIMA 模型可以对时间序列数据集做出假设，例如正态性和平稳性。可以检查这些，并在训练给定模型之前针对给定的数据集引发警告。

## 摘要

在本教程中，您了解了如何在 Python 中搜索 ARIMA 模型的超参数。

具体来说，你学到了：

*   可用于网格搜索 ARIMA 超参数以进行一步滚动预测的过程。
*   如何应用 ARIMA 超参数调整标准单变量时间序列数据集。
*   关于如何进一步改进 ARIMA 超参数网格搜索的思路。

现在轮到你了。

在您喜欢的时间序列数据集上尝试此过程。你得到了什么结果？
在下面的评论中报告您的结果。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。