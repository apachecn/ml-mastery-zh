# 如何使用 Python 对 ARIMA 模型进行手动预测

> 原文： [https://machinelearningmastery.com/make-manual-predictions-arima-models-python/](https://machinelearningmastery.com/make-manual-predictions-arima-models-python/)

自动回归集成移动平均模型或 ARIMA 模型对初学者来说似乎有点吓人。

拉回方法的一个好方法是使用训练有素的模型手动进行预测。这表明 ARIMA 是一个核心的线性回归模型。

使用合适的 ARIMA 模型进行手动预测也可能是项目中的一项要求，这意味着您可以从拟合模型中保存系数，并将它们用作您自己的代码中的配置来进行预测，而无需在生产中使用繁重的 Python 库环境。

在本教程中，您将了解如何使用 Python 中经过训练的 ARIMA 模型进行手动预测。

具体来说，您将学到：

*   如何使用自回归模型进行手动预测。
*   如何使用移动平均模型进行手动预测。
*   如何使用自回归集成移动平均模型进行预测。

让我们潜入。

**更新**：您可能会发现此帖子很有用：

*   [如何使用 Python 中的 ARIMA 进行样本外预测](http://machinelearningmastery.com/make-sample-forecasts-arima-python/)

![How to Make Manual Predictions for ARIMA Models with Python](img/0c095bc3cfe804762a1a51a70c670857.jpg)

如何使用 Python
照片制作 ARIMA 模型的手动预测照片由 [Bernard Spragg 撰写。 NZ](https://www.flickr.com/photos/volvob12b/10133164586/) ，保留一些权利。

## 最低每日温度数据集

该数据集描述了澳大利亚墨尔本市 10 年（1981-1990）的最低日常温度。

单位为摄氏度，有 3,650 个观测值。数据来源被称为澳大利亚气象局。

[您可以从数据市场网站](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990)了解更多信息并下载数据集。

下载数据集并将其放入当前工作目录，文件名为“ _daily-minimum-Temperats.sv_ ”。

**注意**：下载的文件包含一些问号（“？”）字符，必须先将其删除才能使用数据集。在文本编辑器中打开文件并删除“？”字符。同时删除文件中的任何页脚信息。

下面的示例演示了如何将数据集作为 Pandas Series 加载并绘制已加载的数据集。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
series.plot()
pyplot.show()
```

运行该示例会创建时间序列的折线图。

![Minimum Daily Temperatures Dataset Plot](img/9e4be42c288e7703e57cd913fd35b795.jpg)

最低每日温度数据集图

## ARIMA 测试设置

我们将使用一致的测试工具来拟合 ARIMA 模型并评估其预测。

首先，将加载的数据集拆分为训练和测试数据集。大部分数据集用于拟合模型，最后 7 个观察值（一周）作为测试数据集保留以评估拟合模型。

前瞻性验证或滚动预测方法使用如下：

1.  迭代测试数据集中的每个时间步。
2.  在每次迭代中，对所有可用的历史数据训练新的 ARIMA 模型。
3.  该模型用于对第二天进行预测。
4.  存储预测并从测试集中检索“真实”观察并将其添加到历史中以供在下一次迭代中使用。
5.  最后通过计算与测试数据集中的预期值相比所做的所有预测的均方根误差（RMSE）来总结模型的表现。

开发了简单的 AR，MA，ARMA 和 ARMA 模型。它们未经优化，用于演示目的。通过一些调整，您一定能够获得更好的表现。

使用来自 statsmodels Python 库的 ARIMA 实现，并且从拟合模型返回的 ARIMAResults 对象中提取 AR 和 MA 系数。

ARIMA 模型通过 [predict（）](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html)和 [forecast（）](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.forecast.html)函数支持预测。

不过，我们将使用学习系数在本教程中进行手动预测。

这很有用，因为它表明训练的 ARIMA 模型所需的全部是系数。

ARIMA 模型的 statsmodels 实现中的系数不使用截距项。这意味着我们可以通过获取学习系数和滞后值（在 AR 模型的情况下）和滞后残差（在 MA 模型的情况下）的点积来计算输出值。例如：

```py
y = dot_product(ar_coefficients, lags) + dot_product(ma_coefficients, residuals)
```

可以从 aARIMAResults 对象访问学习 ARIMA 模型的系数，如下所示：

*   **AR 系数**：model_fit.arparams
*   **MA 系数**：model_fit.maparams

我们可以使用这些检索到的系数来使用以下手册 _predict（）_ 函数进行预测。

```py
def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat
```

作为参考，您可能会发现以下资源非常有用：

*   [ARIMA API 文档](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARIMA.html)
*   [ARIMAResults API 文档](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.html)
*   [ARIMA statsmodels 源代码](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py)

让我们看一些简单但具体的模型，以及如何使用此测试设置进行手动预测。

## 自回归模型

自回归模型或 AR 是滞后观察的线性回归模型。

具有滞后`k`的 AR 模型可以在 ARIMA 模型中指定如下：

```py
model = ARIMA(history, order=(k,0,0))
```

在此示例中，我们将使用简单的 AR（1）进行演示。

进行预测需要我们从拟合模型中检索 AR 系数，并将它们与观测值的滞后一起使用，并调用上面定义的自定义 _predict（）_ 函数。

下面列出了完整的示例。

```py
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,0))
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef = model_fit.arparams
	yhat = predict(ar_coef, history)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

请注意，ARIMA 实施将自动为时间序列中的趋势建模。这为回归方程式增加了一个常数，我们不需要用于演示目的。我们通过将 _fit（）_ 函数中的'trend'参数设置为'`nc`'为' _no constant_ '来关闭这个便利性。

[fit（）](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.arima_model.ARIMA.fit.html)函数还输出了许多详细消息，我们可以通过将'`disp`'参数设置为'`False`'来关闭。

运行该示例将每次迭代打印预测值和期望值 7 天。印刷的最终 RMSE 显示该简单模型的平均误差约为 1.9 摄氏度。

```py
>predicted=9.738, expected=12.900
>predicted=12.563, expected=14.600
>predicted=14.219, expected=14.000
>predicted=13.635, expected=13.600
>predicted=13.245, expected=13.500
>predicted=13.148, expected=15.700
>predicted=15.292, expected=13.000
Test RMSE: 1.928
```

尝试使用不同顺序的 AR 模型，例如 2 个或更多。

## 移动平均模型

移动平均模型或 MA 是滞后残差的线性回归模型。

可以在 ARIMA 模型中指定滞后为 k 的 MA 模型，如下所示：

```py
model = ARIMA(history, order=(0,0,k))
```

在这个例子中，我们将使用简单的 MA（1）进行演示。

如上所述，进行预测需要我们从拟合模型中检索 MA 系数，并将它们与剩余误差值的滞后一起使用，并调用上面定义的自定义 _predict（）_ 函数。

训练期间的残留误差存储在 ARIMA 模型中`ARIMAResults`对象的'`resid`'参数下。

```py
model_fit.resid
```

下面列出了完整的示例。

```py
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,0,1))
	model_fit = model.fit(trend='nc', disp=False)
	ma_coef = model_fit.maparams
	resid = model_fit.resid
	yhat = predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

运行该示例将每次迭代打印预测值和期望值 7 天，并通过汇总所有预测的 RMSE 结束。

该模型的技能并不是很好，您可以将其作为一个机会，与其他订单一起探索 MA 模型，并使用它们进行手动预测。

```py
>predicted=4.610, expected=12.900
>predicted=7.085, expected=14.600
>predicted=6.423, expected=14.000
>predicted=6.476, expected=13.600
>predicted=6.089, expected=13.500
>predicted=6.335, expected=15.700
>predicted=8.006, expected=13.000
Test RMSE: 7.568
```

您可以看到，当新的观察结果可用时，在`ARIMAResults`对象之外手动跟踪残留误差是多么简单。例如：

```py
residuals = list()
...
error = expected - predicted
residuals.append(error)
```

接下来，让我们将 AR 和 MA 模型放在一起，看看我们如何进行手动预测。

## 自回归移动平均模型

我们现在已经看到了如何为适合的 AR 和 MA 模型进行手动预测。

这些方法可以直接放在一起，以便为更全面的 ARMA 模型进行手动预测。

在这个例子中，我们将拟合 ARMA（1,1）模型，该模型可以在 ARIMA 模型中配置为没有差分的 ARIMA（1,0,1）。

下面列出了完整的示例。

```py
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	yhat = predict(ar_coef, history) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

您可以看到预测（`yhat`）是 AR 系数和滞后观察的点积与 MA 系数和滞后残差之和。

```py
yhat = predict(ar_coef, history) + predict(ma_coef, resid)
```

同样，运行该示例会打印每次迭代的预测值和期望值以及所有预测的汇总 RMSE。

```py
>predicted=11.920, expected=12.900
>predicted=12.309, expected=14.600
>predicted=13.293, expected=14.000
>predicted=13.549, expected=13.600
>predicted=13.504, expected=13.500
>predicted=13.434, expected=15.700
>predicted=14.401, expected=13.000
Test RMSE: 1.405
```

我们现在可以添加差异并显示如何对完整的 ARIMA 模型进行预测。

## 自回归综合移动平均模型

ARIMA 中的 I 代表积分，指的是在线性回归模型中进行预测之前对时间序列观察进行的差分。

在进行手动预测时，我们必须在调用 _predict（）_ 函数之前执行数据集的这种差分。下面是一个实现整个数据集差异的函数。

```py
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
```

简化是在最老的所需滞后值处跟踪观察，并根据需要使用该值来预测预测之前的差异系列。

对于 ARIMA 模型所需的每个差异，可以调用该差异函数一次。

在这个例子中，我们将使用 1 的差异级别，并将其与上一节中的 ARMA 示例相结合，为我们提供 ARIMA（1,1,1）模型。

下面列出了完整的示例。

```py
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	diff = difference(history)
	yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

您可以看到，在使用 AR 系数调用 _predict（）_ 函数之前，滞后观察值是不同的。还将关于这些差异输入值计算残差。

运行该示例会在每次迭代时打印预测值和期望值，并总结所有预测的表现。

```py
>predicted=11.837, expected=12.900
>predicted=13.265, expected=14.600
>predicted=14.159, expected=14.000
>predicted=13.868, expected=13.600
>predicted=13.662, expected=13.500
>predicted=13.603, expected=15.700
>predicted=14.788, expected=13.000
Test RMSE: 1.232
```

## 摘要

在本教程中，您了解了如何使用 Python 对 ARIMA 模型进行手动预测。

具体来说，你学到了：

*   如何对 AR 模型进行手动预测。
*   如何对 MA 模型进行手动预测。
*   如何对 ARMA 和 ARIMA 模型进行手动预测。

您对手动预测有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。