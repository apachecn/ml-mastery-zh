# 如何使用 Python 中的 ARIMA 进行样本外预测

> 原文： [https://machinelearningmastery.com/make-sample-forecasts-arima-python/](https://machinelearningmastery.com/make-sample-forecasts-arima-python/)

在开始使用时间序列数据时，进行样本外预测可能会令人困惑。

statsmodels Python API 提供了执行一步和多步样本外预测的功能。

在本教程中，您将清除在 Python 中使用时间序列数据进行样本外预测时的任何混淆。

完成本教程后，您将了解：

*   如何进行一步到位的样本预测。
*   如何进行多步骤样本外预测。
*  `forecast()`和 _ 之间的差异预测（）_ 的功能。

让我们开始吧。

![How to Make Out-of-Sample Forecasts with ARIMA in Python](img/24eec985fb655f0059ab42526b83f673.jpg)

如何使用 Python 中的 ARIMA 进行样本外预测
照片来自 [dziambel](https://www.flickr.com/photos/141999355@N08/32706989645/) ，保留一些权利。

## 教程概述

本教程分为以下 5 个步骤：

1.  数据集描述
2.  拆分数据集
3.  开发模型
4.  一步到位的样本外预测
5.  多步样本外预测

## 1.最低每日温度数据集

该数据集描述了澳大利亚墨尔本市 10 年（1981-1990）的最低日常温度。

单位为摄氏度，有 3,650 个观测值。数据来源被称为澳大利亚气象局。

[了解有关数据市场](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990)上数据集的更多信息。

使用文件名“ _daily-minimum-Temperats.sv”_ 将最低每日温度数据集下载到当前工作目录。

**注意**：下载的文件包含一些问号（“？”）字符，必须先将其删除才能使用数据集。在文本编辑器中打开文件并删除“？”字符。此外，删除文件中的任何页脚信息。

下面的示例将数据集加载为 Pandas 系列。

```py
# line plot of time series
from pandas import Series
from matplotlib import pyplot
# load dataset
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# display first few rows
print(series.head(20))
# line plot of dataset
series.plot()
pyplot.show()
```

运行该示例将打印已加载数据集的前 20 行。

```py
Date
1981-01-01    20.7
1981-01-02    17.9
1981-01-03    18.8
1981-01-04    14.6
1981-01-05    15.8
1981-01-06    15.8
1981-01-07    15.8
1981-01-08    17.4
1981-01-09    21.8
1981-01-10    20.0
1981-01-11    16.2
1981-01-12    13.3
1981-01-13    16.7
1981-01-14    21.5
1981-01-15    25.0
1981-01-16    20.7
1981-01-17    20.6
1981-01-18    24.8
1981-01-19    17.7
1981-01-20    15.5
```

还创建了时间序列的线图。

![Minimum Daily Temperatures Dataset Line Plot](img/469ab86fb892b6aff829fa6ad5b9c66b.jpg)

最低每日温度数据集线图

## 2.拆分数据集

我们可以将数据集拆分为两部分。

第一部分是我们将用于准备 ARIMA 模型的训练数据集。第二部分是我们假装不可用的测试数据集。正是这些时间步骤我们将在样本之外进行处理。

该数据集包含 1981 年 1 月 1 日至 1990 年 12 月 31 日期间的数据。

我们将从 1990 年 12 月开始将数据集的最后 7 天作为测试数据集，并将这些时间步骤视为样本之外的步骤。

具体是 1990-12-25 到 1990-12-31：

```py
1990-12-25,12.9
1990-12-26,14.6
1990-12-27,14.0
1990-12-28,13.6
1990-12-29,13.5
1990-12-30,15.7
1990-12-31,13.0
```

下面的代码将加载数据集，将其拆分为训练和验证数据集，并将它们分别保存到文件 _dataset.csv_ 和 _validation.csv_ 。

```py
# split the dataset
from pandas import Series
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')
```

运行示例，您现在应该有两个文件可以使用。

dataset.csv 中的最后一个观察是 1990 年平安夜：

```py
1990-12-24,10.0
```

这意味着 1990 年圣诞节及以后是在 _dataset.csv_ 上训练的模型的样本外时间步长。

## 3.开发模型

在本节中，我们将使数据保持静止并开发一个简单的 ARIMA 模型。

数据具有强大的季节性成分。我们可以通过考虑季节性差异来中和这一点并使数据保持不变。也就是说，我们可以观察一天，并从一年前的同一天减去观察结果。

这将产生一个固定的数据集，我们可以从中拟合模型。

```py
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
```

我们可以通过在一年前添加观察值来反转此操作。我们需要对经过季节性调整的数据训练的模型进行预测。

```py
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

我们可以使用 ARIMA 模型。

将强大的 ARIMA 模型拟合到数据不是本文的重点，因此我将选择一个简单的 ARIMA（7,0,7），而不是通过问题分析或[网格搜索参数](http://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)。组态。

我们可以将所有这些放在一起如下：

```py
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = Series.from_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())
```

运行该示例加载数据集，获取季节差异，然后拟合 ARIMA（7,0,7）模型并打印拟合模型的摘要。

```py
                              ARMA Model Results
==============================================================================
Dep. Variable:                      y   No. Observations:                 3278
Model:                     ARMA(7, 1)   Log Likelihood               -8673.748
Method:                       css-mle   S.D. of innovations              3.411
Date:                Mon, 20 Feb 2017   AIC                          17367.497
Time:                        10:28:38   BIC                          17428.447
Sample:                             0   HQIC                         17389.322

==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0132      0.132      0.100      0.921      -0.246       0.273
ar.L1.y        1.1424      0.287      3.976      0.000       0.579       1.706
ar.L2.y       -0.4346      0.154     -2.829      0.005      -0.736      -0.133
ar.L3.y        0.0961      0.042      2.289      0.022       0.014       0.178
ar.L4.y        0.0125      0.029      0.434      0.664      -0.044       0.069
ar.L5.y       -0.0101      0.029     -0.343      0.732      -0.068       0.047
ar.L6.y        0.0119      0.027      0.448      0.654      -0.040       0.064
ar.L7.y        0.0089      0.024      0.368      0.713      -0.038       0.056
ma.L1.y       -0.6157      0.287     -2.146      0.032      -1.178      -0.053
                                    Roots
=============================================================================
                 Real           Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1            1.2234           -0.0000j            1.2234           -0.0000
AR.2            1.2561           -1.0676j            1.6485           -0.1121
AR.3            1.2561           +1.0676j            1.6485            0.1121
AR.4            0.0349           -2.0160j            2.0163           -0.2472
AR.5            0.0349           +2.0160j            2.0163            0.2472
AR.6           -2.5770           -1.3110j            2.8913           -0.4251
AR.7           -2.5770           +1.3110j            2.8913            0.4251
MA.1            1.6242           +0.0000j            1.6242            0.0000
-----------------------------------------------------------------------------
```

我们现在准备探索使用该模型进行样本外预测。

## 4.一步到位的样本外预测

ARIMA 模型非常适合一步预测。

一步预测是从用于拟合模型的可用数据预测序列中的下一个时间步。

在这种情况下，我们对 1990 年圣诞节的一步预测感兴趣：

```py
1990-12-25
```

### 预测功能

statsmodel ARIMAResults 对象提供 [_forecast（）_ 函数](http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.forecast.html)用于进行预测。

默认情况下，此功能可以进行单步样本预测。因此，我们可以直接调用它并进行预测。`forecast()`函数的结果是包含预测值，预测的标准误差和置信区间信息的数组。现在，我们只关注此预测的第一个元素，如下所示。

```py
# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
```

一旦完成，我们可以反转季节性差异并将值转换回原始比例。

```py
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
```

完整示例如下：

```py
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
series = Series.from_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)
```

运行该示例打印 14.8 度，接近 _validation.csv_ 文件中预期的 12.9 度。

```py
Forecast: 14.861669
```

### 预测功能

statsmodel ARIMAResults 对象还提供 [_predict（）_ 函数](http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html)进行预测。

预测函数可用于预测任意样本内和样本外时间步骤，包括下一个样本外预测时间步骤。

预测函数需要指定开始和结束，这些可以是相对于用于拟合模型的训练数据的开始的时间步长的索引，例如：

```py
# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)
```

start 和 end 也可以是日期时间字符串或“日期时间”类型;例如：

```py
start_index = '1990-12-25'
end_index = '1990-12-25'
forecast = model_fit.predict(start=start_index, end=end_index)
```

和

```py
from pandas import datetime
start_index = datetime(1990, 12, 25)
end_index = datetime(1990, 12, 26)
forecast = model_fit.predict(start=start_index, end=end_index)
```

使用时间步长索引以外的任何内容都会导致系统出错，如下所示：

```py
AttributeError: 'NoneType' object has no attribute 'get_loc'
```

也许你会有更多的运气;现在，我坚持时间步骤索引。

完整示例如下：

```py
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy
from pandas import datetime

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
series = Series.from_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)
```

使用`forecast()`函数时，运行该示例将打印与上述相同的预测。

```py
Forecast: 14.861669
```

您可以看到预测功能更灵活。您可以指定样本内或样本外的任何点或连续预测区间。

现在我们知道如何进行一步预测，现在我们可以进行一些多步预测。

## 5.多步骤样本外预测

我们还可以使用`forecast()`和`predict()`函数进行多步预测。

天气数据通常会进行一周（7 天）预测，因此在本节中我们将研究预测接下来的 7 个样本外时间步的最低日常温度。

### 预测功能

`forecast()`函数有一个名为 _ 步骤 _ 的参数，允许您指定预测的时间步数。

默认情况下，对于一步式样本外预测，此参数设置为 1。我们可以将其设置为 7 以获得接下来 7 天的预测。

```py
# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=7)[0]
```

然后，我们可以反转每个预测的时间步骤，一次一个并打印值。请注意，要将 t + 2 的预测值反转，我们需要 t + 1 的反转预测值。在这里，我们将它们添加到名为 history 的列表的末尾，以便在调用`inverse_difference()`时使用。

```py
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1
```

完整示例如下：

```py
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
series = Series.from_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=7)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1
```

运行该示例将打印接下来 7 天的预测。

```py
Day 1: 14.861669
Day 2: 15.628784
Day 3: 13.331349
Day 4: 11.722413
Day 5: 10.421523
Day 6: 14.415549
Day 7: 12.674711
```

### 预测功能

`predict()`函数还可以预测接下来的 7 个样本外时间步长。

使用时间步长索引，我们可以将结束索引指定为未来 6 个以上的时间步长;例如：

```py
# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
```

下面列出了完整的示例。

```py
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
series = Series.from_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1
```

运行该示例会产生与上一节中调用`forecast()`函数相同的结果，如您所料。

```py
Day 1: 14.861669
Day 2: 15.628784
Day 3: 13.331349
Day 4: 11.722413
Day 5: 10.421523
Day 6: 14.415549
Day 7: 12.674711
```

## 摘要

在本教程中，您了解了如何使用 statsmodel 在 Python 中进行样本外预测。

具体来说，你学到了：

*   如何进行一步到位的样本预测。
*   如何进行为期 7 天的多步样本预测。
*   如何在预测时使用`forecast()`和`predict()`函数。

您对样本外预测或此帖有任何疑问吗？在评论中提出您的问题，我会尽力回答。