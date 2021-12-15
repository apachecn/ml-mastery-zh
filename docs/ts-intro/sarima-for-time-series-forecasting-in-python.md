# 使用 Python 编写 SARIMA 时间序列预测

> 原文： [https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)

自回归综合移动平均线（ARIMA）是用于单变量时间序列数据预测的最广泛使用的预测方法之一。

尽管该方法可以处理具有趋势的数据，但它不支持具有季节性组件的时间序列。

支持对该系列季节性组成部分进行直接建模的 ARIMA 扩展称为 SARIMA。

在本教程中，您将发现季节性自回归集成移动平均线（SARIMA）时间序列预测方法，其中包含趋势和季节性的单变量数据。

完成本教程后，您将了解：

*   ARIMA 在季节性数据方面的局限性。
*   ARIMA 的 SARIMA 扩展，明确地模拟单变量数据中的季节性元素。
*   如何使用 Statsmodels 库在 Python 中实现 SARIMA 方法。

让我们开始吧。

*   **更新 Nov / 2018** ：有关使用和网格搜索 SARIMA 超参数的帮助，请参阅此帖子：
    *   [如何在 Python 中搜索用于时间序列预测的 SARIMA 模型超参数](https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/)

![A Gentle Introduction to SARIMA for Time Series Forecasting in Python](img/d319b4c623719496079d0f470af8ece8.jpg)

用于 Python 中时间序列预测的 SARIMA 的温和介绍
[Mario Micklisch](https://www.flickr.com/photos/fvfavo/15438357162/) 的照片，保留一些权利。

## 教程概述

本教程分为四个部分;他们是：

1.  ARIMA 有什么不对
2.  什么是 SARIMA？
3.  如何配置 SARIMA
4.  如何在 Python 中使用 SARIMA

## ARIMA 有什么不对

自回归综合移动平均线（ARIMA）是单变量时间序列数据的预测方法。

顾名思义，它支持自回归和移动平均元素。集成元素指的是差分，允许该方法支持具有趋势的时间序列数据。

ARIMA 的一个问题是它不支持季节性数据。这是一个重复循环的时间序列。

ARIMA 预计数据不是季节性的，或者季节性成分被删除，例如通过季节性差异等方法进行季节性调整。

有关 ARIMA 的更多信息，请参阅帖子：

*   [如何使用 Python 创建用于时间序列预测的 ARIMA 模型](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

另一种方法是使用 SARIMA。

## 什么是 SARIMA？

季节性自回归整合移动平均线，SARIMA 或季节性 ARIMA，是 ARIMA 的扩展，明确支持具有季节性成分的单变量时间序列数据。

它增加了三个新的超参数来指定系列季节性成分的自回归（AR），差分（I）和移动平均（MA），以及季节性周期的附加参数。

> 通过在 ARIMA 中包含额外的季节性术语来形成季节性 ARIMA 模型[...]模型的季节性部分由与模型的非季节性组成非常相似的术语组成，但它们涉及季节性时段的后移。

- 第 242 页，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。

## 如何配置 SARIMA

配置 SARIMA 需要为系列的趋势和季节性元素选择超参数。

### 趋势元素

有三个趋势元素需要配置。

它们与 ARIMA 模型相同;特别：

*   **p** ：趋势自动回归顺序。
*   **d** ：趋势差异顺序。
*   **q** ：趋势均线。

### 季节性元素

有四个不属于 ARIMA 的季节性元素必须配置;他们是：

*   **P** ：季节性自回归顺序。
*   **D** ：季节性差异顺序。
*   **Q** ：季节性移动平均线。
*   **m** ：单个季节性时段的时间步数。

同时，SARIMA 模型的表示法指定为：

```py
SARIMA(p,d,q)(P,D,Q)m
```

指定模型的特定选择超参数的位置;例如：

```py
SARIMA(3,1,0)(1,1,0)12
```

重要的是，`m`参数影响`P`，`D`和`Q`参数。例如，月度数据的 m 为 12 表示每年的季节性周期。

`P`= 1 将利用模型中的第一个季节性偏移观察，例如 t-（m * 1）或 t-12。`P`= 2，将使用最后两个季节性偏移的观测值 t-（m * 1），t-（m * 2）。

类似地，1 的`D`将计算一阶季节差异，并且`Q`= 1 将使用模型中的一阶误差（例如，移动平均值）。

> 季节性 ARIMA 模型使用等于季节数的滞后差异来消除加性季节效应。与滞后 1 差分去除趋势一样，滞后差分引入移动平均项。季节性 ARIMA 模型包括滞后 s 处的自回归和移动平均项。

- 第 142 页，[入门时间序列与 R](https://amzn.to/2smB9LR) ，2009 年。

可以通过仔细分析 ACF 和 PACF 图来选择趋势元素，其中查看最近时间步长（例如，1,2,3）的相关性。

类似地，可以通过查看季节性滞后时间步长的相关性来分析 ACF 和 PACF 图以指定季节模型的值。

有关解释 ACF / PACF 图的更多信息，请参阅帖子：

*   [自相关和部分自相关的温和介绍](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/)

> 季节性 ARIMA 模型可能具有大量参数和术语组合。因此，在拟合数据时尝试各种模型并使用适当的标准选择最佳拟合模型是合适的...

- 第 143-144 页，[介绍时间序列与 R](https://amzn.to/2smB9LR) ，2009 年。

或者，可以在趋势和季节性超参数中使用网格搜索。

有关网格搜索 ARIMA 参数的更多信息，请参阅帖子：

*   [如何使用 Python 网格搜索 ARIMA 模型超参数](https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)

## 如何在 Python 中使用 SARIMA

通过 [Statsmodels 库](http://www.statsmodels.org/dev/statespace.html)在 Python 中支持 S​​ARIMA 时间序列预测方法。

要使用 SARIMA，有三个步骤，它们是：

1.  定义模型。
2.  适合定义的模型。
3.  使用拟合模型进行预测。

让我们依次看一下每一步。

### 1.定义模型

可以通过提供训练数据和大量模型配置参数来创建 SARIMAX 类的实例。

```py
# specify training data
data = ...
# define model
model = SARIMAX(data, ...)
```

该实现称为 SARIMAX 而不是 SARIMA，因为方法名称中的“X”意味着该实现还支持外生变量。

这些是并行时间序列变量，它们不是通过 AR，I 或 MA 过程直接建模，而是作为模型的加权输入提供。

外源变量是可选的，可以通过“`exog`”参数指定。

```py
# specify training data
data = ...
# specify additional data
other_data = ...
# define model
model = SARIMAX(data, exog=other_data, ...)
```

趋势和季节性超参数分别指定为“_ 顺序 _”和“`seasonal_order`”参数的 3 和 4 元素元组。

必须指定这些元素。

```py
# specify training data
data = ...
# define model configuration
my_order = (1, 1, 1)
my_seasonal_order = (1, 1, 1, 12)
# define model
model = SARIMAX(data, order=my_order, seasonal_order=my_seasonal_order, ...)
```

这些是主要的配置元素。

您可能需要配置其他微调参数。在完整的 API 中了解更多信息：

*   [statsmodels.tsa.statespace.sarimax.SARIMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

### 2.适合模型

创建模型后，它可以适合训练数据。

通过调用 [fit（）函数](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html)来拟合模型。

拟合模型返回`SARIMAXResults`类的实例。此对象包含拟合的详细信息，例如数据和系数，以及可用于使用模型的函数。

```py
# specify training data
data = ...
# define model
model = SARIMAX(data, order=..., seasonal_order=...)
# fit model
model_fit = model.fit()
```

可以配置拟合过程的许多元素，一旦您熟悉实现，就值得阅读 API 以查看这些选项。

*   [statsmodels.tsa.statespace.sarimax.SARIMAX.fit API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html)

### 3.进行预测

适合后，该模型可用于进行预测。

可以通过在调用 fit 返回的`SARIMAXResults`对象上调用`forecast()`或`predict()`函数来进行预测。

[forecast（）函数](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.forecast.html)采用单个参数指定要预测的采样时间步数，或者如果未提供参数则采用一步预测。

```py
# specify training data
data = ...
# define model
model = SARIMAX(data, order=..., seasonal_order=...)
# fit model
model_fit = model.fit()
# one step forecast
yhat = model_fit.forecast()
```

`predict()`函数需要指定开始和结束日期或索引。

此外，如果在定义模型时提供了外生变量，则它们也必须在`predict()`函数的预测期内提供。

```py
# specify training data
data = ...
# define model
model = SARIMAX(data, order=..., seasonal_order=...)
# fit model
model_fit = model.fit()
# one step forecast
yhat = model_fit.predict(start=len(data), end=len(data))
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何在 Python 中搜索用于时间序列预测的 SARIMA 模型超参数](https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/)
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

### 用品

*   [维基百科上的自回归综合移动平均线](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

## 摘要

在本教程中，您发现了季节性自回归集成移动平均线（SARIMA），用于包含趋势和季节性的单变量数据进行时间序列预测。

具体来说，你学到了：

*   ARIMA 在季节性数据方面的局限性。
*   ARIMA 的 SARIMA 扩展，明确地模拟单变量数据中的季节性元素。
*   如何使用 Statsmodels 库在 Python 中实现 SARIMA 方法。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。