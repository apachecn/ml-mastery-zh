# Python 中用于时间序列预测的指数平滑的温和介绍

> 原文： [https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/)

指数平滑是单变量数据的时间序列预测方法，可以扩展为支持具有系统趋势或季节性成分的数据。

它是一种强大的预测方法，可用作流行的 Box-Jenkins ARIMA 系列方法的替代方法。

在本教程中，您将发现单变量时间序列预测的指数平滑方法。

完成本教程后，您将了解：

*   什么是指数平滑以及它与其他预测方法的不同之处。
*   指数平滑的三种主要类型以及如何配置它们。
*   如何在 Python 中实现指数平滑。

让我们开始吧。

![A Gentle Introduction to Exponential Smoothing for Time Series Forecasting in Python](img/4f9368483b6fcc2ae9314ffa0d6682e0.jpg)

Python 中时间序列预测的指数平滑的温和介绍
照片由 [Wolfgang Staudt](https://www.flickr.com/photos/wolfgangstaudt/2204054918/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  什么是指数平滑？
2.  指数平滑的类型
3.  如何配置指数平滑
4.  Python 中的指数平滑

## 什么是指数平滑？

指数平滑是单变量数据的时间序列预测方法。

像 Box-Jenkins ARIMA 系列方法这样的时间序列方法开发了一种模型，其中预测是近期过去观察或滞后的加权线性和。

指数平滑预测方法的类似之处在于预测是过去观察的加权和，但模型明确地使用指数减小的权重用于过去的观察。

具体而言，过去的观察以几何减小的比率加权。

> 使用指数平滑方法产生的预测是过去观测的加权平均值，随着观测结果的变化，权重呈指数衰减。换句话说，观察越近，相关重量越高。

- 第 171 页，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。

指数平滑方法可以被视为对等，并且是流行的 Box-Jenkins ARIMA 类时间序列预测方法的替代方法。

总的来说，这些方法有时被称为 ETS 模型，指的是错误，趋势和季节性的显式建模。

## 指数平滑的类型

指数平滑时间序列预测方法有三种主要类型。

一种假设没有系统结构的简单方法，明确处理趋势的扩展，以及增加季节性支持的最先进方法。

### 单指数平滑

单指数平滑，简称 SES，也称为简单指数平滑，是一种没有趋势或季节性的单变量数据的时间序列预测方法。

它需要一个名为`alpha`（`a`）的参数，也称为平滑因子或平滑系数。

该参数控制在先前时间步骤的观察结果的影响以指数方式衰减的速率。 Alpha 通常设置为介于 0 和 1 之间的值。大值意味着模型主要关注最近的过去观察，而较小的值意味着在做出预测时会考虑更多的历史记录。

> 接近 1 的值表示快速学习（即，只有最近的值影响预测），而接近 0 的值表示学习缓慢（过去的观察对预测有很大影响）。

- 第 89 页，[实用时间序列预测与 R](https://amzn.to/2LGKzKm) ，2016 年。

超参数：

*   **Alpha** ：级别的平滑因子。

### 双指数平滑

双指数平滑是指数平滑的扩展，明确增加了对单变量时间序列趋势的支持。

除了用于控制水平平滑因子的`alpha`参数外，还增加了一个额外的平滑因子来控制称为`beta`（_ 的趋势变化影响的衰减。 b_ ）。

该方法支持以不同方式变化的趋势：加法和乘法，取决于趋势分别是线性还是指数。

具有附加趋势的双指数平滑通常被称为 Holt 的线性趋势模型，以 Charles Holt 方法的开发者命名。

*   **附加趋势**：具有线性趋势的双指数平滑。
*   **乘法趋势**：具有指数趋势的双指数平滑。

对于更长距离（多步骤）的预测，这种趋势可能会持续不切实际。因此，随着时间的推移抑制趋势可能是有用的。

阻尼意味着将未来时间步长趋势的大小减小到一条直线（没有趋势）。

> Holt 线性方法产生的预测显示出未来的不变趋势（增加或减少）。更为极端的是指数趋势法产生的预测[...]受此观察的推动[...]引入了一个参数，在未来的某个时间“抑制”趋势为平线。

- 第 183 页，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。

与对趋势本身进行建模一样，我们可以使用相同的原理来抑制趋势，特别是对于线性或指数阻尼效应，可以相加或相乘。阻尼系数`Phi`（`p`）用于控制阻尼率。

*   **添加剂阻尼**：线性地抑制趋势。
*   **乘法阻尼**：以指数方式抑制趋势。

超参数：

*   **Alpha** ：级别的平滑因子。
*   **Beta** ：趋势的平滑因子。
*   **趋势类型**：加法或乘法。
*   **Dampen Type** ：加法或乘法。
*   **Phi** ：阻尼系数。

### 三次指数平滑

三次指数平滑是指数平滑的扩展，明确地增加了对单变量时间序列的季节性支持。

这种方法有时被称为 Holt-Winters Exponential Smoothing，以该方法的两个贡献者命名：Charles Holt 和 Peter Winters。

除了α和β平滑因子之外，还添加了一个新参数，称为`gamma`（`g`），它控制对季节性成分的影响。

与趋势一样，季节性可以被建模为季节性线性或指数变化的加法或乘法过程。

*   **添加季节性**：具有线性季节性的三重指数平滑。
*   **乘法季节性**：具有指数季节性的三次指数平滑。

三指数平滑是指数平滑的最高级变化，通过配置，它还可以开发双指数和单指数平滑模型。

> 作为一种自适应方法，Holt-Winter 的指数平滑允许水平，趋势和季节性模式随时间变化。

- 第 95 页，[实用时间序列预测与 R](https://amzn.to/2LGKzKm) ，2016 年。

此外，为确保正确建模季节性，必须指定季节性时间段（_ 期间 _）中的时间步数。例如，如果系列是每月数据并且每年重复季节性时段，则期间= 12。

超参数：

*   **Alpha** ：级别的平滑因子。
*   **Beta** ：趋势的平滑因子。
*   **Gamma** ：季节性的平滑因子。
*   **趋势类型**：加法或乘法。
*   **Dampen Type** ：加法或乘法。
*   **Phi** ：阻尼系数。
*   **季节性类型**：加法或乘法。
*   **期间**：季节性时间段的时间步长。

## 如何配置指数平滑

可以明确指定所有模型超参数。

对于专家和初学者来说，这可能具有挑战性。

相反，通常使用数值优化来搜索和资助平滑系数（`alpha`，`beta`，`gamma`和`phi`）对于导致最低错误的模型。

> [...]更加稳健和客观的方法来获得任何指数平滑方法中包含的未知参数的值是从观测数据中估计它们。 [...]未知参数和任何指数平滑方法的初始值可以通过最小化 SSE [平方误差之和]来估算。

- 第 177 页，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。

必须明确指定指定趋势和季节性变化类型的参数，例如它们是加性或乘法的天气以及它们是否应该被抑制。

## Python 中的指数平滑

本节介绍如何在 Python 中实现指数平滑。

Statsmodels Python 库中提供了 Python 中的指数平滑的实现。

这些实现基于 Rob Hyndman 和 George Athanasopoulos 的优秀书籍[预测：原理与实践](https://amzn.to/2xlJsfV)，2013 及其在“[预测](https://cran.r-project.org/web/packages/forecast/index.html)”包中的 R 实现方法的描述） 。

### 单指数平滑

可以通过 SimpleExpSmoothing Statsmodels 类在 Python 中实现单指数平滑或简单平滑。

首先，必须实例化`SimpleExpSmoothing`类的实例并传递训练数据。然后调用`fit()`函数提供拟合配置，特别是称为`smoothing_level`的`alpha`值。如果没有提供或设置为 _ 无 _，模型将自动优化该值。

此`fit()`函数返回包含学习系数的`HoltWintersResults`类的实例。可以调用结果对象上的`forecast()`或`predict()`函数做出预测。

例如：

```py
# single exponential smoothing
...
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# prepare data
data = ...
# create class
model = SimpleExpSmoothing(data)
# fit model
model_fit = model.fit(...)
# make prediction
yhat = model_fit.predict(...)
```

### 双指数和三指数平滑

可以使用 ExponentialSmoothing Statsmodels 类在 Python 中实现单指数，双指数和三指数平滑。

首先，必须实例化 ExponentialSmoothing 类的实例，同时指定训练数据和模型的某些配置。

具体而言，您必须指定以下配置参数：

*   **趋势**：趋势分量的类型，作为加法的“_ 加 _”或乘法的“`mul`”。可以通过将趋势设置为“无”来禁用对趋势建模。
*   **阻尼**：是否应该抑制趋势分量，`True`或`False`。
*   **季节性**：季节性成分的类型，为“_ 添加 _”为添加剂或“`mul`”为乘法。可以通过将季节性组件设置为“无”来禁用它。
*   **seasonal_periods** ：季节性时间段内的时间步数，例如每年季节性结构 12 个月 12 个月（[更多](https://robjhyndman.com/hyndsight/seasonal-periods/)）。

然后通过调用`fit()`函数将模型拟合到训练数据上。

此功能允许您指定指数平滑模型的平滑系数或对其进行优化。默认情况下，它们已经过优化（例如 _ 优化= True_ ）。这些系数包括：

*   **smoothing_level** （`alpha`）：该级别的平滑系数。
*   **smoothing_slope** （`beta`）：趋势的平滑系数。
*   **smoothing_seasonal** （`gamma`）：季节性成分的平滑系数。
*   **damping_slope** （`phi`）：阻尼趋势的系数。

另外，拟合函数可以在建模之前执行基本数据准备;特别：

*   **use_boxcox** ：是否执行系列的幂变换（True / False）或指定变换的 lambda。

`fit()`函数将返回包含学习系数的`HoltWintersResults`类的实例。可以调用结果对象上的`forecast()`或`predict()`函数做出预测。

```py
# double or triple exponential smoothing
...
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# prepare data
data = ...
# create class
model = ExponentialSmoothing(data, ...)
# fit model
model_fit = model.fit(...)
# make prediction
yhat = model_fit.predict(...)
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 7 章指数平滑，[预测：原则和实践](https://amzn.to/2xlJsfV)，2013。
*   第 6.4 节。时间序列分析简介，[工程统计手册](https://www.itl.nist.gov/div898/handbook/)，2012。
*   [实际时间序列预测与 R](https://amzn.to/2LGKzKm) ，2016 年。

### API

*   [Statsmodels 时间序列分析](http://www.statsmodels.org/dev/tsa.html)
*   [statsmodels.tsa.holtwinters.SimpleExpSmoothing API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html)
*   [statsmodels.tsa.holtwinters.ExponentialSmoothing API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
*   [statsmodels.tsa.holtwinters.HoltWintersResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html)
*   [预测：时间序列和线性模型 R 包](https://cran.r-project.org/web/packages/forecast/index.html)的预测函数

### 用品

*   [维基百科上的指数平滑](https://en.wikipedia.org/wiki/Exponential_smoothing)

### 摘要

在本教程中，您发现了单变量时间序列预测的指数平滑方法。

具体来说，你学到了：

*   什么是指数平滑以及它与其他预测方法的不同之处。
*   指数平滑的三种主要类型以及如何配置它们。
*   如何在 Python 中实现指数平滑。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。