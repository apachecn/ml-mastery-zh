# 11 Python 中的经典时间序列预测方法（备忘单）

> 原文： [https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)

机器学习方法可用于分类和预测时间序列问题。

在探索时间序列的机器学习方法之前，最好确保使用经典的线性时间序列预测方法。经典时间序列预测方法可能侧重于线性关系，然而，它们是复杂的并且在广泛的问题上表现良好，假设您的数据已经适当准备并且方法配置良好。

在这篇文章中，您将发现一套用于时间序列预测的经典方法，您可以在探索机器学习方法之前测试您的预测问题。

该帖子被构造为一个备忘单，为您提供有关每个方法的足够信息，以便开始使用工作代码示例以及在何处查看有关该方法的更多信息。

所有代码示例都在 Python 中并使用 Statsmodels 库。这个库的 API 对于初学者来说可能很棘手（相信我！），因此以一个工作代码示例作为起点将大大加快您的进度。

这是一个很大的帖子;你可能想要为它添加书签。

让我们开始吧。

![11 Classical Time Series Forecasting Methods in Python (Cheat Sheet)](img/d362f443268654f3bd0585e35541819c.jpg)

11 Python 中的经典时间序列预测方法（备忘单）
[Ron Reiring](https://www.flickr.com/photos/84263554@N00/33076725455/) 的照片，保留一些权利。

## 概观

该备忘单演示了 11 种不同的经典时间序列预测方法;他们是：

1.  自回归（AR）
2.  移动平均线（MA）
3.  自回归移动平均线（ARMA）
4.  自回归综合移动平均线（ARIMA）
5.  季节性自回归整合移动平均线（SARIMA）
6.  具有外生回归量的季节性自回归整合移动平均线（SARIMAX）
7.  向量自回归（VAR）
8.  向量自回归移动平均值（VARMA）
9.  具有外源回归量的向量自回归移动平均值（VARMAX）
10.  简单指数平滑（SES）
11.  霍尔特·温特的指数平滑（HWES）

我错过了您最喜欢的古典时间序列预测方法吗？
请在下面的评论中告诉我。

每种方法都以一致的方式呈现。

这包括：

*   **说明**。对该技术的简短而精确的描述。
*   **Python 代码**。一个简短的工作示例，用于拟合模型并在 Python 中做出预测。
*   **更多信息**。 API 和算法的参考。

每个代码示例都在一个简单的人为数据集上进行演示，该数据集可能适合或不适合该方法。用您的数据替换人为的数据集以测试方法。

请记住：每种方法都需要调整您的具体问题。在很多情况下，我已经有了如何在博客上配置甚至网格搜索参数的示例，请尝试搜索功能。

如果您发现此备忘单有用，请在下面的评论中告诉我。

## 自回归（AR）

自回归（AR）方法将序列中的下一步建模为先前时间步骤的观察的线性函数。

该模型的符号涉及指定模型 p 的顺序作为 AR 函数的参数，例如， AR（P）。例如，AR（1）是一阶自回归模型。

该方法适用于没有趋势和季节性成分的单变量时间序列。

### Python 代码

```py
# AR example
from statsmodels.tsa.ar_model import AR
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.ar_model.AR API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AR.html)
*   [statsmodels.tsa.ar_model.ARResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.ARResults.html)
*   [维基百科上的自回归模型](https://en.wikipedia.org/wiki/Autoregressive_model)

## 移动平均线（MA）

移动平均（MA）方法将序列中的下一步建模为来自先前时间步骤的平均过程的残余误差的线性函数。

移动平均模型与计算时间序列的移动平均值不同。

该模型的表示法涉及将模型 q 的顺序指定为 MA 函数的参数，例如， MA（Q）。例如，MA（1）是一阶移动平均模型。

该方法适用于没有趋势和季节性成分的单变量时间序列。

### Python 代码

我们可以使用 ARMA 类创建 MA 模型并设置零阶 AR 模型。我们必须在 order 参数中指定 MA 模型的顺序。

```py
# MA example
from statsmodels.tsa.arima_model import ARMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARMA(data, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.arima_model.ARMA API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMA.html)
*   [statsmodels.tsa.arima_model.ARMAResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMAResults.html)
*   [维基百科上的移动平均模型](https://en.wikipedia.org/wiki/Moving-average_model)

## 自回归移动平均线（ARMA）

自回归移动平均（ARMA）方法将序列中的下一步建模为先前时间步骤的观测和再造误差的线性函数。

它结合了自回归（AR）和移动平均（MA）模型。

该模型的表示法涉及将 AR（p）和 MA（q）模型的顺序指定为 ARMA 函数的参数，例如，ARMA 函数的参数。 ARMA（p，q）。 ARIMA 模型可用于开发 AR 或 MA 模型。

该方法适用于没有趋势和季节性成分的单变量时间序列。

### Python 代码

```py
# ARMA example
from statsmodels.tsa.arima_model import ARMA
from random import random
# contrived dataset
data = [random() for x in range(1, 100)]
# fit model
model = ARMA(data, order=(2, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.arima_model.ARMA API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMA.html)
*   [statsmodels.tsa.arima_model.ARMAResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMAResults.html)
*   [维基百科上的自回归移动平均模型](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)

## 自回归综合移动平均线（ARIMA）

自回归整合移动平均（ARIMA）方法将序列中的下一步建模为先前时间步长的差异观测值和残差误差的线性函数。

它结合了自回归（AR）和移动平均（MA）模型以及序列的差分预处理步骤，使序列静止，称为积分（I）。

该模型的表示法涉及将 AR（p），I（d）和 MA（q）模型的顺序指定为 ARIMA 函数的参数，例如 ARIMA 函数的参数。 ARIMA（p，d，q）。 ARIMA 模型也可用于开发 AR，MA 和 ARMA 模型。

该方法适用于具有趋势且没有季节性成分的单变量时间序列。

### Python 代码

```py
# ARIMA example
from statsmodels.tsa.arima_model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.arima_model.ARIMA API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.html#statsmodels.tsa.arima_model.ARIMA)
*   [statsmodels.tsa.arima_model.ARIMAResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMAResults.html)
*   [维基百科上的自回归综合移动平均线](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

## 季节性自回归整合移动平均线（SARIMA）

季节性自回归综合移动平均线（SARIMA）方法将序列中的下一步建模为差异观测值，误差，差异季节观测值和先前时间步长的季节误差的线性函数。

它结合了 ARIMA 模型，能够在季节性水平上执行相同的自回归，差分和移动平均建模。

该模型的表示法涉及指定 AR（p），I（d）和 MA（q）模型的顺序作为 ARIMA 函数和 AR（P），I（D），MA（Q）和 m 的参数。季节性参数，例如 SARIMA（p，d，q）（P，D，Q）m 其中“m”是每个季节（季节性时期）的时间步数。 SARIMA 模型可用于开发 AR，MA，ARMA 和 ARIMA 模型。

该方法适用于具有趋势和/或季节性分量的单变量时间序列。

### Python 代码

```py
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.statespace.sarimax.SARIMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
*   [statsmodels.tsa.statespace.sarimax.SARIMAXResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html)
*   [维基百科上的自回归综合移动平均线](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

## 具有外生回归量的季节性自回归整合移动平均线（SARIMAX）

具有外源回归量的季节性自回归整合移动平均值（SARIMAX）是 SARIMA 模型的扩展，其还包括外生变量的建模。

外生变量也称为协变量，可以被认为是并行输入序列，其在与原始序列相同的时间步骤中具有观察结果。初级系列可以称为内源性数据，以将其与外源序列进行对比。对于外源变量的观察结果直接在每个时间步骤包括在模型中，并且不以与主要内源序列相同的方式建模（例如作为 AR，MA 等过程）。

SARIMAX 方法还可用于使用外生变量对包含的模型进行建模，例如 ARX，MAX，ARMAX 和 ARIMAX。

该方法适用于具有趋势和/或季节性成分和外生变量的单变量时间序列。

### Python 代码

```py
# SARIMAX example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.statespace.sarimax.SARIMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
*   [statsmodels.tsa.statespace.sarimax.SARIMAXResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html)
*   [维基百科上的自回归综合移动平均线](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

## 向量自回归（VAR）

向量自回归（VAR）方法使用 AR 模型模拟每个时间序列中的下一步。 AR 是多个并行时间序列的推广，例如，多变量时间序列。

该模型的表示法涉及将 AR（p）模型的顺序指定为 VAR 函数的参数，例如， VAR（P）。

该方法适用于没有趋势和季节性成分的多变量时间序列。

### Python 代码

```py
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
v1 = i + random()
v2 = v1 + random()
row = [v1, v2]
data.append(row)
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.vector_ar.var_model.VAR API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VAR.html)
*   [statsmodels.tsa.vector_ar.var_model.VARResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html)
*   [维基百科](https://en.wikipedia.org/wiki/Vector_autoregression)上的向量自回归

## 向量自回归移动平均值（VARMA）

向量自回归移动平均（VARMA）方法使用 ARMA 模型对每个时间序列中的下一步进行建模。这是 ARMA 对多个并行时间序列的推广，例如多变量时间序列。

该模型的表示法涉及将 AR（p）和 MA（q）模型的顺序指定为 VARMA 函数的参数，例如， VARMA（p，q）。 VARMA 模型也可用于开发 VAR 或 VMA 模型。

该方法适用于没有趋势和季节性成分的多变量时间序列。

### Python 代码

```py
# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
v1 = random()
v2 = v1 + random()
row = [v1, v2]
data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.statespace.varmax.VARMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.varmax.VARMAX.html)
*   [statsmodels.tsa.statespace.varmax.VARMAXResults](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.varmax.VARMAXResults.html)
*   [维基百科](https://en.wikipedia.org/wiki/Vector_autoregression)上的向量自回归

## 具有外源回归量的向量自回归移动平均值（VARMAX）

具有外源回归量的向量自回归移动平均值（VARMAX）是 VARMA 模型的扩展，其还包括外生变量的建模。它是 ARMAX 方法的多变量版本。

外生变量也称为协变量，可以被认为是并行输入序列，其在与原始序列相同的时间步骤中具有观察结果。初级系列被称为内源性数据，以将其与外源序列进行对比。对于外源变量的观察结果直接在每个时间步骤包括在模型中，并且不以与主要内源序列相同的方式建模（例如作为 AR，MA 等过程）。

VARMAX 方法还可用于使用外生变量（如 VARX 和 VMAX）对包含的模型进行建模。

该方法适用于没有趋势和季节性成分以及外生变量的多变量时间序列。

### Python 代码

```py
# VARMAX example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
v1 = random()
v2 = v1 + random()
row = [v1, v2]
data.append(row)
data_exog = [x + random() for x in range(100)]
# fit model
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
data_exog2 = [[100]]
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.statespace.varmax.VARMAX API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.varmax.VARMAX.html)
*   [statsmodels.tsa.statespace.varmax.VARMAXResults](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.varmax.VARMAXResults.html)
*   [维基百科](https://en.wikipedia.org/wiki/Vector_autoregression)上的向量自回归

## 简单指数平滑（SES）

简单指数平滑（SES）方法将下一个时间步长建模为先前时间步长的观测值的指数加权线性函数。

该方法适用于没有趋势和季节性成分的单变量时间序列。

### Python 代码

```py
# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.holtwinters.SimpleExpSmoothing API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html)
*   [statsmodels.tsa.holtwinters.HoltWintersResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html)
*   [维基百科上的指数平滑](https://en.wikipedia.org/wiki/Exponential_smoothing)

## 霍尔特·温特的指数平滑（HWES）

Holt Winter 的指数平滑（HWES）也称为三次指数平滑方法，将下一个时间步长建模为先前时间步长的观测值的指数加权线性函数，并考虑趋势和季节性。

该方法适用于具有趋势和/或季节性分量的单变量时间序列。

### Python 代码

```py
# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
```

### 更多信息

*   [statsmodels.tsa.holtwinters.ExponentialSmoothing API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
*   [statsmodels.tsa.holtwinters.HoltWintersResults API](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html)
*   [维基百科上的指数平滑](https://en.wikipedia.org/wiki/Exponential_smoothing)

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [Statsmodels：时间序列分析 API](http://www.statsmodels.org/dev/tsa.html)
*   [Statsmodels：状态空间方法的时间序列分析](http://www.statsmodels.org/dev/statespace.html)

## 摘要

在这篇文章中，您发现了一套经典的时间序列预测方法，您可以测试和调整时间序列数据集。

我错过了您最喜欢的古典时间序列预测方法吗？
请在下面的评论中告诉我。

您是否在数据集上尝试了这些方法？
请在评论中告诉我你的发现。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。