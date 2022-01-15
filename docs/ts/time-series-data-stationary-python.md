# 如何使用 Python 检查时间序列数据是否是平稳的

> 原文： [https://machinelearningmastery.com/time-series-data-stationary-python/](https://machinelearningmastery.com/time-series-data-stationary-python/)

时间序列不同于更传统的分类和回归预测性建模问题。

时间结构为观察增加了顺序。这种强加的顺序意味着需要专门处理关于这些观察的一致性的重要假设。

例如，在建模时，假设观测的汇总统计量是一致的。在时间序列术语中，我们将此期望称为时间序列是静止的。

通过添加趋势，季节性和其他依赖于时间的结构，可以在时间序列中容易地违反这些假设。

在本教程中，您将了解如何使用 Python 检查时间序列是否固定。

完成本教程后，您将了解：

*   如何使用线图识别明显的静止和非平稳时间序列。
*   如何查看随时间变化的均值和方差等汇总统计量。
*   如何使用具有统计显着性的统计检验来检查时间序列是否静止。

让我们开始吧。

*   **2017 年 2 月更新**：修正了 p 值解释的拼写错误，增加了要点，使其更加清晰。
*   **更新于 May / 2018** ：改进了拒绝与拒绝统计测试的语言。

![How to Check if Time Series Data is Stationary with Python](img/db3db6900fcafd5aae3b32160867fa5e.jpg)

如何使用 Python 检查时间序列数据是否是固定的
照片由 [Susanne Nilsson](https://www.flickr.com/photos/infomastern/16011098565/) 保留，保留一些权利。

## 固定时间序列

静止时间序列中的观察结果不依赖于时间。

如果没有趋势或季节性影响，时间序列是[静止](https://en.wikipedia.org/wiki/Stationary_process)。按时间序列计算的汇总统计量随时间变化是一致的，例如观察值的均值或方差。

当时间序列静止时，可以更容易建模。统计建模方法假定或要求时间序列是静止的以使其有效。

以下是静止的[每日女性出生](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959)数据集的示例。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-total-female-births.csv', header=0)
series.plot()
pyplot.show()
```

运行该示例将创建以下图表。

![Daily Female Births Dataset Plot](img/3cb077e0b35882fd2a17e85387d49613.jpg)

每日女性出生数据集图

## 非定时时间序列

来自非平稳时间序列的观测显示了季节性影响，趋势和依赖于时间指数的其他结构。

像平均值和方差这样的汇总统计量会随着时间的推移而发生变化，从而使模型可能尝试捕获的概念发生偏差。

经典时间序列分析和预测方法涉及通过识别和消除趋势以及消除季节性影响来使非平稳时间序列数据静止。

下面是一个非静止的[航空公司乘客](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!)数据集的示例，显示趋势和季节性组件。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('international-airline-passengers.csv', header=0)
series.plot()
pyplot.show()
```

运行该示例将创建以下图表。

![Non-Stationary Airline Passengers Dataset](img/63a7ce1e92a05a4f3ea7f33cbaa9ce62.jpg)

非固定航空公司乘客数据集

## 固定时间序列的类型

平稳性的概念来自时间序列的理论研究，它在预测时是一个有用的抽象。

如果你深入研究这个话题，你可能会遇到一些更细微的平稳性概念。他们是：

他们是：

*   **固定过程**：生成一系列固定观测的过程。
*   **固定模型**：描述固定观测系列的模型。
*   **Trend Stationary** ：没有趋势的时间序列。
*   **季节性文具**：没有季节性的时间序列。
*   **严格固定**：静止过程的数学定义，特别是观测的联合分布对时移是不变的。

## 固定时间序列与预测

你应该让你的时间序列固定吗？

一般来说，是的。

如果您的时间序列中有明确的趋势和季节性，那么对这些组件进行建模，将其从观察中移除，然后在残差上训练模型。

> 如果我们将静态模型拟合到数据中，我们假设我们的数据是静止过程的实现。因此，我们分析的第一步应该是检查是否有任何趋势或季节性影响的证据，如果有，则删除它们。

- 第 122 页，[介绍时间序列与 R](http://www.amazon.com/dp/0387886974?tag=inspiredalgor-20) 。

统计时间序列方法甚至现代机器学习方法将受益于数据中更清晰的信号。

但…

当经典方法失败时，我们转向机器学习方法。当我们想要更多或更好的结果时。我们不知道如何最好地模拟时间序列数据中的未知非线性关系，并且一些方法在处理非平稳观测或者问题的静态和非静态视图的某种混合时可以产生更好的表现。

这里的建议是将时间序列的属性视为静止或不作为另一个信息源，可以在使用机器学习方法时在时间序列问题中使用特征工程和特征选择。

## 检查平稳性

有许多方法可以检查时间序列（直接观察，残差，否则）是静止的还是非静止的。

1.  **看一下情节**：您可以查看数据的时间序列图，并目视检查是否有任何明显的趋势或季节性。
2.  **摘要统计**：您可以查看季节或随机分区数据的摘要统计量，并检查明显或显着的差异。
3.  **统计检验**：您可以使用统计检验来检查是否满足或已经违反了平稳性的期望。

在上文中，我们已经将每日女性出生和航空旅客数据集分别定为静止和非静止，并且图表显示趋势和季节性成分明显缺乏和存在。

接下来，我们将看一个快速而肮脏的方法来计算和查看我们的时间序列数据集的摘要统计量，以检查它是否是静止的。

## 摘要统计

快速而肮脏的检查以确定您的时间序列是非静止的是查看摘要统计量。

您可以将时间序列拆分为两个（或更多）分区，并比较每个组的均值和方差。如果它们不同并且差异具有统计显着性，则时间序列可能是非平稳的。

接下来，让我们在 Daily Births 数据集上尝试这种方法。

### 每日出生数据集

因为我们正在研究均值和方差，所以我们假设数据符合高斯（也称为钟形曲线或正态）分布。

我们还可以通过观察我们观察的直方图来快速检查这一点。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-total-female-births.csv', header=0)
series.hist()
pyplot.show()
```

运行该示例绘制时间序列中的值的直方图。我们清楚地看到高斯分布的钟形曲线形状，也许右尾更长。

![Histogram of Daily Female Births](img/2993028bc09e6272c250870a12c9b1ab.jpg)

每日女性出生的直方图

接下来，我们可以将时间序列分成两个连续的序列。然后我们可以计算每组数字的均值和方差并比较这些数值。

```py
from pandas import Series
series = Series.from_csv('daily-total-female-births.csv', header=0)
X = series.values
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

运行此示例显示均值和方差值不同，但在同一个球场中。

```py
mean1=39.763736, mean2=44.185792
variance1=49.213410, variance2=48.708651
```

接下来，让我们在 Airline Passengers 数据集上尝试相同的技巧。

### 航空公司乘客数据集

直接切入追逐，我们可以分割我们的数据集并计算每个组的均值和方差。

```py
from pandas import Series
series = Series.from_csv('international-airline-passengers.csv', header=0)
X = series.values
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

运行示例，我们可以看到均值和方差看起来非常不同。

我们有一个非平稳的时间序列。

```py
mean1=182.902778, mean2=377.694444
variance1=2244.087770, variance2=7367.962191
```

也许。

让我们退一步，检查在这种情况下假设高斯分布是否有意义，将时间序列的值绘制为直方图。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('international-airline-passengers.csv', header=0)
series.hist()
pyplot.show()
```

运行该示例表明，值的分布确实看起来不像高斯，因此均值和方差值的意义不大。

这种观察的压扁分布可能是非平稳时间序列的另一个指标。

![Histogram of Airline Passengers](img/2c79f99e2ae6637c85c0264766e1bdda.jpg)

航空公司乘客的直方图

再次回顾时间序列的情节，我们可以看到有一个明显的季节性成分，看起来季节性成分正在增长。

这可能表明一个季节的指数增长。可以使用对数变换将指数变化平坦回到线性关系。

下面是具有时间序列的对数变换的相同直方图。

```py
from pandas import Series
from matplotlib import pyplot
from numpy import log
series = Series.from_csv('international-airline-passengers.csv', header=0)
X = series.values
X = log(X)
pyplot.hist(X)
pyplot.show()
pyplot.plot(X)
pyplot.show()
```

运行该示例，我们可以看到更熟悉的高斯类或类似统一的值分布。

![Histogram Log of Airline Passengers](img/90fdba0cb69f067ea7ad2cec5c3695cd.jpg)

航空公司乘客的直方图记录

我们还创建了对数转换数据的线图，可以看到指数增长似乎减少了，但我们仍然有趋势和季节性元素。

![Line Plot Log of Airline Passengers](img/14637e834cfabca2a7ba7b395c44b18e.jpg)

航空公司乘客的线路图

我们现在可以计算对数变换数据集的值的均值和标准差。

```py
from pandas import Series
from matplotlib import pyplot
from numpy import log
series = Series.from_csv('international-airline-passengers.csv', header=0)
X = series.values
X = log(X)
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

运行示例显示了每组的平均值和标准偏差值，这些值又相似但不相同。

也许，仅从这些数字来看，我们就会说时间序列是静止的，但我们坚信在审查线图时并非如此。

```py
mean1=5.175146, mean2=5.909206
variance1=0.068375, variance2=0.049264
```

这是一种快速而肮脏的方法，很容易被愚弄。

我们可以使用统计检验来检查两个高斯随机变量样本之间的差异是真实的还是统计的侥幸。我们可以探索统计显着性检验，例如 Student t 检验，但由于值之间的序列相关性，事情变得棘手。

在下一节中，我们将使用统计测试来明确评论单变量时间序列是否是静止的。

## 增强 Dickey-Fuller 测试

统计测试对您的数据做出了强有力的假设。它们只能用于通知零假设可被拒绝或未被拒绝的程度。必须解释结果才能使给定问题有意义。

然而，他们可以提供快速检查和确认证据，证明您的时间序列是静止的或非静止的。

[增强 Dickey-Fuller 检验](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)是一种称为[单位根检验](https://en.wikipedia.org/wiki/Unit_root_test)的统计检验。

单位根检验背后的直觉是它确定趋势定义时间序列的强度。

有许多单位根测试，Augmented Dickey-Fuller 可能是更广泛使用的之一。它使用自回归模型并优化跨多个不同滞后值的信息标准。

测试的零假设是时间序列可以由单位根表示，它不是静止的（具有一些时间相关的结构）。替代假设（拒绝零假设）是时间序列是静止的。

*   **空假设（H0）**：如果未能被拒绝，则表明时间序列具有单位根，这意味着它是非平稳的。它有一些时间依赖的结构。
*   **替代假设（H1）**：零假设被拒绝;它表明时间序列没有单位根，这意味着它是静止的。它没有时间依赖的结构。

我们使用测试中的 p 值来解释这个结果。低于阈值的 p 值（例如 5％或 1％）表明我们拒绝零假设（静止），否则高于阈值的 p 值表明我们未能拒绝零假设（非静止）。

*   **p 值＆gt; 0.05** ：未能拒绝原假设（H0），数据具有单位根并且是非平稳的。
*   **p 值＆lt; = 0.05** ：拒绝原假设（H0），数据没有单位根并且是静止的。

下面是在 Daily Female Births 数据集上计算 Augmented Dickey-Fuller 测试的示例。 statsmodels 库提供实现测试的 [adfuller（）](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html)函数。

```py
from pandas import Series
from statsmodels.tsa.stattools import adfuller
series = Series.from_csv('daily-total-female-births.csv', header=0)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

运行该示例将打印测试统计值-4。这个统计量越负，我们就越有可能拒绝零假设（我们有一个固定的数据集）。

作为输出的一部分，我们得到一个查找表，以帮助确定 ADF 统计量。我们可以看到，我们的统计值-4 小于-3.449 的值，1％。

这表明我们可以拒绝具有小于 1％的显着性水平的零假设（即，结果是统计侥幸的低概率）。

拒绝原假设意味着该过程没有单位根，反过来，时间序列是静止的或没有时间依赖的结构。

```py
ADF Statistic: -4.808291
p-value: 0.000052
Critical Values:
	5%: -2.870
	1%: -3.449
	10%: -2.571
```

我们可以在 Airline Passenger 数据集上执行相同的测试。

```py
from pandas import Series
from statsmodels.tsa.stattools import adfuller
series = Series.from_csv('international-airline-passengers.csv', header=0)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

运行该示例给出了与上面不同的图片。检验统计量是正的，这意味着我们不太可能拒绝零假设（它看起来是非平稳的）。

将测试统计量与临界值进行比较，看起来我们不得不拒绝零假设，即时间序列是非平稳的并且具有时间依赖性结构。

```py
ADF Statistic: 0.815369
p-value: 0.991880
Critical Values:
	5%: -2.884
	1%: -3.482
	10%: -2.579
```

让我们再次对数据集进行日志转换，使值的分布更加线性，更好地满足此统计检验的预期。

```py
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from numpy import log
series = Series.from_csv('international-airline-passengers.csv', header=0)
X = series.values
X = log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

运行该示例显示测试统计量的负值。

我们可以看到该值大于临界值，这意味着我们不能拒绝零假设，反过来说时间序列是非平稳的。

```py
ADF Statistic: -1.717017
p-value: 0.422367
	5%: -2.884
	1%: -3.482
	10%: -2.579
```

## 摘要

在本教程中，您了解了如何使用 Python 检查时间序列是否固定。

具体来说，你学到了：

*   时间序列数据静止用于统计建模方法甚至一些现代机器学习方法的重要性。
*   如何使用线图和基本摘要统计来检查时间序列是否静止。
*   如何计算和解释统计显着性检验以检查时间序列是否静止。

您对固定和非固定时间序列或此帖有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。