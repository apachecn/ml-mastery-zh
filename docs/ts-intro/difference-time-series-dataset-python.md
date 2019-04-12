# 如何将时间序列数据集与 Python 区分开来

> 原文： [https://machinelearningmastery.com/difference-time-series-dataset-python/](https://machinelearningmastery.com/difference-time-series-dataset-python/)

差分是时间序列中流行且广泛使用的数据变换。

在本教程中，您将了解如何使用 Python 将差异操作应用于时间序列数据。

完成本教程后，您将了解：

*   关于差分运算，包括滞后差和差分顺序的配置。
*   如何开发差分操作的手动实现。
*   如何使用内置的 Pandas 差分功能。

让我们开始吧。

![How to Difference a Time Series Dataset with Python](img/43b7f72d612ed1d6a13cbddfc4b212fb.jpg)

如何区分时间序列数据集与 Python
照片由 [Marcus](https://www.flickr.com/photos/tempoworld/7326465464/) ，保留一些权利。

## 为什么差异时间序列数据？

差分是一种转换时间序列数据集的方法。

它可用于消除序列对时间的依赖性，即所谓的时间依赖性。这包括趋势和季节性等结构。

> 差异可以通过消除时间序列水平的变化来帮助稳定时间序列的均值，从而消除（或减少）趋势和季节性。

- 第 215 页，[预测：原则与实践](http://www.amazon.com/dp/0987507109?tag=inspiredalgor-20)

通过从当前观察中减去先前的观察来执行差分。

```py
difference(t) = observation(t) - observation(t-1)
```

以这种方式，可以计算一系列差异。

### 滞后差异

将连续观察之间的差异称为滞后-1 差异。

可以调整滞后差异以适应特定的时间结构。

对于具有季节性成分的时间序列，滞后可以预期为季节性的周期（宽度）。

### 差异订单

在执行差分运算之后，例如在非线性趋势的情况下，时间结构可能仍然存在。

这样，差分过程可以重复多次，直到所有时间依赖性都被消除。

执行差分的次数称为差分顺序。

## 洗发水销售数据集

该数据集描述了 3 年期间每月洗发水的销售数量。

单位是销售计数，有 36 个观察。原始数据集归功于 Makridakis，Wheelwright 和 Hyndman（1998）。

[您可以在此处下载并了解有关数据集的更多信息](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)。

下面的示例加载并创建已加载数据集的图。

```py
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.plot()
pyplot.show()
```

运行该示例将创建在数据中显示清晰线性趋势的图。

![Shampoo Sales Dataset Plot](img/1999b84051dad90ee6a5edd3548510d9.jpg)

洗发水销售数据集图

## 手动差分

我们可以手动区分数据集。

这涉及开发一个创建差异数据集的新功能。该函数将遍历提供的序列并以指定的间隔或滞后计算差异值。

以下名为 _difference（）_ 的函数实现了此过程。

```py
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
```

我们可以看到该函数在指定的时间间隔后小心地开始差异数据集，以确保实际上可以计算差值。定义默认间隔或滞后值 1。这是一个合理的默认值。

进一步的改进是还能够指定执行差分操作的次序或次数。

以下示例将手动 _ 差异（）_ 功能应用于 Shampoo Sales 数据集。

```py
from pandas import read_csv
from pandas import datetime
from pandas import Series
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
diff = difference(X)
pyplot.plot(diff)
pyplot.show()
```

运行该示例将创建差异数据集并绘制结果。

![Manually Differenced Shampoo Sales Dataset](img/c0348350af1cd8ebd4fd3f785abb83ec.jpg)

手动差异的洗发水销售数据集

## 自动差分

Pandas 库提供了自动计算数据集差异的功能。

在[系列](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.diff.html)和 [DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html) 对象上都提供了 _diff（）_ 功能。

与上一节中手动定义的差异函数一样，它需要一个参数来指定间隔或滞后，在本例中称为 _ 周期 _。

下面的示例演示了如何在 Pandas Series 对象上使用内置差异函数。

```py
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
diff = series.diff()
pyplot.plot(diff)
pyplot.show()
```

与上一节一样，运行该示例会绘制差异数据集。

除了需要更少的代码之外，使用 Pandas 功能的好处是它可以维护差异系列的日期时间信息。

![Automatic Differenced Shampoo Sales Dataset](img/db8368260f3902ac16301eab9f5ed730.jpg)

自动差异洗发水销售数据集

## 摘要

在本教程中，您了解了如何使用 Python 将差异操作应用于时间序列数据。

具体来说，你学到了：

*   关于差异操作，包括滞后和顺序的配置。
*   如何手动实现差异变换。
*   如何使用内置的 Pandas 实现差异变换。

您对差异或此帖有任何疑问吗？
在下面的评论中提出您的问题。