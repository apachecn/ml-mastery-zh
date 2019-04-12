# 如何重构时间序列预测问题

> 原文： [https://machinelearningmastery.com/reframe-time-series-forecasting-problem/](https://machinelearningmastery.com/reframe-time-series-forecasting-problem/)

您不必按原样建立时间序列预测问题的模型。

有许多方法可以重新构建您的预测问题，既可以简化预测问题，又可能会暴露更多或不同的信息进行建模。重构最终可以产生更好和/或更稳健的预测。

在本教程中，您将了解如何使用 Python 重新构建时间序列预测问题。

完成本教程后，您将了解：

*   如何将时间序列预测问题重新定义为备用回归问题。
*   如何将时间序列预测问题重新定义为分类预测问题。
*   如何使用其他时间范围重新构建时间序列预测问题。

让我们开始吧。

![How to Reframe Your Time Series Forecasting Problem](img/321d91d71a312f939629c15033288c8c.jpg)

如何重构您的时间序列预测问题
照片由 [Sean MacEntee](https://www.flickr.com/photos/smemon/14373987202/) ，保留一些权利。

## 重塑问题的好处

重新定义您的问题是探索对预测内容的替代观点。

探索时间序列预测问题的备用框架有两个潜在的好处：

1.  简化您的问题。
2.  提供集合预测的基础。

这两种好处最终都会导致更加熟练和/或更强大的预测。

### 1.简化您的问题

也许预测项目中最大的胜利可能来自重构问题。

这是因为预测问题的结构和类型比数据变换的选择，模型的选择或模型超参数的选择具有更大的影响。

它是项目中最大的杠杆，必须仔细考虑。

### 2.集合预报

除了改变您正在处理的问题之外，重构还起到了另一个作用：它可以为您提供一组可以建模的不同但高度相关的问题。

这样做的好处是框架可能不同，需要在数据准备和建模方法上有所不同。

对同一问题的不同观点的模型可以从输入中捕获不同的信息，并且反过来导致熟练的预测，但是以不同的方式。这些预测可以在整体中组合以产生更熟练或更强大的预测。

在本教程中，我们将探讨您可以考虑重构时间序列预测问题的三种不同方法。

在我们深入研究之前，让我们看一个简单的单变量时间序列问题，即预测每日最低温度作为讨论的背景。

## 最低每日温度数据集

该数据集描述了澳大利亚墨尔本市 10 年（1981-1990）的最低日常温度。

单位为摄氏度，有 3,650 个观测值。数据来源被称为澳大利亚气象局。

[了解有关数据市场](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990)上数据集的更多信息。

使用文件名“ _daily-minimum-Temperats.sv_ ”将最低每日温度下载到当前工作目录。

**注意**：下载的文件包含一些问号（“？”）字符，必须先将其删除才能使用数据集。在文本编辑器中打开文件并删除“？”字符。同时删除文件中的任何页脚信息。

下面的示例将数据集加载为 Pandas 系列。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
print(series.head())
series.plot()
pyplot.show()
```

运行该示例将打印已加载数据集的前 5 行。

```py
Date Temperature
1981-01-01 20.7
1981-01-02 17.9
1981-01-03 18.8
1981-01-04 14.6
1981-01-05 15.8
```

还创建了时间序列的线图。

![Minimum Daily Temperature Dataset](img/0c7d6e112b4f8795da2d94531768c7ff.jpg)

最低每日温度数据集

## 朴素的时间序列预测

朴素的方法是按原样预测问题。

作为参考，我们将这称为朴素的时间序列预测。

在这种情况下，可以删除季节性信息以使序列季节性静止。

然后可以基于滞后观察的一些函数来建模时间序列。

例如：

```py
Temp(t+1) = B0 + B1*Temp(t-1) + B2*Temp(t-2) ... Bn*Temp(t-n)
```

其中 _Temp（t + 1）_ 是预测序列中的下一个温度， _B0_ 到 _Bn_ 是从训练数据和 _Temp（ t-1）_ 至 _Temp（tn）_ 是滞后观察。

这可能是很好的甚至是许多问题所要求的。

风险在于，如何构建问题的先入为主的想法影响了数据收集，反过来可能限制了结果。

## 回归框架

大多数时间序列预测问题是回归问题，需要预测实值输出。

以下是 5 种不同的方法，可以将此预测问题重新表述为替代回归问题：

*   预测与前一天相比最低温度的变化。
*   预测相对于过去 14 天平均值的最低温度。
*   预测相对于去年同月平均值的最低温度。
*   预测最低温度四舍五入到最接近的 5 摄氏度。
*   预测未来 7 天的平均最低温度。

使温度相对是一个线性变换，可能不会使问题更简单，更容易预测，但它可能会动摇新的想法，甚至可能会考虑新的数据来源。

它还可以帮助您更清楚地思考预测的准确性以及预测值的实际要求。

转换预测问题的粒度确实会改变问题的难度，并且如果问题的要求允许这样的重新定义，则非常有用。

下面是重新设定最低每日温度预测问题的示例，以预测每日温度四舍五入到最接近的 5 度。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
from math import floor
# load data
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# round forecast to nearest 5
for i in range(len(dataframe['t+1'])):
	dataframe['t+1'][i] = int(dataframe['t+1'][i] / 5) * 5.0
print(dataframe.head(5))
```

运行该示例将打印重构问题的前 5 行。

该问题定义为前一天的最低温度，以摄氏度为单位，最小值为最接近的 5 度。

```py
    t-1   t+1
0   NaN  20.0
1  20.7  15.0
2  17.9  15.0
3  18.8  10.0
4  14.6  15.0
```

## 分类框架

分类涉及预测分类或标签输出（如“热”和“冷”）。

以下是将此预测问题重新定义为分类问题的 5 种不同方式：

*   预测最低温度是冷，中温还是暖。
*   预测最低温度的变化是小还是大。
*   预测最低温度是否为每月最低温度。
*   预测最低值是高于还是低于上一年的最低值。
*   预测未来 7 天的最低温度是上升还是下降。

转向分类可以简化预测问题。

这种方法打开了标签和二进制分类框架的想法。

输出变量的原生回归表示意味着大多数分类框架可能保持序数结构（例如冷，中，热）。意味着正在预测的类之间存在有序关系，这在预测“狗”和“猫”等标签时可能不是这种情况。

序数关系允许硬分类问题以及可以事后舍入到特定类别的整数预测问题。

下面是将最低每日温度预测问题转换为分类问题的示例，其中每个温度值是冷，中或热的序数值。这些标签映射到整数值，定义如下：

*   0（冷）：＆lt; 10 摄氏度。
*   1（中等）：&gt; = 10 且＆lt; 25 摄氏度。
*   2（热）：&gt; = 25 摄氏度。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
from math import floor
# load data
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# round forecast to nearest 5
for i in range(len(dataframe['t+1'])):
	value = dataframe['t+1'][i]
	if value < 10.0:
		dataframe['t+1'][i] = 0
	elif value >= 25.0:
		dataframe['t+1'][i] = 2
	else:
		dataframe['t+1'][i] = 1
print(dataframe.head(5))
```

运行该示例将打印重构问题的前 5 行。

给定前一天的最低温度（摄氏度），目标是将温度预测为冷，中或热（分别为 0,1,2）。

```py
    t-1  t+1
0   NaN  1.0
1  20.7  1.0
2  17.9  1.0
3  18.8  1.0
4  14.6  1.0
```

## 时间地平线框架

可以改变的另一个轴是时间范围。

时间范围是未来预测的时间步数。

以下是 5 种不同的方法，可以将此预测问题重新表述为不同的时间范围：

*   预测未来 7 天的最低温度。
*   预测 30 天内的最低温度。
*   预测下个月的平均最低温度。
*   预测下一周将具有最低最低温度的日期。
*   预测一年的最低温度值。

您很容易理解需要一步预测的想法。

专注于围绕时间范围重新解决问题迫使您思考点与多步预测以及未来需要考虑的距离。

您可能能够对未来进行预测，但技能可能会有所不同，进一步降低到您预测的未来。在思考预测的视野时，还要考虑预测的最低可接受表现。

下面的示例转换最低每日温度预测问题，以预测接下来 7 天的最低温度。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
from math import floor
# load data
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values, values.shift(-1),
	values.shift(-2), values.shift(-3), values.shift(-4), values.shift(-5),
	values.shift(-6)], axis=1)
dataframe.columns = ['t-1', 't+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7']
print(dataframe.head(14))
```

运行该示例将打印转换后的数据集的前 14 条记录。

问题定义为：给定前一天的最低日摄入温度，以摄氏度为单位，预测接下来 7 天的最低日常温度。

```py
     t-1   t+1   t+2   t+3   t+4   t+5   t+6   t+7
0    NaN  20.7  17.9  18.8  14.6  15.8  15.8  15.8
1   20.7  17.9  18.8  14.6  15.8  15.8  15.8  17.4
2   17.9  18.8  14.6  15.8  15.8  15.8  17.4  21.8
3   18.8  14.6  15.8  15.8  15.8  17.4  21.8  20.0
4   14.6  15.8  15.8  15.8  17.4  21.8  20.0  16.2
5   15.8  15.8  15.8  17.4  21.8  20.0  16.2  13.3
6   15.8  15.8  17.4  21.8  20.0  16.2  13.3  16.7
7   15.8  17.4  21.8  20.0  16.2  13.3  16.7  21.5
8   17.4  21.8  20.0  16.2  13.3  16.7  21.5  25.0
9   21.8  20.0  16.2  13.3  16.7  21.5  25.0  20.7
10  20.0  16.2  13.3  16.7  21.5  25.0  20.7  20.6
11  16.2  13.3  16.7  21.5  25.0  20.7  20.6  24.8
12  13.3  16.7  21.5  25.0  20.7  20.6  24.8  17.7
13  16.7  21.5  25.0  20.7  20.6  24.8  17.7  15.5
```

## 摘要

在本教程中，您了解了如何使用 Python 重新构建时间序列预测问题。

具体来说，你学到了：

*   如何设计时间序列问题的替代回归表示。
*   如何将预测问题构建为分类问题。
*   如何为预测问题设计备用时间范围。

您知道其他方法来重构您的时间序列预测问题吗？
在下面的评论中分享？

你有任何问题吗？
在下面的评论中询问他们，我会尽力回答。