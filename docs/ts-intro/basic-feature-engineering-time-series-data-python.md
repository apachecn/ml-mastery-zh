# Python 中基于时间序列数据的基本特征工程

> 原文： [https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

在我们开始使用机器学习算法之前，必须将时间序列数据重新构建为监督学习数据集。

时间序列中没有输入和输出功能的概念。相反，我们必须选择要预测的变量，并使用特征工程来构建将用于对未来时间步骤进行预测的所有输入。

在本教程中，您将了解如何使用 Python 对时间序列数据执行特征工程，以使用机器学习算法对时间序列问题进行建模。

完成本教程后，您将了解：

*   特征工程时间序列数据的基本原理和目标。
*   如何开发基于日期时间的基本输入功能。
*   如何开发更复杂的滞后和滑动窗口汇总统计功能。

让我们潜入。

*   **2017 年 6 月更新**：修正了扩展窗口代码示例中的拼写错误。

![Basic Feature Engineering With Time Series Data in Python](img/6b1bb5739d5ddef1fd2fe5f18d7b6803.jpg)

基于 Python 时间序列数据的基本特征工程
[JoséMorcilloValenciano](https://www.flickr.com/photos/jamorcillov/6108532064/) 的照片，保留一些权利。

## 时间序列的特征工程

必须转换时间序列数据集以将其建模为监督学习问题。

这看起来像是这样的：

```py
time 1, value 1
time 2, value 2
time 3, value 3
```

对于看起来像这样的东西：

```py
input 1, output 1
input 2, output 2
input 3, output 3
```

这样我们就可以训练有监督的学习算法。

输入变量在机器学习领域也称为特征，我们面前的任务是从我们的时间序列数据集创建或发明新的输入特征。理想情况下，我们只需要最有助于学习方法的输入特征来模拟我们想要预测的输入（ **X** ）和输出（ **y** ）之间的关系。

在本教程中，我们将介绍可以从时间序列数据集中创建的三类功能：

1.  **日期时间特征**：这些是每个观察的时间步长本身的组成部分。
2.  **滞后特征**：这些是先前时间步长的值。
3.  **窗口特征**：这些是先前时间步长的固定窗口上的值的摘要。

在我们深入研究从时间序列数据创建输入要素的方法之前，让我们首先回顾一下特征工程的目标。

## 特征工程的目标

[特征工程](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)的目标是在新输入特征和监督学习算法建模的输出特征之间提供强大且理想的简单关系。

实际上，我们正在移动复杂性。

输入和输出数据之间的关系存在复杂性。在时间序列的情况下，没有输入和输出变量的概念;我们也必须发明这些并从头开始构建有监督的学习问题。

我们可以依靠复杂模型的能力来破译问题的复杂性。如果我们能够更好地揭示数据中输入和输出之间的内在关系，我们可以使这些模型的工作更容易（甚至使用更简单的模型）。

困难在于我们不知道我们试图揭示的输入和输出之间潜在的固有功能关系。如果我们知道，我们可能不需要机器学习。

相反，我们唯一的反馈是在监督学习数据集上开发的模型的表现或我们创建的问题的“视图”。实际上，最好的默认策略是使用所有可用知识从时间序列数据集中创建许多优秀的数据集，并使用模型表现（和其他项目要求）来帮助确定您的问题的优秀特性和良好视图。

为清楚起见，我们将重点关注示例中的单变量（一个变量）时间序列数据集，但这些方法同样适用于多变量时间序列问题。接下来，让我们看一下我们将在本教程中使用的数据集。

## 最低每日温度数据集

在这篇文章中，我们将使用最低每日温度数据集。

该数据集描述了澳大利亚墨尔本 10 年（1981-1990）的最低日常温度。

单位为摄氏度，有 3,650 个观测值。数据来源被称为澳大利亚气象局。

下面是前 5 行数据的示例，包括标题行。

```py
"Date","Temperature"
"1981-01-01",20.7
"1981-01-02",17.9
"1981-01-03",18.8
"1981-01-04",14.6
"1981-01-05",15.8
```

下面是从数据市场获取的整个数据集的图表。

![Minimum Daily Temperatures](img/294be07b72998925edffe0f805907812.jpg)

最低每日温度

数据集显示趋势增加，可能还有一些季节性组件。

[在此处下载并了解有关数据集的更多信息](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990)。

**注意**：下载的文件包含一些问号（“？”）字符，必须先将其删除才能使用数据集。在文本编辑器中打开文件并删除“？”字符。同时删除文件中的任何页脚信息。

## 日期时间功能

让我们从一些我们可以使用的最简单的功能开始。

这些是每次观察日期/时间的特征。事实上，这些可以简单地开始，并进入相当复杂的领域特定领域。

我们可以开始的两个特征是每个观察的整数月和日。我们可以想象，有监督的学习算法可能能够使用这些输入来帮助梳理出一年中的时间或季节性类型的季节性信息。

我们提出的监督学习问题是预测月和日的每日最低温度，如下：

```py
Month, Day, Temperature
Month, Day, Temperature
Month, Day, Temperature
```

我们可以使用 Pandas 来做到这一点。首先，时间序列作为 Pandas _ 系列 _ 加载。然后，我们为转换后的数据集创建一个新的 Pandas`DataFrame`。

接下来，每次添加一列，其中从系列中的每个观察的时间戳信息中提取月和日信息。

下面是执行此操作的 Python 代码。

```py
from pandas import Series
from pandas import DataFrame
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]
print(dataframe.head(5))
```

运行此示例将打印转换后的数据集的前 5 行。

```py
month day temperature
0 1 1 20.7
1 1 2 17.9
2 1 3 18.8
3 1 4 14.6
4 1 5 15.8
```

仅使用月和日信息来预测温度并不复杂，并且可能导致模型不佳。然而，这些信息与其他工程特征相结合可能最终会产生更好的模型。

您可以枚举时间戳的所有属性，并考虑对您的问题可能有用的内容，例如：

*   一天的分钟数。
*   一天中的一小时。
*   营业时间与否。
*   周末与否。
*   一年中的季节。
*   一年中的业务季度。
*   夏令时与否。
*   公共假期与否。
*   闰年与否。

从这些示例中，您可以看到您不限于原始整数值。您也可以使用二进制标记功能，例如观察是否在公共假日录制。

在最低温度数据集的情况下，季节可能更相关。它正在创建这样的特定于域的功能，这些功能更有可能为您的模型增加价值。

基于日期时间的功能是一个良好的开端，但在以前的时间步骤中包含值通常更有用。这些被称为滞后值，我们将在下一节中介绍添加这些功能。

## 滞后特征

滞后特征是时间序列预测问题转化为监督学习问题的经典方式。

最简单的方法是在给定前一时间（t-1）的值的情况下预测下一次（t + 1）的值。具有移位值的监督学习问题如下所示：

```py
Value(t-1), Value(t+1)
Value(t-1), Value(t+1)
Value(t-1), Value(t+1)
```

Pandas 库提供 [shift（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)，以帮助从时间序列数据集创建这些移位或滞后特征。将数据集移动 1 会创建 t-1 列，为第一行添加 NaN（未知）值。没有移位的时间序列数据集表示 t + 1。

让我们以一个例子来具体化。温度数据集的前 3 个值分别为 20.7,17.9 和 18.8。因此，前 3 个观测值的移位和未移位温度列表如下：

```py
Shifted, Original
NaN, 20.7
20.7, 17.9
17.9, 18.8
```

我们可以使用 [concat（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html)沿着列轴（_ 轴= 1_ ）将移位列连接到一个新的 DataFrame 中。

综合这些，下面是为我们的日常温度数据集创建滞后特征的示例。从加载的序列中提取值，并创建这些值的移位和未移位列表。为清楚起见，每列也在`DataFrame`中命名。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))
```

运行该示例将使用滞后功能打印新数据集的前 5 行。

```py
t-1 t+1
0 NaN 20.7
1 20.7 17.9
2 17.9 18.8
3 18.8 14.6
4 14.6 15.8
```

您可以看到我们必须丢弃第一行才能使用数据集来训练监督学习模型，因为它没有足够的数据可供使用。

滞后特征的添加称为滑动窗口方法，在这种情况下窗口宽度为 1.就好像我们在每个观察的时间序列中滑动焦点，只关注窗口宽度内的内容。

我们可以扩展窗口宽度并包含更多滞后功能。例如，下面的上述情况被修改为包括最后 3 个观察值以预测下一个时间步的值。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-3', 't-2', 't-1', 't+1']
print(dataframe.head(5))
```

运行此示例将打印新滞后数据集的前 5 行。

```py
t-3 t-2 t-1 t+1
0 NaN NaN NaN 20.7
1 NaN NaN 20.7 17.9
2 NaN 20.7 17.9 18.8
3 20.7 17.9 18.8 14.6
4 17.9 18.8 14.6 15.8
```

同样，您可以看到我们必须丢弃没有足够数据来训练监督模型的前几行。

滑动窗口方法的一个难点是为您的问题制作窗口的大小。

也许一个好的起点是执行灵敏度分析并尝试一组不同的窗口宽度，从而创建一组不同的数据集“视图”，并查看哪些结果表现更好的模型。会有一个收益递减点。

另外，为什么要停止使用线性窗口？也许您需要上周，上个月和去年的滞后值。同样，这归结于特定领域。

在温度数据集的情况下，来自前一年或前几年的同一天的滞后值可能是有用的。

我们可以使用窗口做更多事情，而不是包含原始值。在下一节中，我们将介绍包含在窗口中汇总统计信息的功能。

## 滚动窗口统计

添加原始滞后值之外的步骤是添加前一时间步的值的摘要。

我们可以计算滑动窗口中值的汇总统计数据，并将这些统计数据包含在数据集中。也许最有用的是前几个值的平均值，也称为滚动均值。

例如，我们可以计算前两个值的平均值，并使用它来预测下一个值。对于温度数据，我们必须等待 3 个时间步，然后才能使用 2 个值来取平均值，然后才能使用该值来预测第 3 个值。

例如：

```py
mean(t-2, t-1), t+1
mean(20.7, 17.9), 18.8
19.3, 18.8
```

Pandas 提供了 [rolling（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html)，它在每个时间步都创建了一个带有值窗口的新数据结构。然后，我们可以在为每个时间步骤收集的值窗口上执行统计函数，例如计算平均值。

首先，必须改变系列。然后可以创建滚动数据集，并在每个窗口上计算两个值的平均值。

以下是前三个滚动窗口中的值：

```py
#, Window Values
1, NaN
2, NaN, 20.7
3, 20.7, 17.9
```

这表明我们在第 3 行之前不会有可用的数据。

最后，与上一节一样，我们可以使用`concat()`函数构建一个只包含新列的新数据集。

下面的示例演示了如何使用窗口大小为 2 的 Pandas 执行此操作。

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
temps = DataFrame(series.values)
shifted = temps.shift(1)
window = shifted.rolling(window=2)
means = window.mean()
dataframe = concat([means, temps], axis=1)
dataframe.columns = ['mean(t-2,t-1)', 't+1']
print(dataframe.head(5))
```

运行该示例将打印新数据集的前 5 行。我们可以看到前两行没用。

*   第一个 NaN 是由系列的转变创造的。
*   第二个因为 NaN 不能用于计算平均值。
*   最后，第三行显示了用于预测 18.8 系列中第 3 个值的 19.30（平均值 20.7 和 17.9）的预期值。

```py
mean(t-2,t-1) t+1
0 NaN 20.7
1 NaN 17.9
2 19.30 18.8
3 18.35 14.6
4 16.70 15.8
```

我们可以计算更多的统计数据，甚至可以用不同的数学方法计算“窗口”的定义。

下面是另一个示例，显示窗口宽度为 3，数据集包含更多摘要统计信息，特别是窗口中的最小值，平均值和最大值。

您可以在代码中看到我们明确指定滑动窗口宽度作为命名变量。这使我们可以在计算系列的正确位移和指定`rolling()`函数的窗口宽度时使用它。

在这种情况下，窗口宽度为 3 表示我们必须将系列向前移动 2 个时间步长。这使得前两行为 NaN。接下来，我们需要计算每个窗口有 3 个值的窗口统计信息。在我们甚至从窗口中的系列中获得足够的数据以开始计算统计数据之前，它需要 3 行。前 5 个窗口中的值如下：

```py
#, Window Values
1, NaN
2, NaN, NaN
3, NaN, NaN, 20.7
4, NaN, 20.7, 17.9
5, 20.7, 17.9, 18.8
```

这表明我们不会期望至少在第 5 行（数组索引 4）之前可用的数据

```py
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
temps = DataFrame(series.values)
width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(window=width)
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
```

运行代码将打印新数据集的前 5 行。

我们可以检查第 5 行（数组索引 4）上值的正确性。我们可以看到确实 17.9 是最小值，20.7 是[20.7,17.9,18.8]窗口中值的最大值。

```py
min mean max t+1
0 NaN NaN NaN 20.7
1 NaN NaN NaN 17.9
2 NaN NaN NaN 18.8
3 NaN NaN NaN 14.6
4 17.9 19.133333 20.7 15.8
```

## 扩展窗口统计

另一种可能有用的窗口包括该系列中的所有先前数据。

这称为扩展窗口，可以帮助跟踪可观察数据的范围。与`DataFrame`上的`rolling()`函数一样，Pandas 提供 [expand（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.expanding.html)，它收集每个时间步的所有先前值的集合。

可以汇总这些先前数字列表并将其作为新功能包括在内。例如，下面是系列前 5 个步骤的展开窗口中的数字列表：

```py
#, Window Values
1, 20.7
2, 20.7, 17.9,
3, 20.7, 17.9, 18.8
4, 20.7, 17.9, 18.8, 14.6
5, 20.7, 17.9, 18.8, 14.6, 15.8
```

同样，您可以看到我们必须转换系列一次性步骤，以确保我们希望预测的输出值从这些窗口值中排除。因此输入窗口如下所示：

```py
#, Window Values
1, NaN
2, NaN, 20.7
3, NaN, 20.7, 17.9,
4, NaN, 20.7, 17.9, 18.8
5, NaN, 20.7, 17.9, 18.8, 14.6
```

值得庆幸的是，统计计算不包括扩展窗口中的 NaN 值，这意味着不需要进一步修改。

下面是计算每日温度数据集上展开窗口的最小值，平均值和最大值的示例。

```py
# create expanding window features
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
temps = DataFrame(series.values)
window = temps.expanding()
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
```

运行该示例将打印数据集的前 5 行。

检查扩展的最小值，平均值和最大值的点显示具有预期效果的示例。

```py
    min       mean   max   t+1
0  20.7  20.700000  20.7  17.9
1  17.9  19.300000  20.7  18.8
2  17.9  19.133333  20.7  14.6
3  14.6  18.000000  20.7  15.8
4  14.6  17.560000  20.7  15.8
```

## 摘要

在本教程中，您了解了如何使用特征工程将时间序列数据集转换为用于机器学习的监督学习数据集。

具体来说，你学到了：

*   特征工程时间序列数据的重要性和目标。
*   如何开发基于日期时间和滞后的功能。
*   如何开发滑动和展开窗口摘要统计功能。

**你知道时间序列的更多特征工程方法吗？** 请在下面的评论中告诉我。

**你有什么问题吗？** 在下面的评论中提出您的问题，我会尽力回答。