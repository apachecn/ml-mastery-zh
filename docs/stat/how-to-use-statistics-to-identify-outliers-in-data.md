# 如何使用统计量识别数据中的异常值

> 原文： [https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)

建模时，清理数据样本以确保观察结果最能代表问题非常重要。

有时，数据集可以包含超出预期范围的极值，而不像其他数据。这些被称为异常值，通常可以通过理解甚至去除这些异常值来改进机器学习建模和模型技能。

在本教程中，您将发现有关异常值和两种统计方法的更多信息，可用于从数据集中识别和过滤异常值。

完成本教程后，您将了解：

*   异常值是数据集中不太可能的观察结果，可能有许多原因之一。
*   该标准偏差可用于识别高斯或类高斯数据中的异常值。
*   无论分布如何，四分位数范围都可用于识别数据中的异常值。

让我们开始吧。

*   **更新 May / 2018** ：修正了通过异常值限制过滤样本时的错误。谢谢 Yishai E 和彼得。

![How to Use Statistics to Identify Outliers in Data](img/88660edc68a1bce41465108bfae8af1f.jpg)

如何使用统计量识别数据中的异常值
照片由 [Jeff Richardson](https://www.flickr.com/photos/richo7/7176319689/) 提供，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  什么是异常值？
2.  测试数据集
3.  标准偏差法
4.  四分位距离法

## 什么是异常值？

异常值是一种与其他观察结果不同的观察结果。

它很少见，或不同，或者不适合某种方式。

异常值可能有很多原因，例如：

*   测量或输入错误。
*   数据损坏。
*   真正的离群观察（例如篮球中的迈克尔乔丹）。

由于每个数据集的具体情况，通常没有精确的方法来定义和识别异常值。相反，您或域专家必须解释原始观察并决定值是否为异常值。

尽管如此，我们可以使用统计方法来识别在可用数据的情况下看起来很少或不太可能的观察结果。

这并不意味着所识别的值是异常值，应予以删除。但是，本教程中描述的工具可以帮助您了解可能需要重新审视的罕见事件。

一个好的建议是考虑绘制已识别的离群值，也许在非离群值的背景下，以查看异常值是否存在任何系统关系或模式。如果存在，也许它们不是异常值并且可以解释，或者可能更系统地识别异常值本身。

## 测试数据集

在我们查看异常值识别方法之前，让我们定义一个可用于测试方法的数据集。

我们将生成从高斯分布中抽取的 10,000 个随机数，平均值为 50，标准差为 5。

从高斯分布绘制的数字将具有异常值。也就是说，凭借分布本身，将会有一些与我们可以识别为异常值的平均罕见值相距很远的值。

我们将使用 _randn（）_ 函数生成平均值为 0 且标准差为 1 的随机高斯值，然后将结果乘以我们自己的标准差并添加平均值以将值转换为首选范围。

伪随机数生成器被播种以确保每次运行代码时我们都得到相同的数字样本。

```py
# generate gaussian data
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# summarize
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
```

运行该示例将生成样本，然后打印均值和标准差。正如预期的那样，这些值非常接近预期值。

```py
mean=50.049 stdv=4.994
```

## 标准偏差法

如果我们知道样本中的值分布是高斯分布或类高斯分布，我们可以使用样本的标准偏差作为识别异常值的截止值。

高斯分布具有以下特性：可以使用与均值的标准偏差来可靠地总结样本中的值的百分比。

例如，在平均值的一个标准偏差内将覆盖 68％的数据。

因此，如果平均值为 50 且标准差为 5，如上面的测试数据集中那样，则样本中 45 到 55 之间的所有数据将占数据样本的约 68％。如果我们扩展范围如下，我们可以覆盖更多的数据样本：

*   1 标准偏差均值：68％
*   2 与标准差的标准偏差：95％
*   3 与标准差的标准偏差：99.7％

超出 3 个标准偏差的值是分布的一部分，但在 370 个样本中约有 1 个是不太可能或罕见的事件。

与平均值的三个标准偏差是在实践中用于识别高斯或高斯分布的异常值的常见截止。对于较小的数据样本，可以使用 2 个标准偏差（95％）的值，对于较大的样本，可以使用 4 个标准偏差（99.9％）的值。

让我们用一个有效的例子来具体化。

有时，首先将数据标准化（例如，以零均值和单位方差的 Z 分数），以便可以使用标准 Z 分数截止值来执行异常值检测。这是一种方便，一般不需要，我们将在此处以数据的原始比例进行计算以使事情清楚。

我们可以计算给定样本的平均值和标准差，然后计算用于识别异常值的截止值，其与平均值的偏差超过 3 个标准偏差。

```py
# calculate summary statistics
data_mean, data_std = mean(data), std(data)
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
```

然后我们可以将异常值识别为超出定义的下限和上限的那些示例。

```py
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
```

或者，我们可以从样本中过滤掉那些不在定义范围内的值。

```py
# remove outliers
outliers_removed = [x for x in data if x > lower and x < upper]
```

我们可以将这一切与我们在上一节中准备的样本数据集放在一起。

下面列出了完整的示例。

```py
# identify outliers with standard deviation
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate summary statistics
data_mean, data_std = mean(data), std(data)
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))
```

运行该示例将首先打印已识别的异常值的数量，然后打印不是异常值的观察数量，从而演示如何分别识别和过滤异常值。

```py
Identified outliers: 29
Non-outlier observations: 9971
```

到目前为止，我们只讨论了具有高斯分布的单变量数据，例如：一个变量。如果您有多变量数据，则可以使用相同的方法，例如：具有多个变量的数据，每个变量具有不同的高斯分布。

如果你有两个变量，你可以想象在两个维度中定义椭圆的边界。落在椭圆之外的观察将被视为异常值。在三个维度中，这将是椭圆体，依此类推到更高的维度。

或者，如果您对域有更多了解，可能会通过超出数据维度的一个或一个子集的限制来识别异常值。

## 四分位距离法

并非所有数据都是正常的或正常的，足以将其视为从高斯分布中提取。

总结非高斯分布数据样本的一个很好的统计量是 Interquartile Range，简称 IQR。

IQR 计算为数据的第 75 百分位数和第 25 百分位数之间的差异，并在框和晶须图中定义框。

请记住，可以通过对观察值进行排序并在特定指数处选择值来计算百分位数。第 50 个百分位数是中间值，或偶数个例子的两个中间值的平均值。如果我们有 10,000 个样本，那么第 50 个百分位将是第 5000 个和第 5001 个值的平均值。

我们将百分位数称为四分位数（“_ 夸脱 _”意思是 4）因为数据通过第 25,50 和 75 位值分为四组。

IQR 定义了中间 50％的数据或数据的主体。

IQR 可用于通过定义样本值的限制来识别异常值，这些样本值是 IQR 低于第 25 百分位数或高于第 75 百分位数的因子`k`。因子`k`的共同值是值 1.5。当在盒子和须状图的背景下描述时，因子 k 为 3 或更大可用于识别极端异常值或“_ 远离 _”的值。

在盒子和须状图上，这些限制被绘制为从盒子中绘制的胡须（或线条）上的栅栏。超出这些值的值将绘制为点。

我们可以使用 _ 百分位数（）_ NumPy 函数计算数据集的百分位数，该函数采用数据集和所需百分位数的规格。然后可以将 IQR 计算为第 75 百分位数和第 25 百分位数之间的差值。

```py
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
```

然后，我们可以将异常值的截止值计算为 IQR 的 1.5 倍，并从第 25 个百分位数中减去此截止值，并将其加到第 75 个百分位数，以给出数据的实际限制。

```py
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
```

然后，我们可以使用这些限制来识别异常值。

```py
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
```

我们还可以使用限制来过滤数据集中的异常值。

```py
outliers_removed = [x for x in data if x > lower and x < upper]
```

我们可以将所有这些结合在一起并演示测试数据集上的过程。

The complete example is listed below.

```py
# identify outliers with interquartile range
from numpy.random import seed
from numpy.random import randn
from numpy import percentile
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))
```

首先运行该示例打印已识别的第 25 和第 75 百分位数以及计算的 IQR。打印所识别的异常值的数量，然后是非异常值观察的数量。

```py
Percentiles: 25th=46.685, 75th=53.359, IQR=6.674
Identified outliers: 81
Non-outlier observations: 9919
```

通过依次计算数据集中每个变量的限制，并将异常值作为落在矩形或超矩形之外的观察值，可以将该方法用于多变量数据。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   开发自己的高斯测试数据集并在直方图上绘制异常值和非异常值。
*   在使用非高斯分布生成的单变量数据集上测试基于 IQR 的方法。
*   选择一种方法并创建一个函数，该函数将过滤掉具有任意维数的给定数据集的异常值。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何识别数据中的异常值](https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/)

### API

*   [seed（）NumPy API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)
*   [randn（）NumPy API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)
*   [mean（）NumPy API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)
*   [std（）NumPy API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
*   [百分位（）NumPy API](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)

### 用品

*   [维基百科上的异常值](https://en.wikipedia.org/wiki/Outlier)
*   [维基百科上的异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)
*   维基百科上的 [68-95-99.7 规则](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)
*   [四分位数范围](https://en.wikipedia.org/wiki/Interquartile_range)
*   [维基百科上的箱子图](https://en.wikipedia.org/wiki/Box_plot)

### 摘要

在本教程中，您发现了异常值和两种统计方法，可用于从数据集中识别和过滤异常值。

具体来说，你学到了：

*   异常值是数据集中不太可能的观察结果，可能有许多原因之一。
*   该标准偏差可用于识别高斯或类高斯数据中的异常值。
*   无论分布如何，四分位数范围都可用于识别数据中的异常值。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。