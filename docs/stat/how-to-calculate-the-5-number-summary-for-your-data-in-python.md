# 如何在Python中计算数据的5位数摘要

> 原文： [https://machinelearningmastery.com/how-to-calculate-the-5-number-summary-for-your-data-in-python/](https://machinelearningmastery.com/how-to-calculate-the-5-number-summary-for-your-data-in-python/)

数据汇总提供了一种方便的方法来描述数据样本中的所有值，只需几个统计值。

平均值和标准差用于汇总具有高斯分布的数据，但如果您的数据样本具有非高斯分布，则可能没有意义，甚至可能具有误导性。

在本教程中，您将发现用于描述数据样本分布的五个数字摘要，而不假设特定的数据分布。

完成本教程后，您将了解：

*   数据汇总（例如计算均值和标准差）仅对高斯分布有意义。
*   五个数字摘要可用于描述具有任何分布的数据样本。
*   如何计算Python中的五位数摘要。

让我们开始吧。

![How to Calculate the 5-Number Summary for Your Data in Python](img/db9b0efb2136a8324488a6f902d7c46f.jpg)

如何在Python中计算数据的5位数摘要
照片由 [Masterbutler](https://www.flickr.com/photos/alwbutler/7456018222/) ，保留一些权利。

## 教程概述

本教程分为4个部分;他们是：

1.  非参数数据摘要
2.  五位数总结
3.  如何计算五位数汇总
4.  使用五位数摘要

## 非参数数据摘要

数据汇总技术提供了一种使用一些关键测量来描述数据分布的方法。

最常见的数据汇总示例是计算具有高斯分布的数据的均值和标准差。仅使用这两个参数，您就可以理解并重新创建数据的分布。数据摘要可以压缩几十或几百万个别观察。

问题是，您不能轻易计算出没有高斯分布的数据的均值和标准差。从技术上讲，您可以计算这些数量，但它们并未总结数据分布;事实上，他们可能会产生误导。

对于没有高斯分布的数据，您可以使用五个数字摘要汇总数据样本。

## 五位数总结

五个数字摘要或简称为5个数字的摘要是非参数数据摘要技术。

它有时被称为Tukey 5号码摘要，因为它是由John Tukey推荐的。它可用于描述具有任何分布的数据的数据样本的分布。

> 作为一般用途的标准摘要，5号摘要提供了正确的详细信息量。

- 第37页，[理解稳健和探索性数据分析](https://amzn.to/2Gp2sNW)，2000。

五位数摘要涉及5个汇总统计量的计算：即：

*   **中位数**：样本中的中间值，也称为第50个百分位数或第2个四分位数。
*   **第一四分位数**：第25个百分点。
*   **第三四分位数**：第75个百分点。
*   **最小**：样本中的最小观察值。
*   **最大值**：样本中最大的观察值。

四分位数是一个点上的观察值，有助于将有序数据样本分成四个大小相等的部分。中位数或第二四分位数将有序数据样本分成两部分，第一和第三四分位数将这些半部分分成四部分。

百分位数是在有助于将有序数据样本分成100个相等大小的部分的点处的观察值。四分位数通常也表示为百分位数。

四分位数和百分位数值都是排名统计的示例，可以在具有任何分布的数据样本上计算。它们用于快速总结分布中有多少数据落后于给定观察值。例如，一半的观​​察结果落在分布的中位数之前和之前。

注意，四分位数也在[框和须状图](https://en.wikipedia.org/wiki/Box_plot)中计算，这是一种非参数方法，用于图形化地总结数据样本的分布。

## 如何计算五位数汇总

计算五位数摘要包括查找每个四分位数的观测值以及数据样本的最小和最大观测值。

如果四分位数的有序数据样本中没有特定值，例如，如果有偶数个观察值并且我们试图找到中位数，那么我们可以计算两个最接近的值的平均值，例如两个中间价值观。

我们可以使用[百分位（）](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html) NumPy函数在Python中计算任意百分位数值。我们可以使用此函数来计算第1，第2（中位数）和第3四分位数值。该函数采用观察数组和浮点值来指定要在0到100范围内计算的百分位数。它还可以采用百分位数值列表来计算多个百分位数;例如：

```py
quartiles = percentile(data, [25, 50, 75])
```

默认情况下，如果需要，函数将计算观察值之间的线性插值（平均值），例如在计算具有偶数值的样本的中值的情况下。

NumPy函数min（）和max（）可用于返回数据样本中的最小值和最大值;例如：

```py
data_min, data_max = data.min(), data.max()
```

我们可以把所有这些放在一起。

下面的示例生成从0到1之间的均匀分布绘制的数据样本，并使用五个数字摘要对其进行汇总。

```py
# calculate a 5-number summary
from numpy import percentile
from numpy.random import rand
# generate data sample
data = rand(1000)
# calculate quartiles
quartiles = percentile(data, [25, 50, 75])
# calculate min/max
data_min, data_max = data.min(), data.max()
# print 5-number summary
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)
```

运行该示例将生成数据样本并计算五个数字摘要以描述样本分布。

我们可以看到观察的传播接近我们的预期，第50百分位数为0.27，第50百分位数为0.53，第75百分位数为0.76，接近理想值分别为0.25,0.50和0.75。

```py
Min: 0.000
Q1: 0.277
Median: 0.532
Q3: 0.766
Max: 1.000
```

## 使用五位数摘要

可以针对具有任何分布的数据样本计算五个数字摘要。

这包括具有已知分布的数据，例如高斯分布或类高斯分布。

我建议总是计算五个数字的摘要，并且只能继续分发特定的摘要，例如高斯的均值和标准差，以便您可以识别数据所属的分布。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   在机器学习项目中描述三个示例，其中可以计算五个数字摘要。
*   生成具有高斯分布的数据样本并计算五个数字摘要。
*   编写一个函数来计算任何数据样本的5个数字摘要。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [理解稳健和探索性数据分析](https://amzn.to/2Gp2sNW)，2000。

### API

*   [numpy.percentile（）API](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)
*   [numpy.ndarray.min（）API](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.ndarray.min.html)
*   [numpy.ndarray.max（）API](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.ndarray.max.html)

### 用品

*   [维基百科上的五个数字摘要](https://en.wikipedia.org/wiki/Five-number_summary)
*   [维基百科上的四分之一](https://en.wikipedia.org/wiki/Quartile)
*   [维基百科上的百分位数](https://en.wikipedia.org/wiki/Percentile)

## 摘要

在本教程中，您发现了五个数字摘要，用于描述数据样本的分布，而不假设特定的数据分布。

具体来说，你学到了：

*   数据汇总（例如计算均值和标准差）仅对高斯分布有意义。
*   五个数字摘要可用于描述具有任何分布的数据样本。
*   如何计算Python中的五位数摘要。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。