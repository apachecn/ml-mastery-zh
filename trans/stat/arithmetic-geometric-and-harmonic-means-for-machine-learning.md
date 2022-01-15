# 机器学习的算术、几何和调和方法

> 原文:[https://machinelearning master . com/算术-几何和调和-机器学习平均值/](https://machinelearningmastery.com/arithmetic-geometric-and-harmonic-means-for-machine-learning/)

最后更新于 2020 年 8 月 19 日

计算变量或数字列表的平均值是机器学习中常见的操作。

这是一项您可以每天直接使用的操作，例如汇总数据时，也可以间接使用，例如拟合模型时大型过程中的一个较小步骤。

平均值是平均值的同义词，平均值是一个代表概率分布中最可能值的数字。因此，根据您使用的数据类型，有多种不同的方法来计算平均值。

如果你对数据使用了错误的平均值，这可能会让你出错。在使用性能指标评估模型时，您也可以输入一些更奇特的平均值计算，例如 G 均值或 F 度量。

在本教程中，您将发现算术平均值、几何平均值和调和平均值之间的差异。

完成本教程后，您将知道:

*   中心趋势概括了变量最可能的值，平均值是计算平均值的常用名称。
*   如果值具有相同的单位，算术平均值是合适的，而如果值具有不同的单位，几何平均值是合适的。
*   如果数据值是具有不同度量的两个变量(称为比率)的比率，则调和平均值是合适的。

**用我的新书[机器学习统计](https://machinelearningmastery.com/statistics_for_machine_learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Arithmetic, Geometric, and Harmonic Means for Machine Learning](img/368512aea4f3c78150d356d943f50601.png)

机器学习的算术、几何和调和方法
摄影:T2 马尼拉，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  什么是平均值？
2.  等差中项
3.  几何平均值
4.  调和平均值
5.  如何选择正确的平均值？

## 什么是平均值？

[中心趋势](https://en.wikipedia.org/wiki/Central_tendency)是一个单一的数字，代表了数字列表中最常见的值。

更确切地说，是[概率分布](https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/)中概率最高的值描述了一个变量可能具有的所有可能值。

有许多方法可以计算数据样本的中心趋势，例如根据值计算的**平均值**、**模式**，这是数据分布中最常见的值，或者**中值**，如果数据样本中的所有值都是有序的，这是中间值。

[平均值](https://en.wikipedia.org/wiki/Average)是平均值的常用术语。它们可以互换使用。

平均值不同于中位数和模式，因为它是根据数据计算的中心趋势的度量。因此，根据数据类型计算平均值有不同的方法。

你可能会遇到的三种常见的平均值计算类型是**算术平均值**、**几何平均值**和**调和平均值**。还有其他手段，还有很多更中心的倾向度量，但这三种手段也许是最常见的(例如所谓的[毕达哥拉斯的意思](https://en.wikipedia.org/wiki/Pythagorean_means))。

让我们依次仔细看看平均值的每个计算。

## 等差中项

[算术平均值](https://en.wikipedia.org/wiki/Arithmetic_mean)计算为数值之和除以数值总数，称为 n

*   算术平均值= (x1 + x2 + … + xN) / N

计算算术平均值的一种更方便的方法是计算值的总和，并将其乘以值的倒数(1 比 N)；例如:

*   算术平均值= (1/N) * (x1 + x2 + … + xN)

当数据样本中的所有值都具有相同的度量单位时，算术平均值是合适的，例如所有数字都是高度、美元或英里等。

计算算术平均值时，值可以是正数、负数或零。

如果观测值样本包含异常值(特征空间中远离所有其他值的几个值)，或者对于具有非高斯分布(例如多个峰值，即所谓的多模态概率分布)的数据，算术平均值很容易失真。

算术平均值在机器学习中总结变量时很有用，例如报告最可能的值。当变量具有高斯或类似高斯的数据分布时，这更有意义。

算术平均值可以使用[平均值()NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)来计算。

下面的例子演示了如何计算 10 个数的算术平均值。

```
# example of calculating the arithmetic mean
from numpy import mean
# define the dataset
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# calculate the mean
result = mean(data)
print('Arithmetic Mean: %.3f' % result)
```

运行该示例计算算术平均值并报告结果。

```
Arithmetic Mean: 4.500
```

## 几何平均值

[几何平均值](https://en.wikipedia.org/wiki/Geometric_mean)计算为所有值乘积的第 N 根，其中 N 为值的个数。

*   几何平均值= N 根(x1 * x2 *……* xN)

例如，如果数据只包含两个值，则这两个值的乘积的平方根就是几何平均值。对于三个值，使用立方根，依此类推。

当数据包含具有不同度量单位的值时，几何平均值是合适的，例如，一些度量是高度，一些是美元，一些是英里等。

几何平均值不接受负值或零值，例如所有值都必须为正值。

机器学习中几何平均的一个常见例子是所谓的 [G-Mean](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) (几何平均)度量的计算，该度量是模型评估度量，被计算为灵敏度和特异性度量的几何平均。

可以使用 [gmean() SciPy 函数](https://scipy.github.io/devdocs/generated/scipy.stats.gmean.html)计算几何平均值。

下面的例子演示了如何计算 10 个数的几何平均值。

```
# example of calculating the geometric mean
from scipy.stats import gmean
# define the dataset
data = [1, 2, 3, 40, 50, 60, 0.7, 0.88, 0.9, 1000]
# calculate the mean
result = gmean(data)
print('Geometric Mean: %.3f' % result)
```

运行该示例计算几何平均值并报告结果。

```
Geometric Mean: 7.246
```

## 调和平均值

谐波平均值的计算方法是数值数量 *N* 除以数值倒数之和(每个数值 1)。

*   谐波平均值= N / (1/x1 + 1/x2 + … + 1/xN)

如果只有两个值(x1 和 x2)，谐波平均值的简化计算公式如下:

*   调和平均值= (2 * x1 * x2) / (x1 + x2)

如果数据由速率组成，谐波平均值是合适的平均值。

回想一下，a [率](https://en.wikipedia.org/wiki/Rate_(mathematics))是两个具有不同度量的量之间的比率，例如速度、加速度、频率等。

在机器学习中，我们在评估模型时会有比率，例如预测中的真阳性率或假阳性率。

谐波平均值不采用负值或零值的速率，例如，所有速率都必须为正值。

在机器学习中使用调和平均值的一个常见例子是计算[F-测度](https://en.wikipedia.org/wiki/F1_score)(也是 F1-测度或 Fbeta-测度)；这是一个模型评估指标，计算为精确度和召回率指标的调和平均值。

可以使用 [hmean() SciPy 函数](https://scipy.github.io/devdocs/generated/scipy.stats.hmean.html)计算谐波平均值。

下面的例子演示了如何计算九个数的调和平均值。

```
# example of calculating the harmonic mean
from scipy.stats import hmean
# define the dataset
data = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
# calculate the mean
result = hmean(data)
print('Harmonic Mean: %.3f' % result)
```

运行该示例计算调和平均值并报告结果。

```
Harmonic Mean: 0.350
```

## 如何选择正确的平均值？

我们回顾了计算变量或数据集平均值的三种不同方法。

算术平均值是最常用的平均值，尽管在某些情况下可能不合适。

每种方法适用于不同类型的数据；例如:

*   **如果值具有相同的单位**:使用算术平均值。
*   **如果数值有不同的单位**:使用几何平均值。
*   **如果数值是速率**:使用谐波平均值。

例外情况是，如果数据包含负值或零值，则不能直接使用几何和调和平均值。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 蜜蜂

*   [numpy.mean API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html) 。
*   [scipy.stats.gmean API](https://scipy.github.io/devdocs/generated/scipy.stats.gmean.html) 。
*   [scipy . stats . hmen API](https://scipy.github.io/devdocs/generated/scipy.stats.hmean.html)。

### 文章

*   [平均值，维基百科](https://en.wikipedia.org/wiki/Average)。
*   [中央倾向，维基百科](https://en.wikipedia.org/wiki/Central_tendency)。
*   [算术平均值，维基百科](https://en.wikipedia.org/wiki/Arithmetic_mean)。
*   [几何平均值，维基百科](https://en.wikipedia.org/wiki/Geometric_mean)。
*   [调和的意思，维基百科](https://en.wikipedia.org/wiki/Harmonic_mean)。

## 摘要

在本教程中，您发现了算术平均值、几何平均值和调和平均值之间的差异。

具体来说，您了解到:

*   中心趋势概括了变量最可能的值，平均值是计算平均值的常用名称。
*   如果值具有相同的单位，算术平均值是合适的，而如果值具有不同的单位，几何平均值是合适的。
*   如果数据值是具有不同度量的两个变量(称为比率)的比率，则调和平均值是合适的。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。