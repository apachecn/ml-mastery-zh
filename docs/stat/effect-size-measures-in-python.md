# Python中效果大小度量的温和介绍

> 原文： [https://machinelearningmastery.com/effect-size-measures-in-python/](https://machinelearningmastery.com/effect-size-measures-in-python/)

统计假设检验报告了假设观察结果的可能性，例如变量之间没有关联或组间没有差异。

如果关联或差异具有统计学意义，则假设检验不会评估效果的大小。这突出了对计算和报告结果的标准方法的需求。

效应大小方法是指用于量化实验结果中效应大小的一套统计工具，可用于补充统计假设检验的结果。

在本教程中，您将发现用于量化结果大小的效果大小和效果大小度量。

完成本教程后，您将了解：

*   在实验结果中计算和报告效应大小的重要性。
*   效应大小度量用于量化变量之间的关联，例如Pearson相关系数。
*   用于量化组间差异的效应大小度量，例如Cohen的d度量。

让我们开始吧。

![A Gentle Introduction to Effect Size Measures in Python](img/d61aa2b531f0504780eb0b0103ef13e6.jpg)

Python中效果大小测量的温和介绍
[scott1346](https://www.flickr.com/photos/bluecorvette/8345369485/) 的照片，保留一些权利。

## 教程概述

本教程分为三个部分;他们是：

1.  需要报告效果大小
2.  什么是效果大小？
3.  如何计算效果大小

## 需要报告效果大小

一旦从业者熟悉统计方法，通常会专注于量化结果的可能性。

这通常可以通过统计假设检验的结果的计算和表示来看待p值和显着性水平。

在结果呈现中经常被忽略的一个方面是实际量化差异或关系，称为效果。很容易忘记实验的目的是量化效果。

> 研究调查的主要产品是效果大小的一个或多个度量，而不是P值。

- [我学到的东西（到目前为止）](https://tech.me.holycross.edu/files/2015/03/Cohen_1990.pdf)，1990。

统计检验只能评论是否存在影响。它没有评论效果的大小。实验结果可能很重要，但影响很小，几乎没有后果。

> 结果在统计上显着且微不足道是可能的，并且不幸的是很常见。结果在统计上也可能是非显着且重要的。

- 第4页，[影响大小的基本指南：统计力量，Meta分析和研究结果的解释](https://amzn.to/2JDcwSe)，2010。

忽略效果呈现的问题在于它可以使用临时测量来计算或者甚至完全被忽略并留给读者解释。这是一个很大的问题，因为量化效果的大小对于解释结果至关重要。

## 什么是效果大小？

效应大小是指预期在群体中发生的效果或结果的大小或大小。

效果大小是根据数据样本估算的。

效果大小方法是指用于计算效果大小的统计工具的集合。通常，效应大小测量的领域被简称为“_效应大小_”，以注意该领域的普遍关注。

基于要量化的效果类型，将效应大小统计方法组织成组是常见的。计算效果大小的两组主要方法是：

*   **协会**。用于量化变量之间的关联的统计方法（例如，相关性）。
*   **差异**。量化变量之间差异的统计方法（例如均值之间的差异）。

> 效果可以是在组（例如治疗组和未治疗组）之间比较中显示的治疗结果，或者它可以描述两个相关变量（例如治疗剂量和健康）之间的关联程度。

- 第5页，[影响大小的基本指南：统计力量，Meta分析和研究结果的解释](https://amzn.to/2JDcwSe)，2010。

必须解释效果大小计算的结果，并且它取决于所使用的特定统计方法。必须根据解释的目标选择一项措施。三种计算结果包括：

*   **标准化结果**。效果大小具有标准比例，允许一般地解释它而不管应用（例如，科恩的计算）。
*   **原始单位结果**。效果大小可以使用变量的原始单位，这可以有助于域内的解释（例如，两个样本均值之间的差异）。
*   **单位免费结果**。效果大小可以不具有诸如计数或比例（例如，相关系数）的单位。

> 因此，效果大小可以指群组平均值或绝对效果大小之间的原始差异，以及标准化效果度量，其被计算以将效果转换为易于理解的量表。当研究中的变量具有内在意义（例如，睡眠小时数）时，绝对效应大小是有用的。

- [使用效果大小 - 或为什么P值不够](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/)，2012。

使用多种方法报告效果大小以帮助您的研究结果的不同类型的读者可能是个好主意。

> 有时，最好以原始单位报告结果，以便于读者理解，以及在未来的元分析中易于包含的一些标准化措施。

- 第41页，[了解新统计：影响大小，置信区间和元分析](https://amzn.to/2v0wKSI)，2011。

效应大小不会取代统计假设检验的结果。相反，效果大小补充了测试。理想情况下，假设检验和效应大小计算的结果将并排显示。

*   **假设检验**：量化给定假设（零假设）观察数据的可能性。
*   **效果大小**：假设效果存在，量化效果的大小。

## 如何计算效果大小

效应大小的计算可以是样本平均值的计算或两个平均值之间的绝对差值。它也可以是更精细的统计计算。

在本节中，我们将查看关联和差异的一些常见效果大小计算。方法的例子不完整;可能有100种方法可用于计算效果大小。

### 计算关联效果大小

变量之间的关联通常被称为效应大小方法的“ _r家族_”。

这个名字来自于计算效果大小的最常用方法，称为Pearson相关系数，也称为Pearson's r。

Pearson相关系数衡量两个实值变量之间的线性关联程度。它是一种无单位效应大小度量，可以按标准方式解释，如下所示：

*   -1.0：完美的负面关系。
*   -0.7：强烈的负面关系
*   -0.5：适度的负面关系
*   -0.3：弱关系
*   0.0：没有关系。
*   0.3：弱关系
*   0.5：适度的积极关系
*   0.7：积极的关系
*   1.0：完美的积极关系。

Pearson的相关系数可以使用 [pearsonr（）SciPy函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)在Python中计算。

下面的例子演示了Pearson相关系数的计算，以量化两个随机高斯数样本之间关联的大小，其中一个样本与第二个样本有很强的关系。

```py
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
# seed random number generator
seed(1)
# prepare data
data1 = 10 * randn(10000) + 50
data2 = data1 + (10 * randn(10000) + 50)
# calculate Pearson's correlation
corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)
```

运行该示例计算并打印两个数据样本之间的Pearson相关性。我们可以看到效果显示样本之间存在强烈的正相关关系。

```py
Pearson’s correlation: 0.712
```

用于计算关联效应大小的另一种非常流行的方法是r平方测量，或r ^ 2，也称为确定系数。它总结了另一个变量解释的一个变量的方差比例。

### 计算差异效应大小

组之间的差异通常被称为效应大小方法的“ _d家族_”。

这个名字来自于计算群体平均值之间差异的最常用方法，称为Cohen's d。

科恩测量两个高斯分布变量的均值之间的差异。它是一个标准分数，总结了标准偏差数量的差异。由于分数是标准化的，因此有一个表格可以解释结果，总结如下：

*   **小效应尺寸**：d = 0.20
*   **中等效应大小**：d = 0.50
*   **大效应尺寸**：d = 0.80

Python中没有提供Cohen的计算;我们可以手动计算。

计算两个样本的平均值之间的差异如下：

```py
d = (u1 - u2) / s
```

_d_ 是Cohen的d，`u1`是第一个样本的平均值，`u2`是第二个样本的平均值，`s`]是两个样品的合并标准偏差。

两个独立样本的合并标准偏差可以计算如下：

```py
s = sqrt(((n1 - 1) . s1^2 + (n2 - 1) . s2^2) / (n1 + n2 - 2))
```

是合并的标准偏差，`n1`和`n2`是第一个样本和第二个样本和 _s1 ^ 2_ 的大小和 _s2 ^ 2_ 是第一个和第二个样本的方差。减法是对自由度数的调整。

下面的函数将计算两个实值变量样本的Cohen d度量。 NumPy函数 [mean（）](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)和 [var（）](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)分别用于计算样本均值和方差。

```py
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s
```

下面的例子计算了具有不同均值的两个随机高斯变量样本的Cohen d度量。

设计该实例使得平均值相差半个标准差，并且两个样本具有相同的标准偏差。

```py
# calculate the Cohen's d between two samples
from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

# seed random number generator
seed(1)
# prepare data
data1 = 10 * randn(10000) + 60
data2 = 10 * randn(10000) + 55
# calculate cohen's d
d = cohend(data1, data2)
print('Cohens d: %.3f' % d)
```

运行该示例计算并打印Cohen的效果大小。

我们可以看到，正如预期的那样，均值之间的差异是被解释为中等效应大小的一个标准差的一半。

```py
Cohen's d: 0.500
```

量化差异效应大小的另外两种流行方法是：

*   **优势比**。测量一种治疗结果与另一种治疗相比的可能性。
*   **相对风险比**。测量一种治疗方法与另一种治疗方法相比的可能性。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   在研究论文中查找报告效果大小以及统计显着性结果的示例。
*   实现一个函数来计算配对样本的Cohen d并在测试数据集上进行演示。
*   实施并演示另一种差异效应指标，例如赔率或风险比率。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [使用效果大小 - 或为什么P值不够](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/)，2012。
*   [我学到的东西（到目前为止）](https://tech.me.holycross.edu/files/2015/03/Cohen_1990.pdf)，1990。

### 图书

*   [影响大小的基本指南：统计力量，Meta分析和研究结果的解释](https://amzn.to/2JDcwSe)，2010。
*   [了解新统计：影响大小，置信区间和元分析](https://amzn.to/2v0wKSI)，2011。
*   [行为科学的统计功效分析](https://amzn.to/2GNcmtu)，1988。

### API

*   [scipy.stats.pearsonr（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
*   [numpy.var（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)
*   [numpy.mean（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)

### 用品

*   [维基百科上的效果大小](https://en.wikipedia.org/wiki/Effect_size)

## 摘要

在本教程中，您发现了用于量化结果大小的效果大小和效果大小度量。

具体来说，你学到了：

*   在实验结果中计算和报告效应大小的重要性。
*   效应大小度量用于量化变量之间的关联，例如Pearson相关系数。
*   用于量化组间差异的效应大小度量，例如Cohen的d度量。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。