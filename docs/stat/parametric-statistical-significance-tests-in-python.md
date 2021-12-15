# 如何在 Python 中计算参数统计显着性检验

> 原文： [https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)

参数统计方法通常意味着假设数据样本具有高斯分布的那些方法。

在应用机器学习中，我们需要比较数据样本，特别是样本的平均值。也许要看一种技术在一个或多个数据集上的表现是否优于另一种技术。为了量化这个问题并解释结果，我们可以使用参数假设检验方法，如 T 检验和方差分析。

在本教程中，您将发现参数统计显着性检验，用于量化两个或多个数据样本均值之间的差异。

完成本教程后，您将了解：

*   Student's t 检验用于量化两个独立数据样本的平均值之间的差异。
*   配对 Student's t 检验用于量化两个依赖数据样本的平均值之间的差异。
*   ANOVA 和重复测量 ANOVA 用于检查 2 个或更多个数据样本的平均值之间的相似性或差异。

让我们开始吧。

*   **更新于 May / 2018** ：改进了拒绝与拒绝统计测试的语言。

![Introduction to Use Parametric Statistical Significance Tests in Python](img/de6aac23c1ed12e90fbf070bb12d77e6.jpg)

Python 中使用参数统计意义测试介绍
照片由 [nestor ferraro](https://www.flickr.com/photos/nestorferraro/11970850885/) 拍摄，保留一些权利。

## 教程概述

1.  参数统计显着性检验
2.  测试数据
3.  学生的 t-测试
4.  配对 T 检验
5.  方差检验分析
6.  重复测量方差分析测试

## 参数统计显着性检验

参数统计测试假设数据样本来自特定的人口分布。

它们通常指的是假设高斯分布的统计检验。由于数据适合此分布是如此常见，因此更常用的是参数统计方法。

我们可能有两个或更多数据样本的典型问题是它们是否具有相同的分布。参数统计显着性检验是假设数据来自相同的高斯分布的统计方法，即具有相同均值和标准差的数据分布：分布的参数。

通常，每个测试计算一个测试统计量，必须用统计学中的一些背景解释并更深入地了解统计测试本身。测试还返回一个 p 值，可用于解释测试结果。 p 值可以被认为是在给定基本假设（零假设）的情况下观察两个数据样本的概率，即两个样本是从具有相同分布的群体中抽取的。

可以在称为α的所选显着性水平的上下文中解释 p 值。 alpha 的常见值为 5％或 0.05。如果 p 值低于显着性水平，则测试表明有足够的证据拒绝零假设，并且样本可能来自具有不同分布的群体。

*   **p &lt;= alpha** ：拒绝零假设，不同分布。
*   **p＆gt; alpha** ：无法拒绝原假设，相同的分布。

## 测试数据

在我们查看特定的参数显着性检验之前，让我们首先定义一个测试数据集，我们可以用它来演示每个测试。

我们将生成从不同分布中抽取的两个样本。每个样本将从高斯分布中抽取。

我们将使用 [randn（）NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)在每个样本中生成 100 个高斯随机数的样本，平均值为 0，标准差为 1.第一个样本中的观察值被缩放为具有均值标准偏差为 50，标准偏差为 5.将第二个样品中的观察结果缩放为平均值为 51，标准偏差为 5。

我们希望统计检验发现样本来自不同的分布，尽管每个样本 100 个观察值的小样本量会给这个决定增加一些噪音。

完整的代码示例如下所示。

```py
# generate gaussian data samples
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
# seed the random number generator
seed(1)
# generate two sets of univariate observations
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
```

运行该示例生成数据样本，然后计算并打印每个样本的均值和标准差，确认它们的不同分布。

```py
data1: mean=50.303 stdv=4.426
data2: mean=51.764 stdv=4.660
```

## 学生的 t-测试

[T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)是一个统计假设检验，两个独立的数据样本已知具有高斯分布，具有相同的高斯分布，以 William Gosset 命名，使用化名“`Student`]“。

> 最常用的 t 检验之一是独立样本 t 检验。当您想要比较给定变量上两个独立样本的均值时，可以使用此测试。

- 第 93 页，[普通英语统计](http://amzn.to/2H8nE7A)，第三版，2010 年。

测试的假设或零假设是两个群体的平均值相等。拒绝这一假设表明，有足够的证据表明人口的均值是不同的，反过来说，分布是不相等的。

*   **无法拒绝 H0** ：样本分布相等。
*   **拒绝 H0** ：样本分布不相等。

学生的 t 检验可通过 [ttest_ind（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)在 Python 中获得。该函数将两个数据样本作为参数，并返回计算的统计量和 p 值。

我们可以证明学生对测试问题的 t 检验，期望测试发现两个独立样本之间的分布差异。完整的代码示例如下所示。

```py
# Student's t-test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = ttest_ind(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行该示例计算生成的数据样本的 Student t 检验并打印统计和 p 值。

对统计量的解释发现样本均值不同，其显着性至少为 5％。

```py
Statistics=-2.262, p=0.025
Different distributions (reject H0)
```

## 配对 T 检验

我们可能希望比较以某种方式相关的两个数据样本之间的平均值。

例如，数据样本可以表示对同一对象的两个独立测量或评估。这些数据样本是重复的或依赖的，并称为配对样本或重复测量。

由于样本不是独立的，我们不能使用学生的 t 检验。相反，我们必须使用测试的修改版本来纠正数据样本依赖的事实，称为配对 Student's t 检验。

> 依赖样本 t 检验也用于比较单个因变量上的两个均值。然而，与独立样本测试不同，依赖样本 t 检验用于比较单个样本或两个匹配或配对样本的平均值。

- 第 94 页，[普通英语统计](http://amzn.to/2H8nE7A)，第三版，2010 年。

该测试被简化，因为它不再假设观察结果之间存在差异，观察是在对同一受试者或受试者进行治疗之前和之后成对进行的。

测试的默认假设或零假设是样本之间的平均值没有差异。拒绝零假设表明有足够的证据表明样本均值不同。

*   **无法拒绝 H0** ：配对样本分布相等。
*   **拒绝 H0** ：配对样本分布不相等。

配对 Student's t 检验可以使用 [ttest_rel（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)在 Python 中实现。与未配对版本一样，该函数将两个数据样本作为参数，并返回计算的统计值和 p 值。

我们可以在测试数据集上演示配对 Student's t 检验。虽然样本是独立的，而不是配对的，但我们可以假装为了证明观察结果并计算统计量。

下面列出了完整的示例。

```py
# Paired Student's t-test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_rel
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = ttest_rel(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行该示例计算并打印测试统计和 p 值。对结果的解释表明样品具有不同的平均值，因此具有不同的分布。

```py
Statistics=-2.372, p=0.020
Different distributions (reject H0)
```

## 方差检验分析

有时我们可能会有多个独立的数据样本。

我们可以对每个数据样本组合成对执行 Student t 检验，以了解哪些样本具有不同的均值。如果我们只关心所有样品是否具有相同的分布，那么这可能是繁重的。

为了回答这个问题，我们可以使用[方差分析](https://en.wikipedia.org/wiki/Analysis_of_variance)或简称 ANOVA。 ANOVA 是一种统计检验，假设两组或更多组的平均值相等。如果证据表明情况并非如此，则拒绝零假设，并且至少一个数据样本具有不同的分布。

*   **无法拒绝 H0** ：所有样本分布均相等。
*   **拒绝 H0** ：一个或多个样本分布不相等。

重要的是，测试只能评论所有样本是否相同;它不能量化哪些样本不同或多少。

> 单因素方差分析（单向 ANOVA）的目的是比较一个因变量上两个或多个组（自变量）的均值，以查看组均值是否彼此显着不同。

- 第 105 页，[普通英语统计](http://amzn.to/2H8nE7A)，第三版，2010 年。

该测试要求数据样本是高斯分布，样本是独立的，并且所有数据样本具有相同的标准偏差。

可以使用 [f_oneway（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)在 Python 中执行 ANOVA 测试。该函数将两个或多个数据样本作为参数，并返回测试统计和 f 值。

我们可以修改我们的测试问题，使两个样本具有相同的均值，第三个样本具有略微不同的均值。然后我们期望测试发现至少一个样本具有不同的分布。

The complete example is listed below.

```py
# Analysis of Variance test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import f_oneway
# seed the random number generator
seed(1)
# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52
# compare samples
stat, p = f_oneway(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行该示例计算并打印测试统计量和 p 值。

对 p 值的解释正确地拒绝了表示一个或多个样本均值不同的零假设。

```py
Statistics=3.655, p=0.027
Different distributions (reject H0)
```

## 重复测量方差分析测试

我们可能有多个相关或依赖的数据样本。

例如，我们可以在不同时间段对主题重复相同的测量。在这种情况下，样本将不再是独立的;相反，我们将有多个配对样本。

我们可以多次重复成对学生的 t 检验。或者，我们可以使用单个测试来检查所有样本是否具有相同的均值。可以使用 ANOVA 测试的变体，修改以测试超过 2 个样品。该测试称为重复测量 ANOVA 测试。

默认假设或零假设是所有配对样本具有相同的均值，因此具有相同的分布。如果样本表明情况并非如此，则拒绝零假设，并且一个或多个配对样本具有不同的均值。

*   **无法拒绝 H0** ：所有配对样本分布均相等。
*   **拒绝 H0** ：一个或多个配对样本分布不相等。

> 然而，重复测量 ANOVA 与配对 t 检验相比具有许多优点。首先，通过重复测量 ANOVA，我们可以检查在两个以上时间点测量的因变量的差异，而使用独立 t 检验，我们只能比较两个时间点的因变量的得分。

- 第 131 页，[普通英语统计](http://amzn.to/2H8nE7A)，第三版，2010 年。

不幸的是，在撰写本文时，SciPy 中没有可用的重复测量 ANOVA 测试版本。希望这个测试很快就会加入。

如果您在项目中需要它并且能够寻找并找到替代实现，我提到此测试是否完整。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   更新所有示例以对具有相同分布的数据样本进行操作。
*   根据每个测试的要求和行为，创建一个流程图，用于选择三个统计显着性检验中的每一个。
*   考虑 3 个在机器学习项目中比较数据样本的情况，假设样本的非高斯分布，并建议可以在每种情况下使用的测试类型。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [普通英语统计](http://amzn.to/2H8nE7A)，第三版，2010 年。

### API

*   [numpy.random.seed（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)
*   [numpy.random.randn（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)
*   [scipy.stats.ttest_ind（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
*   [scipy.stats.ttest_rel（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
*   [scipy.stats.f_oneway（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)

### 用品

*   [维基百科的统计意义](https://en.wikipedia.org/wiki/Statistical_significance)
*   [维基百科上的 T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)
*   [维基百科上的配对差异测试](https://en.wikipedia.org/wiki/Paired_difference_test)
*   [维基百科](https://en.wikipedia.org/wiki/Analysis_of_variance)的方差分析
*   [在维基百科](https://en.wikipedia.org/wiki/Repeated_measures_design)上重复测量设计

## 摘要

在本教程中，您发现了参数统计显着性检验，用于量化两个或多个数据样本均值之间的差异。

具体来说，你学到了：

*   Student's t 检验用于量化两个独立数据样本的平均值之间的差异。
*   配对 Student's t 检验用于量化两个依赖数据样本的平均值之间的差异。
*   ANOVA 和重复测量 ANOVA 用于检查两个或更多个数据样本的均值之间的相似性或差异。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。