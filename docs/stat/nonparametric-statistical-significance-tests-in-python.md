# Python中非参数统计显着性检验简介

> 原文： [https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)

在应用机器学习中，我们经常需要确定两个数据样本是否具有相同或不同的分布。

我们可以使用统计显着性检验来回答这个问题，这些检验可以量化样本具有相同分布的可能性。

如果数据没有熟悉的高斯分布，我们必须求助于非参数版本的显着性检验。这些测试以类似的方式操作，但是无分布，要求在可以执行测试之前首先将实值数据转换为等级数据。

在本教程中，您将发现非参数统计测试，您可以使用它来确定数据样本是否来自具有相同或不同分布的群体。

完成本教程后，您将了解：

*   用于比较独立数据样本的Mann-Whitney U检验：Student t检验的非参数版本。
*   用于比较配对数据样本的Wilcoxon符号秩检验：配对Student t检验的非参数版本。
*   Kruskal-Wallis H和Friedman测试用于比较两个以上的数据样本：ANOVA的非参数版本和重复测量ANOVA测试。

让我们开始吧。

*   **更新于May / 2018** ：改进了拒绝与拒绝统计测试的语言。

![Introduction to Nonparametric Statistical Significance Tests in Python](img/1a8e46cdba4be3371ae1f4c72c30b6d6.jpg)

Python中的非参数统计显着性检验介绍
[Jirka Matousek](https://www.flickr.com/photos/jirka_matousek/9220286695/) 的照片，保留一些权利。

## 教程概述

本教程分为6个部分;他们是：

1.  非参数统计显着性检验
2.  测试数据
3.  Mann-Whitney U测试
4.  威尔科克森签名等级测试
5.  Kruskal-Wallis H测试
6.  弗里德曼测试

## 非参数统计显着性检验

[非参数统计](https://en.wikipedia.org/wiki/Nonparametric_statistics)是那些不假设对数据进行特定分配的方法。

通常，它们指的是不采用高斯分布的统计方法。它们被开发用于序数或区间数据，但实际上也可以用于数据样本中的实值观测值的排序，而不是观察值本身。

关于两个或更多数据集的常见问题是它们是否不同。具体而言，它们的集中趋势（例如平均值或中值）之间的差异是否具有统计学意义。

对于没有使用非参数统计显着性检验的高斯分布的数据样本，可以回答这个问题。这些测试的零假设通常假设两个样本都来自具有相同分布的群体，因此具有相同的群体参数，例如平均值或中值。

如果在计算两个或更多样本的显着性检验后，零假设被拒绝，则表明有证据表明样本来自不同的群体，反过来是样本估计的人口参数之间的差异，例如平均值或中位数可能很重要。

这些测试通常用于模型技能分数的样本，以确认机器学习模型之间的技能差异是显着的。

通常，每个测试计算一个测试统计量，必须用统计学中的一些背景解释并对统计测试本身有更深入的了解。测试还返回一个p值，可用于解释测试结果。 p值可以被认为是在给定基本假设（零假设）的情况下观察两个数据样本的概率，即两个样本是从具有相同分布的群体中抽取的。

可以在称为α的所选显着性水平的上下文中解释p值。 alpha的常见值为5％或0.05。如果p值低于显着性水平，则测试表明有足够的证据拒绝零假设，并且样本可能来自具有不同分布的群体。

*   **p &lt;= alpha** ：拒绝H0，分布不同。
*   **p＆gt; alpha** ：无法拒绝H0，相同的分布。

## 测试数据集

在我们查看特定的非参数重要性测试之前，让我们首先定义一个测试数据集，我们可以用它来演示每个测试。

我们将生成从不同分布中抽取的两个样本。为简单起见，我们将从高斯分布中抽取样本，尽管如上所述，我们在本教程中审查的测试是针对我们不知道或假设任何特定分布的数据样本。

我们将使用 [randn（）NumPy函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)在每个样本中生成100个高斯随机数的样本，平均值为0，标准差为1.第一个样本中的观察值被缩放为具有均值标准偏差为50，标准偏差为5.将第二个样品中的观察结果缩放为平均值为51，标准偏差为5。

我们希望统计检验发现样本来自不同的分布，尽管每个样本100个观察值的小样本量会给这个决定增加一些噪音。

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

## Mann-Whitney U测试

Mann-Whitney U检验是一种非参数统计显着性检验，用于确定是否从具有相同分布的群体中抽取两个独立样本。

该测试以Henry Mann和Donald Whitney的名字命名，虽然它有时被称为Wilcoxon-Mann-Whitney测试，也以Frank Wilcoxon命名，他也开发了测试的变体。

> 将两个样本组合在一起并排序。策略是确定来自两个样本的值是否在等级排序中随机混合，或者如果它们在组合时聚集在相对的末端。随机排名顺序意味着两个样本没有不同，而一个样本值的集群将指示它们之间的差异。

- 第58页，[非统计学家的非参数统计：循序渐进的方法](http://amzn.to/2CZcXBz)，2009年。

默认假设或零假设是数据样本的分布之间没有差异。拒绝这一假设表明样本之间可能存在一些差异。更具体地，该测试确定来自一个样本的任何随机选择的观察是否同样可能大于或小于另一个分布中的样本。如果违反，则表明不同的分布。

*   **无法拒绝H0** ：样本分布相等。
*   **拒绝H0** ：样本分布不相等。

为使测试有效，每个数据样本至少需要20次观察。

我们可以使用 [mannwhitneyu（）SciPy函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)在Python中实现Mann-Whitney U测试。这些函数将两个数据样本作为参数。它返回测试统计和p值。

下面的示例演示了测试数据集上的Mann-Whitney U测试。

```py
# Mann-Whitney U test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
```

运行该示例计算数据集上的测试并打印统计量和p值。

p值强烈表明样本分布是不同的，如预期的那样。

```py
Statistics=4025.000, p=0.009
Different distribution (reject H0)
```

## 威尔科克森签名等级测试

在某些情况下，数据样本可以配对。

有许多原因可能是这种情况，例如，样品是相关的或[以某种方式匹配](https://en.wikipedia.org/wiki/Paired_difference_test)或代表相同技术的两次测量。更具体地，每个样本是独立的，但来自相同的群体。

机器学习中的配对样本的示例可以是在不同数据集上评估的相同算法，或者在完全相同的训练和测试数据上评估的不同算法。

样品不是独立的，因此不能使用Mann-Whitney U检验。相反，使用 [Wilcoxon符号秩检验](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)，也称为Wilcoxon T检验，以Frank Wilcoxon命名。它相当于配对T检验，但是对于排序数据而不是具有高斯分布的实值数据。

> Wilcoxon签名等级测试是一种非参数统计程序，用于比较配对或相关的两个样本。与Wilcoxon签名等级测试相对应的参数等名称包括Student's t检验，匹配对的t检验，配对样本的t检验或依赖样本的t检验。

- 第38-39页，[非统计学家的非参数统计：循序渐进的方法](http://amzn.to/2CZcXBz)，2009。

测试的默认假设，即零假设，即两个样本具有相同的分布。

*   **无法拒绝H0** ：样本分布相等。
*   **拒绝H0** ：样本分布不相等。

For the test to be effective, it requires at least 20 observations in each data sample.

Wilcoxon符号秩检验可以使用 [wilcoxon（）SciPy函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)在Python中实现。该函数将两个样本作为参数，并返回计算的统计量和p值。

下面是完整的示例，演示了对测试问题的Wilcoxon符号秩检验的计算。这两个样本在技术上不配对，但我们可以假装它们是为了证明这个重要性测试的计算。

```py
# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = wilcoxon(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
```

运行该示例计算并打印统计量并打印结果。

p值被强烈解释，表明样本是从不同的分布中提取的。

```py
Statistics=1886.000, p=0.028
Different distribution (reject H0)
```

## Kruskal-Wallis H测试

当使用显着性检验时，例如Mann-Whitney U和Wilcoxon符号秩检验，数据样本之间的比较必须成对进行。

如果您有许多数据样本并且您只对两个或更多样本是否具有不同的分布感兴趣，则这可能是低效的。

Kruskal-Wallis检验是单因素方差分析或简称ANOVA的非参数版本。它以该方法的开发者命名，William Kruskal和Wilson Wallis。该测试可用于确定两个以上的独立样本是否具有不同的分布。它可以被认为是Mann-Whitney U检验的推广。

默认假设或零假设是所有数据样本都来自同一分布。具体而言，所有群体的人口中位数相等。拒绝零假设表明有足够的证据表明一个或多个样本支配另一个样本，但测试并未指出哪些样本或多少样本。

> 当Kruskal-Wallis H-检验导致显着结果时，则至少一个样品与其他样品不同。但是，测试不能确定差异发生的位置。而且，它没有确定出现了多少差异。为了确定样本对之间的特定差异，研究人员可以使用样本对比或事后测试来分析特定样本对的显着差异。 Mann-Whitney U检验是在各个样本集之间进行样本对比的有用方法。

- 第100页，[非统计学家的非参数统计：循序渐进的方法](http://amzn.to/2CZcXBz)，2009。

*   **无法拒绝H0** ：所有样本分布均相等。
*   **拒绝H0** ：一个或多个样本分布不相等。

每个数据样本必须是独立的，具有5个或更多个观察值，并且数据样本的大小可以不同。

我们可以更新测试问题以获得3个数据样本，而不是2个，其中两个具有相同的样本均值。鉴于一个样本不同，我们期望测试发现差异并拒绝原假设。

```py
# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52
```

Kruskal-Wallis H-test可以使用 [kruskal（）SciPy函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)在Python中实现。它将两个或多个数据样本作为参数，并返回测试统计量和p值作为结果。

下面列出了完整的示例。

```py
# Kruskal-Wallis H-test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import kruskal
# seed the random number generator
seed(1)
# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52
# compare samples
stat, p = kruskal(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行该示例计算测试并打印结果。

解释p值，正确地拒绝所有样本具有相同分布的零假设。

```py
Statistics=6.051, p=0.049
Different distributions (reject H0)
```

## 弗里德曼测试

与前面的示例一样，我们可能有两个以上的不同样本，并且对所有样本是否具有相同分布感兴趣。

如果样品以某种方式配对，例如重复测量，那么Kruskal-Wallis H测试将是不合适的。相反，可以使用 [Friedman测试](https://en.wikipedia.org/wiki/Friedman_test)，命名为Milton Friedman。

弗里德曼检验是方差检验的重复测量分析的非参数形式，或重复测量方差分析。该测试可以被认为是对两个以上样品的Kruskal-Wallis H检验的推广。

默认假设或零假设是多个配对样本具有相同的分布。拒绝零假设表明配对样本中的一个或多个具有不同的分布。

*   **无法拒绝H0** ：配对样本分布相等。
*   **拒绝H0** ：配对样本分布不相等。

该测试假设两个或更多配对数据样本，每组10个或更多样本。

> 弗里德曼检验是一种非参数统计程序，用于比较两个以上相关的样本。与该测试等效的参数是重复测量方差分析（ANOVA）。当Friedman测试导致显着结果时，至少一个样品与其他样品不同。

- 第79-80页，[非统计学家的非参数统计：循序渐进的方法](http://amzn.to/2CZcXBz)，2009。

我们可以使用 [friedmanchisquare（）SciPy函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html)在Python中实现Friedman测试。此函数将要比较的数据样本作为参数，并返回计算的统计量和p值。

可以在与前一节中使用的测试数据集相同的变体上证明该显着性检验。即三个样本，两个具有相同的总体平均值，一个具有略微不同的平均值。虽然样本没有配对，但我们希望测试发现并非所有样本都具有相同的分布。

The complete code example is listed below.

```py
# Friedman test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare
# seed the random number generator
seed(1)
# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52
# compare samples
stat, p = friedmanchisquare(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行该示例计算三个数据样本的测试并打印测试统计和p值。

对p值的解释正确地表明至少一个样本具有不同的分布。

```py
Statistics=9.360, p=0.009
Different distributions (reject H0)
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   更新所有示例以对具有相同分布的数据样本进行操作。
*   根据每个测试的要求和行为，创建一个流程图，用于选择每个统计显着性检验。
*   考虑3个在机器学习项目中比较数据样本的情况，假设样本的非高斯分布，并建议可以在每种情况下使用的测试类型。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [非统计人员的非参数统计：循序渐进的方法](http://amzn.to/2CZcXBz)，2009年。

### API

*   [numpy.random.seed（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)
*   [numpy.random.randn（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)
*   [scipy.stats.mannwhitneyu（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
*   [scipy.stats.wilcoxon（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
*   [scipy.stats.kruskal（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
*   [scipy.stats.friedmanchisquare（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)

### 用品

*   [维基百科上的非参数统计](https://en.wikipedia.org/wiki/Nonparametric_statistics)
*   [维基百科上的配对差异测试](https://en.wikipedia.org/wiki/Paired_difference_test)
*   [维基百科上的Mann-Whitney U测试](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
*   [Wilcoxon对维基百科的签名等级测试](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
*   [Kruskal-Wallis对维基百科的单因素方差分析](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)
*   [弗里德曼在维基百科上的测试](https://en.wikipedia.org/wiki/Friedman_test)

## 摘要

在本教程中，您发现了非参数统计测试，可用于确定数据样本是否来自具有相同或不同分布的群体。

具体来说，你学到了：

*   用于比较独立数据样本的Mann-Whitney U检验：Student t检验的非参数版本。
*   用于比较配对数据样本的Wilcoxon符号秩检验：配对Student t检验的非参数版本。
*   Kruskal-Wallis H和Friedman测试用于比较两个以上的数据样本：ANOVA的非参数版本和重复测量ANOVA测试。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。