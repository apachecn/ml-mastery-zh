# 如何在 Python 中计算非参数秩相关性

> 原文： [https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/](https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/)

相关性是两个变量之间关联的度量。

当两个变量都具有良好理解的高斯分布时，很容易计算和解释。当我们不知道变量的分布时，我们必须使用非参数秩相关方法。

在本教程中，您将发现用于量化具有非高斯分布的变量之间的关联的等级相关方法。

完成本教程后，您将了解：

*   排名相关方法如何工作以及方法是否可用。
*   如何在 Python 中计算和解释 Spearman 的秩相关系数。
*   如何在 Python 中计算和解释 Kendall 的秩相关系数。

让我们开始吧。

## 教程概述

本教程分为 4 个部分;他们是：

1.  等级相关
2.  测试数据集
3.  斯皮尔曼的秩相关
4.  肯德尔的秩相关

## 等级相关

相关性是指两个变量的观测值之间的关联。

变量可能具有正关联，这意味着随着一个变量的值增加，另一个变量的值也增加。该关联也可以是否定的，意味着随着一个变量的值增加，其他变量的值减小。最后，关联可能是中性的，这意味着变量不相关。

相关性量化这种关联，通常作为值-1 到 1 之间的度量，完全负相关和完全正相关。计算的相关性被称为“_ 相关系数 _。”然后可以解释该相关系数以描述测量。

请参阅下表以帮助解释相关系数。

![Table of Correlation Coefficient Values and Their Interpretation](img/9df1f69048629bc7373155535c3a80df.jpg)

相关系数值表及其解释
取自“非统计学家的非参数统计：逐步法”。

可以使用诸如 Pearson 相关的标准方法来计算每个具有高斯分布的两个变量之间的相关性。此过程不能用于没有高斯分布的数据。相反，必须使用等级相关方法。

[秩相关](https://en.wikipedia.org/wiki/Rank_correlation)是指使用值之间的序数关系而不是特定值来量化变量之间的关联的方法。序数据是具有标签值并具有顺序或等级关系的数据;例如：'_ 低 _'，'_ 训练基 _'和'_ 高 _'。

可以针对实值变量计算秩相关性。这是通过首先将每个变量的值转换为等级数据来完成的。这是值的排序位置，并赋予整数排名值。然后可以计算秩相关系数以量化两个排序变量之间的关联。

因为没有假设值的分布，所以秩相关方法被称为无分布相关或非参数相关。有趣的是，秩相关度量通常被用作其他统计假设检验的基础，例如确定两个样本是否可能来自相同（或不同）的人口分布。

秩相关方法通常以研究人员或开发该方法的研究人员的名字命名。秩相关方法的四个例子如下：

*   斯皮尔曼的秩相关。
*   肯德尔的秩相关。
*   Goodman 和 Kruskal 的秩相关。
*   萨默斯的秩相关。

在接下来的部分中，我们将仔细研究两种更常见的排名相关方法：Spearman 和 Kendall。

## 测试数据集

在我们演示秩相关方法之前，我们必须首先定义一个测试问题。

在本节中，我们将定义一个简单的双变量数据集，其中每个变量是从均匀分布（例如非高斯分布）绘制的，第二个变量的值取决于第一个值的值。

具体而言，从均匀分布中抽取 1,000 个随机浮点值的样本，并缩放到 0 到 20 的范围。从 0 到 10 之间的均匀分布中抽取 1,000 个随机浮点值的第二个样本，并将其添加到第一个创建关联的示例。

```py
# prepare data
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
```

下面列出了完整的示例。

```py
# generate related variables
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
seed(1)
# prepare data
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
# plot
pyplot.scatter(data1, data2)
pyplot.show()
```

运行该示例将生成数据样本并绘制散点图上的点。

我们可以清楚地看到每个变量具有均匀分布，并且通过从图的左下到右上角的点的对角分组可以看到变量之间的正相关。

![Scatter Plot of Associated Variables Drawn From a Uniform Distribution](img/5526515d599bf88315d47cc795fab9d4.jpg)

从均匀分布绘制的相关变量的散点图

## 斯皮尔曼的秩相关

[斯皮尔曼的等级相关](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)以查尔斯斯皮尔曼命名。

它也可以称为斯皮尔曼相关系数，并用小写希腊字母 rho（p）表示。因此，它可以被称为 Spearman 的 rho。

该统计方法量化了排序变量与单调函数相关联的程度，意味着增加或减少的关系。作为统计假设检验，该方法假设样本不相关（不能拒绝 H0）。

> Spearman 等级相关是一种统计过程，旨在测量顺序测量尺度上两个变量之间的关系。

- 第 124 页，[非统计学家的非参数统计：循序渐进的方法](https://amzn.to/2HevldG)，2009。

Spearman 等级相关性的直觉是它使用秩值而不是实际值来计算 Pearson 相关性（例如，相关性的参数度量）。 Pearson 相关性是通过两个变量的方差或扩展归一化的两个变量之间的协方差（或平均观测值的预期差异）的计算。

Spearman 的等级相关性可以使用 [spearmanr（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)在 Python 中计算。

该函数将两个实值样本作为参数，并返回介于-1 和 1 之间的相关系数以及用于解释系数重要性的 p 值。

```py
# calculate spearman's correlation
coef, p = spearmanr(data1, data2)
```

我们可以在测试数据集上演示 Spearman 的等级相关性。我们知道数据集中的变量之间存在很强的关联，我们希望 Spearman 的测试能够找到这种关联。

The complete example is listed below.

```py
# calculate the spearman's correlation between two variables
from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr
# seed random number generator
seed(1)
# prepare data
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
# calculate spearman's correlation
coef, p = spearmanr(data1, data2)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p)
```

运行该示例计算测试数据集中两个变量之间的 Spearman 相关系数。

统计检验报告与 0.9 的值呈强正相关。 p 值接近于零，这意味着在给定样本不相关的情况下观察数据的可能性是非常不可能的（例如 95％置信度），并且我们可以拒绝样本不相关的零假设。

```py
Spearmans correlation coefficient: 0.900
Samples are correlated (reject H0) p=0.000
```

## 肯德尔的秩相关

[Kendall 的等级相关](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)以 Maurice Kendall 的名字命名。

它也被称为肯德尔的相关系数，系数通常用小写希腊字母 tau（t）表示。反过来，测试可能被称为肯德尔的头。

测试的直觉是它计算两个样本之间匹配或一致排名数的标准化分数。因此，该测试也称为肯德尔的一致性测试。

Kendall 的秩相关系数可以使用 [kendalltau（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)在 Python 中计算。测试将两个数据样本作为参数，并返回相关系数和 p 值。作为统计假设检验，该方法假设（H0）两个样本之间没有关联。

```py
# calculate kendall's correlation
coef, p = kendalltau(data1, data2)
```

我们可以在测试数据集上演示计算，我们确实希望报告显着的正关联。

The complete example is listed below.

```py
# calculate the kendall's correlation between two variables
from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau
# seed random number generator
seed(1)
# prepare data
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
# calculate kendall's correlation
coef, p = kendalltau(data1, data2)
print('Kendall correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p)
```

运行该示例将 Kendall 的相关系数计算为 0.7，这是高度相关的。

与 Spearman 测试一样，p 值接近于零（并打印为零），这意味着我们可以放心地拒绝样本不相关的零假设。

```py
Kendall correlation coefficient: 0.709
Samples are correlated (reject H0) p=0.000
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   列举三个示例，其中计算非参数相关系数在机器学习项目期间可能是有用的。
*   更新每个示例以计算从非高斯分布中提取的不相关数据样本之间的相关性。
*   加载标准机器学习数据集并计算所有变量之间的成对非参数相关性。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [非统计人员的非参数统计：循序渐进的方法](https://amzn.to/2HevldG)，2009 年。
*   [应用非参数统计方法](https://amzn.to/2GCKnfW)，第四版，2007。
*   [秩相关方法](https://amzn.to/2JofYzY)，1990。

### API

*   [scipy.stats.spearmanr（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
    [scipy.stats.kendalltau（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)

### 用品

*   [维基百科上的非参数统计](https://en.wikipedia.org/wiki/Nonparametric_statistics)
*   [维基百科上的排名相关](https://en.wikipedia.org/wiki/Rank_correlation)
*   [Spearman 在维基百科上的等级相关系数](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
*   [维基百科上的肯德尔等级相关系数](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
*   [Goodman 和 Kruskal 在维基百科上的伽玛](https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_gamma)
*   [维基百科上的 Somers'D](https://en.wikipedia.org/wiki/Somers%27_D)

## 摘要

在本教程中，您发现了用于量化具有非高斯分布的变量之间的关联的等级相关方法。

具体来说，你学到了：

*   排名相关方法如何工作以及方法是否可用。
*   如何在 Python 中计算和解释 Spearman 的秩相关系数。
*   如何在 Python 中计算和解释 Kendall 的秩相关系数。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。