# 15 个 Python 中的统计假设检验（备忘单）

> 原文： [https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)

#### 在
应用机器学习中需要的 15 个统计假设检验的快速参考指南，以及 Python 中的示例代码。

尽管您可以使用数百种统计假设检验，但您可能只需要在机器学习项目中使用一小部分。

在这篇文章中，您将发现一个备忘单，用于机器学习项目的最流行的统计假设检验，其中包含使用 Python API 的示例。

每个统计测试都以一致的方式呈现，包括：

*   测试的名称。
*   测试的内容是什么。
*   测试的关键假设。
*   如何解释测试结果。
*   用于使用测试的 Python API。

注意，当涉及诸如预期的数据分布或样本大小之类的假设时，给定测试的结果可能会优雅地降级，而不是在违反假设时立即变得不可用。

通常，数据样本需要代表域并且足够大以将它们的分布暴露给分析。

在某些情况下，可以校正数据以满足假设，例如通过去除异常值来将近似正态分布校正为正常，或者在样本具有不同方差时使用统计检验中的自由度校正来命名为二例子。

最后，对于给定的关注点可能存在多个测试，例如，常态。我们无法通过统计量获得清晰的问题答案;相反，我们得到概率答案。因此，我们可以通过不同方式考虑问题来得出同一问题的不同答案。因此，对于我们可能对数据提出的一些问题，需要进行多种不同的测试。

让我们开始吧。

*   **更新 Nov / 2018** ：更好地概述了所涵盖的测试。

![Statistical Hypothesis Tests in Python Cheat Sheet](img/e75b65d84f0c276d384372821a4a100f.jpg)

Python 备忘单中的统计假设检验
[davemichuda](https://www.flickr.com/photos/36137232@N00/4800239195/) 的照片，保留一些权利。

## 教程概述

本教程分为四个部分;他们是：

1.  **正态性测试**
    1.  Shapiro-Wilk 测试
    2.  D'Agostino 的 K ^ 2 测试
    3.  安德森 - 达林测试
2.  **相关性测试**
    1.  皮尔逊的相关系数
    2.  斯皮尔曼的秩相关
    3.  肯德尔的秩相关
    4.  卡方测试
3.  **参数统计假设检验**
    1.  学生的 t 检验
    2.  配对学生的 t 检验
    3.  方差检验分析（ANOVA）
    4.  重复测量方差分析测试
4.  **非参数统计假设检验**
    1.  Mann-Whitney U 测试
    2.  威尔科克森签名等级测试
    3.  Kruskal-Wallis H 测试
    4.  弗里德曼测试

## 1.正态性测试

本节列出了可用于检查数据是否具有高斯分布的统计测试。

### Shapiro-Wilk 测试

测试数据样本是否具有高斯分布。

假设

*   每个样本中的观察是独立的并且相同地分布（iid）。

解释

*   H0：样本具有高斯分布。
*   H1：样本没有高斯分布。

Python 代码

```py
from scipy.stats import shapiro
data1 = ....
stat, p = shapiro(data)
```

更多信息

*   [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
*   维基百科上的 [Shapiro-Wilk 测试](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)

### D'Agostino 的 K ^ 2 测试

Tests whether a data sample has a Gaussian distribution.

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。

Interpretation

*   H0：样本具有高斯分布。
*   H1：样本没有高斯分布。

Python Code

```py
from scipy.stats import normaltest
data1 = ....
stat, p = normaltest(data)
```

More Information

*   [scipy.stats.normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
*   [D'Agostino 在维基百科上的$ K $平方测试](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)

### 安德森 - 达林测试

Tests whether a data sample has a Gaussian distribution.

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。

Interpretation

*   H0：样本具有高斯分布。
*   H1：样本没有高斯分布。

Python Code

```py
from scipy.stats import anderson
data1 = ....
result = anderson(data)
```

More Information

*   [scipy.stats.anderson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)
*   [维基百科上的 Anderson-Darling 测试](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)

## 2.相关性测试

本节列出了可用于检查两个样本是否相关的统计检验。

### 皮尔逊的相关系数

测试两个样本是否具有线性关系。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   每个样本中的观察结果通常是分布的。
*   每个样本中的观察结果具有相同的方差。

Interpretation

*   H0：两个样本是独立的。
*   H1：样本之间存在依赖关系。

Python Code

```py
from scipy.stats import pearsonr
data1, data2 = ...
corr, p = pearsonr(data1, data2)
```

More Information

*   [scipy.stats.pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
*   [Pearson 在维基百科上的相关系数](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

### 斯皮尔曼的秩相关

测试两个样本是否具有单调关系。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。

Interpretation

*   H0：两个样本是独立的。
*   H1：样本之间存在依赖关系。

Python Code

```py
from scipy.stats import spearmanr
data1, data2 = ...
corr, p = spearmanr(data1, data2)
```

More Information

*   [scipy.stats.spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
*   [Spearman 在维基百科上的等级相关系数](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

### 肯德尔的秩相关

Tests whether two samples have a monotonic relationship.

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。

Interpretation

*   H0：两个样本是独立的。
*   H1：样本之间存在依赖关系。

Python Code

```py
from scipy.stats import kendalltau
data1, data2 = ...
corr, p = kendalltau(data1, data2)
```

More Information

*   [scipy.stats.kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
*   [维基百科上的肯德尔等级相关系数](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

### 卡方测试

测试两个分类变量是相关的还是独立的。

Assumptions

*   用于计算列联表的观察是独立的。
*   列联表的每个单元格中有 25 个或更多个例子。

Interpretation

*   H0：两个样本是独立的。
*   H1：样本之间存在依赖关系。

Python Code

```py
from scipy.stats import chi2_contingency
table = ...
stat, p, dof, expected = chi2_contingency(table)
```

More Information

*   [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
*   [维基百科上的卡方测试](https://en.wikipedia.org/wiki/卡方 _test)

## 3.参数统计假设检验

本节列出了可用于比较数据样本的统计测试。

### 学生的 t 检验

测试两个独立样本的均值是否显着不同。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   每个样本中的观察结果通常是分布的。
*   每个样本中的观察结果具有相同的方差。

Interpretation

*   H0：样本的平均值相等。
*   H1：样本的均值不相等。

Python Code

```py
from scipy.stats import ttest_ind
data1, data2 = ...
stat, p = ttest_ind(data1, data2)
```

More Information

*   [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
*   [维基百科上的 T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)

### 配对学生的 t 检验

测试两个配对样本的均值是否显着不同。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   每个样本中的观察结果通常是分布的。
*   每个样本中的观察结果具有相同的方差。
*   每个样本的观察结果是成对的。

Interpretation

*   H0：样本的平均值相等。
*   H1：样本的均值不相等。

Python Code

```py
from scipy.stats import ttest_rel
data1, data2 = ...
stat, p = ttest_rel(data1, data2)
```

More Information

*   [scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
*   [维基百科上的 T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)

### 方差检验分析（ANOVA）

测试两个或多个独立样本的均值是否显着不同。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   每个样本中的观察结果通常是分布的。
*   每个样本中的观察结果具有相同的方差。

Interpretation

*   H0：样本的平均值相等。
*   H1：样品的一种或多种方法是不相等的。

Python Code

```py
from scipy.stats import f_oneway
data1, data2, ... = ...
stat, p = f_oneway(data1, data2, ...)
```

More Information

*   [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
*   [维基百科](https://en.wikipedia.org/wiki/Analysis_of_variance)的方差分析

### 重复测量方差分析测试

测试两个或更多配对样本的均值是否显着不同。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   每个样本中的观察结果通常是分布的。
*   每个样本中的观察结果具有相同的方差。
*   每个样本的观察结果是成对的。

Interpretation

*   H0：样本的平均值相等。
*   H1：样品的一种或多种方法是不相等的。

Python Code

目前在 Python 中不支持。

More Information

*   [维基百科](https://en.wikipedia.org/wiki/Analysis_of_variance)的方差分析

## 4.非参数统计假设检验

### Mann-Whitney U 测试

测试两个独立样本的分布是否相等。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。

Interpretation

*   H0：两个样本的分布相等。
*   H1：两个样本的分布不相等。

Python Code

```py
from scipy.stats import mannwhitneyu
data1, data2 = ...
stat, p = mannwhitneyu(data1, data2)
```

More Information

*   [scipy.stats.mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
*   [维基百科上的 Mann-Whitney U 测试](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)

### 威尔科克森签名等级测试

测试两个配对样本的分布是否相等。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。
*   每个样本的观察结果是成对的。

Interpretation

*   H0：两个样本的分布相等。
*   H1：两个样本的分布不相等。

Python Code

```py
from scipy.stats import wilcoxon
data1, data2 = ...
stat, p = wilcoxon(data1, data2)
```

More Information

*   [scipy.stats.wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
*   [Wilcoxon 对维基百科的签名等级测试](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

### Kruskal-Wallis H 测试

测试两个或多个独立样本的分布是否相等。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。

Interpretation

*   H0：所有样本的分布相等。
*   H1：一个或多个样本的分布不相等。

Python Code

```py
from scipy.stats import kruskal
data1, data2, ... = ...
stat, p = kruskal(data1, data2, ...)
```

More Information

*   [scipy.stats.kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
*   [Kruskal-Wallis 对维基百科的单因素方差分析](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)

### 弗里德曼测试

测试两个或更多配对样本的分布是否相等。

Assumptions

*   每个样本中的观察是独立的并且相同地分布（iid）。
*   可以对每个样本中的观察进行排序。
*   每个样本的观察结果是成对的。

Interpretation

*   H0：所有样本的分布相等。
*   H1：一个或多个样本的分布不相等。

Python Code

```py
from scipy.stats import friedmanchisquare
data1, data2, ... = ...
stat, p = friedmanchisquare(data1, data2, ...)
```

More Information

*   [scipy.stats.friedmanchisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
*   [弗里德曼在维基百科上的测试](https://en.wikipedia.org/wiki/Friedman_test)

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [Python 中正常性测试的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
*   [如何使用相关来理解变量之间的关系](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)
*   [如何在 Python 中使用参数统计显着性检验](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)
*   [统计假设检验的温和介绍](https://machinelearningmastery.com/statistical-hypothesis-tests/)

## 摘要

在本教程中，您发现了可能需要在机器学习项目中使用的关键统计假设检验。

具体来说，你学到了：

*   在不同情况下使用的测试类型，例如正态性检查，变量之间的关系以及样本之间的差异。
*   每个测试的关键假设以及如何解释测试结果。
*   如何使用 Python API 实现测试。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。

我是否错过了其中一项列出的测试的重要统计测试或关键假设？
请在下面的评论中告诉我。