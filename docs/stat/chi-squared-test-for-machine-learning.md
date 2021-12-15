# 浅谈机器学习的卡方测试

> 原文： [https://machinelearningmastery.com/chi-squared-test-for-machine-learning/](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)

应用机器学习中的常见问题是确定输入特征是否与要预测的结果相关。

这是特征选择的问题。

在输入变量也是分类的分类问题的情况下，我们可以使用统计测试来确定输出变量是依赖还是独立于输入变量。如果是独立的，则输入变量是可能与问题无关并从数据集中删除的要素的候选者。

Pearson 的卡方统计假设是分类变量之间独立性检验的一个例子。

在本教程中，您将发现用于量化分类变量对的独立性的卡方统计假设检验。

完成本教程后，您将了解：

*   可以使用列联表来汇总成对的分类变量。
*   卡方检验可以将观察到的列联表与预期表进行比较，并确定分类变量是否独立。
*   如何计算和解释 Python 中分类变量的卡方检验。

让我们开始吧。

*   **更新 Jun / 2018** ：从测试中解释临界值的小错误修复（感谢 Andrew）。

![A Gentle Introduction to the 卡方 Test for Machine Learning](img/e986eb1bedf061678da5260395bae17f.jpg)

机器学习卡方测试的温和介绍
[NC 湿地](https://www.flickr.com/photos/ncwetlands/38431877722/)的照片，保留一些权利

## 教程概述

本教程分为 3 个部分;他们是：

1.  列联表
2.  皮尔逊的卡方测试
3.  示例卡方测试

## 列联表

分类变量是可以采用一组标签之一的变量。

一个例子可能是性，可以概括为男性或女性。变量是'`sex`'，变量的标签或因子是'`male`'和'`female`'在这种情况下。

我们可能希望查看分类变量的摘要，因为它与另一个分类变量有关。例如，性和兴趣，其中兴趣可能有标签'_ 科学 _'，'_ 数学 _'或'_ 艺术 _'。我们可以从收集到的关于这两个分类变量的人收集观察结果;例如：

```py
Sex,	Interest
Male,	Art
Female,	Math
Male, 	Science
Male,	Math
...
```

我们可以在一个表中汇总所收集的观察结果，其中一个变量对应于列，另一个变量对应于行。表中的每个单元格对应于与行和列类别对应的观察的计数或频率。

历史上，这种形式的两个分类变量的表汇总称为[列联表](https://en.wikipedia.org/wiki/Contingency_table)。

例如，具有人为计数的 _ 性别=行 _ 和 _ 兴趣=列 _ 表可能如下所示：

```py
        Science,	Math,	Art
Male         20,      30,    15
Female       20,      15,    30
```

该表由 Karl Pearson 称为列联表，因为其目的是帮助确定一个变量是否依赖于另一个变量。例如，对数学或科学的兴趣是否取决于性别，还是它们是独立的？

仅从表格中确定这是具有挑战性的;相反，我们可以使用称为 Pearson 的卡方测试的统计方法。

## 皮尔逊的卡方测试

Pearson 的卡方测试，或简称卡方测试，以 Karl Pearson 命名，尽管测试有变化。

卡方测试是一种统计假设检验，假设（零假设）分类变量的观察频率与分类变量的预期频率匹配。该测试计算具有卡方分布的统计量，以希腊大写字母 Chi（X）命名为“ki”，如风筝中所示。

鉴于上面的性/兴趣例子，一个类别（例如男性和女性）的观察数量可能相同或不同。尽管如此，我们可以计算每个兴趣小组中观察的预期频率，并查看按性别划分的利益是否会产生相似或不同的频率。

卡方测试用于列联表，首先计算组的预期频率，然后确定组的划分（称为观察频率）是否与预期频率匹配。

测试的结果是具有卡方分布的测试统计量，并且可以被解释为拒绝或不能拒绝观察到的和预期的频率相同的假设或零假设。

> 当观测频率远离预期频率时，总和中的相应项很大;当两者接近时，这个词很小。较大的 X ^ 2 值表明观察到的和预期的频率相差很远。 X ^ 2 的小值意味着相反：观察到的接近预期。所以 X ^ 2 确实测量了观测频率和预期频率之间的距离。

- 第 525 页，[统计](http://amzn.to/2u44zll)，第四版，2007 年。

如果观察到的和预期的频率相似，变量的水平不相互作用，则变量被认为是独立的。

> 卡方检验的独立性通过比较您收集的分类编码数据（称为观察到的频率）与您预期在表中每个单元格中获得的频率（称为预期频率）进行比较。 。

- 第 162 页，[普通英语统计](http://amzn.to/2IFyS4P)，第三版，2010 年。

我们可以在卡方分布的背景下解释检验统计量，并具有必要的自由度数，如下所示：

*   **如果统计＆gt; =临界值**：显着结果，拒绝原假设（H0），依赖。
*   **如果统计＆lt;临界值**：不显着的结果，不能拒绝零假设（H0），独立。

卡方分布的自由度是根据列联表的大小计算的：

```py
degrees of freedom: (rows - 1) * (cols - 1)
```

根据 p 值和选择的显着性水平（alpha），测试可以解释如下：

*   **如果 p 值＆lt; = alpha** ：显着结果，则拒绝原假设（H0），依赖。
*   **如果 p 值&gt; alpha** ：不显着的结果，不能拒绝零假设（H0），独立。

为了使测试有效，在列联表的每个单元格中至少需要五次观察。

接下来，让我们看看我们如何计算卡方检验。

## 示例卡方测试

可以使用 [chi2_contingency（）SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)在 Python 中计算 Pearson 的卡方检验。

该函数将数组作为输入，表示两个分类变量的列联表。它返回计算的统计值和解释的 p 值以及计算的自由度和预期频率表。

```py
stat, p, dof, expected = chi2_contingency(table)
```

我们可以通过从卡方分布中检索概率和自由度数的临界值来解释统计量。

例如，可以使用 95％的概率，这表明在测试假设变量是独立的情况下很可能发现测试结果。如果统计量小于或等于临界值，我们可能无法拒绝此假设，否则可能会被拒绝。

```py
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
```

我们还可以通过将 p 值与选定的显着性水平进行比较来解释 p 值，该显着性水平为 5％，通过反转临界值解释中使用的 95％概率来计算。

```py
# interpret p-value
alpha = 1.0 - prob
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
```

我们可以将所有这些结合在一起，并使用设计的列联表来演示卡方显着性检验。

下面定义了一个列联表，每个人口（行）的观察数量不同，但每个群体（列）的比例相似。鉴于相似的比例，我们期望测试发现组是相似的并且变量是独立的（不能拒绝零假设，或 H0）。

```py
table = [	[10, 20, 30],
			[6,  9,  17]]
```

下面列出了完整的示例。

```py
# 卡方 test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# contingency table
table = [	[10, 20, 30],
			[6,  9,  17]]
print(table)
stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
```

首先运行示例打印列联表。计算测试并将自由度（`dof`）报告为 2，这是有道理的：

```py
degrees of freedom: (rows - 1) * (cols - 1)
degrees of freedom: (2 - 1) * (3 - 1)
degrees of freedom: 1 * 2
degrees of freedom: 2
```

接下来，打印计算出的预期频率表，我们可以看到，通过数字的眼球检查，确实观察到的列联表似乎确实匹配。

计算并解释临界值，发现变量确实是独立的（未能拒绝 H0）。对 p 值的解释得出了同样的结论。

```py
[[10, 20, 30], [6, 9, 17]]

dof=2

[[10.43478261 18.91304348 30.65217391]
 [ 5.56521739 10.08695652 16.34782609]]

probability=0.950, critical=5.991, stat=0.272
Independent (fail to reject H0)

significance=0.050, p=0.873
Independent (fail to reject H0)
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   更新卡方检验以使用您自己的列联表。
*   编写一个函数来报告两个分类变量的观察结果的独立性
*   加载包含分类变量的标准机器学习数据集，并报告每个变量的独立性。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 14 章，独立卡方检验，[简明英语统计](http://amzn.to/2IFyS4P)，第 3 版，2010 年。
*   第 28 章，卡方检验，[统计](http://amzn.to/2u44zll)，第四版，2007 年。

### API

*   [scipy.stats.chisquare（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
*   [scipy.stats.chi2_contingency（）API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
*   [sklearn.feature_selection.chi2（）API](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)

## 用品

*   [维基百科上的卡方测试](https://en.wikipedia.org/wiki/卡方 _test)
*   [Pearson 对维基百科的卡方测试](https://en.wikipedia.org/wiki/Pearson%27s_ 卡方 _test)
*   [维基百科上的列联表](https://en.wikipedia.org/wiki/Contingency_table)
*   [chi 测试如何用于机器学习中的特征选择？关于 Quora](https://www.quora.com/How-is-chi-test-used-for-feature-selection-in-machine-learning)

## 摘要

在本教程中，您发现了用于量化分类变量对的独立性的卡方统计假设检验。

具体来说，你学到了：

*   可以使用列联表来汇总成对的分类变量。
*   卡方检验可以将观察到的列联表与预期表进行比较，并确定分类变量是否独立。
*   如何计算和解释 Python 中分类变量的卡方检验。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。