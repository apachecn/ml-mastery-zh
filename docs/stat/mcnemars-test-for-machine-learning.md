# 如何计算 McNemar 检验来比较两种机器学习分类器

> 原文： [https://machinelearningmastery.com/mcnemars-test-for-machine-learning/](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)

统计假设检验的选择对于解释机器学习结果是一个具有挑战性的开放性问题。

在他 1998 年被广泛引用的论文中，Thomas Dietterich 在那些训练多份分类器模型昂贵或不切实际的案例中推荐了 McNemar 的测试。

这描述了深度学习模型的当前情况，这些模型非常大并且在大型数据集上进行训练和评估，通常需要数天或数周来训练单个模型。

在本教程中，您将了解如何使用 McNemar 的统计假设检验来比较单个测试数据集上的机器学习分类器模型。

完成本教程后，您将了解：

*   McNemar 测试对于训练昂贵的模型的建议，适合大型深度学习模型。
*   如何将两个分类器的预测结果转换为列联表，以及如何使用该表来计算 McNemar 测试中的统计量。
*   如何用 Python 计算 McNemar 的测试并解释和报告结果。

让我们开始吧。

![How to Calculate McNemar's Test for Two Machine Learning Classifiers](img/d5401d2995454ada4c17b76798243936.jpg)

如何计算 McNemar 对两台机器学习量词的测试
[Mark Kao](https://www.flickr.com/photos/67415843@N05/37883206606/) 的照片，保留一些权利。

## 教程概述

本教程分为五个部分;他们是：

1.  深度学习的统计假设检验
2.  列联表
3.  麦克尼玛的测试统计
4.  解读 McNemar 的分类器测试
5.  McNemar 的 Python 测试

## 深度学习的统计假设检验

在他 1998 年重要且广泛引用的关于使用统计假设检验来比较标题为“[近似统计检验比较监督分类学习算法](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197)”的论文中，Thomas Dietterich 建议使用 McNemar 检验。

具体而言，建议在那些被比较的算法只能评估一次的情况下进行测试，例如，在一个测试集上，而不是通过重采样技术重复评估，例如 k 折交叉验证。

> 对于只能执行一次的算法，McNemar 的测试是唯一具有可接受的 I 类错误的测试。

- [用于比较监督分类学习算法的近似统计检验](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197)，1998。

具体而言，Dietterich 的研究涉及不同统计假设检验的评估，其中一些检验采用重采样方法的结果。该研究的关注点是 [I 型错误](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)，即统计检验报告实际上没有效果时的效果（假阳性）。

可以比较基于单个测试集的模型的统计测试是现代机器学习的重要考虑因素，特别是在深度学习领域。

深度学习模型通常很大，可以在非常大的数据集上运行。总之，这些因素可能意味着在快速的现代硬件上对模型的训练可能需要数天甚至数周。

这排除了重采样方法的实际应用以比较模型，并建议需要使用可以对在单个测试数据集上评估训练模型的结果进行操作的测试。

McNemar 的测试可能是评估这些大型和慢速训练深度学习模型的合适测试。

## 列联表

McNemar 的测试按照列联表进行。

在我们深入测试之前，让我们花一点时间来理解如何计算两个分类器的列联表。

[列联表](https://en.wikipedia.org/wiki/Contingency_table)是两个分类变量的列表或计数。在 McNemar 测试的情况下，我们感兴趣的是二进制变量正确/不正确或是/否对于对照和治疗或两种情况。这被称为 2×2 列联表。

列表的列联表乍一看可能不直观。让我们通过一个有效的例子来具体化。

考虑一下我们有两个训练有素的分类器。每个分类器对测试数据集中的 10 个示例中的每个示例进行二进制类预测。预测被评估并确定为正确或不正确。

然后我们可以在表格中总结这些结果，如下所示：

```py
Instance,	Classifier1 Correct,	Classifier2 Correct
1			Yes						No
2			No						No
3			No						Yes
4			No						No
5			Yes						Yes
6			Yes						Yes
7			Yes						Yes
8			No						No
9			Yes						No
10			Yes						Yes
```

我们可以看到 Classifier1 得到 6 个正确，或者精度为 60％，Classifier2 得到 5 个正确，或者在测试集上有 50％的准确度。

该表现在可以简化为列联表。

列联表依赖于两个分类器都在完全相同的训练数据上训练并在完全相同的测试数据实例上进行评估的事实。

列联表具有以下结构：

```py
						Classifier2 Correct,	Classifier2 Incorrect
Classifier1 Correct 	??						??
Classifier1 Incorrect 	?? 						??
```

对于表中第一个单元格，我们必须将 Classifier1 得到正确且 Classifier2 正确的测试实例总数相加。例如，两个分类器正确预测的第一个实例是实例编号 5.两个分类器正确预测的实例总数为 4。

考虑这一点的另一种更具编程性的方法是在上面的结果表中总结是/否的每个组合。

```py
						Classifier2 Correct,	Classifier2 Incorrect
Classifier1 Correct 	Yes/Yes					Yes/No
Classifier1 Incorrect 	No/Yes 					No/No
```

将结果组织成一个列联表如下：

```py
						Classifier2 Correct,	Classifier2 Incorrect
Classifier1 Correct 	4						2
Classifier1 Incorrect 	1 						3
```

## 麦克尼玛的测试统计

[McNemar 的测试](https://en.wikipedia.org/wiki/McNemar%27s_test)是一个配对的非参数或无分布的统计假设检验。

它也不像其他一些统计假设检验那么直观。

McNemar 的测试是检查两个案件之间的分歧是否匹配。从技术上讲，这被称为列联表的同质性（特别是边际同质性）。因此，McNemar 的测试是一种应变表的同质性测试。

该试验广泛用于医学中以比较治疗对照的效果。

在比较两个二分类算法方面，测试是评论两个模型是否以相同的方式不同意。它没有评论一个模型是否比另一个模型更准确或更准确或更容易出错。当我们查看统计量的计算方式时，这一点很清楚。

McNemar 的检验统计量计算如下：

```py
statistic = (Yes/No - No/Yes)^2 / (Yes/No + No/Yes)
```

其中 Yes / No 是 Classifier1 正确且 Classifier2 不正确的测试实例的数量，No / Yes 是 Classifier1 不正确且 Classifier2 正确的测试实例的计数。

测试统计量的这种计算假定计算中使用的列联表中的每个单元具有至少 25 的计数。检验统计量具有 1 自由度的卡方分布。

我们可以看到只使用列联表的两个元素，特别是在计算测试统计量时不使用是/是和否/否元素。因此，我们可以看到统计量报告了两个模型之间不同的正确或不正确的预测，而不是准确率或错误率。在了解统计量的发现时，这一点非常重要。

测试的默认假设或零假设是两个案例不同意相同的数量。如果零假设被拒绝，则表明有证据表明案例在不同方面存在分歧，即分歧存在偏差。

给定显着性水平的选择，通过测试计算的 p 值可以解释如下：

*   **p＆gt; α**：不能拒绝 H0，在分歧上没有差异（例如治疗没有效果）。
*   **p &lt;=α**：排斥 H0，不一致的显着差异（例如治疗有效）。

## 解读 McNemar 的分类器测试

重要的是花点时间清楚地理解如何在两个机器学习分类器模型的上下文中解释测试结果。

计算 McNemar 测试时使用的两个术语捕获了两个模型的误差。具体而言，列联表中的 No / Yes 和 Yes / No 单元格。该测试检查这两个单元格中的计数之间是否存在显着差异。就这些。

如果这些单元格具有相似的计数，则向我们显示两个模型在大小相同的比例中产生错误，仅在测试集的不同实例上。在这种情况下，测试结果不会很明显，零假设也不会被拒绝。

> 在零假设下，两种算法应该具有相同的错误率......

— [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithm](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197), 1998.

如果这些单元格具有不相似的计数，则表明两个模型不仅产生不同的错误，而且实际上在测试集上具有不同的相对错误比例。在这种情况下，测试的结果将是显着的，我们将拒绝零假设。

> 因此，我们可以拒绝零假设，以支持两种算法在特定训练中训练时具有不同表现的假设

— [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithm](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197), 1998.

我们可以总结如下：

*   **无法拒绝空假设**：分类器在测试集上的错误比例相似。
*   **拒绝空假设**：分类器在测试集上具有不同的错误比例。

在执行测试并找到显着结果之后，报告效果统计测量以量化该发现可能是有用的。例如，一个自然的选择是报告优势比，或列联表本身，尽管这两者都假定一个复杂的读者。

报告测试集上两个分类器之间的差异可能很有用。在这种情况下，请小心您的声明，因为重要测试不会报告模型之间的误差差异，只会报告模型之间误差比例的相对差异。

最后，在使用 McNemar 的测试时，Dietterich 强调了必须考虑的两个重要限制。他们是：

### 1.没有训练集或模型可变性的度量。

通常，模型行为根据用于拟合模型的特定训练数据而变化。

这是由于模型与特定训练实例的相互作用以及学习期间随机性的使用。将模型拟合到多个不同的训练数据集上并评估技能，就像重采样方法一样，提供了一种测量模型方差的方法。

如果可变性的来源很小，则该测试是合适的。

> 因此，只有当我们认为这些可变性来源很小时，才应该应用 McNemar 的测试。

— [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithm](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197), 1998.

### 2.较少的模型直接比较

在单个测试集上评估两个分类器，并且预期测试集小于训练集。

这与使用重采样方法的假设检验不同，因为在评估过程中，如果不是全部数据集的更多（如果不是全部）可用作测试集（从统计角度介绍其自身的问题）。

这提供了较少的机会来比较模型的表现。它要求测试集合适当地代表域，通常意味着测试数据集很大。

## McNemar 的 Python 测试

可以使用 [mcnemar（）Statsmodels 函数](http://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html)在 Python 中实现 McNemar 的测试。

该函数将列联表作为参数，并返回计算的测试统计量和 p 值。

根据数据量，有两种方法可以使用统计量。

如果表中有一个单元用于计算计数小于 25 的测试统计量，则使用测试的修改版本，使用二项分布计算精确的 p 值。这是测试的默认用法：

```py
stat, p = mcnemar(table, exact=True)
```

或者，如果在列联表中计算测试统计量时使用的所有单元具有 25 或更大的值，则可以使用测试的标准计算。

```py
stat, p = mcnemar(table, exact=False, correction=True)
```

我们可以在上面描述的示例列联表上计算 McNemar。这个列联表在两个不同的单元中都有一个小的计数，因此必须使用确切的方法。

下面列出了完整的示例。

```py
# Example of calculating the mcnemar test
from statsmodels.stats.contingency_tables import mcnemar
# define contingency table
table = [[4, 2],
		 [1, 3]]
# calculate mcnemar test
result = mcnemar(table, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')
```

运行该示例计算列联表上的统计值和 p 值并打印结果。

我们可以看到，该测试强烈证实两种情况之间的分歧差别很小。零假设没有被拒绝。

当我们使用测试来比较分类器时，我们指出两个模型之间的分歧没有统计学上的显着差异。

```py
statistic=1.000, p-value=1.000
Same proportions of errors (fail to reject H0)
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   在机器学习中找到一篇利用 McNemar 统计假设检验的研究论文。
*   更新代码示例，以便列联表显示两种情况之间不一致的显着差异。
*   实现一个函数，该函数将根据提供的列联表使用正确版本的 McNemar 测试。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [关于相关比例或百分比之间差异的采样误差的注释](https://link.springer.com/article/10.1007/BF02295996)，1947。
*   [用于比较监督分类学习算法的近似统计检验](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017197)，1998。

### API

*   [statsmodels.stats.contingency_tables.mcnemar（）API](http://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html)

### 用品

*   [McNemar 对维基百科的测试](https://en.wikipedia.org/wiki/McNemar%27s_test)
*   [维基百科上的 I 型和 II 型错误](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
*   [维基百科上的列联表](https://en.wikipedia.org/wiki/Contingency_table)

## 摘要

在本教程中，您了解了如何使用 McNemar 的测试统计假设检验来比较单个测试数据集上的机器学习分类器模型。

具体来说，你学到了：

*   McNemar 测试对于训练昂贵的模型的建议，适合大型深度学习模型。
*   如何将两个分类器的预测结果转换为列联表，以及如何使用该表来计算 McNemar 测试中的统计量。
*   如何用 Python 计算 McNemar 的测试并解释和报告结果。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。