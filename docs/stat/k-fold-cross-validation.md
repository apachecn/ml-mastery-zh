# k 折交叉验证的温和介绍

> 原文： [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)

交叉验证是用于估计机器学习模型技能的统计方法。

它通常用于应用机器学习，以比较和选择给定预测性建模问题的模型，因为它易于理解，易于实现，并且导致技能估计通常具有比其他方法更低的偏差。

在本教程中，您将发现 k-fold 交叉验证程序的简要介绍，用于估计机器学习模型的技能。

完成本教程后，您将了解：

*   该 k 折交叉验证是用于估计模型对新数据的技能的过程。
*   您可以使用常用的策略为数据集选择 k 的值。
*   在交叉验证中存在常用的变体，例如 scikit-learn 中可用的分层和重复。

让我们开始吧。

![A Gentle Introduction to k-fold Cross-Validation](img/dcac3b090d0c4f8db464e3220bc71cb3.jpg)

k-fold 交叉验证的温和介绍
[Jon Baldock](https://www.flickr.com/photos/jbaldock/3586229318/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  k-fold 交叉验证
2.  配置 k
3.  工作示例
4.  交叉验证 API
5.  交叉验证的变化

## k-fold 交叉验证

交叉验证是一种重采样程序，用于评估有限数据样本的机器学习模型。

该过程有一个名为 k 的参数，它指的是给定数据样本要分组的组数。因此，该过程通常称为 k 折交叉验证。当选择 k 的特定值时，可以使用它代替模型参考中的 k，例如 k = 10 变为 10 倍交叉验证。

交叉验证主要用于应用机器学习，以估计机器学习模型对未见数据的技能。也就是说，使用有限样本来估计模型在用于对模型训练期间未使用的数据做出预测时通常如何执行。

这是一种流行的方法，因为它易于理解，并且因为它通常比其他方法（例如简单的训练/测试分割）导致对模型技能的偏差或不太乐观的估计。

一般程序如下：

1.  随机随机播放数据集。
2.  将数据集拆分为 k 个组
3.  对于每个独特的组：
    1.  将该组作为保留或测试数据集
    2.  将剩余的组作为训练数据集
    3.  在训练集上拟合模型并在测试集上进行评估
    4.  保留评估分数并丢弃模型
4.  使用模型评估分数样本总结模型的技能

重要的是，数据样本中的每个观察结果都被分配给一个单独的组，并在该过程的持续时间内保留在该组中。这意味着每个样本都有机会在保持集 1 中使用并用于训练模型 k-1 次。

> 该方法涉及将观察组随机地划分为大小相等的 k 组或折叠。第一个折叠被视为验证集，并且该方法适合剩余的 k-1 倍。

- 第 181 页，[统计学习导论](http://amzn.to/2FkHqvW)，2013。

同样重要的是，在拟合模型之前的任何数据准备都发生在循环内的 CV 分配的训练数据集上而不是更广泛的数据集上。这也适用于超参数的任何调整。在循环内未能执行这些操作可能导致[数据泄漏](https://machinelearningmastery.com/data-leakage-machine-learning/)和模型技能的乐观估计。

> 尽管统计方法学家付出了最大努力，但用户经常无意中偷看测试数据，从而使他们的结果无效。

- 第 708 页，[人工智能：现代方法（第 3 版）](http://amzn.to/2thrWHq)，2009。

k 次交叉验证运行的结果通常用模型技能得分的平均值来概括。优良作法是包括技能分数方差的度量，例如标准偏差或标准误差。

## 配置 k

必须仔细选择 k 值作为数据样本。

选择不当的 k 值可能会导致对模型技能的错误代表性概念，例如具有高方差的分数（可能会根据用于拟合模型的数据而发生很大变化）或高偏差，（例如高估模型的技巧）。

选择 k 值的三种常用策略如下：

*   **代表性**：选择 k 的值使得每个训练/测试组的数据样本足够大以在统计上代表更广泛的数据集。
*   **k = 10** ：k 的值固定为 10，这是通过实验发现的值，通常导致具有低偏差的模型技能估计，适度方差。
*   **k = n** ：k 的值固定为 n，其中 n 是数据集的大小，以便为每个测试样本提供在保留数据集中使用的机会。这种方法称为留一交叉验证。

> k 的选择通常是 5 或 10，但没有正式的规则。随着 k 变大，训练集和重采样子集之间的大小差异变小。随着这种差异减小，技术的偏差变小

- 第 70 页， [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。

k = 10 的值在应用机器学习领域非常普遍，如果您在努力为数据集选择值，则建议使用 k = 10。

> 总而言之，在 k 折交叉验证中存在与 k 选择相关的偏方差权衡。通常，考虑到这些考虑因素，使用 k = 5 或 k = 10 执行 k 折交叉验证，因为这些值已经凭经验显示以产生既不受过高偏差也不受非常高方差影响的测试误差率估计。

- 第 184 页，[统计学习导论](http://amzn.to/2FkHqvW)，2013。

如果选择的 k 值不能均匀地分割数据样本，则一个组将包含其余的示例。优选地将数据样本分成具有相同数量的样本的 k 个组，使得模型技能分数的样本都是等效的。

## 工作示例

为了使交叉验证过程具体，让我们看一个有效的例子。

想象一下，我们有一个包含 6 个观察结果的数据样本：

```py
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
```

第一步是为 k 选择一个值，以确定用于拆分数据的折叠数。在这里，我们将使用 k = 3 的值。这意味着我们将对数据进行洗牌，然后将数据分成 3 组。因为我们有 6 个观测值，每个群体将有相同数量的 2 个观测值。

例如：

```py
Fold1: [0.5, 0.2]
Fold2: [0.1, 0.3]
Fold3: [0.4, 0.6]
```

然后我们可以使用该样本，例如评估机器学习算法的技能。

每个折叠都训练和评估三个模型，有机会成为保持测试集。

For example:

*   **Model1** ：在 Fold1 + Fold2 上训练，在 Fold3 上测试
*   **Model2** ：在 Fold2 + Fold3 上训练，在 Fold1 上​​测试
*   **Model3** ：在 Fold1 + Fold3 上训练，在 Fold2 上测试

然后在评估模型之后将它们丢弃，因为它们已经达到了目的。

收集每个模型的技能分数并汇总使用。

## 交叉验证 API

我们不必手动实现 k 折交叉验证。 scikit-learn 库提供了一个实现，可以将给定的数据样本分开。

可以使用`KFold()`scikit-learn 类。它取决于分裂的数量，是否对样本进行混洗，以及在混洗之前使用的伪随机数生成器的种子。

例如，我们可以创建一个实例，将数据集拆分为 3 倍，在拆分之前进行混洗，并为伪随机数生成器使用值 1。

```py
kfold = KFold(3, True, 1)
```

然后可以在提供数据样本作为参数的类上调用 _split（）_ 函数。重复调用，拆分将返回每组训练和测试集。具体而言，返回包含索引的数组到观察的原始数据样本中，以用于每次迭代的训练和测试集。

例如，我们可以使用创建的`KFold`实例枚举数据样本的索引拆分，如下所示：

```py
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (train, test))
```

我们可以将所有这些与我们在前一部分的工作示例中使用的小数据集结合在一起。

```py
# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))
```

运行该示例将打印为每个训练和测试集选择的特定观察值。索引直接用于原始数据数组以检索观察值。

```py
train: [0.1 0.4 0.5 0.6], test: [0.2 0.3]
train: [0.2 0.3 0.4 0.6], test: [0.1 0.5]
train: [0.1 0.2 0.3 0.5], test: [0.4 0.6]
```

有用的是，scikit-learn 中的 k-fold 交叉验证实现作为更广泛方法中的组件操作提供，例如网格搜索模型超参数和对数据集上的模型评分。

然而，`KFold`类可以直接使用，以便在建模之前分割数据集，使得所有模型将使用相同的数据分割。如果您正在处理非常大的数据样本，这将特别有用。在算法中使用相同的拆分可以为您稍后可能希望对数据执行的统计测试带来好处。

## 交叉验证的变化

k 折交叉验证程序有许多变化。

三种常用的变体如下：

*   **训练/测试分割**：取一个极端，k 可以设置为 1，这样就可以创建单个训练/测试分裂来评估模型。
*   **LOOCV** ：从另一个极端来看，k 可以设置为数据集中观察的总数，使得每个观察结果都有机会被保留在数据集之外。这称为留一交叉验证，或简称 LOOCV。
*   **分层**：将数据分成折叠可以通过诸如确保每个折叠具有相同比例的具有给定分类值的观察值（例如类结果值）的标准来控制。这称为分层交叉验证。
*   **重复**：这是 k 次交叉验证程序重复 n 次的地方，其中重要的是，数据样本在每次重复之前进行混洗，这导致样本的不同分裂。

scikit-learn 库提供了一套交叉验证实现。您可以在[模型选择 API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) 中查看完整列表。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   查找 3 个机器学习研究论文，使用 10 的值进行 k 折交叉验证。
*   使用 k-fold 交叉验证编写自己的函数来分割数据样本。
*   开发示例以演示 scikit-learn 支持的每种主要交叉验证类型。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何在 Python 中从零开始实现重采样方法](https://machinelearningmastery.com/implement-resampling-methods-scratch-python/)
*   [使用重采样](https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/)评估 Python 中机器学习算法的表现
*   [测试和验证数据集有什么区别？](https://machinelearningmastery.com/difference-test-validation-datasets/)
*   [机器学习中的数据泄漏](https://machinelearningmastery.com/data-leakage-machine-learning/)

### 图书

*   [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。
*   [统计学习导论](http://amzn.to/2FkHqvW)，2013 年。
*   [人工智能：现代方法（第 3 版）](http://amzn.to/2thrWHq)，2009。

### API

*   [sklearn.model_selection.KFold（）API](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
    [sklearn.model_selection：模型选择 API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

### 用品

*   [维基百科上的重新取样（统计）](https://en.wikipedia.org/wiki/Resampling_(statistics))
*   [维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))的交叉验证（统计）

## 摘要

在本教程中，您发现了 k-fold 交叉验证程序的简要介绍，用于估计机器学习模型的技能。

具体来说，你学到了：

*   该 k 折交叉验证是用于估计模型对新数据的技能的过程。
*   您可以使用常用的策略为数据集选择 k 的值。
*   交叉验证中常用的变体，例如分层和重复，可用于 scikit-learn。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。