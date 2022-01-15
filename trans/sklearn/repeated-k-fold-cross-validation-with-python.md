# Python 中用于模型评估的重复 k 折交叉验证

> 原文：<https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/>

最后更新于 2020 年 8 月 26 日

k-fold 交叉验证过程是一种评估数据集上机器学习算法或配置表现的标准方法。

k 倍交叉验证程序的一次运行可能会导致模型表现的噪声估计。不同的数据分割可能会导致截然不同的结果。

重复的 k 折交叉验证提供了一种提高机器学习模型估计表现的方法。这包括多次重复交叉验证程序，并报告所有运行的所有折叠的平均结果。该平均结果有望成为数据集上模型的真实未知潜在平均表现的更准确估计，使用标准误差进行计算。

在本教程中，您将发现模型评估的重复 k 倍交叉验证。

完成本教程后，您将知道:

*   k 倍交叉验证单次运行报告的平均表现可能会有噪声。
*   重复的 k 折交叉验证提供了一种减少平均模型表现估计误差的方法。
*   如何在 Python 中使用重复的 k 折交叉验证来评估机器学习模型。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Repeated k-Fold Cross-Validation for Model Evaluation in Python](img/748e76709443cfaa0d189a0e60cf0982.png)

Python 中模型评估的重复 k-Fold 交叉验证
图片由 [lina smith](https://flickr.com/photos/linasmith/4720796600/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  k 折叠交叉验证
2.  重复 k 折叠交叉验证
3.  Python 中重复的 k 折叠交叉验证

## k 折叠交叉验证

使用 k 折交叉验证在数据集上评估机器学习模型是很常见的。

k 折叠交叉验证过程将有限的数据集分成 k 个不重叠的折叠。k 个折叠中的每一个都有机会用作保留测试集，而所有其他折叠一起用作训练数据集。在 k 个保持测试集上，对总共 k 个模型进行拟合和评估，并报告平均表现。

有关 k-fold 交叉验证过程的更多信息，请参见教程:

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

使用 Sklearn 机器学习库可以轻松实现 k-fold 交叉验证过程。

首先，让我们定义一个可以作为本教程基础的综合分类数据集。

[make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)可用于创建合成二进制分类数据集。我们将配置它生成 1000 个样本，每个样本有 20 个输入特征，其中 15 个有助于目标变量。

下面的示例创建并汇总了数据集。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例会创建数据集，并确认它包含 1，000 个样本和 10 个输入变量。

伪随机数发生器的固定种子确保我们每次生成数据集时获得相同的样本。

```py
(1000, 20) (1000,)
```

接下来，我们可以使用 k-fold 交叉验证来评估这个数据集上的模型。

我们将评估一个[logisticreduce](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)模型，并使用 [KFold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html)执行交叉验证，配置为洗牌数据集并设置 k=10，这是一个流行的默认值。

[cross_val_score()函数](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)将用于执行评估，获取数据集和交叉验证配置，并返回为每个折叠计算的分数列表。

下面列出了完整的示例。

```py
# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会创建数据集，然后使用 10 倍交叉验证对其进行逻辑回归模型评估。然后报告数据集的平均分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到该模型实现了大约 86.8%的估计分类准确率。

```py
Accuracy: 0.868 (0.032)
```

现在我们已经熟悉了 k-fold 交叉验证，让我们看看重复该过程的扩展。

## 重复 k 折叠交叉验证

通过 k 倍交叉验证对模型表现的估计可能会有噪声。

这意味着每次运行该过程时，可以实现将数据集分成 k 个不同的折叠，而表现分数的分布也可能不同，从而导致模型表现的不同平均估计。

从一次 k 倍交叉验证到另一次 k 倍交叉验证的估计表现差异取决于所使用的模型和数据集本身。

模型表现的噪声估计可能令人沮丧，因为可能不清楚应该使用哪个结果来比较和选择最终模型来解决问题。

降低估计模型表现中的噪声的一个解决方案是增加 k 值。这将减少模型估计表现的偏差，尽管会增加方差:例如，将结果更多地与评估中使用的特定数据集联系起来。

另一种方法是多次重复 k 折叠交叉验证过程，并报告所有折叠和所有重复的平均表现。这种方法通常被称为重复的 k 倍交叉验证。

> …重复的 k-fold 交叉验证将程序重复【…】次。例如，如果 10 倍交叉验证重复 5 次，将使用 50 个不同的搁置集来评估模型功效。

—第 70 页，[应用预测建模](https://amzn.to/3aeBRyH)，2013 年。

重要的是，k-fold 交叉验证过程的每个重复必须在分割成不同折叠的同一数据集上执行。

重复 k 折交叉验证的好处是以拟合和评估更多模型为代价来改进平均模型表现的估计。

常见的重复数包括 3、5 和 10。例如，如果使用 10 倍交叉验证的 3 次重复来估计模型表现，这意味着需要拟合和评估(3 * 10)或 30 个不同的模型。

*   **合适的**:对于小数据集和简单模型(如线性)。

因此，这种方法适用于计算成本不高的小型到中等规模的数据集和/或模型。这表明该方法可能适用于线性模型，而不适用于像深度学习神经网络这样的慢拟合模型。

像 k-fold 交叉验证本身一样，重复的 k-fold 交叉验证很容易并行化，其中每个折叠或每个重复的交叉验证过程可以在不同的内核或不同的机器上执行。

## Python 中重复的 k 折叠交叉验证

Sklearn Python 机器学习库通过 [RepeatedKFold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)提供了重复 k 折交叉验证的实现。

主要参数是折叠数( *n_splits* )，即 k 折叠交叉验证中的“ *k* ”，以及重复数( *n_repeats* )。

k 的良好默认值是 k=10。

重复次数的良好默认值取决于数据集上模型表现估计的噪声程度。值为 3、5 或 10 的重复可能是一个好的开始。可能不需要超过 10 次的重复。

```py
...
# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

下面的例子演示了我们的测试数据集的重复 k 倍交叉验证。

```py
# evaluate a logistic regression model using repeated k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会创建数据集，然后使用具有三次重复的 10 倍交叉验证对数据集的逻辑回归模型进行评估。然后报告数据集的平均分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了大约 86.7%的估计分类准确率，这低于之前报告的 86.8%的单次运行结果。这可能表明单次运行结果可能是乐观的，三次重复的结果可能是对真实均值模型表现的更好估计。

```py
Accuracy: 0.867 (0.031)
```

重复 k 倍交叉验证的预期是，重复平均值将是比单一 k 倍交叉验证程序的结果更可靠的模型表现估计。

这可能意味着更少的统计噪声。

衡量这一点的一种方法是比较不同重复次数下平均成绩的分布。

我们可以想象，数据集上的模型有一个真正未知的潜在平均表现，重复的 k 倍交叉验证运行估计了这个平均值。我们可以使用称为标准误差的统计工具，根据真正未知的潜在平均表现来估计平均表现的误差。

[标准误差](https://en.wikipedia.org/wiki/Standard_error)可以为给定样本量提供误差量或误差传播的指示，该误差量或误差传播可以从样本均值到潜在的未知总体均值。

标准误差可以计算如下:

*   标准误差=样本标准偏差/ sqrt(重复次数)

我们可以使用 [sem() scipy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html)计算样品的标准误差。

理想情况下，我们希望选择与其他重复次数相比，既能最小化标准误差又能稳定平均估计表现的重复次数。

下面的例子通过用 10 倍交叉验证报告模型表现来证明这一点，该过程重复 1 到 15 次。

考虑到[大数定律](https://machinelearningmastery.com/a-gentle-introduction-to-the-law-of-large-numbers-in-machine-learning/)，我们预计程序的更多重复将导致平均模型表现的更精确估计。虽然，这些试验不是独立的，所以潜在的统计原理变得具有挑战性。

```py
# compare the number of repeats for repeated k-fold cross-validation
from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	model = LogisticRegression()
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# configurations to test
repeats = range(1,16)
results = list()
for r in repeats:
	# evaluate using a given number of repeats
	scores = evaluate_model(X, y, r)
	# summarize
	print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
	# store
	results.append(scores)
# plot the results
pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()
```

运行该示例使用不同重复次数的 10 倍交叉验证来报告平均和标准误差分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，与其他结果相比，一个重复的默认值似乎是乐观的，准确率约为 86.80%，而不是 86.73%，并且随着重复次数的不同而降低。

我们可以看到平均值似乎在 86.5%左右。我们可以将此作为模型表现的稳定估计值，然后依次选择 5 或 6 个看起来最接近此值的重复。

查看标准误差，我们可以看到它随着重复次数的增加而减少，并在大约 9 或 10 次重复时稳定在大约 0.003 的值，尽管 5 次重复达到 0.005 的标准误差，是单次重复的一半。

```py
>1 mean=0.8680 se=0.011
>2 mean=0.8675 se=0.008
>3 mean=0.8673 se=0.006
>4 mean=0.8670 se=0.006
>5 mean=0.8658 se=0.005
>6 mean=0.8655 se=0.004
>7 mean=0.8651 se=0.004
>8 mean=0.8651 se=0.004
>9 mean=0.8656 se=0.003
>10 mean=0.8658 se=0.003
>11 mean=0.8655 se=0.003
>12 mean=0.8654 se=0.003
>13 mean=0.8652 se=0.003
>14 mean=0.8651 se=0.003
>15 mean=0.8653 se=0.003
```

创建一个方框和触须图来总结每个重复次数的分数分布。

橙色线表示分布的中间值，绿色三角形表示算术平均值。如果这些符号(值)一致，则表明存在合理的对称分布，平均值可以很好地捕捉中心趋势。

这可能会为您的测试工具选择合适的重复次数提供额外的启发。

考虑到这一点，对这个选择的测试工具和算法使用五次重复似乎是一个不错的选择。

![Box and Whisker Plots of Classification Accuracy vs Repeats for k-Fold Cross-Validation](img/4c23d812c9753375f84f78b42716396b.png)

k 倍交叉验证的分类准确度与重复次数的方框图和须图

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)
*   [如何固定不平衡分类的 k 折交叉验证](https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/)

### 蜜蜂

*   [sklearn.model_selection。KFold API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 。
*   [sklearn.model_selection。重复应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)。
*   [sklearn.model_selection。离开应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)。
*   [sklearn . model _ selection . cross _ val _ score API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

### 文章

*   [交叉验证(统计)，维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))。
*   [标准错误，维基百科](https://en.wikipedia.org/wiki/Standard_error)。

## 摘要

在本教程中，您发现了模型评估的重复 k 倍交叉验证。

具体来说，您了解到:

*   k 倍交叉验证单次运行报告的平均表现可能会有噪声。
*   重复的 k 折交叉验证提供了一种减少平均模型表现估计误差的方法。
*   如何在 Python 中使用重复的 k 折交叉验证来评估机器学习模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。