# 带标签扩散的半监督学习

> 原文：<https://machinelearningmastery.com/semi-supervised-learning-with-label-spreading/>

**半监督学习**是指试图同时利用已标记和未标记训练数据的算法。

半监督学习算法不像监督学习算法那样只能从有标签的训练数据中学习。

半监督学习的一种流行方法是创建一个连接训练数据集中的示例的图，并通过图的边缘传播已知的标签来标记未标记的示例。这种半监督学习方法的一个例子是用于分类预测建模的**标签传播算法**。

在本教程中，您将发现如何将标签扩散算法应用于半监督学习分类数据集。

完成本教程后，您将知道:

*   标签传播半监督学习算法如何工作的直觉。
*   如何使用监督学习算法开发半监督分类数据集并建立性能基线。
*   如何开发和评估标签传播算法，并使用模型输出来训练监督学习算法。

我们开始吧。

![Semi-Supervised Learning With Label Spreading](img/d5f04cc49b9544bec4c4697a73ebdc75.png)

带有标签传播的半监督学习
图片由 [Jernej Furman](https://www.flickr.com/photos/91261194@N06/44897768224/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  标签扩展算法
2.  半监督分类数据集
3.  半监督学习中的标签扩散

## 标签扩展算法

标签扩散是一种半监督学习算法。

该算法是由周等人在 2003 年的论文《具有局部和全局一致性的学习》中提出的

半监督学习的更广泛方法的直觉是，输入空间中的邻近点应该具有相同的标签，并且输入空间中的相同结构或流形中的点应该具有相同的标签。

> 半监督学习问题的关键是一致性的先验假设，这意味着:(1)附近的点很可能有相同的标签；以及(2)相同结构上的点(通常称为簇或流形)可能具有相同的标签。

——[具有局部和全局一致性的学习](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)，2003。

标签传播的灵感来自实验心理学的一项技术，叫做传播激活网络。

> 这个算法可以从实验心理学传播激活网络的角度直观地理解。

——[具有局部和全局一致性的学习](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)，2003。

数据集中的点根据它们在输入空间中的相对距离连接成一个图形。图的权重矩阵是对称归一化的，很像[谱聚类](https://machinelearningmastery.com/clustering-algorithms-with-python/)。信息通过图形传递，图形适于捕捉输入空间中的结构。

该方法与半监督学习的标签传播算法非常相似。

> 另一个类似的标签传播算法由 Zhou 等人给出:在每一步，节点 I 接收来自其邻居 j 的贡献(由边(I，j)的归一化权重加权)，以及由其初始值给出的额外的小贡献

—第 196 页，[半监督学习](https://amzn.to/3fVfO3O)，2006。

收敛后，基于传递最多信息的节点应用标签。

> 最后，每个未标记点的标签被设置为在迭代过程中接收到最多信息的类别。

——[具有局部和全局一致性的学习](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)，2003。

现在我们已经熟悉了标签传播算法，让我们看看如何在项目中使用它。首先，我们必须定义一个半监督分类数据集。

## 半监督分类数据集

在本节中，我们将为半监督学习定义一个数据集，并在该数据集上建立一个性能基线。

首先，我们可以使用[make _ classion()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个合成分类数据集。

我们将用两个类(二进制分类)和两个输入变量以及 1000 个示例来定义数据集。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
```

接下来，我们将数据集分割成训练数据集和测试数据集，分割比例为 50-50(例如，每组 500 行)。

```py
...
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
```

最后，我们将训练数据集再次分成两部分，一部分有标签，另一部分我们假装没有标签。

```py
...
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
```

将这些联系在一起，下面列出了准备半监督学习数据集的完整示例。

```py
# prepare semi-supervised learning dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# summarize training set size
print('Labeled Train Set:', X_train_lab.shape, y_train_lab.shape)
print('Unlabeled Train Set:', X_test_unlab.shape, y_test_unlab.shape)
# summarize test set size
print('Test Set:', X_test.shape, y_test.shape)
```

运行该示例准备数据集，然后总结三个部分的形状。

结果证实，我们有一个 500 行的测试数据集、一个 250 行的标记训练数据集和 250 行的未标记数据。

```py
Labeled Train Set: (250, 2) (250,)
Unlabeled Train Set: (250, 2) (250,)
Test Set: (500, 2) (500,)
```

一个有监督的学习算法只有 250 行来训练一个模型。

半监督学习算法将具有 250 个标记行以及 250 个未标记行，这些行可以以多种方式用于改进标记的训练数据集。

接下来，我们可以使用仅适用于标记训练数据的监督学习算法，在半监督学习数据集上建立性能基线。

这一点很重要，因为我们期望半监督学习算法的性能优于仅适用于标记数据的监督学习算法。如果不是这样，那么半监督学习算法就没有技巧。

在这种情况下，我们将使用逻辑回归算法来拟合训练数据集的标记部分。

```py
...
# define model
model = LogisticRegression()
# fit model on labeled dataset
model.fit(X_train_lab, y_train_lab)
```

然后，该模型可用于对整个保持测试数据集进行预测，并使用分类精确率进行评估。

```py
...
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

将这些联系在一起，下面列出了在半监督学习数据集上评估监督学习算法的完整示例。

```py
# baseline performance on the semi-supervised learning dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# define model
model = LogisticRegression()
# fit model on labeled dataset
model.fit(X_train_lab, y_train_lab)
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

运行该算法使模型适合标记的训练数据集，并在保持数据集上对其进行评估，并打印分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到该算法实现了大约 84.8%的分类准确率。

我们期望一种有效的半监督学习算法能获得比这更好的精确率。

```py
Accuracy: 84.800
```

接下来，让我们探索如何将标签扩散算法应用于数据集。

## 半监督学习中的标签扩散

标签传播算法可通过[标签传播类](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html)在 scikit-learn Python 机器学习库中获得。

通过调用 *fit()* 函数，该模型可以像任何其他分类模型一样进行拟合，并通过 *predict()* 函数用于对新数据进行预测。

```py
...
# define model
model = LabelSpreading()
# fit model on training dataset
model.fit(..., ...)
# make predictions on hold out test set
yhat = model.predict(...)
```

重要的是，提供给 *fit()* 函数的训练数据集必须包括有序编码的已标记示例(按照正常情况)和标记为-1 的未标记示例。

然后，作为拟合模型的一部分，模型将确定未标记示例的标签。

模型拟合后，训练数据集中已标记和未标记数据的估计标签可通过*标签预标记*类上的“*转导 _* ”属性获得。

```py
...
# get labels for entire training dataset data
tran_labels = model.transduction_
```

现在我们已经熟悉了如何在 scikit-learn 中使用标签传播算法，让我们看看如何将其应用于我们的半监督学习数据集。

首先，我们必须准备训练数据集。

我们可以将训练数据集的输入数据连接成一个数组。

```py
...
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
```

然后，我们可以为训练数据集中未标记部分的每一行创建一个-1 值(未标记)的列表。

```py
...
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
```

然后，该列表可以与来自训练数据集标记部分的标签连接起来，以对应于训练数据集的输入数组。

```py
...
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))
```

我们现在可以在整个训练数据集上训练*标签预定义*模型。

```py
...
# define model
model = LabelSpreading()
# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
```

接下来，我们可以使用该模型对保持数据集进行预测，并使用分类精确率评估该模型。

```py
...
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

将这些联系在一起，下面列出了在半监督学习数据集上评估标签扩散的完整示例。

```py
# evaluate label spreading on the semi-supervised learning dataset
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))
# define model
model = LabelSpreading()
# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

运行该算法使模型适合整个训练数据集，并在保持数据集上对其进行评估，并打印分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到标签传播模型实现了大约 85.4%的分类精确率，这略高于仅在实现了大约 84.8%的精确率的标签训练数据集上的逻辑回归拟合。

```py
Accuracy: 85.400
```

目前为止一切顺利。

对于半监督模型，我们可以使用的另一种方法是获取训练数据集的估计标签，并拟合监督学习模型。

回想一下，我们可以从标签传播模型中检索整个训练数据集的标签，如下所示:

```py
...
# get labels for entire training dataset data
tran_labels = model.transduction_
```

然后，我们可以使用这些标签以及所有输入数据来训练和评估监督学习算法，例如逻辑回归模型。

希望适合整个训练数据集的监督学习模型将获得比单独的半监督学习模型更好的性能。

```py
...
# define supervised learning model
model2 = LogisticRegression()
# fit supervised learning model on entire training dataset
model2.fit(X_train_mixed, tran_labels)
# make predictions on hold out test set
yhat = model2.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

将这些联系在一起，下面列出了使用估计的训练集标签来训练和评估监督学习模型的完整示例。

```py
# evaluate logistic regression fit on label spreading for semi-supervised learning
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))
# define model
model = LabelSpreading()
# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
# get labels for entire training dataset data
tran_labels = model.transduction_
# define supervised learning model
model2 = LogisticRegression()
# fit supervised learning model on entire training dataset
model2.fit(X_train_mixed, tran_labels)
# make predictions on hold out test set
yhat = model2.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))
```

运行该算法将半监督模型拟合到整个训练数据集上，然后将监督学习模型拟合到具有推断标签的整个训练数据集上，并在保持数据集上对其进行评估，打印分类精确率。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，半监督模型跟随监督模型的分层方法在保持数据集上实现了大约 85.8%的分类精确率，略好于单独使用的半监督学习算法，该算法实现了大约 85.6%的精确率。

```py
Accuracy: 85.800
```

**通过调优 LabelSpreading 模型的超参数，能达到更好的效果吗？**
让我知道你在下面的评论中发现了什么。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [半监督学习导论](https://amzn.to/37niYJw)，2009。
*   第十一章:标签传播与二次准则，[半监督学习](https://amzn.to/3fVfO3O)，2006。

### 报纸

*   [具有局部和全局一致性的学习](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)，2003。

### 蜜蜂

*   [sklearn.semi_supervised。标签预发布 API](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html) 。
*   [第 1.14 节。半监督，Scikit-学习用户指南](https://scikit-learn.org/stable/modules/label_propagation.html)。
*   [sklearn . model _ selection . train _ test _ split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
*   [sklearn.linear_model。物流配送应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
*   [sklearn . datasets . make _ classification API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。

### 文章

*   [半监督学习，维基百科](https://en.wikipedia.org/wiki/Semi-supervised_learning)。

## 摘要

在本教程中，您发现了如何将标签扩散算法应用于半监督学习分类数据集。

具体来说，您了解到:

*   标签传播半监督学习算法如何工作的直觉。
*   如何使用监督学习算法开发半监督分类数据集并建立性能基线。
*   如何开发和评估标签传播算法，并使用模型输出来训练监督学习算法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。