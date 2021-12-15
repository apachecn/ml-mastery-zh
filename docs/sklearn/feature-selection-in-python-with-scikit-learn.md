# 使用 Scikit-Learn 在 Python 中进行特征选择

> 原文： [https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/](https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/)

并非所有数据属性都是相同的。对于数据集中的属性或列，更多并不总是更好。

在本文中，您将了解如何在使用 [scikit-learn 库](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library")创建机器学习模型之前选择数据中的属性。

**更新**：有关 Python 中功能选择的最新教程，请参阅帖子：

*   [Python 机器学习的特征选择](http://machinelearningmastery.com/feature-selection-machine-learning-python/)

[![feature selection](img/17392bfd8704b46a8fe5e5c53f82378b.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/feature-selection.jpg)

通过功能选择减少您的选项
照片由 [Josh Friedman](https://www.flickr.com/photos/joshfriedmantravel/4935712614) ，保留一些权利

## 选择功能

特征选择是一个过程，您可以自动选择数据中对您感兴趣的预测变量或输出贡献最大的那些特征。

在数据中包含太多不相关的功能会降低模型的准确性。在建模数据之前执行特征选择的三个好处是：

*   **减少过拟合**：冗余数据越少意味着根据噪声做出决策的机会就越少。
*   **提高准确度**：误导性较差的数据意味着建模精度提高。
*   **减少训练**时间：数据越少意味着算法训练越快。

scikit-learn Python 库提供的两种不同的特征选择方法是递归特征消除和特征重要性排序。

## 递归特征消除

递归特征消除（RFE）方法是特征选择方法。它的工作原理是递归删除属性并在剩余的属性上构建模型。它使用模型精度来识别哪些属性（和属性组合）对预测目标属性的贡献最大。

此秘籍显示在 Iris floweres 数据集上使用 RFE 以选择 3 个属性。

Recursive Feature Elimination Python

```
# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
```

有关更多信息，请参阅 API 文档中的 [RFE 方法](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)。

## 功能重要性

使用决策树集合（如随机森林或额外树）的方法也可以计算每个属性的相对重要性。这些重要性值可用于通知特征选择过程。

此秘籍显示了虹膜花数据集的额外树组合的构造以及相对特征重要性的显示。

Feature Importance with datasets.load_iris() # fit an Extra Python

```
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.feature_importances_)
```

有关更多信息，请参阅 API 文档中的 [ExtraTreesClassifier 方法](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)。

## 摘要

特征选择方法可以为您提供有关特定问题的特征的相对重要性或相关性的有用信息。您可以使用此信息创建数据集的过滤版本，并提高模型的准确性。

在这篇文章中，您发现了两个可以使用 scikit-learn 库在 Python 中应用的特征选择方法。