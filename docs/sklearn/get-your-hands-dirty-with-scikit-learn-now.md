# scikit-learn 中的机器学习算法秘籍

> 原文： [https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/](https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)

你必须弄脏你的手。

您可以阅读所有博客文章并观看世界上的所有视频，但在您开始练习之前，您实际上并不会真正开始学习机器。

[scikit-learn Python 库](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library")很容易启动和运行。尽管如此，我看到很多初学者犹豫不决。在这篇博文中，我想给出一些使用 scikit-learn 进行一些监督分类算法的非常简单的例子。

[![mean-shift clustering algorithm](img/d4cccaa0dbd532c10d1f94d11b71eace.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/04/plot_mean_shift_1.png)

## Scikit-Learn Recipes

您不需要了解并使用 scikit-learn 中的所有算法，至少在开始时，选择一个或两个（或少数）并仅使用这些算法。

在这篇文章中，您将看到 5 个监督分类算法的秘籍应用于 scikit-learn 库提供的小标准数据集。

秘籍是有原则的。每个例子是：

*   **Standalone** ：每个代码示例都是一个独立，完整且可执行的秘籍。
*   **Just Code** ：每个秘籍的重点都放在代码上，对机器学习理论的阐述很少。
*   **简单**：秘籍提供了常见的用例，这可能是你想要做的。
*   **一致**：所有代码示例都是一致的，并遵循相同的代码模式和样式约定。

秘籍不会探索给定算法的参数。它们提供了一个框架，您可以将其复制并粘贴到文件，项目或 python REPL 中并立即开始播放。

这些秘籍向您展示了您现在可以开始练习 scikit-learn。别推迟了。

## Logistic 回归

Logistic 回归将逻辑模型与数据拟合，并对事件的概率（0 到 1 之间）做出预测。

该秘籍显示了逻辑回归模型与虹膜数据集的拟合。因为这是一个多分类问题，逻辑回归使得预测在 0 和 1 之间，所以使用了一对一方案（每个类一个模型）。

Logistic Regression Python

```
# Logistic Regression
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

有关更多信息，请参阅 Logistic 回归的 [API 参考，以获取有关配置算法参数的详细信息。另请参阅用户指南](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)的 [Logistic 回归部分。](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

## 朴素贝叶斯

朴素贝叶斯使用贝叶斯定理来模拟每个属性与类变量的条件关系。

该秘籍显示了朴素贝叶斯模型与虹膜数据集的拟合。

Gaussian Naive Bayes Python

```
# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

有关配置算法参数的详细信息，请参阅高斯朴素贝叶斯的 [API 参考。另请参阅用户指南](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)的 [Naive Bayes 部分。](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)

## k-最近邻

k-最近邻（kNN）方法通过将类似情况定位到给定数据实例（使用相似性函数）并返回最相似数据实例的平均或大部分来做出预测。 kNN 算法可用于分类或回归。

该秘籍显示了使用 kNN 模型对虹膜数据集做出预测。

k-Nearest Neighbor Python

```
# k-Nearest Neighbor
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# load iris the datasets
dataset = datasets.load_iris()
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

有关配置算法参数的详细信息，请参阅 k-Nearest Neighbor 的 [API 参考。另请参阅用户指南](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)的 [k-Nearest Neighbor 部分。](http://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## 分类和回归树

分类和回归树（CART）是通过进行分割来构建的，这些分割可以最好地分离正在进行的类或预测的数据。 CART 算法可用于分类或回归。

此秘籍显示使用 CART 模型对虹膜数据集做出预测。

Decision Tree Classifier Python

```
# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

有关更多信息，请参阅 CART 的 [API 参考，以获取有关配置算法参数的详细信息。另请参阅用户指南](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)的[决策树部分。](http://scikit-learn.org/stable/modules/tree.html#tree)

## 支持向量机

支持向量机（SVM）是一种在转换后的问题空间中使用点的方法，它最好将类分成两组。一对一方法支持多个类的分类。 SVM 还通过使用最小允许误差量对函数建模来支持回归。

该秘籍显示了使用 SVM 模型对虹膜数据集做出预测。

Support Vector Machine Python

```
# Support Vector Machine
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
# load the iris datasets
dataset = datasets.load_iris()
# fit a SVM model to the data
model = SVC()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

有关更多信息，请参阅 SVM 的 [API 参考，以获取有关配置算法参数的详细信息。另请参阅用户指南](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)的 [SVM 部分。](http://scikit-learn.org/stable/modules/svm.html#svm)

## 摘要

在这篇文章中，您已经看到了 5 个独立的秘籍，展示了一些最受欢迎和最强大的监督分类问题。

每个示例都少于 20 行，您可以立即复制和粘贴并开始使用 scikit-learn。停止阅读并开始练习。选择一个秘籍并运行它，然后开始播放参数，看看它对结果有什么影响。