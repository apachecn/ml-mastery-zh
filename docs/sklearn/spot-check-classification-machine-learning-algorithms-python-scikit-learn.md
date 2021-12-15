# 使用 Python 和 scikit-learn 采样检查分类机器学习算法

> 原文： [https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/](https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/)

采样检查是一种发现哪些算法在您的机器学习问题上表现良好的方法。

您无法预先知道哪种算法最适合您的问题。你必须尝试一些方法，并将注意力集中在那些证明自己最有希望的方法上。

在这篇文章中，您将发现 6 种机器学习算法，您可以在使用 scikit-learn 在 Python 中检查分类问题时使用这些算法。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Spot-Check Classification Machine Learning Algorithms in Python with scikit-learn](img/f7fe09a754f0418de7d5a42e83a87ac3.jpg)

用 scikit-learn
照片分析机器学习算法 [Masahiro Ihara](https://www.flickr.com/photos/forever5yearsold/2808759067/) ，保留一些权利

## 算法现场检查

您无法预先知道哪种算法最适合您的数据集。

您必须使用反复试验来发现一个简短的算法列表，这些算法可以很好地解决您的问题，然后您可以加倍并进一步调整。我称这个过程现场检查。

问题不是：

> 我应该在数据集上使用什么算法？

相反，它是：

> 我应该在哪些算法上检查我的数据集？

您可以猜测哪些算法可能对您的数据集做得很好，这可能是一个很好的起点。

我建议尝试混合使用算法，看看哪种方法能够很好地选择数据中的结构。

*   尝试混合算法表示（例如实例和树）。
*   尝试混合使用学习算法（例如，学习相同类型的表示的不同算法）。
*   尝试混合使用建模类型（例如线性和非线性函数或参数和非参数）。

让我们具体一点。在下一节中，我们将介绍可用于在 Python 中检查下一个机器学习项目的算法。

## 算法概述

我们将看一下您可以检查数据集的 6 种分类算法。

2 线性机器学习算法：

1.  逻辑回归
2.  线性判别分析

4 种非线性机器学习算法：

1.  K 最近邻
2.  朴素贝叶斯
3.  分类和回归树
4.  支持向量机

每个秘籍都在[皮马印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上发表（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。这是一个二分类问题，其中所有属性都是数字。

每个秘籍都是完整且独立的。这意味着您可以将其复制并粘贴到您自己的项目中并立即开始使用它。

使用 10 倍交叉验证的测试工具用于演示如何检查每个机器学习算法，并且使用平均准确度测量来指示算法表现。

这些秘籍假设您了解每种机器学习算法以及如何使用它们。我们不会进入每个算法的 API 或参数化。

## 线性机器学习算法

本节演示了如何使用两种线性机器学习算法的最小秘籍：逻辑回归和线性判别分析。

### 1\. 逻辑回归

逻辑回归假定数值输入变量的高斯分布，并且可以模拟二分类问题。

您可以使用 [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 类构建逻辑回归模型。

```
# Logistic Regression Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

运行该示例打印平均估计精度。

```
0.76951469583
```

### 2.线性判别分析

线性判别分析或 LDA 是用于二元和多分类的统计技术。它也假定数值输入变量的高斯分布。

您可以使用 [LinearDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) 类构建 LDA 模型。

```
# LDA Classification
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearDiscriminantAnalysis()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example prints the mean estimated accuracy.

```
0.773462064252
```

## 非线性机器学习算法

本节演示了如何使用 4 种非线性机器学习算法的最小秘籍。

### 1\. K 最近邻

K 最近邻（或 KNN）使用距离度量来查找新实例的训练数据中的 K 个最相似的实例，并将邻居的平均结果作为预测。

您可以使用 [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 类构建 KNN 模型。

```
# KNN Classification
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
random_state = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example prints the mean estimated accuracy.

```
0.726555023923
```

### 2.朴素的贝叶斯

朴素贝叶斯计算每个类的概率以及给定每个输入值的每个类的条件概率。假设它们都是独立的（简单或朴素的假设），对新数据估计这些概率并相乘。

当使用实值数据时，假设高斯分布使用[高斯概率密度函数](https://en.wikipedia.org/wiki/Normal_distribution)容易地估计输入变量的概率。

您可以使用 [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) 类构建朴素贝叶斯模型。

```
# Gaussian Naive Bayes Classification
import pandas
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example prints the mean estimated accuracy.

```
0.75517771702
```

### 3.分类和回归树

分类和回归树（CART 或仅决策树）根据训练数据构造二叉树。通过评估训练数据中每个属性和每个属性的每个值来贪婪地选择分裂点，以便最小化成本函数（如 [Gini](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) ）。

您可以使用 [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 类构建 CART 模型。

```
# CART Classification
import pandas
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example prints the mean estimated accuracy.

```
0.692600820232
```

### 4.支持向量机

支持向量机（或 SVM）寻求最佳分隔两个类的行。那些最接近最佳分隔类的行的数据实例称为支持向量，并影响放置行的位置。 SVM 已扩展为支持多个类。

特别重要的是通过内核参数使用不同的内核函数。默认情况下使用功能强大的[径向基函数](https://en.wikipedia.org/wiki/Radial_basis_function)。

您可以使用 [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) 类构建 SVM 模型。

```
# SVM Classification
import pandas
from sklearn import model_selection
from sklearn.svm import SVC
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = SVC()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example prints the mean estimated accuracy.

```
0.651025290499
```

## 摘要

在这篇文章中，您发现了 6 种机器学习算法，您可以使用 scikit-learn 在 Python 中对您的分类问题进行采样检查。

具体来说，您学会了如何进行抽查：

2 线性机器学习算法

1.  逻辑回归
2.  线性判别分析

4 种非线性机器学习算法

1.  K 最近邻
2.  朴素贝叶斯
3.  分类和回归树
4.  支持向量机

您对现场检查机器学习算法或此帖子有任何疑问吗？在下面的评论部分提出您的问题，我会尽力回答。