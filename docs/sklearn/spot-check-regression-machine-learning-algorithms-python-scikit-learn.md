# 使用 Python 和 scikit-learn 抽样检查回归机器学习算法

> 原文： [https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/](https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/)

抽样检查是一种发现哪些算法在您的机器学习问题上表现良好的方法。

您无法预先知道哪种算法最适合您的问题。你必须尝试一些方法，并将注意力集中在那些证明自己最有希望的方法上。

在这篇文章中，您将发现 6 种机器学习算法，您可以在使用 scikit-learn 在 Python 中检查回归问题时使用这些算法。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Spot-Check Regression Machine Learning Algorithms in Python with scikit-learn](img/e67465c4445862a8ff9339a0116cd811.jpg)

使用 scikit-learn
照片通过 [frankieleon](https://www.flickr.com/photos/armydre2008/8004733173/) 在 Python 中使用 Spot-Check 回归机器学习算法，保留一些权利。

## 算法概述

我们将看一下您可以检查数据集的 7 种分类算法。

4 线性机器学习算法：

1.  线性回归
2.  岭回归
3.  LASSO 线性回归
4.  弹性网络回归

3 种非线性机器学习算法：

1.  K 最近邻
2.  分类和回归树
3.  支持向量机

每个秘籍都在 [Boston House Price 数据集](https://archive.ics.uci.edu/ml/datasets/Housing)上进行演示。这是一个回归问题，其中所有属性都是数字的（更新：[从这里下载数据](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data)）。

每个秘籍都是完整且独立的。这意味着您可以将其复制并粘贴到您自己的项目中并立即开始使用它。

使用具有 10 倍交叉验证的测试工具来演示如何检查每个机器学习算法，并且使用均方误差测量来指示算法表现。注意，均方误差值是反转的（负）。这是所使用的`cross_val_score()`函数的一个怪癖，它要求所有算法指标按升序排序（值越大越好）。

这些秘籍假设您了解每种机器学习算法以及如何使用它们。我们不会进入每个算法的 API 或参数化。

## 线性机器学习算法

本节提供了如何使用 4 种不同的线性机器学习算法在 Python 中使用 scikit-learn 进行回归的示例。

### 1.线性回归

线性回归假设输入变量具有高斯分布。还假设输入变量与输出变量相关，并且它们彼此之间不高度相关（称为共线性的问题）。

您可以使用 [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) 类构建线性回归模型。

```
# Linear Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

运行该示例提供了均方误差的估计。

```
-34.7052559445
```

### 2.岭回归

岭回归是线性回归的扩展，其中损失函数被修改以最小化模型的复杂度，其被测量为系数值的总平方值（也称为 l2 范数）。

您可以使用 [Ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) 类构建岭回归模型。

```
# Ridge Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import Ridge
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = Ridge()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

运行该示例提供了均方误差的估计。

```
-34.0782462093
```

### 3\. LASSO 回归

最小绝对收缩和选择算子（或简称 LASSO）是线性回归的修改，如岭回归，其中损失函数被修改以最小化模型的复杂度，测量为系数值的总和绝对值（也称为 l1-规范）。

您可以使用 [Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) 类构建 LASSO 模型。

```
# Lasso Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import Lasso
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = Lasso()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

Running the example provides an estimate of the mean squared error.

```
-34.4640845883
```

### 4.弹性网络回归

ElasticNet 是一种正则化回归形式，它结合了岭回归和 LASSO 回归的特性。它试图通过使用 l2 范数（和平方系数值）和 l1 范数（和绝对系数值）惩罚模型来最小化回归模型的复杂性（回归系数的大小和数量）。

您可以使用 ElasticNet 类构建 [ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) 模型。

```
# ElasticNet Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

Running the example provides an estimate of the mean squared error.

```
-31.1645737142
```

## 非线性机器学习算法

本节提供了如何使用 3 种不同的非线性机器学习算法在 Python 中使用 scikit-learn 进行回归的示例。

### 1\. K 最近邻

K 最近邻（或 KNN）在训练数据集中为新数据实例定位 K 个最相似的实例。从 K 个邻居中，将平均或中值输出变量作为预测。值得注意的是使用的距离度量（_ 度量 _ 参数）。默认情况下使用 [Minkowski 距离](https://en.wikipedia.org/wiki/Minkowski_distance)，它是欧几里德距离（当所有输入具有相同比例时使用）和曼哈顿距离（当输入变量的比例不同时）的推广。

您可以使用 [KNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) 类构建用于回归的 KNN 模型。

```
# KNN Regression
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

Running the example provides an estimate of the mean squared error.

```
-107.28683898
```

### 2.分类和回归树

决策树或分类和回归树（已知的 CART）使用训练数据来选择分割数据的最佳点，以便最小化成本度量。回归决策树的默认成本度量标准是在标准参数中指定的均方误差。

您可以使用 [DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) 类为回归创建 CART 模型。

```
# Decision Tree Regression
import pandas
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

Running the example provides an estimate of the mean squared error.

```
-35.4906027451
```

### 3.支持向量机

支持向量机（SVM）是为二分类而开发的。该技术已被扩展用于称为支持向量回归（SVR）的预测实值问题。与分类示例一样，SVR 建立在 [LIBSVM 库](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)之上。

您可以使用 [SVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) 类为回归创建 SVM 模型。

```
# SVM Regression
import pandas
from sklearn import model_selection
from sklearn.svm import SVR
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = SVR()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

Running the example provides an estimate of the mean squared error.

```
-91.0478243332
```

## 摘要

在这篇文章中，您使用 scikit-learn 在 Python 中发现了用于回归的机器学习秘籍。

具体来说，您了解到：

4 Linear Machine Learning Algorithms:

*   线性回归
*   岭回归
*   LASSO 线性回归
*   弹性网络回归

3 Nonlinear Machine Learning Algorithms:

*   K 最近邻
*   分类和回归树
*   支持向量机

您对回归机器学习算法或这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。