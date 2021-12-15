# 用于评估 Python 中机器学习算法的度量标准

> 原文： [https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)

您选择用于评估机器学习算法的指标非常重要。

度量的选择会影响如何测量和比较机器学习算法的表现。它们会影响您如何权衡结果中不同特征的重要性以及您选择哪种算法的最终选择。

在本文中，您将了解如何使用 scikit-learn 在 Python 中选择和使用不同的机器学习表现指标。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Metrics To Evaluate Machine Learning Algorithms in Python](img/f219de3133049abf483c571db3e07320.jpg)

用于评估 Python 中的机器学习算法的度量
照片由[FerrousBüller](https://www.flickr.com/photos/lumachrome/4898510377)拍摄，保留一些权利。

## 关于秘籍

本文使用 Python 和 scikit-learn 中的小代码秘籍演示了各种不同的机器学习评估指标。

每个秘籍都是独立设计的，因此您可以将其复制并粘贴到项目中并立即使用。

针对分类和回归类型的机器学习问题演示了度量标准。

*   对于分类指标， [Pima 印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)的发病被用作示范。这是一个二分类问题，其中所有输入变量都是数字的（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。
*   对于回归指标， [Boston House Price 数据集](https://archive.ics.uci.edu/ml/datasets/Housing)用作演示。这是一个回归问题，其中所有输入变量也是数字的（更新：[从这里下载数据](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data)）。

在每个秘籍中，数据集直接从 [UCI 机器学习库](https://archive.ics.uci.edu/ml/index.html)下载。

所有秘籍都会评估相同的算法，分类的 Logistic 回归和回归问题的线性回归。 10 倍交叉验证测试工具用于演示每个指标，因为这是您将采用不同算法评估指标的最可能情况。

这些秘籍中的一个警告是 [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html) 函数用于报告每个秘籍中的表现。它允许使用将要讨论的不同评分指标，但报告所有得分以便可以对它们进行排序升序（最高分是最好的）。

一些评估指标（如均方误差）是自然下降的分数（最小分数最好），因此`cross_val_score()`函数报告为负数。这一点很重要，因为有些分数会被报告为负数，根据定义，它们永远不会是负数。

您可以在页面上了解更多关于 scikit-learn 支持的机器学习算法表现指标[模型评估：量化预测质量](http://scikit-learn.org/stable/modules/model_evaluation.html)。

让我们继续评估指标。

## 分类指标

分类问题可能是最常见的机器学习问题类型，因此有无数的度量标准可用于评估这些问题的预测。

在本节中，我们将介绍如何使用以下指标：

1.  分类准确率。
2.  对数损失。
3.  ROC 曲线下面积。
4.  混乱矩阵。
5.  分类报告。

### 1.分类准确率

分类准确度是作为所有预测的比率而作出的正确预测的数量。

这是分类问题最常见的评估指标，也是最被误用的。它实际上只适用于每个类中存在相同数量的观测值（这种情况很少发生）并且所有预测和预测误差同样重要，而事实并非如此。

以下是计算分类准确度的示例。

```
# Cross Validation Classification Accuracy
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
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
```

您可以看到报告的比率。可以通过将该值乘以 100 将其转换为百分比，从而使准确度得分大约为 77％。

```
Accuracy: 0.770 (0.048)
```

### 2.对数损失

对数损失（或 logloss）是用于评估给定类的成员概率的预测的表现度量。

0 和 1 之间的标量概率可以被视为算法预测的置信度的度量。正确或不正确的预测会与预测的置信度成比例地得到奖励或惩罚。

您可以在分类维基百科文章的[损失函数中了解更多关于对数的信息。](https://en.wikipedia.org/wiki/Loss_functions_for_classification)

以下是计算 Pima Indians 糖尿病数据集开始时 Logistic 回归预测的 logloss 的示例。

```
# Cross Validation Classification LogLoss
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
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
```

较小的 logloss 更好，0 表示完美的 logloss。如上所述，当使用`cross_val_score()`函数时，度量被反转为上升。

```
Logloss: -0.493 (0.047)
```

### 3\. ROC 曲线下的面积

ROC 曲线下面积（或简称 AUC）是二分类问题的表现指标。

AUC 代表模型区分正面和负面类别的能力。面积为 1.0 表示完美地预测所有预测的模型。 0.5 的面积表示随机的模型。 [了解更多有关 ROC 的信息](http://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)。

ROC 可以分解为敏感性和特异性。二分类问题实际上是敏感性和特异性之间的权衡。

*   敏感度是真正的正面率，也称为召回率。它是实际正确预测的正（第一）类的数字实例。
*   特异性也称为真正的负面率。是负类（第二）类中实际预测的实例数是否正确。

您可以在维基百科页面上了解有关 [ROC 的更多信息。](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

以下示例提供了计算 AUC 的演示。

```
# Cross Validation Classification ROC AUC
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
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
```

您可以看到 AUC 相对接近 1 且大于 0.5，这表明预测中有一些技巧。

```
AUC: 0.824 (0.041)
```

### 4.混淆矩阵

混淆矩阵是具有两个或更多类的模型的准确率的便利表示。

该表提供了关于 x 轴的预测和 y 轴上的准确度结果。表格的单元格是机器学习算法所做的预测次数。

例如，机器学习算法可以预测 0 或 1，并且每个预测实际上可以是 0 或 1.对于 0 实际为 0 的预测出现在用于预测= 0 和实际= 0 的单元格中，而对于 0 的预测是 0 实际上 1 出现在单元格中，用于预测= 0 和实际= 1。等等。

您可以在维基百科文章上了解有关[混淆矩阵的更多信息。](https://en.wikipedia.org/wiki/Confusion_matrix)

下面是通过测试集上的模型计算一组预测的混淆矩阵的示例。

```
# Cross Validation Classification Confusion Matrix
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
```

虽然阵列的打印没有标题，但您可以看到大多数预测都落在矩阵的对角线上（这是正确的预测）。

```
[[141  21]
 [ 41  51]]
```

### 5.分类报告

在处理分类问题时，Scikit-learn 确实提供了便利报告，使您可以使用多种方法快速了解模型的准确率。

`classification_report()`函数显示每个类的精度，召回率，f1 分数和支持。

下面的示例演示了有关二分类问题的报告。

```
# Cross Validation Classification Report
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
```

您可以看到该算法的良好预测和召回。

```
             precision    recall  f1-score   support

        0.0       0.77      0.87      0.82       162
        1.0       0.71      0.55      0.62        92

avg / total       0.75      0.76      0.75       254
```

## 回归指标

在本节中，将回顾 3 个用于评估回归机器学习问题预测的最常用指标：

1.  平均绝对误差。
2.  均方误差。
3.  R ^ 2。

### 1.平均绝对误差

平均绝对误差（或 MAE）是预测值与实际值之间的绝对差值之和。它给出了预测错误的概念。

该度量给出了误差幅度的概念，但不知道方向（例如，过度或低于预测）。

您可以在 Wikipedia 上了解有关[平均绝对误差的更多信息。](https://en.wikipedia.org/wiki/Mean_absolute_error)

以下示例演示了计算波士顿房价数据集的平均绝对误差。

```
# Cross Validation Regression MAE
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
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
```

值 0 表示没有错误或完美预测。与 logloss 一样，该指标由`cross_val_score()`函数反转。

```
MAE: -4.005 (2.084)
```

### 2.均方误差

均方误差（或 MSE）非常类似于平均绝对误差，因为它提供了误差幅度的总体思路。

取均方误差的平方根将单位转换回输出变量的原始单位，对描述和表示有意义。这称为均方根误差（或均方根）。

您可以在维基百科上了解有关[均方误差的更多信息。](https://en.wikipedia.org/wiki/Mean_squared_error)

以下示例提供了计算均方误差的演示。

```
# Cross Validation Regression MSE
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
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
```

此度量标准也被反转，因此结果会增加。如果您有兴趣计算 RMSE，请记住在取平方根之前取绝对值。

```
MSE: -34.705 (45.574)
```

### 3\. R ^ 2 公制

R ^ 2（或 R Squared）度量提供了一组预测与实际值的拟合优度的指示。在统计文献中，该度量被称为确定系数。

对于不适合和完美贴合，这是 0 到 1 之间的值。

您可以在维基百科上了解更多关于[决定系数的文章。](https://en.wikipedia.org/wiki/Coefficient_of_determination)

下面的示例提供了计算一组预测的平均 R ^ 2 的演示。

```
# Cross Validation Regression R^2
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
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
```

您可以看到预测与实际值的拟合度较差，其值接近零且小于 0.5。

```
R^2: 0.203 (0.595)
```

## 摘要

在这篇文章中，您发现了可用于评估机器学习算法的指标。

您了解了 3 个分类指标：

*   准确率。
*   对数损失。
*   ROC 曲线下面积。

另外 2 种分类预测结果的便捷方法：

*   混乱矩阵。
*   分类报告。

和 3 个回归指标：

*   平均绝对误差。
*   均方误差。
*   R ^ 2。

您对评估机器学习算法或此帖子的指标有任何疑问吗？在评论中提出您的问题，我会尽力回答。