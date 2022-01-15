# 如何在 Python 中开发 LARS 回归模型

> 原文：<https://machinelearningmastery.com/lars-regression-with-python/>

最后更新于 2020 年 10 月 25 日

回归是一项建模任务，包括预测给定输入的数值。

线性回归是回归的标准算法，假设输入和目标变量之间存在线性关系。线性回归的扩展包括在训练期间对损失函数增加惩罚，以鼓励具有较小系数值的更简单的模型。这些扩展被称为正则化线性回归或惩罚线性回归。

套索回归是一种流行的正则化线性回归，包括 L1 惩罚。这具有缩小那些对预测任务贡献不大的输入变量的系数的效果。

**最小角度回归**或 **LARS** 简称提供了一种替代的、有效的方法来拟合不需要任何超参数的套索正则化回归模型。

在本教程中，您将发现如何在 Python 中开发和评估 LARS 回归模型。

完成本教程后，您将知道:

*   LARS 回归提供了一种训练 Lasso 正则化线性回归模型的替代方法，该模型在训练期间对损失函数增加了惩罚。
*   如何评估 LARS 回归模型，并使用最终模型对新数据进行预测。
*   如何使用交叉验证版本的估计器为新数据集自动配置 LARS 回归模型。

我们开始吧。

![How to Develop LARS Regression Models in Python](img/2e740defbbf3603ca665837e52f7dd33.png)

如何在 Python 中开发 LARS 回归模型
图片由 [Nicolas Raymond](https://flickr.com/photos/82955120@N05/20449610383/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  LARS 回归
2.  LARS 回归的例子
3.  调整 LARS 超参数

## LARS 回归

[线性回归](https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/)是指假设输入变量和目标变量之间存在线性关系的模型。

对于单个输入变量，这种关系是一条线，对于更高的维度，这种关系可以被认为是连接输入变量和目标变量的超平面。模型的系数是通过优化过程找到的，该过程寻求最小化预测值( *yhat* )和预期目标值( *y* )之间的误差平方和。

*   损失=总和 i=0 至 n(y _ I–yhat_i)^2

线性回归的一个问题是模型的估计系数会变大，使模型对输入敏感，并且可能不稳定。对于观察值(样本)很少或输入预测值(T0)比样本(T2)n(T3)多的问题(所谓的[T5 p(T8)T9 n 问题 T7)更是如此。](https://machinelearningmastery.com/how-to-handle-big-p-little-n-p-n-in-machine-learning/)

解决回归模型稳定性的一种方法是改变损失函数，以包括具有大系数的模型的额外成本。在训练过程中使用这些修正损失函数的线性回归模型统称为惩罚线性回归。

一种流行的惩罚是基于绝对系数值的总和来惩罚模型。这被称为 L1 惩罚。L1 惩罚使所有系数的大小最小化，并允许一些系数最小化到零值，这从模型中移除了预测器。

*   l1 _ 罚分=总和 j=0 至 p ABS(β_ j)

L1 罚函数最小化了所有系数的大小，并允许任何系数达到零值，从而有效地从模型中移除输入特征。这是一种自动特征选择方法。

> …惩罚绝对值的结果是，对于λ的某个值，某些参数实际上被设置为 0。因此套索产生同时使用正则化来改进模型和进行特征选择的模型。

—第 125 页，[应用预测建模](https://amzn.to/38uGJOl)，2013 年。

这个惩罚可以加到线性回归的成本函数中，称为[最小绝对收缩和选择算子](https://en.wikipedia.org/wiki/Lasso_(statistics)) (LASSO)，或者更常见的简称为“ *Lasso* ”(带标题情况)。

套索使用最小二乘损失训练程序训练模型。

**最小角度回归**，简称 LAR 或 LARS，是解决拟合惩罚模型优化问题的一种替代方法。从技术上讲，LARS 是回归特征选择的前向逐步版本，可适用于 Lasso 模型。

与套索不同，它不需要控制损失函数中惩罚权重的超参数。相反，权重由 LARS 自动发现。

> …最小角度回归(LARS)，是一个包含套索和类似模型的广泛框架。LARS 模型可以用来更有效地拟合套索模型，尤其是在高维问题中。

—第 126 页，[应用预测建模](https://amzn.to/38uGJOl)，2013 年。

现在我们已经熟悉了 LARS 惩罚回归，让我们看看一个成功的例子。

## LARS 回归的例子

在本节中，我们将演示如何使用 LARS 回归算法。

首先，让我们介绍一个标准回归数据集。我们将使用房屋数据集。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的平均绝对误差(MAE)。一个表现最好的模型可以在大约 1.9 的相同测试线束上实现 MAE。这提供了此数据集的预期表现范围。

该数据集包括预测美国波士顿郊区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例将数据集下载并加载为熊猫数据框，并总结了数据集的形状和前五行数据。

```py
# load and summarize the housing dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())
```

运行该示例确认了 506 行数据、13 个输入变量和一个数字目标变量(总共 14 个)。我们还可以看到，所有的输入变量都是数字。

```py
(506, 14)
        0     1     2   3      4      5   ...  8      9     10      11    12    13
0  0.00632  18.0  2.31   0  0.538  6.575  ...   1  296.0  15.3  396.90  4.98  24.0
1  0.02731   0.0  7.07   0  0.469  6.421  ...   2  242.0  17.8  396.90  9.14  21.6
2  0.02729   0.0  7.07   0  0.469  7.185  ...   2  242.0  17.8  392.83  4.03  34.7
3  0.03237   0.0  2.18   0  0.458  6.998  ...   3  222.0  18.7  394.63  2.94  33.4
4  0.06905   0.0  2.18   0  0.458  7.147  ...   3  222.0  18.7  396.90  5.33  36.2

[5 rows x 14 columns]
```

scikit-learn Python 机器学习库通过 [Lars 类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)提供了 LARS 惩罚回归算法的实现。

```py
...
# define model
model = Lars()
```

我们可以使用[重复 10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)在住房数据集上评估 LARS 回归模型，并报告数据集上的平均绝对误差(MAE)。

```py
# evaluate an lars regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lars
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lars()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估住房数据集上的 LARS 回归算法，并报告 10 倍交叉验证的三次重复的平均 MAE。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型实现了大约 3.432 的 MAE。

```py
Mean MAE: 3.432 (0.552)
```

我们可能会决定使用 LARS 回归作为我们的最终模型，并根据新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用 *predict()* 函数，传入新的数据行来实现。

我们可以用下面列出的一个完整的例子来演示这一点。

```py
# make a prediction with a lars regression model on the dataset
from pandas import read_csv
from sklearn.linear_model import Lars
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lars()
# fit model
model.fit(X, y)
# define new data
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted: %.3f' % yhat)
```

运行该示例符合模型，并对新的数据行进行预测。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

```py
Predicted: 29.904
```

接下来，我们可以看看配置模型超参数。

## 调整 LARS 超参数

作为 LARS 训练算法的一部分，它会自动发现 Lasso 算法中使用的 lambda 超参数的最佳值。

这个超参数在 scikit-learn 套索和 LARS 的实现中被称为“ *alpha* ”参数。

然而，自动发现最佳模型和*α*超参数的过程仍然基于单个训练数据集。

另一种方法是在训练数据集的多个子集上拟合模型，并在褶皱上选择最佳的内部模型配置，在这种情况下是*α*的值。通常，这被称为交叉验证估计。

scikit-learn 库提供了 LARS 的交叉验证版本，用于通过 [LarsCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html)为 *alpha* 找到更可靠的值。

下面的示例演示了如何拟合 *LarsCV* 模型并报告通过交叉验证找到的*α*值

```py
# use automatically configured the lars regression algorithm
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import LarsCV
from sklearn.model_selection import RepeatedKFold
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LarsCV(cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
```

运行该示例符合使用重复交叉验证的 *LarsCV* 模型，并报告在运行中发现的最佳*α*值。

```py
alpha: 0.001623
```

这一版本的 LARS 模型在实践中可能会更加稳健。

我们可以使用与上一节相同的程序对其进行评估，尽管在这种情况下，每个模型拟合都基于通过内部重复 k 倍交叉验证(例如交叉验证估计量的交叉验证)找到的超参数。

下面列出了完整的示例。

```py
# evaluate an lars cross-validation regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LarsCV
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LarsCV(cv=cv, n_jobs=-1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例将使用重复交叉验证来评估模型超参数的交叉验证估计。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到，与上一节中的 3.432 相比，我们获得了稍好的结果。

```py
Mean MAE: 3.374 (0.558)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [统计学习的要素](https://amzn.to/3aBTnMV)，2016。
*   [应用预测建模](https://amzn.to/38uGJOl)，2013。

### 蜜蜂

*   [线性模型，sci kit-学习](https://scikit-learn.org/stable/modules/linear_model.html)。
*   [sklearn.linear_model。拉斯 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html) 。
*   [sklearn.linear_model。拉斯 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html) 。

### 文章

*   [最小角度回归，维基百科](https://en.wikipedia.org/wiki/Least-angle_regression)。

## 摘要

在本教程中，您发现了如何在 Python 中开发和评估 LARS 回归模型。

具体来说，您了解到:

*   LARS 回归提供了一种训练 Lasso 正则化线性回归模型的替代方法，该模型在训练期间对损失函数增加了惩罚。
*   如何评估 LARS 回归模型，并使用最终模型对新数据进行预测。
*   如何使用交叉验证版本的估计器为新数据集自动配置 LARS 回归模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。