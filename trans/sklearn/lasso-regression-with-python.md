# 如何在 Python 中开发 LASSO 回归模型

> 原文：<https://machinelearningmastery.com/lasso-regression-with-python/>

最后更新于 2021 年 10 月 6 日

回归是一项建模任务，包括预测给定输入的数值。

线性回归是回归的标准算法，假设输入和目标变量之间存在线性关系。线性回归的扩展调用了在训练期间对损失函数增加惩罚，这鼓励具有较小系数值的更简单的模型。这些扩展被称为正则化线性回归或惩罚线性回归。

**套索回归**是一种流行的正则化线性回归，包括 L1 惩罚。这具有缩小那些对预测任务贡献不大的输入变量的系数的效果。这种损失允许一些系数值达到零值，从而允许从模型中有效地移除输入变量，提供了一种自动特征选择。

在本教程中，您将发现如何在 Python 中开发和评估套索回归模型。

完成本教程后，您将知道:

*   套索回归是线性回归的扩展，它在训练期间给损失函数增加了正则化惩罚。
*   如何评估套索回归模型并使用最终模型对新数据进行预测。
*   如何通过网格搜索和自动为新数据集配置套索回归模型。

我们开始吧。

![How to Develop LASSO Regression Models in Python](img/ec009573b166e5f451a23a3008cf249b.png)

如何用 Python 开发 LASSO 回归模型
图片由[菲尔杜比](https://flickr.com/photos/126654539@N08/24743046671/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  套索回归
2.  套索回归的例子
3.  调谐套索超参数

## 套索回归

[线性回归](https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/)是指假设输入变量和目标变量之间存在线性关系的模型。

对于单个输入变量，这种关系是一条线，对于更高的维度，这种关系可以被认为是连接输入变量和目标变量的超平面。模型的系数是通过优化过程找到的，该过程寻求最小化预测值( *yhat* )和预期目标值( *y* )之间的误差平方和。

*   损失=总和 i=0 至 n(y _ I–yhat_i)^2

线性回归的一个问题是模型的估计系数会变大，使模型对输入敏感，并且可能不稳定。对于观测值很少(*样本*)或样本数比输入预测值( *p* )或变量(所谓的 *p > > n 问题*)少( *n* )的问题尤其如此。

解决回归模型稳定性的一种方法是改变损失函数，以包括具有大系数的模型的额外成本。在训练过程中使用这些修正损失函数的线性回归模型统称为惩罚线性回归。

一种流行的惩罚是基于绝对系数值的总和来惩罚模型。这被称为 L1 惩罚。L1 惩罚使所有系数的大小最小化，并允许一些系数最小化到零值，这从模型中移除了预测器。

*   l1 _ 罚分=总和 j=0 至 p ABS(β_ j)

L1 罚函数最小化了所有系数的大小，并允许任何系数达到零值，从而有效地从模型中移除输入特征。

这是一种自动特征选择。

> …惩罚绝对值的结果是，对于λ的某个值，某些参数实际上被设置为 0。因此套索产生同时使用正则化来改进模型和进行特征选择的模型。

—第 125 页，[应用预测建模](https://amzn.to/38uGJOl)，2013 年。

这种惩罚可以加到线性回归的成本函数中，称为[最小绝对收缩和选择算子正则化](https://en.wikipedia.org/wiki/Lasso_(statistics)) (LASSO)，或者更常见的是，简称为“ *Lasso* ”(带标题情况)。

> 岭回归的一种流行替代方法是最小绝对收缩和选择算子模型，通常称为套索。

—第 124 页，[应用预测建模](https://amzn.to/38uGJOl)，2013 年。

使用名为“*λ*”的超参数来控制损失函数的惩罚权重。默认值 1.0 将给予处罚全部权重；值 0 不包括罚款。非常小的*λ*值，例如 1e-3 或更小，是常见的。

*   lasso _ loss = loss+(λ* L1 _ 罚分)

现在我们已经熟悉了 Lasso 惩罚回归，让我们来看看一个成功的例子。

## 套索回归的例子

在本节中，我们将演示如何使用套索回归算法。

首先，让我们介绍一个标准回归数据集。我们将使用房屋数据集。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的平均绝对误差(MAE)。一个性能最好的模型可以在大约 1.9 的相同测试线束上实现 MAE。这提供了此数据集的预期性能范围。

该数据集包括预测美国波士顿市郊住宅区的房价。

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

scikit-learn Python 机器学习库通过[套索类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)提供了套索惩罚回归算法的实现。

令人困惑的是， *lambda* 术语可以在定义类时通过“ *alpha* 参数进行配置。默认值为 1.0 或全额罚款。

```py
...
# define model
model = Lasso(alpha=1.0)
```

我们可以使用[重复 10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估房屋数据集上的套索回归模型，并报告数据集上的平均绝对误差(MAE)。

```py
# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lasso(alpha=1.0)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估房屋数据集上的套索回归算法，并报告 10 倍交叉验证的三次重复的平均 MAE。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型实现了大约 3.711 的 MAE。

```py
Mean MAE: 3.711 (0.549)
```

我们可能会决定使用套索回归作为我们的最终模型，并根据新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用 *predict()* 函数，传入新的数据行来实现。

我们可以用下面列出的一个完整的例子来演示这一点。

```py
# make a prediction with a lasso regression model on the dataset
from pandas import read_csv
from sklearn.linear_model import Lasso
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lasso(alpha=1.0)
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
Predicted: 30.998
```

接下来，我们可以看看配置模型超参数。

## 调谐套索超参数

我们如何知道*α= 1.0*的默认超参数适合我们的数据集？

我们没有。

相反，测试一套不同的配置并发现什么最适合我们的数据集是一个很好的做法。

一种方法是在对数 10 标度上网格搜索从 1e-5 到 100 的α值，并发现什么最适合数据集。另一种方法是测试 0.0 到 1.0 之间的值，网格间距为 0.01。在这种情况下，我们将尝试后者。

下面的例子使用 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 类和我们定义的值网格来演示这一点。

```py
# grid search hyperparameters for lasso regression
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

您可能会看到一些可以安全忽略的警告，例如:

```py
Objective did not converge. You might want to increase the number of iterations.
```

在这种情况下，我们可以看到我们获得了比默认的 3.379 vs. 3.711 稍好的结果。忽略标志；出于优化目的，该库使 MAE 为负。

我们可以看到模型给惩罚分配了一个 0.01 的*α*权重。

```py
MAE: -3.379
Config: {'alpha': 0.01}
```

scikit-learn 库还提供了一个内置的算法版本，可以通过 [LassoCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)自动找到好的超参数。

为了使用该类，模型按照正常方式适合训练数据集，并且在训练过程中自动调整超参数。然后，拟合模型可用于进行预测。

默认情况下，模型将测试 100 个*α*值。我们可以通过设置“*阿尔法*”参数，将它更改为 0 到 1 之间的值网格，间距为 0.01，就像我们在前面的示例中所做的那样。

下面的例子演示了这一点。

```py
# use automatically configured the lasso regression algorithm
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
```

运行该示例符合模型，并使用交叉验证发现给出最佳结果的超参数。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到模型选择了α= 0.0 的超参数。这与我们通过手动网格搜索发现的不同，可能是由于搜索或选择配置的系统方式。

```py
alpha: 0.000000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [统计学习的要素](https://amzn.to/3aBTnMV)，2016。
*   [应用预测建模](https://amzn.to/38uGJOl)，2013。

### 蜜蜂

*   [线性模型，sci kit-学习](https://scikit-learn.org/stable/modules/linear_model.html)。
*   [sklearn.linear_model。套索 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) 。
*   [硬化. linear_model .lasocv API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)。

### 文章

*   [拉索(统计)，维基百科](https://en.wikipedia.org/wiki/Lasso_(statistics))。

## 摘要

在本教程中，您发现了如何在 Python 中开发和评估套索回归模型。

具体来说，您了解到:

*   套索回归是线性回归的扩展，它在训练期间给损失函数增加了正则化惩罚。
*   如何评估套索回归模型并使用最终模型对新数据进行预测。
*   如何通过网格搜索和自动为新数据集配置套索回归模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。