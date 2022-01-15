# 如何用 Python 开发弹性网络回归模型

> 原文：<https://machinelearningmastery.com/elastic-net-regression-in-python/>

回归是一项建模任务，包括预测给定输入的数值。

线性回归是回归的标准算法，假设输入和目标变量之间存在线性关系。线性回归的扩展包括在训练期间对损失函数增加惩罚，以鼓励具有较小系数值的更简单的模型。这些扩展被称为正则化线性回归或惩罚线性回归。

**弹性网**是一种流行的正则化线性回归，它结合了两种流行的罚函数，特别是 L1 和 L2 罚函数。

在本教程中，您将发现如何在 Python 中开发弹性网正则化回归。

完成本教程后，您将知道:

*   弹性网是线性回归的扩展，它在训练期间给损失函数增加了正则化惩罚。
*   如何评估弹性网络模型，并使用最终模型对新数据进行预测。
*   如何通过网格搜索和自动为新数据集配置弹性网络模型。

我们开始吧。

![How to Develop Elastic Net Regression Models in Python](img/63ba74f8b3419022f95eb7b1c419c149.png)

如何用 Python 开发弹性网络回归模型
图片由[菲尔杜比](https://flickr.com/photos/126654539@N08/30388901788/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  弹性网络回归
2.  弹性网络回归示例
3.  调整弹性网络超参数

## 弹性网络回归

线性回归是指假设输入变量和目标变量之间呈线性关系的模型。

对于单个输入变量，这种关系是一条线，对于更高的维度，这种关系可以被认为是连接输入变量和目标变量的超平面。模型的系数是通过优化过程找到的，该过程寻求最小化预测值( *yhat* )和预期目标值( *y* )之间的误差平方和。

*   损失=总和 i=0 至 n(y _ I–yhat_i)^2

线性回归的一个问题是模型的估计系数会变大，使模型对输入敏感，并且可能不稳定。对于观测值很少(*样本*)或样本数多于输入预测值( *p* )或变量(所谓的 *p > > n 问题*)的问题来说尤其如此。

解决回归模型稳定性的一种方法是改变损失函数，以包括具有大系数的模型的额外成本。在训练过程中使用这些修正损失函数的线性回归模型统称为惩罚线性回归。

一种流行的惩罚是基于系数值的平方和来惩罚模型。这被称为 L2 处罚。L2 罚函数最小化了所有系数的大小，尽管它阻止从模型中移除任何系数。

*   l2 _ 罚分=总和 j=0 至 p beta_j^2

另一种流行的惩罚是基于绝对系数值的总和来惩罚模型。这被称为 L1 惩罚。L1 惩罚使所有系数的大小最小化，并允许一些系数最小化到零值，这从模型中移除了预测器。

*   l1 _ 罚分=总和 j=0 至 p ABS(β_ j)

弹性网是一个惩罚线性回归模型，包括 L1 和 L2 在训练中的惩罚。

使用“统计学习的要素”“一个超参数”*α*”中的术语来指定 L1 和 L2 的每个处罚的权重。Alpha 值介于 0 和 1 之间，用于加权 L1 罚分，1 减去 alpha 值用于加权 L2 罚分。

*   弹性 _ 净 _ 惩罚=(α* L1 _ 惩罚)+((1–α)* L2 _ 惩罚)

例如，α值为 0.5 将为损失函数提供 50%的罚分。α值为 0 表示 L2 处罚的全部权重，值为 1 表示 L1 处罚的全部权重。

> 参数α决定了处罚的组合，并且通常是基于定性的理由预先选择的。

—第 663 页，[统计学习的要素](https://amzn.to/3aBTnMV)，2016。

好处是弹性网允许两种惩罚的平衡，这可以比在某些问题上有一种或另一种惩罚的模型产生更好的表现。

提供了另一个超参数，称为“*λ*”，它控制两个损失之和对损失函数的加权。默认值 1.0 用于使用完全加权惩罚；值 0 不包括罚款。非常小的 lambada 值很常见，例如 1e-3 或更小。

*   弹性 _ 净 _ 损失=损失+(λ*弹性 _ 净 _ 损失)

现在我们已经熟悉了弹性网惩罚回归，让我们来看一个工作的例子。

## 弹性网络回归示例

在本节中，我们将演示如何使用弹性网回归算法。

首先，让我们介绍一个标准回归数据集。我们将使用房屋数据集。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的平均绝对误差(MAE)。一个表现最好的模型可以在大约 1.9 的相同测试线束上实现 MAE。这提供了此数据集的预期表现范围。

该数据集包括预测美国波士顿郊区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

以下示例下载并加载数据集作为 Pandas [数据框](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)，并总结数据集的形状和前五行数据。

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

运行该示例确认了 506 行数据、13 个输入变量和一个数字目标变量(总共 14 个)。

我们还可以看到，所有的输入变量都是数字。

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

Sklearn Python 机器学习库通过 [ElasticNet 类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)提供了弹性网络惩罚回归算法的实现。

令人困惑的是，*α*超参数可以通过控制 l1 和 L2 罚分贡献的“ *l1_ratio* 参数设置，而*λ*超参数可以通过控制两个罚分之和对损失函数贡献的“*α*参数设置。

默认情况下， *l1_ratio* 使用相等的余额 0.5，alpha 使用 1.0 的全权重。

```py
...
# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

我们可以使用[重复 10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估房屋数据集上的弹性网模型，并报告数据集上的平均绝对误差(MAE)。

```py
# evaluate an elastic net model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会评估房屋数据集上的弹性网络算法，并报告 10 倍交叉验证的三次重复的平均 MAE。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型实现了大约 3.682 的 MAE。

```py
Mean MAE: 3.682 (0.530)
```

我们可能会决定使用弹性网作为我们的最终模型，并根据新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用 *predict()* 函数，传入新的数据行来实现。

我们可以用下面列出的一个完整的例子来演示这一点。

```py
# make a prediction with an elastic net model on the dataset
from pandas import read_csv
from sklearn.linear_model import ElasticNet
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
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

```py
Predicted: 31.047
```

接下来，我们可以看看如何配置模型超参数。

## 调整弹性网络超参数

我们如何知道 alpha=1.0 和 l1_ratio=0.5 的默认超参数对我们的数据集有什么好处？

我们没有。

相反，测试一套不同的配置并发现什么最有效是一个很好的做法。

一种方法是以 0.1 或 0.01 的间隔网格搜索 0 到 1 之间的 L1 _ ratio 值，以对数-10 的标度网格搜索 1e-5 到 100 之间的 T2α值，并找出最适合数据集的值。

下面的例子使用 [GridSearchCV](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 类和我们定义的值网格来演示这一点。

```py
# grid search hyperparameters for the elastic net
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = ElasticNet()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

您可能会看到一些可以安全忽略的警告，例如:

```py
Objective did not converge. You might want to increase the number of iterations.
```

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到我们获得了比默认的 3.378 vs. 3.682 稍好的结果。忽略标志；出于优化目的，该库使 MAE 为负。

我们可以看到，该模型将 0.01 的 alpha 权重分配给了处罚，并且只关注 L2 处罚。

```py
MAE: -3.378
Config: {'alpha': 0.01, 'l1_ratio': 0.97}
```

Sklearn 库还提供了一个内置的算法版本，可以通过[elastic cnetcv](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)类自动找到好的超参数。

要使用这个类，首先要对数据集进行拟合，然后用它来进行预测。它会自动找到合适的超参数。

默认情况下，模型将测试 100 个阿尔法值，并使用默认比率。我们可以通过“ *l1_ratio* ”和“*alpha*”参数来指定我们自己要测试的值列表，就像我们手动网格搜索一样。

下面的例子演示了这一点。

```py
# use automatically configured elastic net algorithm
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ratios = arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
print('l1_ratio_: %f' % model.l1_ratio_)
```

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

同样，您可能会看到一些可以安全忽略的警告，例如:

```py
Objective did not converge. You might want to increase the number of iterations.
```

在这种情况下，我们可以看到选择了 0.0 的α，从损失函数中去掉了两个惩罚。

这与我们通过手动网格搜索发现的不同，可能是由于搜索或选择配置的系统方式。

```py
alpha: 0.000000
l1_ratio_: 0.470000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [统计学习的要素](https://amzn.to/3aBTnMV)，2016。
*   [应用预测建模](https://amzn.to/38uGJOl)，2013。

### 蜜蜂

*   [sklearn.linear_model。弹力网 API](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) 。
*   [硬化. linear_model .弹性体 CV API](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html) 。

### 文章

*   [弹性网正则化，维基百科](https://en.wikipedia.org/wiki/Elastic_net_regularization)。

## 摘要

在本教程中，您发现了如何在 Python 中开发弹性网正则化回归。

具体来说，您了解到:

*   弹性网是线性回归的扩展，它在训练期间给损失函数增加了正则化惩罚。
*   如何评估弹性网络模型，并使用最终模型对新数据进行预测。
*   如何通过网格搜索和自动为新数据集配置弹性网络模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。