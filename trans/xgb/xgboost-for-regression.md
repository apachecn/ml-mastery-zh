# 用于回归的 XGBoost

> 原文：<https://machinelearningmastery.com/xgboost-for-regression/>

极限梯度增强(XGBoost)是一个开源库，它提供了梯度增强算法的高效和有效的实现。

在开发和最初发布后不久，XGBoost 就成为了热门方法，并且经常成为在机器学习竞赛中赢得一系列问题解决方案的关键组件。

回归预测建模问题涉及预测一个数值，如一美元金额或高度。 **XGBoost** 可直接用于**回归预测建模**。

在本教程中，您将发现如何在 Python 中开发和评估 XGBoost 回归模型。

完成本教程后，您将知道:

*   XGBoost 是梯度增强的有效实现，可用于回归预测建模。
*   如何使用重复 k 倍交叉验证的最佳实践技术评估 XGBoost 回归模型？
*   如何拟合最终模型，并利用它对新数据进行预测。

我们开始吧。

![XGBoost for Regression](img/8a7444e0be0160ba458a2346053e0628.png)

回归的 xboost
图片由 [chas B](https://www.flickr.com/photos/tarquingemstone/16264896191/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  极限梯度助推
2.  XGBoost 回归 API
3.  XGBoost 回归示例

## 极限梯度助推

**梯度提升**是指一类可用于分类或回归预测建模问题的集成机器学习算法。

集成是由决策树模型构建的。树被一次一个地添加到集合中，并且适合于校正由先前模型产生的预测误差。这是一种称为 boosting 的集成机器学习模型。

使用任意可微损失函数和梯度下降优化算法拟合模型。这给这项技术起了一个名字，“梯度增强”，因为随着模型的拟合，损失梯度被最小化，很像一个神经网络。

有关渐变增强的更多信息，请参见教程:

*   [机器学习梯度增强算法的简单介绍](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

极限梯度增强，简称 XGBoost，是梯度增强算法的一个高效开源实现。因此，XGBoost 是一个算法、一个开源项目和一个 Python 库。

它最初是由[陈天棋](https://www.linkedin.com/in/tianqi-chen-679a9856/)开发的，并由陈和在他们 2016 年的论文《XGBoost:一个可扩展的树木提升系统》中进行了描述

它被设计为既有计算效率(例如，执行速度快)，又非常有效，可能比其他开源实现更有效。

使用 XGBoost 的两个主要原因是执行速度和模型表现。

XGBoost 在分类和回归预测建模问题上主导结构化或表格数据集。证据是，它是 Kaggle 竞争数据科学平台上竞争赢家的 go-to 算法。

> 在 2015 年 Kaggle 博客上发布的 29 个挑战获胜解决方案 3 中，有 17 个解决方案使用了 XGBoost。[……]该系统的成功也在 KDDCup 2015 中得到了见证，在该赛事中，XGBoost 被前 10 名中的每一个获胜团队所使用。

——[xboost:一个可扩展的树提升系统](https://arxiv.org/abs/1603.02754)，2016。

现在我们已经熟悉了什么是 XGBoost 以及它为什么重要，让我们更仔细地看看如何在我们的回归预测建模项目中使用它。

## XGBoost 回归 API

xboost 可以作为一个独立的库安装，并且可以使用 scikit-learn API 开发一个 xboost 模型。

第一步是安装尚未安装的 XGBoost 库。这可以在大多数平台上使用 pip python 包管理器来实现；例如:

```py
sudo pip install xgboost
```

然后，您可以通过运行以下脚本来确认 XGBoost 库安装正确，并且可以使用。

```py
# check xgboost version
import xgboost
print(xgboost.__version__)
```

运行该脚本将打印您安装的 XGBoost 库的版本。

您的版本应该相同或更高。如果没有，您必须升级 XGBoost 库的版本。

```py
1.1.1
```

您可能对最新版本的库有问题。这不是你的错。

有时，库的最新版本会带来额外的要求，或者可能不太稳定。

如果您在尝试运行上述脚本时确实有错误，我建议降级到 1.0.1 版(或更低版本)。这可以通过指定要安装到 pip 命令的版本来实现，如下所示:

```py
sudo pip install xgboost==1.0.1
```

如果您需要开发环境的特定说明，请参阅教程:

*   [XGBoost 安装指南](https://xgboost.readthedocs.io/en/latest/build.html)

尽管我们将通过 scikit-learn 包装类使用这个方法:[xgbreversor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)和 [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) ，但是 XGBoost 库有自己的自定义 API。这将允许我们使用 scikit-learn 机器学习库中的全套工具来准备数据和评估模型。

一个 XGBoost 回归模型可以通过创建一个*xgbreversor*类的实例来定义；例如:

```py
...
# create an xgboost regression model
model = XGBRegressor()
```

您可以为类构造函数指定超参数值来配置模型。

可能最常见的配置超参数如下:

*   **n _ estimates**:集合中的树的数量，经常增加，直到看不到进一步的改进。
*   **max_depth** :每棵树的最大深度，往往取值在 1 到 10 之间。
*   **eta** :用于对每个模型进行加权的学习率，通常设置为 0.3、0.1、0.01 或更小的值。
*   **子样本**:每个树中使用的样本(行)数量，设置为 0 到 1 之间的值，通常为 1.0 以使用所有样本。
*   **colsample_bytree** :每个树中使用的特征(列)数量，设置为 0 到 1 之间的值，通常为 1.0 以使用所有特征。

例如:

```py
...
# create an xgboost regression model
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
```

好的超参数值可以通过对给定数据集的反复试验或系统实验来找到，例如在一系列值中使用网格搜索。

随机性用于模型的构建。这意味着算法每次在相同的数据上运行时，可能会产生稍微不同的模型。

当使用具有随机学习算法的机器学习算法时，最好通过在多次运行或重复交叉验证中平均它们的表现来评估它们。当拟合最终模型时，可能需要增加树的数量，直到模型的方差在重复评估中减小，或者拟合多个最终模型并对它们的预测进行平均。

让我们看看如何为回归开发一个 XGBoost 集成。

## XGBoost 回归示例

在本节中，我们将研究如何为标准回归预测建模数据集开发一个 XGBoost 模型。

首先，让我们介绍一个标准回归数据集。

我们将使用房屋数据集。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的[平均绝对误差(MAE)](https://machinelearningmastery.com/regression-metrics-for-machine-learning/) 。一个表现最好的模型可以在大约 1.9 的相同测试线束上实现 MAE。这提供了此数据集的预期表现范围。

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

接下来，让我们评估一个关于这个问题的带有默认超参数的回归 XGBoost 模型。

首先，我们可以将加载的数据集分成输入和输出列，用于训练和评估预测模型。

```py
...
# split data into input and output columns
X, y = data[:, :-1], data[:, -1]
```

接下来，我们可以创建一个具有默认配置的模型实例。

```py
...
# define model
model = XGBRegressor()
```

我们将使用 3 次重复和 10 次重复的重复 k-fold 交叉验证的最佳实践来评估模型。

这可以通过使用 [RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) 类配置评估过程，并调用 [cross_val_score()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) 使用该过程评估模型并收集分数来实现。

将使用均方差(MAE)评估模型表现。注意，MAE 在 scikit-learn 库中被设为负，这样它可以被最大化。因此，我们可以忽略符号，假设所有错误都是正的。

```py
...
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

一旦评估完毕，我们就可以报告模型的估计表现，用于对这个问题的新数据进行预测。

在这种情况下，因为分数被设为负值，所以我们可以使用[绝对()NumPy 函数](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html)将分数设为正值。

然后，我们使用分数分布的平均值和标准偏差来报告表现的统计摘要，这是另一种好的做法。

```py
...
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
```

将这些联系在一起，下面列出了评估住房回归预测建模问题的 XGBoost 模型的完整示例。

```py
# evaluate an xgboost regression model on the housing dataset
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split data into input and output columns
X, y = data[:, :-1], data[:, -1]
# define model
model = XGBRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
```

运行该示例评估住房数据集上的 XGBoost 回归算法，并报告 10 倍交叉验证的三次重复的平均 MAE。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了大约 2.1 的 MAE。

这是一个不错的分数，比基线好，意味着模型有技巧，接近 1.9 的最佳分数。

```py
Mean MAE: 2.109 (0.320)
```

我们可能会决定使用 XGBoost 回归模型作为我们的最终模型，并对新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用 *predict()* 函数，传入新的数据行来实现。

例如:

```py
...
# make a prediction
yhat = model.predict(new_data)
```

我们可以用下面列出的一个完整的例子来演示这一点。

```py
# fit a final xgboost model on the housing dataset and make a prediction
from numpy import asarray
from pandas import read_csv
from xgboost import XGBRegressor
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split dataset into input and output columns
X, y = data[:, :-1], data[:, -1]
# define model
model = XGBRegressor()
# fit model
model.fit(X, y)
# define new data
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
new_data = asarray([row])
# make a prediction
yhat = model.predict(new_data)
# summarize prediction
print('Predicted: %.3f' % yhat)
```

运行该示例符合模型，并对新的数据行进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型预测的值约为 24。

```py
Predicted: 24.019
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [Python 中的极限梯度增强(XGBoost)集成](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)
*   [使用 Scikit-Learn、XGBoost、LightGBM 和 CatBoost 进行梯度增强](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
*   [标准机器学习数据集的最佳结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [如何使用 XGBoost 进行时间序列预测](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)

### 报纸

*   [XGBoost:一个可扩展的树木助推系统](https://arxiv.org/abs/1603.02754)，2016。

### 蜜蜂

*   [XGBoost 安装指南](https://xgboost.readthedocs.io/en/latest/build.html)
*   [xboost。xgbreversor API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)。
*   [sklearn.model_selection。重复应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)。
*   [sklearn . model _ selection . cross _ val _ score API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

## 摘要

在本教程中，您发现了如何在 Python 中开发和评估 XGBoost 回归模型。

具体来说，您了解到:

*   XGBoost 是梯度增强的有效实现，可用于回归预测建模。
*   如何使用重复 k 倍交叉验证的最佳实践技术评估 XGBoost 回归模型？
*   如何拟合最终模型，并利用它对新数据进行预测。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。