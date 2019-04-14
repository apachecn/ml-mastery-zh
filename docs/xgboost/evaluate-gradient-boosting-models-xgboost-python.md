# 如何在 Python 中使用 XGBoost 评估梯度提升模型

> 原文： [https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/](https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/)

开发预测模型的目标是开发一个对看不见的数据准确的模型。

这可以使用统计技术来实现，其中训练数据集被仔细地用于估计模型在新的和未看到的数据上的表现。

在本教程中，您将了解如何使用 Python 中的 XGBoost 评估梯度提升模型的表现。

完成本教程后，您将了解到。

*   如何使用训练和测试数据集评估 XGBoost 模型的表现。
*   如何使用 k-fold 交叉验证评估 XGBoost 模型的表现。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映 scikit-learn API 版本 0.18.1 中的更改​​。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Evaluate Gradient Boosting Models with XGBoost in Python](img/22202cde2c6a7833a222b951de920b6b.jpg)

如何在 Python 中使用 XGBoost 评估梯度提升模型
照片由 [Timitrius](https://www.flickr.com/photos/nox_noctis_silentium/5526750448/) ，保留一些权利。

## 使用训练和测试集评估 XGBoost 模型

我们可以用来评估机器学习算法表现的最简单方法是使用不同的训练和测试数据集。

我们可以将原始数据集分成两部分。在第一部分训练算法，然后对第二部分进行预测，并根据预期结果评估预测。

拆分的大小可能取决于数据集的大小和细节，尽管通常使用 67％的数据进行训练，剩余的 33％用于测试。

该算法评估技术很快。它非常适用于大型数据集（数百万条记录），其中有强有力的证据表明数据的两个分裂都代表了潜在的问题。由于速度的原因，当您正在调查的算法训练缓慢时，使用此方法很有用。

这种技术的缺点是它可能具有很大的差异。这意味着训练和测试数据集的差异可能导致模型精度估计的有意义差异。

我们可以使用 scikit-learn 库中的 **train_test_split（）**函数将数据集拆分为训练和测试集。例如，我们可以将数据集拆分为 67％和 33％的分组，用于训练和测试集，如下所示：

```py
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
```

下面使用 [Pima 印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)开始提供完整的代码清单，假设它位于当前工作目录中（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。具有默认配置的 XGBoost 模型适合训练数据集并在测试数据集上进行评估。

```py
# train-test split evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例总结了测试集上模型的表现。

```py
Accuracy: 77.95%
```

## 使用 k-fold 交叉验证评估 XGBoost 模型

交叉验证是一种可用于估计机器学习算法表现的方法，其方差小于单个训练测试集拆分。

它的工作原理是将数据集分成 k 部分（例如 k = 5 或 k = 10）。每次分割数据称为折叠。该算法在 k-1 折叠上进行训练，其中一个被扣住并在保持的背部折叠上进行测试。重复这一过程，以便数据集的每个折叠都有机会成为阻碍测试集。

运行交叉验证后，您最终得到 k 个不同的表现分数，您可以使用均值和标准差来总结。

结果是在给定测试数据的情况下，对新数据的算法表现进行更可靠的估计。它更准确，因为算法在不同数据上被多次训练和评估。

k 的选择必须允许每个测试分区的大小足够大以成为问题的合理样本，同时允许对算法的训练测试评估的足够重复以提供对看不见的数据的算法表现的公平估计。 。对于数千或数万个观测值中的适度大小的数据集，k 值为 3,5 和 10 是常见的。

我们可以使用 scikit-learn 中提供的 k-fold 交叉验证支持。首先，我们必须创建 [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html) 对象，指定折叠的数量和数据集的大小。然后，我们可以将此方案与特定数据集一起使用。来自 scikit-learn 的 **cross_val_score（）**函数允许我们使用交叉验证方案评估模型，并返回每个折叠上训练的每个模型的分数列表。

```py
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
```

下面提供了用于评估具有 k 折交叉验证的 XGBoost 模型的完整代码清单，以确保完整性。

```py
# k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = xgboost.XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此示例总结了数据集上默认模型配置的表现，包括平均值和标准差分类精度。

```py
Accuracy: 76.69% (7.11%)
```

如果您有许多类用于分类类型预测建模问题，或者类是不平衡的（一个类的实例比另一个类多得多），那么在执行交叉验证时创建分层折叠可能是个好主意。

这具有在执行交叉验证评估时在每个折叠中强制执行与在整个训练数据集中相同的类分布的效果。 scikit-learn 库在 [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html) 类中提供此功能。

下面是修改为使用分层交叉验证来评估 XGBoost 模型的相同示例。

```py
# stratified k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = xgboost.XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此示例将生成以下输出。

```py
Accuracy: 76.95% (5.88%)
```

## 什么技术使用时

*   通常，k 折交叉验证是用于评估机器学习算法在看不见的数据上的表现的金标准，其中 k 设置为 3,5 或 10。
*   当存在大量类或每个类的实例不平衡时，使用分层交叉验证来强制执行类分发。
*   使用慢速算法时，使用训练/测试分割有利于提高速度，并在使用大型数据集时产生具有较低偏差的表现估计。

最好的建议是试验并找到一种快速解决问题的技术，并产生可用于制定决策的合理表现估算。

如果有疑问，请对回归问题使用 10 倍交叉验证，并对分类问题进行 10 倍交叉验证。

## 摘要

在本教程中，您了解了如何通过估计 XGBoost 模型在未见数据上的执行情况来评估它们。

具体来说，你学到了：

*   如何将数据集拆分为训练和测试子集以进行训练和评估模型的表现。
*   如何在数据集的不同子集上创建 k XGBoost 模型并平均得分以获得更稳健的模型表现估计。
*   启发式帮助您在问题中选择训练测试拆分和 k 折交叉验证。

您对如何评估 XGBoost 模型或该帖子的表现有任何疑问吗？在下面的评论中提出您的问题，我会尽力回答。