# 使用 scikit-learn 在 Python 中集成机器学习算法

> 原文： [https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)

合奏可以提高数据集的准确率。

在这篇文章中，您将了解如何使用 scikit-learn 在 Python 中创建一些最强大的集合类型。

本案例研究将引导您完成 Boosting，Bagging 和 Majority Voting，并向您展示如何继续提高您自己数据集上模型的准确率。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Ensemble Machine Learning Algorithms in Python with scikit-learn](img/67233e2ae9078c9eb7a16ef0f1bb45fa.jpg)

使用 scikit-learn
照片的[照片由美国陆军乐队](https://www.flickr.com/photos/usarmyband/6830189060/)，部分权利保留。

## 将模型预测结合到集合预测中

结合不同模型预测的三种最流行的方法是：

*   **套袋**。从训练数据集的不同子样本构建多个模型（通常是相同类型）。
*   **提升**。构建多个模型（通常是相同类型），每个模型都学习如何修复链中先前模型的预测误差。
*   **投票**。构建多个模型（通常具有不同类型）和简单统计（如计算均值）用于组合预测。

这篇文章不会解释这些方法。

它假设您通常熟悉机器学习算法和集合方法，并且您正在寻找有关如何在 Python 中创建集合的信息。

## 关于秘籍

这篇文章中的每个秘籍都是独立设计的。这样您就可以将其复制并粘贴到项目中并立即开始使用。

来自 UCI 机器学习库的标准分类问题用于演示每个集成算法。这是[皮马印第安人糖尿病数据集的发病](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。这是一个二分类问题，其中所有输入变量都是数字的并且具有不同的比例。

每个集成算法使用 10 倍交叉验证进行演示，这是一种标准技术，用于估计任何机器学习算法对未见数据的表现。

## 套袋算法

Bootstrap 聚合或装袋涉及从训练数据集中获取多个样本（替换）并为每个样本训练模型。

最终输出预测在所有子模型的预测中取平均值。

本节涉及的三种装袋模型如下：

1.  袋装决策树
2.  随机森林
3.  额外的树木

### 1.袋装决策树

Bagging 在具有高差异的算法中表现最佳。一个流行的例子是决策树，通常是在没有修剪的情况下构建的。

在下面的示例中，请参阅使用 BaggingClassifier 和分类和回归树算法（ [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) ）的示例。共创造了 100 棵树。

```
# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

运行该示例，我们可以获得对模型准确率的可靠估计。

```
0.770745044429
```

### 2.随机森林

随机森林是袋装决策树的扩展。

训练数据集的样本是替换的，但树的构造方式会减少各个分类器之间的相关性。具体而言，不是贪婪地选择树的构造中的最佳分裂点，而是仅考虑每个分裂的随机特征子集。

您可以使用 [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 类构建随机森林模型进行分类。

下面的示例提供了一个随机森林的示例，用于对 100 棵树进行分类，并从随机选择的 3 个特征中选择分割点。

```
# Random Forest Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

运行该示例提供了分类准确度的平均估计。

```
0.770727956254
```

### 3.额外的树木

额外树木是套袋的另一种修改，其中随机树是根据训练数据集的样本构建的。

您可以使用 [ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) 类构建 Extra Trees 模型以进行分类。

下面的示例演示了额外的树，树的数量设置为 100，并且从 7 个随机要素中选择了分割。

```
# Extra Trees Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example provides a mean estimate of classification accuracy.

```
0.760269993165
```

## 提升算法

提升集成算法会创建一系列模型，尝试在序列中纠正模型之前的错误。

一旦创建，模型做出预测，可以通过其显示的准确度加权，并将结果组合以创建最终输出预测。

两种最常见的增强集成机器学习算法是：

1.  AdaBoost 的
2.  随机梯度提升

### 1\. AdaBoost

AdaBoost 可能是第一个成功的增强集成算法。它通常通过对数据集中的实例进行加权来对其进行分类是多么容易或困难，从而允许算法在后续模型的构造中支付或不太关注它们。

您可以使用 [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 类构建用于分类的 AdaBoost 模型。

下面的示例演示了使用 AdaBoost 算法按顺序构造 30 个决策树。

```
# AdaBoost Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example provides a mean estimate of classification accuracy.

```
0.76045796309
```

### 2.随机梯度提升

随机梯度增强（也称为梯度增强机器）是最复杂的整体技术之一。它也是一种被证明可能是通过集合提高表现的最佳技术。

您可以使用 [GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) 类构建一个 Gradient Boosting 模型进行分类。

下面的示例演示了随机梯度提升用于 100 棵树的分类。

```
# Stochastic Gradient Boosting Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Running the example provides a mean estimate of classification accuracy.

```
0.764285714286
```

## 投票合奏

投票是结合多种机器学习算法的预测的最简单方法之一。

它的工作原理是首先从训练数据集中创建两个或更多独立模型。然后，当被要求对新数据做出预测时，可以使用投票分类器来包装模型并平均子模型的预测。

可以对子模型的预测进行加权，但是手动或甚至启发式地指定分类器的权重是困难的。更高级的方法可以学习如何最好地对来自子模型的预测进行加权，但这称为堆叠（堆叠聚合），目前未在 scikit-learn 中提供。

您可以使用 [VotingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) 类创建用于分类的投票集合模型。

下面的代码提供了将逻辑回归，分类和回归树以及支持向量机的预测组合在一起用于分类问题的示例。

```
# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
```

Running the example provides a mean estimate of classification accuracy.

```
0.712166780588
```

## 摘要

在这篇文章中，您发现了集成机器学习算法，用于提高模型在问题上的表现。

你了解到：

*   Bagging Ensembles 包括袋装决策树，随机森林和额外树木。
*   推动包括 AdaBoost 和随机梯度提升在内的合奏。
*   Voting Ensembles 用于平均任意模型的预测。

您对 scikit-learn 中的整体机器学习算法或合奏有任何疑问吗？在评论中提出您的问题，我会尽力回答。