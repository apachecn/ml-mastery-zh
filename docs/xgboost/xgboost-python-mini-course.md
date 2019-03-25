# 在Python中开始使用XGBoost的7步迷你课程

> 原文： [https://machinelearningmastery.com/xgboost-python-mini-course/](https://machinelearningmastery.com/xgboost-python-mini-course/)

### XGBoost使用Python迷你课程。

XGBoost是梯度增强的一种实现，用于赢得机器学习竞赛。

它很强大，但很难开始。

在这篇文章中，您将发现使用Python的XGBoost 7部分速成课程。

这个迷你课程专为已经熟悉scikit-learn和SciPy生态系统的Python机器学习从业者而设计。

让我们开始吧。

*   **2017年1月更新**：已更新，以反映scikit-learn API版本0.18.1中的更改​​。
*   **更新March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![XGBoost With Python Mini-Course](img/8c06c6e9b7a4dcc217ead3a0c339fb54.jpg)

XGBoost与Python迷你课程
照片由 [Teresa Boardman](https://www.flickr.com/photos/tboard/6566015625/) ，保留一些权利。

（**提示**：_你可能想打印或书签这个页面，以便以后再参考_。）

## 这个迷你课程是谁？

在我们开始之前，让我们确保您在正确的位置。以下列表提供了有关本课程设计对象的一般指导原则。

如果你没有完全匹配这些点，请不要惊慌，你可能只需要在一个或另一个区域刷新以跟上。

*   **开发人员知道如何编写一些代码**。这意味着使用Python完成任务并了解如何在工作站上设置SciPy生态系统（先决条件）对您来说并不是什么大问题。它并不意味着你是一个向导编码器，但它确实意味着你不怕安装软件包和编写脚本。
*   **知道一点机器学习的开发人员**。这意味着您了解机器学习的基础知识，如交叉验证，一些算法和偏差 - 方差权衡。这并不意味着你是一个机器学习博士，只是你知道地标或知道在哪里查找它们。

这个迷你课程不是XGBoost的教科书。没有方程式。

它将带您从一个熟悉Python的小机器学习的开发人员到能够获得结果并将XGBoost的强大功能带到您自己的项目中的开发人员。

## 迷你课程概述（期待什么）

这个迷你课程分为7个部分。

每节课的目的是让普通开发人员大约30分钟。你可能会更快完成一些，而其他人可能会选择更深入，花更多时间。

您可以根据需要快速或慢速完成每个部分。舒适的时间表可能是在一周的时间内每天完成一节课。强烈推荐。

您将在接下来的7节课中讨论的主题如下：

*   **第01课**：Gradient Boosting简介。
*   **第02课**：XGBoost简介。
*   **第03课**：开发你的第一个XGBoost模型。
*   **第04课**：监控性能和提前停止。
*   **第05课**：功能与XGBoost的重要性。
*   **第06课**：如何配置梯度提升。
*   **第07课**：XGBoost Hyperparameter Tuning。

这将是一件很有趣的事情。

你将不得不做一些工作，一点点阅读，一点研究和一点点编程。您想了解XGBoost吗？

（**提示**：_这些课程的帮助可以在这个博客上找到，使用搜索功能_。）

如有任何问题，请在下面的评论中发布。

在评论中分享您的结果。

挂在那里，不要放弃！

## 第01课：梯度提升简介

梯度提升是构建预测模型的最强大技术之一。

提升的想法来自于弱学习者是否可以被修改为变得更好的想法。应用程序取得巨大成功的第一个实现提升的是Adaptive Boosting或简称AdaBoost。 AdaBoost中的弱学习者是决策树，只有一个分裂，称为决策树桩的短缺。

AdaBoost和相关算法在统计框架中重铸，并被称为梯度增强机器。统计框架将推进作为一个数值优化问题，其目标是通过使用类似过程的梯度下降添加弱学习者来最小化模型的损失，因此得名。

Gradient Boosting算法涉及三个要素：

1.  **要优化的损失函数**，例如用于分类的交叉熵或用于回归问题的均方误差。
2.  **做出预测的弱学习者**，例如贪婪构建的决策树。
3.  **一个加法模型，**用于添加弱学习者以最小化损失函数。

为了纠正所有先前树木的残留误差，将新的弱学习者添加到模型中。结果是一个强大的预测建模算法，可能比随机森林更强大。

在下一课中，我们将仔细研究梯度提升的XGBoost实现。

## 第02课：XGBoost简介

XGBoost是为速度和性能而设计的梯度提升决策树的实现。

XGBoost代表e **X** treme **G** radient **Boosti** ng。

它由陈天琪开发，激光专注于计算速度和模型性能，因此几乎没有多余的装饰。

除了支持该技术的所有关键变体之外，真正感兴趣的是通过精心设计实施所提供的速度，包括：

*   **在训练期间使用所有CPU内核构建树的并行化**。
*   **分布式计算**用于使用一组机器训练超大型模型。
*   **非核心计算**适用于不适合内存的超大型数据集。
*   **缓存优化**的数据结构和算法，以充分利用硬件。

传统上，梯度提升实现很慢，因为必须构造每个树并将其添加到模型中的顺序性质。

XGBoost开发中的性能已经成为最好的预测建模算法之一，现在可以利用硬件平台的全部功能，或者您可能在云中租用的超大型计算机。

因此，XGBoost一直是竞争机器学习的基石，是赢家赢得和推荐的技术。例如，以下是一些最近的Kaggle比赛获奖者所说的话：

> 作为越来越多的Kaggle比赛的赢家，XGBoost再次向我们展示了一个值得在您的工具箱中使用的全面算法。

- [Dato Winners的采访](http://goo.gl/AHkmWx)

> 如有疑问，请使用xgboost。

- [Avito Winner的采访](http://goo.gl/sGyGtu)

在下一课中，我们将使用Python开发我们的第一个XGBoost模型。

## 第03课：开发您的第一个XGBoost模型

假设您有一个可用的SciPy环境，可以使用pip轻松安装XGBoost。

例如：

```
sudo pip install xgboost
```

您可以在 [XGBoost安装说明](http://xgboost.readthedocs.io/en/latest/build.html)中了解有关在您的平台上安装和构建XGBoost的更多信息。

XGBoost模型可以使用包装类直接在scikit-learn框架中使用， **XGBClassifier** 用于分类， **XGBRegressor** 用于回归问题。

这是在Python中使用XGBoost的推荐方法。

从 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)下载Pima Indians糖尿病数据集[更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。它是二进制分类的一个很好的测试数据集，因为所有输入变量都是数字的，这意味着问题可以直接建模而无需数据准备。

我们可以通过构造它并调用 **model.fit（）**函数来训练XGBoost模型进行分类：

```
model = XGBClassifier()
model.fit(X_train, y_train)
```

然后可以通过在新数据上调用 **model.predict（）**函数来使用该模型进行预测。

```
y_pred = model.predict(X_test)
```

我们可以将这些结合起来如下：

```
# First XGBoost model for Pima Indians dataset
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
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

在下一课中，我们将研究如何使用早期停止来限制过度拟合。

## 第04课：监控性能和提前停止

XGBoost模型可以在训练期间评估和报告模型的测试集上的性能。

它通过在训练模型和指定详细输出（ **verbose = True** ）时调用 **model.fit（）**时指定测试数据集和评估指标来支持此功能。

例如，我们可以在训练XGBoost模型时报告独立测试集（ **eval_set** ）上的二进制分类错误率（**错误**），如下所示：

```
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
```

使用此配置运行模型将在添加每个树后报告模型的性能。例如：

```
...
[89] validation_0-error:0.204724
[90] validation_0-error:0.208661
```

一旦没有对模型进行进一步改进，我们就可以使用此评估来停止培训。

我们可以通过在调用 **model.fit（）**时将 **early_stopping_rounds** 参数设置为在停止训练之前验证数据集未见改进的迭代次数来完成此操作。

下面提供了使用Pima Indians Onset of Diabetes数据集的完整示例。

```
# exmaple of early stopping
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
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

在下一课中，我们将研究如何使用XGBoost计算特征的重要性

## 第05课：使用XGBoost进行功能重要性

使用诸如梯度增强之类的决策树方法的集合的好处是它们可以从训练的预测模型自动提供特征重要性的估计。

经过训练的XGBoost模型可自动计算预测建模问题的特征重要性。

这些重要性分数可在训练模型的 **feature_importances_** 成员变量中找到。例如，它们可以直接打印如下：

```
print(model.feature_importances_)
```

XGBoost库提供了一个内置函数来绘制按其重要性排序的特征。

该函数称为 **plot_importance（）**，可以按如下方式使用：

```
plot_importance(model)
pyplot.show()
```

这些重要性分数可以帮助您确定要保留或丢弃的输入变量。它们也可以用作自动特征选择技术的基础。

下面提供了使用Pima Indians Onset of Diabetes数据集绘制特征重要性分数的完整示例。

```
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()
```

在下一课中，我们将研究启发式算法，以便最好地配置梯度增强算法。

## 第06课：如何配置梯度提升

渐变助推是应用机器学习最强大的技术之一，因此很快成为最受欢迎的技术之一。

但是，如何为您的问题配置梯度提升？

在原始梯度提升论文中发表了许多配置启发式方法。它们可以概括为：

*   学习率或收缩率（XGBoost中的 **learning_rate** ）应设置为0.1或更低，较小的值将需要添加更多树。
*   树的深度（XGBoost中的 **tree_depth** ）应该在2到8的范围内配置，其中对于更深的树没有多少益处。
*   行采样（XGBoost中的**子样本**）应配置在训练数据集的30％到80％的范围内，并且与未采样的100％的值进行比较。

这些是配置模型时的一个很好的起点。

一个好的通用配置策略如下：

1.  运行默认配置并查看培训和验证数据集上的学习曲线图。
2.  如果系统过度学习，则降低学习率和/或增加树木数量。
3.  如果系统学习不足，可以通过提高学习率和/或减少树木数量来加快学习速度。

[Owen Zhang](http://goo.gl/OqIRIc) ，前Kaggle排名第一的竞争对手，现在是Data Robot的首席技术官提出了一个配置XGBoost的有趣策略。

他建议将树木的数量设置为目标值，如100或1000，然后调整学习率以找到最佳模型。这是快速找到好模型的有效策略。

在下一节和最后一节中，我们将看一个调整XGBoost超参数的示例。

## 第07课：XGBoost超参数调整

scikit-learn框架提供了搜索参数组合的功能。

此功能在 **GridSearchCV** 类中提供，可用于发现配置模型以获得最佳性能的最佳方法。

例如，我们可以定义一个树的数量（ **n_estimators** ）和树大小（ **max_depth** ）的网格，通过将网格定义为：

```
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
```

然后使用10倍交叉验证评估每个参数组合：

```
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
result = grid_search.fit(X, label_encoded_y)
```

然后，我们可以查看结果，以确定最佳组合以及改变参数组合的一般趋势。

这是将XGBoost应用于您自己的问题时的最佳做法。要考虑调整的参数是：

*   树木的数量和大小（ **n_estimators** 和 **max_depth** ）。
*   学习率和树木数量（ **learning_rate** 和 **n_estimators** ）。
*   行和列子采样率（**子样本**， **colsample_bytree** 和 **colsample_bylevel** ）。

下面是调整Pima Indians Onset of Diabetes数据集中 **learning_rate** 的完整示例。

```
# Tune learning_rate
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# grid search
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
```

## XGBoost学习迷你课程评论

恭喜你，你做到了。做得好！

花点时间回顾一下你走了多远：

*   您了解了梯度增强算法和XGBoost库。
*   您开发了第一个XGBoost模型。
*   您学习了如何使用早期停止和功能重要性等高级功能。
*   您学习了如何配置梯度增强模型以及如何设计受控实验来调整XGBoost超参数。

不要轻视这一点，你在很短的时间内走了很长的路。这只是您在Python中使用XGBoost的旅程的开始。继续练习和发展你的技能。

你喜欢这个迷你课吗？你有任何问题或疑点吗？
发表评论让我知道。