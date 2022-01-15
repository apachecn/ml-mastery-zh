# LOOCV 评估机器学习算法

> 原文:[https://machinelearning master . com/loocv-for-evaluation-machine-learning-algorithms/](https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/)

最后更新于 2020 年 8 月 26 日

当机器学习算法用于对未用于训练模型的数据进行预测时，使用**留一交叉验证**或 **LOOCV** 程序来估计机器学习算法的性能。

这是一个计算成本很高的过程，尽管它会产生模型性能的可靠且无偏的估计。虽然使用起来很简单，也不需要指定配置，但是有时不应该使用该过程，例如当您有一个非常大的数据集或一个计算成本很高的模型要评估时。

在本教程中，您将发现如何使用省去交叉验证来评估机器学习模型。

完成本教程后，您将知道:

*   当您有一个小数据集，或者当模型性能的准确估计比方法的计算成本更重要时，省略交叉验证过程是合适的。
*   如何使用 scikit-learn 机器学习库执行省去交叉验证程序。
*   如何使用留一交叉验证评估用于分类和回归的机器学习算法。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![LOOCV for Evaluating Machine Learning Algorithms](img/f1465ceb1909859785ed84d34ef3f072.png)

LOOCV 评价机器学习算法
图片由[希瑟·哈维](https://flickr.com/photos/smilygrl/4771680957/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  LOOCV 模型评价
2.  科学工具包中的 LOOCV 程序-学习
3.  LOOCV 评估机器学习模型
    1.  LOOCV 分类协会
    2.  回归的 LOOCV

## LOOCV 模型评价

交叉验证，或 k 倍交叉验证，是在对模型训练期间未使用的数据进行预测时，用于估计机器学习算法性能的过程。

交叉验证有一个超参数“ *k* ”，它控制数据集被分割成的子集的数量。分割后，每个子集都有机会用作测试集，而所有其他子集一起用作训练数据集。

这意味着 k 重交叉验证涉及拟合和评估 *k* 模型。这反过来提供了模型在数据集上的性能的 k 个估计，可以使用汇总统计数据(如平均值和标准偏差)来报告。然后，该分数可用于比较并最终选择用作数据集的“*最终模型*”的模型和配置。

k 的典型值为 k=3、k=5 和 k=10，其中 10 代表最常见的值。这是因为，给定广泛的测试，与其他 k 值和单一训练测试分割相比，10 倍交叉验证在模型性能估计中提供了低计算成本和低偏差的良好平衡。

有关 k-fold 交叉验证的更多信息，请参见教程:

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

留一交叉验证，或 LOOCV，是 k 倍交叉验证的一种配置，其中 *k* 被设置为数据集中的示例数。

LOOCV 是 k 倍交叉验证的极端版本，计算成本最高。它要求为训练数据集中的每个示例创建和评估一个模型。

如此多的拟合和评估模型的好处是对模型性能的更可靠的估计，因为每一行数据都有机会表示整个测试数据集。

考虑到计算成本，LOOCV 不适合非常大的数据集，如超过数万或数十万个例子，也不适合拟合成本高的模型，如神经网络。

*   **不要使用 LOOCV** :大数据集或昂贵的模型来适应。

给定模型性能的改进估计，当模型性能的精确估计是关键时，LOOCV 是合适的。当数据集很小时，例如少于数千个示例时，这种情况尤其明显，这可能导致训练过程中的模型过拟合以及对模型性能的有偏差的估计。

此外，假设没有使用训练数据集的采样，这种估计过程是确定性的，不像训练测试分割和其他 k 倍交叉验证确认那样提供模型性能的随机估计。

*   **使用 LOOCV** :小数据集或当估计模型性能至关重要时。

一旦使用 LOOCV 评估了模型，并选择了最终模型和配置，最终模型就适合所有可用数据，并用于对新数据进行预测。

现在我们已经熟悉了 LOOCV 过程，让我们看看如何在 Python 中使用该方法。

## 科学工具包中的 LOOCV 程序-学习

scikit-leave Python 机器学习库通过 [LeaveOneOut 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)提供了 LOOCV 的实现。

该方法没有配置，因此，没有提供任何参数来创建类的实例。

```
...
# create loocv procedure
cv = LeaveOneOut()
```

创建后，可以调用 *split()* 函数并提供要枚举的数据集。

每次迭代将从提供的数据集中返回可用于训练和测试集的行索引。

```
...
for train_ix, test_ix in cv.split(X):
	...
```

这些索引可用于数据集数组的输入( *X* )和输出( *y* )列，以分割数据集。

```
...
# split data
X_train, X_test = X[train_ix, :], X[test_ix, :]
y_train, y_test = y[train_ix], y[test_ix]
```

训练集可用于拟合模型，测试集可用于通过首先进行预测并根据预测值和期望值计算性能度量来评估模型。

```
...
# fit model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
# evaluate model
yhat = model.predict(X_test)
```

可以从每个评估中保存分数，并且可以呈现模型性能的最终平均估计。

我们可以将这些联系在一起，并演示如何使用 LOOCV 评估一个用于合成二进制分类数据集的 [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 模型，该数据集是使用 [make_blobs()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)创建的。

下面列出了完整的示例。

```
# loocv to manually evaluate the performance of a random forest classifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# create dataset
X, y = make_blobs(n_samples=100, random_state=1)
# create loocv procedure
cv = LeaveOneOut()
# enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]
	# fit model
	model = RandomForestClassifier(random_state=1)
	model.fit(X_train, y_train)
	# evaluate model
	yhat = model.predict(X_test)
	# store
	y_true.append(y_test[0])
	y_pred.append(yhat[0])
# calculate accuracy
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)
```

手动运行该示例可以估计合成数据集上随机森林分类器的性能。

假设数据集有 100 个示例，这意味着创建了数据集的 100 个训练/测试拆分，数据集的每一行都有机会用作测试集。同样，创建并评估了 100 个模型。

然后报告所有预测的分类准确率，在这种情况下为 99%。

```
Accuracy: 0.990
```

手动枚举折叠的缺点是速度慢，并且包含大量可能引入错误的代码。

使用 LOOCV 评估模型的另一种方法是使用 [cross_val_score()函数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

该函数通过“ *cv* ”参数获取模型、数据集和实例化的 LOOCV 对象集。然后返回准确度分数的样本，可通过计算平均值和标准偏差进行汇总。

我们还可以将“ *n_jobs* ”参数设置为-1 来使用所有的 CPU 内核，大大降低了拟合和评估这么多模型的计算成本。

下面的例子演示了在同一合成数据集上使用 LOOCV 使用 *cross_val_score()* 函数评估*随机森林分类器*。

```
# loocv to automatically evaluate the performance of a random forest classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# create dataset
X, y = make_blobs(n_samples=100, random_state=1)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestClassifier(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会自动估计合成数据集上随机森林分类器的性能。

所有折叠的平均分类精度与我们之前的手动估计相匹配。

```
Accuracy: 0.990 (0.099)
```

现在我们已经熟悉了如何使用 LeaveOneOut 类，让我们看看如何使用它来评估真实数据集上的机器学习模型。

## LOOCV 评估机器学习模型

在本节中，我们将探索使用 LOOCV 过程来评估标准分类和回归预测建模数据集上的机器学习模型。

### LOOCV 分类协会

我们将演示如何使用 LOOCV 评估声纳数据集上的随机森林算法。

声纳数据集是一个标准的机器学习数据集，包括 208 行具有 60 个数字输入变量的数据和一个具有两个类值的目标变量，例如二进制分类。

该数据集包括预测声纳回波显示的是岩石还是模拟地雷。

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集描述(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```
# summarize the sonar dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 208 行数据，60 个输入变量。

```
(208, 60) (208,)
```

我们现在可以用 LOOCV 来评估一个模型。

首先，加载的数据集必须分成输入和输出组件。

```
...
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

接下来，我们定义 LOOCV 程序。

```
...
# create loocv procedure
cv = LeaveOneOut()
```

然后我们可以定义要评估的模型。

```
...
# create model
model = RandomForestClassifier(random_state=1)
```

然后使用 *cross_val_score()* 函数枚举褶皱，拟合模型，然后进行预测和评估。然后，我们可以报告模型性能的平均值和标准偏差。

```
...
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

将这些联系在一起，完整的示例如下所示。

```
# loocv evaluate random forest on the sonar dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestClassifier(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例首先加载数据集，并确认输入和输出元素中的行数。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

然后使用 LOOCV 对模型进行评估，当对新数据进行预测时，估计的性能具有大约 82.2%的准确性。

```
(208, 60) (208,)
Accuracy: 0.822 (0.382)
```

### 回归的 LOOCV

我们将演示如何使用 LOOCV 评估住房数据集上的随机森林算法。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

该数据集包括预测美国波士顿郊区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

以下示例将数据集下载并加载为熊猫数据框，并总结了数据集的形状。

```
# load and summarize the housing dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
```

运行该示例确认了 506 行数据、13 个输入变量和单个数值目标变量(总共 14 个)。

```
(506, 14)
```

我们现在可以用 LOOCV 来评估一个模型。

首先，加载的数据集必须分成输入和输出组件。

```
...
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

接下来，我们定义 LOOCV 程序。

```
...
# create loocv procedure
cv = LeaveOneOut()
```

然后我们可以定义要评估的模型。

```
...
# create model
model = RandomForestRegressor(random_state=1)
```

然后使用 *cross_val_score()* 函数枚举褶皱，拟合模型，然后进行预测和评估。然后，我们可以报告模型性能的平均值和标准偏差。

在这种情况下，我们使用适用于回归的平均绝对误差(MAE)性能指标。

```
...
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force positive
scores = absolute(scores)
# report performance
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

将这些联系在一起，完整的示例如下所示。

```
# loocv evaluate random forest on the housing dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestRegressor(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force positive
scores = absolute(scores)
# report performance
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例首先加载数据集，并确认输入和输出元素中的行数。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

该模型使用 LOOCV 进行评估，在对新数据进行预测时，该模型的性能平均绝对误差约为 2.180(千美元)。

```
(506, 13) (506,)
MAE: 2.180 (2.346)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

### 蜜蜂

*   [交叉验证:评估评估者绩效，scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) 。
*   [sklearn.model_selection。离开应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)。
*   [sklearn . model _ selection . cross _ val _ score API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

## 摘要

在本教程中，您发现了如何使用省去交叉验证来评估机器学习模型。

具体来说，您了解到:

*   当您有一个小数据集，或者当模型性能的准确估计比方法的计算成本更重要时，省略交叉验证过程是合适的。
*   如何使用 scikit-learn 机器学习库执行省去交叉验证程序。
*   如何使用留一交叉验证评估用于分类和回归的机器学习算法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。