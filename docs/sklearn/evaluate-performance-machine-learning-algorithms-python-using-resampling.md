# 使用重采样评估 Python 中机器学习算法的表现

> 原文： [https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/](https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/)

您需要知道算法在看不见的数据上的表现。

评估算法表现的最佳方法是对已经知道答案的新数据做出预测。第二种最好的方法是使用称为重采样方法的统计学中的巧妙技术，这些技术允许您准确估计算法对新数据的执行情况。

在这篇文章中，您将了解如何使用 Python 和 scikit-learn 中的重采样方法估计机器学习算法的准确率。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。
*   **2017 年 10 月更新**：更新了用于 Python 3 的打印语句。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Evaluate the Performance of Machine Learning Algorithms in Python using Resampling](img/69cda3d5c8e60d018a5e1e893a01d509.jpg)

使用重新取样
照片由 [Doug Waldron](https://www.flickr.com/photos/dougww/2453670430/) 评估 Python 中机器学习算法的表现，保留一些权利。

## 关于秘籍

本文使用 Python 中的小代码秘籍演示了重采样方法。

每个秘籍都是独立设计的，因此您可以将其复制并粘贴到项目中并立即使用。

每个秘籍中都使用 [Pima 印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)。这是一个二分类问题，其中所有输入变量都是数字。在每个秘籍中，它直接从 [UCI 机器学习库](http://archive.ics.uci.edu/ml/)下载（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。您可以根据需要将其替换为您自己的数据集。

## 评估您的机器学习算法

为什么不能在数据集上训练机器学习算法，并使用来自同一数据集的预测来评估机器学习算法？

简单的答案是过拟合。

想象一个算法，记住它显示的每个观察。如果您在用于训练算法的相同数据集上评估您的机器学习算法，那么这样的算法将在训练数据集上具有完美的分数。但它对新数据的预测将是可怕的。

我们必须在不用于训练算法的数据上评估我们的机器学习算法。

评估是一种估计，我们可以用它来讨论我们认为算法在实践中实际可行的程度。它不是表现的保证。

一旦我们估算了算法的表现，我们就可以在整个训练数据集上重新训练最终算法，并为运行使用做好准备。

接下来，我们将研究四种不同的技术，我们可以使用这些技术来分割训练数据集，并为我们的机器学习算法创建有用的表现估计：

1.  训练和测试集。
2.  K 折交叉验证。
3.  保留一次交叉验证。
4.  重复随机测试 - 训练分裂。

我们将从称为 Train 和 Test Sets 的最简单方法开始。

## 1.分成训练和测试装置

我们可以用来评估机器学习算法表现的最简单方法是使用不同的训练和测试数据集。

我们可以将原始数据集分成两部分。在第一部分训练算法，对第二部分做出预测，并根据预期结果评估预测。

拆分的大小可能取决于数据集的大小和细节，尽管通常使用 67％的数据进行训练，剩余的 33％用于测试。

该算法评估技术非常快。它非常适用于大型数据集（数百万条记录），其中有强有力的证据表明数据的两个分裂都代表了潜在的问题。由于速度的原因，当您正在调查的算法训练缓慢时，使用此方法很有用。

这种技术的缺点是它可能具有很大的差异。这意味着训练和测试数据集的差异可能导致准确度估计的有意义差异。

在下面的示例中，我们将数据 Pima Indians 数据集拆分为 67％/ 33％，用于训练和测试，并评估 Logistic 回归模型的准确率。

```
# Evaluate using a train and a test set
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
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
result = model.score(X_test, Y_test)
print("Accuracy: %.3f%%" % (result*100.0))
```

我们可以看到该模型的估计精度约为 75％。请注意，除了指定拆分的大小外，我们还指定了随机种子。由于数据的分割是随机的，我们希望确保结果是可重复的。通过指定随机种子，我们确保每次运行代码时都得到相同的随机数。

如果我们想要将此结果与另一个机器学习算法或具有不同配置的相同算法的估计精度进行比较，这一点很重要。为了确保比较是苹果换苹果，我们必须确保他们接受相同数据的训练和测试。

```
Accuracy: 75.591%
```

## 2\. K 折交叉验证

交叉验证是一种可用于估计机器学习算法表现的方法，其方差小于单个训练测试集拆分。

它的工作原理是将数据集分成 k 部分（例如 k = 5 或 k = 10）。每次分割数据称为折叠。该算法在 k-1 折叠上进行训练，其中一个被扣住并在保持的背部折叠上进行测试。重复这一过程，以便数据集的每个折叠都有机会成为阻碍测试集。

运行交叉验证后，您最终得到 k 个不同的表现分数，您可以使用均值和标准差来总结。

结果是在给定测试数据的情况下，对新数据的算法表现进行更可靠的估计。它更准确，因为算法在不同数据上被多次训练和评估。

k 的选择必须允许每个测试分区的大小足够大以成为问题的合理样本，同时允许对算法的训练测试评估的足够重复以提供对看不见的数据的算法表现的公平估计。 。对于数千或数万条记录中的适度大小的数据集，k 值为 3,5 和 10 是常见的。

在下面的示例中，我们使用 10 倍交叉验证。

```
# Evaluate using Cross Validation
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_instances = len(X)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
```

您可以看到我们报告了绩效指标的均值和标准差。在总结表现测量时，总结测量的分布是一种好的做法，在这种情况下假设高斯表现分布（非常合理的假设）并记录平均值和标准偏差。

```
Accuracy: 76.951% (4.841%)
```

## 3.保留一次交叉验证

您可以配置交叉验证，以便折叠的大小为 1（k 设置为数据集中的观察数）。交叉验证的这种变化称为留一交叉验证。

结果是可以总结大量的表现测量，以便更加合理地估计模型在看不见的数据上的准确率。缺点是它可能是计算上比 k 折交叉验证更昂贵的过程。

在下面的示例中，我们使用了一次性交叉验证。

```
# Evaluate using Leave One Out Cross Validation
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
loocv = model_selection.LeaveOneOut()
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
```

您可以在标准差中看到，得分比上述 k 倍交叉验证结果具有更多的方差。

```
Accuracy: 76.823% (42.196%)
```

## 4.重复随机测试 - 训练分裂

k 折交叉验证的另一个变化是创建数据的随机分割，如上面描述的训练/测试分裂，但重复多次分割和评估算法的过程，如交叉验证。

这具有使用训练/测试分割的速度以及 k 倍交叉验证的估计表现的方差的减少。您也可以根据需要重复此过程。不利的一面是，重复可能包括训练中的大部分相同数据或从运行到运行的测试拆分，从而在评估中引入冗余。

下面的示例将数据拆分为 67％/ 33％的训练/测试拆分，并重复该过程 10 次。

```
# Evaluate using Shuffle Split Cross Validation
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_samples = 10
test_size = 0.33
num_instances = len(X)
seed = 7
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
```

我们可以看到，绩效指标的分布与上面的 k 折交叉验证相同。

```
Accuracy: 76.496% (1.698%)
```

## 什么技术使用时

*   通常，k 折交叉验证是用于评估机器学习算法在看不见的数据上的表现的金标准，其中 k 设置为 3,5 或 10。
*   使用慢速算法时，使用训练/测试分割有利于提高速度，并在使用大型数据集时产生具有较低偏差的表现估计。
*   当试图平衡估计表现，模型训练速度和数据集大小的方差时，诸如留一交叉验证和重复随机分裂等技术可能是有用的中间体。

最好的建议是试验并找到一种快速解决问题的技术，并产生可用于制定决策的合理表现估算。如有疑问，请使用 10 倍交叉验证。

## 摘要

在这篇文章中，您发现了可用于估计机器学习算法表现的统计技术，称为重新采样。

具体来说，您了解到：

1.  训练和测试集。
2.  交叉验证。
3.  保留一次交叉验证。
4.  重复随机测试 - 训练分裂。

您对重新采样方法或此帖有任何疑问吗？在评论中提出您的问题，我会尽力回答。