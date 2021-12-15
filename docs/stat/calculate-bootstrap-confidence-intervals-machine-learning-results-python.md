# 如何计算Python中机器学习结果的Bootstrap置信区间

> 原文： [https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)

重要的是既展示机器学习模型的预期技能，又展示该模型技能的置信区间。

置信区间提供了一系列模型技能，以及在对新数据做出预测时模型技能落在范围之间的可能性。例如，分类准确率的95％可能性在70％和75％之间。

计算机器学习算法置信区间的有效方法是使用引导程序。这是用于估计可用于计算经验置信区间的统计数据的一般技术，无论技能分数的分布如何（例如，非高斯分布）

在本文中，您将了解如何使用引导程序计算机器学习算法表现的置信区间。

阅读这篇文章后，你会知道：

*   如何使用引导程序估计统计信息的置信区间。
*   如何应用此方法来评估机器学习算法。
*   如何实现用于估计Python中置信区间的bootstrap方法。

让我们开始吧。

*   **2017年6月更新**：修正了为numpy.percentile（）提供错误值的错误。谢谢Elie Kawerk。
*   **更新March / 2018** ：更新了数据集文件的链接。

![How to Calculate Bootstrap Confidence Intervals For Machine Learning Results in Python](img/14c678555f6f330945f25df0e047316d.jpg)

如何计算Python中机器学习结果的Bootstrap置信区间
照片由 [Hendrik Wieduwilt](https://www.flickr.com/photos/hendrikwieduwilt/13744081435/) ，保留一些权利。

## Bootstrap置信区间

使用引导程序计算置信区间包括两个步骤：

1.  计算统计数据
2.  计算置信区间

### 1.计算统计人口

第一步是使用引导程序多次重新采样原始数据并计算感兴趣的统计量。

使用替换对数据集进行采样。这意味着每次从原始数据集中选择一个项目时，都不会将其删除，从而允许再次为该样本选择该项目。

统计数据在样本上计算并存储，以便我们建立一个感兴趣的统计数据。

引导重复的数量定义了估计的方差，越多越好，通常是数百或数千。

我们可以使用以下伪代码演示此步骤。

```py
statistics = []
for i in bootstraps:
	sample = select_sample_with_replacement(data)
	stat = calculate_statistic(sample)
	statistics.append(stat)
```

### 2.计算置信区间

现在我们有了感兴趣的统计数据，我们可以计算置信区间。

这是通过首先排序统计数据，然后在置信区间选择所选百分位数的值来完成的。在这种情况下选择的百分位称为alpha。

例如，如果我们对95％的置信区间感兴趣，则α将为0.95，我们将选择2.5％百分位数的值作为下限，将97.5％百分位数作为感兴趣统计量的上限。

例如，如果我们从1,000个bootstrap样本计算1,000个统计数据，则下限将是第25个值，上限将是第975个值，假设统计列表已订购。

在这里，我们计算一个非参数置信区间，它不对统计分布的函数形式做出任何假设。该置信区间通常称为经验置信区间。

我们可以用下面的伪代码来证明这一点。

```py
ordered = sort(statistics)
lower = percentile(ordered, (1-alpha)/2)
upper = percentile(ordered, alpha+((1-alpha)/2))
```

## Bootstrap模型表现

引导程序可用于评估机器学习算法的表现。

每次迭代采样的大小可以限制为可用数据的60％或80％。这意味着将会有一些未包含在样本中的样本。这些被称为袋（OOB）样品。

然后可以在每个自举迭代的数据样本上训练模型，并在袋外样本上进行评估，以给出可以收集的表现统计量，并且可以从中计算置信区间。

我们可以使用以下伪代码演示此过程。

```py
statistics = []
for i in bootstraps:
	train, test = select_sample_with_replacement(data, size)
	model = train_model(train)
	stat = evaluate_model(test)
	statistics.append(stat)
```

## 计算分类准确率置信区间

本节演示如何使用引导程序使用Python机器学习库scikit-learn计算实际数据集上的机器学习算法的经验置信区间。

本节假定您已安装Pandas，NumPy和Matplotlib。如果您在设置环境方面需要帮助，请参阅教程：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

首先，下载 [Pima Indians数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)并将其放在当前工作目录中，文件名为“pima _-_ indians _-diabetes.data.csv_ ”（更新： [在这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

我们将使用Pandas加载数据集。

```py
# load dataset
data = read_csv('pima-indians-diabetes.data.csv', header=None)
values = data.values
```

接下来，我们将配置引导程序。我们将使用1,000次自举迭代并选择一个50％的数据集大小的样本。

```py
# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)
```

接下来，我们将迭代引导程序。

将使用sklearn中的 [resample（）函数](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)替换样本。检索未包含在样本中的任何行并将其用作测试数据集。接下来，决策树分类器适合样本并在测试集上进行评估，计算分类分数，并添加到跨所有引导收集的分数列表中。

```py
# run bootstrap
stats = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
```

收集分数后，将创建直方图以了解分数的分布。我们通常期望这种分布是高斯分布，也许是偏差与均值周围的对称方差。

最后，我们可以使用[百分位数（）NumPy函数](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)计算经验置信区间。使用95％置信区间，因此选择2.5和97.5百分位数的值。

综合这些，下面列出了完整的例子。

```py
import numpy
from pandas import read_csv
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
# load dataset
data = read_csv('pima-indians-diabetes.data.csv', header=None)
values = data.values
# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	print(score)
	stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
```

运行该示例会在每次引导迭代时打印分类精度。

创建1000个准确度分数的直方图，显示类似高斯分布。

![Distribution of Classification Accuracy Using the Bootstrap](img/58225babd47d6836760c7e90f3ed5a65.jpg)

使用Bootstrap分配分类准确度

最后，报告置信区间，表明置信区间64.4％和73.0％有95％的可能性涵盖模型的真实技能。

```py
...
0.646288209607
0.682203389831
0.668085106383
0.673728813559
0.686021505376
95.0 confidence interval 64.4% and 73.0%
```

该相同方法可用于计算任何其他误差分数的置信区间，例如回归算法的均方根误差。

## 进一步阅读

本节提供有关引导程序和引导程序置信区间的其他资源。

*   [Bootstrap简介](http://www.amazon.com/dp/0412042312?tag=inspiredalgor-20)，1996
*   [Bootstrap置信区间](https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214)，统计科学，1996
*   第5.2.3节，Bootstrap置信区间，[人工智能的经验方法](http://www.amazon.com/dp/0262032252?tag=inspiredalgor-20)
*   维基百科上的 [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
*   第4.4节重采样技术，[应用预测建模](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20)

## 摘要

在这篇文章中，您了解了如何使用引导程序来计算机器学习算法的置信区间。

具体来说，你学到了：

*   如何计算数据集中统计量的置信区间的自举估计值。
*   如何应用引导程序来评估机器学习算法。
*   如何计算Python中机器学习算法的bootstrap置信区间。

您对置信区间有任何疑问吗？
在下面的评论中提出您的问题。