# 如何评估深度学习模型的性能

> 原文： [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

我经常看到从业者对如何评估深度学习模型表示困惑。

从以下问题中可以看出这一点：

*   我应该使用什么随机种子？
*   我需要随机种子吗？
*   为什么我在后续运行中得不到相同的结果？

在这篇文章中，您将发现可用于评估深度学习模型的过程以及使用它的基本原理。

您还将发现有用的相关统计数据，您可以计算这些统计数据以显示模型性能的技巧，例如标准偏差，标准误差和置信区间。

让我们开始吧！

![How to Evaluate the Skill of Deep Learning Models](img/8e1689ec640a31d8358af77078ac6bfb.png)


照片由 [Allagash Brewing](https://www.flickr.com/photos/allagashbrewing/14612890354/)提供，并保留所属权利。

## 初学者易犯的错误

您将在训练数据集上拟合您的模型并且在测试数据集上评估您的模型，然后做出模型特性的报告。

也许您会使用 k 折交叉验证来评估模型，报告关于改进模型性能的技巧。

这是初学者常犯的错误。

看起来你做的是正确的事情，但有一个关键问题是你没有考虑到：

**深度学习模型是随机的。**

人工神经网络在拟合数据集时是随机性的，例如在每个迭代期间随机清洗数据，每个随机梯度下降期间随机初始化权重。

这意味着每次相同的模型拟合相同的数据时，它可能会给出不同的预测，从而具有不同的性能。

## 评估模型的技巧

（_模型方差控制_）

我们可能没有所有的数据，如果有，我们就不需要做出预测。

通常情况下，我们有一个有限的数据样本，我们需要利用这些数据拟合出最好的模型。

### 使用训练测试拆分

我们通过将数据分成两部分来做到这一点，在数据的第一部分拟合模型或特定模型配置，并使用拟合后的模型对其余部分做出预测，然后评估这些预测的性能，这中技巧被称为训练测试分割，我们使用该技能评估模型对新数据做出预测时在实践中的性能。

例如，这里有一些用于使用训练测试分割来评估模型的伪代码：

```py
train, test = split(data)
model = fit(train.X, train.y)
predictions = model.predict(test.X)
skill = compare(test.y, predictions)
```

如果您有大量数据或需要对非常慢的模型进行训练，训练测试分割是一种很好的方法，但由于数据的随机性（模型的方差），模型的最终性能分数会很混乱。

这意味着拟合不同数据的相同模型将给出不同的模型性能分数。

### 使用 k-fold 交叉验证

我们通常可以使用 k-fold 交叉验证等技术来加强这一点，并更准确地估计模型行性能，这是一种系统地将可用数据分成 k 重折叠，在k-1折上训练数据以拟合模型，在保持折叠上进行评估模型，并对每个折叠重复此过程的技术。

这导致 k 个不同的模型具有 k 个不同的预测集合，并且反过来具有 k 个不同的性能分数。

例如，这里有一些使用 k 折交叉验证评估模型的伪代码：

```py
scores = list()
for i in k:
	train, test = split_old(data, i)
	model = fit(train.X, train.y)
	predictions = model.predict(test.X)
	skill = compare(test.y, predictions)
	scores.append(skill)
```

技能分数更有用，因为我们可以采用均值并报告模型的平均预期表现，这可能更接近实际模型的实际表现。例如：

```py
mean_skill = sum(scores) / count(scores)
```

我们还可以使用 mean_skill 计算标准偏差，以了解 mean_skill 周围的平均分数差异：

```py
standard_deviation = sqrt(1/count(scores) * sum( (score - mean_skill)^2 ))
```

## 评估随机模型的性能
（_控制模型稳定性_）

一些随机模型，如深度神经网络，增加了一个额外的随机源。

这种额外的随机性使得模型在学习时具有更大的灵活性，但也可能会使模型更不稳定（例如，当在相同数据上训练相同模型时会有不同的结果）。

这与模型方的差不同，模型的方差通常是当在不同数据上训练相同模型时，模型方差给出不同的结果。

为了得到随机模型性能的可靠估计，我们必须考虑这个额外的方差来源并且我们必须控制它。

### 固定随机种子

一种方法是每次模型拟合时使用相同的随机数，我们可以通过固定系统使用的随机数种子然后评估或拟合模型来做到这一点。例如：

```py
seed(1)
scores = list()
for i in k:
	train, test = split_old(data, i)
	model = fit(train.X, train.y)
	predictions = model.predict(test.X)
	skill = compare(test.y, predictions)
	scores.append(skill)
```

这在每次运行代码或都需要相同的结果时，非常适合教程和演示。

这中做法是不稳定的，不建议用于评估模型。

如下文章所示：

*   [在机器学习中拥抱随机性](http://machinelearningmastery.com/randomness-in-machine-learning/)
*   [如何使用 Keras](http://machinelearningmastery.com/reproducible-results-neural-networks-keras/) 获得可重现的结果

### 重复评估实验

更强大的方法是重复多次评估非随机模型的实验。

例如：

```py
scores = list()
for i in repeats:
	run_scores = list()
	for j in k:
		train, test = split_old(data, j)
		model = fit(train.X, train.y)
		predictions = model.predict(test.X)
		skill = compare(test.y, predictions)
		run_scores.append(skill)
	scores.append(mean(run_scores))
```

注意，我们计算估计的平均模型技能的平均值，即所谓的[宏均值](https://en.wikipedia.org/wiki/Grand_mean)。

这是我推荐的评估深度学习模型技能的程序。

因为重复通常次数>=30，所以我们可以很容易地计算出平均模型性能的标准误差，即模型性能得分的估计平均值与未知的实际平均模型技能的差异（例如，mean_skill 的差值会有多大）

```py
standard_error = standard_deviation / sqrt(count(scores))
```

此外，我们可以使用 standard_error 来计算 mean_skill 的置信区间，假设结果的分布是高斯分布，您可以通过查看直方图，Q-Q 图或对收集的分数使用统计检验来检查。

例如，计算95％左右的间隔是平均性能的指标（1.96 *标准误差）。

```py
interval = standard_error * 1.96
lower_interval = mean_skill - interval
upper_interval = mean_skill + interval
```

与使用大均值的标准误差相比，还有其他可能在统计上更稳健的计算置信区间的方法，例如：

*   计算[二项式比例置信区间](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)。
*   使用自举到[估计经验置信区间](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Deriving_confidence_intervals_from_the_bootstrap_distribution)。

## 神经网络有多不稳定？

这取决于您的问题，网络和配置。

我建议进行敏感性分析以找出答案。

在同一数据上多次（30,100 或数千）评估相同的模型，只改变随机数生成器的种子。

然后检查所产生性能分数的均值和标准差，标准偏差（分数与平均分数的平均距离）将让您了解模型的不稳定程度。

### 多少次重复？

我建议至少 30，也许 100，甚至数千，仅限于你的时间和计算机资源，以及递减的回馈（例如，mean_skill 上的标准误差）。

更为严格地说，我建议进行一个实验，研究对估计模型技能的影响与重复次数的影响以及标准误差的计算（平均估计性能与真实基本总体平均值相差多少）。

## 进一步阅读

*   [在机器学习中拥抱随机性](http://machinelearningmastery.com/randomness-in-machine-learning/)
*   [如何训练最终机器学习模型](http://machinelearningmastery.com/train-final-machine-learning-model/)
*   [比较不同种类的交叉验证](http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm)
*   [人工智能的经验方法](http://www.amazon.com/dp/0262032252?tag=inspiredalgor-20)，Cohen，1995。
*   维基百科上的[标准错误](https://en.wikipedia.org/wiki/Standard_error)

## 摘要

在这篇文章中，您发现了如何评估深度学习模型的技能。

具体来说，你学到了：

*   初学者在评估深度学习模型时常犯的错误。
*   使用重复 k 倍交叉验证来评估深度学习模型的基本原理。
*   如何计算相关的模型技能统计数据，例如标准差，标准误差和置信区间。

您对估算深度学习模型的技能有任何疑问吗？
请在评论中发表您的问题，我会尽力回答。