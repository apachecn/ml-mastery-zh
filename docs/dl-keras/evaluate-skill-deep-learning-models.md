# 如何评价深度学习模型的技巧

> 原文： [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

我经常看到从业者对如何评估深度学习模型表示困惑。

从以下问题中可以看出这一点：

*   我应该使用什么随机种子？
*   我需要随机种子吗？
*   为什么我在后续运行中得不到相同的结果？

在这篇文章中，您将发现可用于评估深度学习模型的过程以及使用它的基本原理。

您还将发现有用的相关统计数据，您可以计算这些统计数据以显示模型的技能，例如标准偏差，标准误差和置信区间。

让我们开始吧。

![How to Evaluate the Skill of Deep Learning Models](img/8e1689ec640a31d8358af77078ac6bfb.png)

如何评估深度学习模型的技巧
照片由 [Allagash Brewing](https://www.flickr.com/photos/allagashbrewing/14612890354/) ，保留一些权利。

## 初学者的错误

您将模型拟合到训练数据并在测试数据集上进行评估，然后报告技能。

也许您使用 k 折交叉验证来评估模型，然后报告模型的技能。

这是初学者犯的错误。

看起来你做的是正确的事情，但有一个关键问题是你没有考虑到：

**深度学习模型是随机的。**

人工神经网络在适应数据集时使用随机性，例如随机初始权重和随机梯度下降期间每个训练时期的数据随机混洗。

这意味着每次相同的模型适合相同的数据时，它可能会给出不同的预测，从而具有不同的整体技能。

## 估算模型技能
（_ 模型方差控制 _）

我们没有所有可能的数据;如果我们这样做，我们就不需要做出预测。

我们有一个有限的数据样本，我们需要发现最好的模型。

### 使用训练测试拆分

我们通过将数据分成两部分来做到这一点，在数据的第一部分拟合模型或特定模型配置，并使用拟合模型对其余部分进行预测，然后评估这些预测的技能。这被称为训练测试分割，我们使用该技能估计模型在对新数据进行预测时在实践中的表现。

例如，这里有一些用于使用训练测试分割来评估模型的伪代码：

```py
train, test = split(data)
model = fit(train.X, train.y)
predictions = model.predict(test.X)
skill = compare(test.y, predictions)
```

如果您有大量数据或非常慢的模型进行训练，训练测试分割是一种很好的方法，但由于数据的随机性（模型的方差），模型的最终技能分数会很嘈杂。 。

这意味着适合不同数据的相同模型将给出不同的模型技能分数。

### 使用 k-fold 交叉验证

我们通常可以收紧这一点，并使用 k-fold 交叉验证等技术获得更准确的模型技能估算。这是一种系统地将可用数据分成 k 重折叠，将模型拟合在 k-1 折叠上，在保持折叠上进行评估，并对每个折叠重复此过程的技术。

这导致 k 个不同的模型具有 k 个不同的预测集合，并且反过来具有 k 个不同的技能分数。

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

## 估计随机模型的技能
（_ 控制模型稳定性 _）

随机模型，如深度神经网络，增加了一个额外的随机源。

这种额外的随机性使得模型在学习时具有更大的灵活性，但可以使模型更不稳定（例如，当在相同数据上训练相同模​​型时的不同结果）。

这与模型方差不同，当在不同数据上训练相同模​​型时，模型方差给出不同的结果。

为了得到随机模型技能的可靠估计，我们必须考虑这个额外的方差来源;我们必须控制它。

### 修复随机种子

一种方法是每次模型拟合时使用相同的随机性。我们可以通过修复系统使用的随机数种子然后评估或拟合模型来做到这一点。例如：

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

当每次运行代码时需要相同的结果时，这对于教程和演示很有用。

这很脆弱，不建议用于评估模型。

看帖子：

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

这是我推荐的估算深度学习模型技能的程序。

因为重复通常&gt; = 30，所以我们可以很容易地计算出平均模型技能的标准误差，即模型技能得分的估计平均值与未知的实际平均模型技能的差异（例如，mean_skill 可能有多差）

```py
standard_error = standard_deviation / sqrt(count(scores))
```

此外，我们可以使用 standard_error 来计算 mean_skill 的置信区间。这假设结果的分布是高斯分布，您可以通过查看直方图，Q-Q 图或对收集的分数使用统计检验来检查。

例如，95％的间隔是平均技能周围的（1.96 *标准误差）。

```py
interval = standard_error * 1.96
lower_interval = mean_skill - interval
upper_interval = mean_skill + interval
```

与使用大均值的标准误差相比，还有其他可能更加统计上更稳健的计算置信区间的方法，例如：

*   计算[二项式比例置信区间](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)。
*   使用自举到[估计经验置信区间](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Deriving_confidence_intervals_from_the_bootstrap_distribution)。

## 神经网络有多不稳定？

这取决于您的问题，网络和配置。

我建议进行敏感性分析以找出答案。

在同一数据上多次（30,100 或数千）评估相同的模型，只改变随机数生成器的种子。

然后检查所产生技能分数的均值和标准差。标准偏差（平均得分与平均得分的平均距离）将让您了解模型的不稳定程度。

### 多少重复？

我建议至少 30，也许 100，甚至数千，仅限于你的时间和计算机资源，以及递减的回报（例如，mean_skill 上的标准错误）。

更严格的是，我建议进行一项实验，研究估计模型技能对重复次数的影响和标准误差的计算（平均估计表现与真实潜在人口平均值的差异）。

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
在评论中发表您的问题，我会尽力回答。