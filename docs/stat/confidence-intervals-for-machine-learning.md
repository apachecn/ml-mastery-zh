# 机器学习中的置信区间

> 原文： [https://machinelearningmastery.com/confidence-intervals-for-machine-learning/](https://machinelearningmastery.com/confidence-intervals-for-machine-learning/)

许多机器学习涉及估计机器学习算法在看不见的数据上的表现。

置信区间是量化估计不确定性的一种方式。它们可用于在群体参数上添加界限或可能性，例如平均值，从群体的独立观察样本估计。

在本教程中，您将发现置信区间以及如何在实践中计算置信区间。

完成本教程后，您将了解：

*   置信区间是对总体参数的估计的界限。
*   可以直接计算分类方法的估计技能的置信区间。
*   可以使用引导程序以无分布的方式估计任意任意总体统计量的置信区间。

让我们开始吧。

*   **更新 June / 2018** ：修复了引导代码示例部分采样中的拼写错误。

![Confidence Intervals for Machine Learning](img/376c11a99795ebb0c589f41afc9292b1.jpg)

机器学习的置信区间
[Paul Balfe](https://www.flickr.com/photos/paul_e_balfe/34633468352/) 的照片，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  什么是置信区间？
2.  分类准确率的时间间隔
3.  非参数置信区间

## 什么是置信区间？

置信区间是人口变量估计值的界限。它是一个区间统计量，用于量化估计的不确定性。

> 包含人口或过程未知特征的置信区间。感兴趣的数量可以是人口属性或“参数”，例如人口或过程的平均值或标准差。

- 第 3 页，[统计间隔：从业者和研究人员指南](http://amzn.to/2G8w3IL)，2017 年。

置信区间不同于描述从分布中采样的数据范围的容差区间。它也不同于描述单个观察的界限的预测间隔。相反，置信区间提供总体参数的界限，例如平均值，标准偏差或类似值。

在应用机器学习中，我们可能希望在预测模型的技能呈现中使用置信区间。

例如，置信区间可用于表示分类模型的技能，可以表述为：

_ 给定样本，范围 x 到 y 有 95％的可能性涵盖真实的模型准确率。_

要么

_ 模型的准确度为 x +/- y，置信水平为 95％。_

置信区间也可用于呈现回归预测模型的误差;例如：

_ 范围 x 到 y 有 95％的可能性涵盖了模型的真实误差。_

or

_ 模型的误差在 95％置信水平下为 x +/- y。_

尽管使用了其他较不常见的值，例如 90％和 99.7％，但在呈现置信区间时，95％置信度的选择非常常见。在实践中，您可以使用您喜欢的任何值。

> 95％置信区间（CI）是根据我们的数据计算出的一系列值，很可能包括我们估计的人口数的真实值。

- 第 4 页，[新统计学概论：估计，开放科学及其他](http://amzn.to/2HhrT0w)，2016 年。

置信区间的值是其量化估计的不确定性的能力。它提供了下限和上限以及可能性。单独作为半径测量，置信区间通常被称为误差范围，并且可以用于通过使用[误差条](https://en.wikipedia.org/wiki/Error_bar)以图形方式描绘图表上的估计的不确定性。

通常，从中得出估计的样本越大，估计越精确，置信区间越小（越好）。

*   **较小的置信区间**：更精确的估计。
*   **更大的置信区间**：估计不太精确。

> 我们还可以说，CI 告诉我们估计的准确程度，误差幅度是我们的精确度量。短 CI 意味着一个小的误差范围，我们有一个相对精确的估计[...]长 CI 意味着很大的误差，我们的精度很低

— Page 4, [Introduction to the New Statistics: Estimation, Open Science, and Beyond](http://amzn.to/2HhrT0w), 2016.

置信区间属于称为估计统计的统计领域，可用于呈现和解释实验结果，而不是统计显着性检验或除此之外。

> 估算为分析和解释结果提供了更具信息性的方法。 [...]知道和思考效应的幅度和精确度对于定量科学比考虑观察至少那个极端数据的可能性更有用，假设绝对没有效果。

- [估算统计量应取代 2016 年的显着性检验](https://www.nature.com/articles/nmeth.3729)。

在实践中，使用统计显着性检验可能优选置信区间。

原因是从业者和利益相关者更容易直接与域名相关联。它们也可以被解释并用于比较机器学习模型。

> 这些不确定性估计有两个方面。首先，间隔使模型的消费者了解模型的好坏。 [...]通过这种方式，置信区间有助于衡量比较模型时可用证据的权重。置信区间的第二个好处是促进模型之间的权衡。如果两个模型的置信区间显着重叠，则表明两者之间的（统计）等价，并且可能提供支持不太复杂或更可解释的模型的理由。

- 第 416 页， [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。

现在我们知道置信区间是什么，让我们看一下我们可以用它来计算预测模型的几种方法。

## 分类准确率的时间间隔

分类问题是在给定一些输入数据的情况下预测标签或类结果变量的问题。

通常使用分类准确度或分类误差（精度的倒数）来描述分类预测模型的技能。例如，在 75％的时间内对类结果变量进行正确预测的模型具有 75％的分类准确度，计算公式如下：

```py
accuracy = total correct predictions / total predictions made * 100
```

可以基于训练期间模型未看到的保持数据集（例如验证或测试数据集）来计算该准确度。

分类精度或分类误差是[比例](https://en.wikipedia.org/wiki/Proportionality_(mathematics))或比率。它描述了模型所做的正确或不正确预测的比例。每个预测都是二元决策，可能是正确的或不正确的。从技术上讲，这被称为[伯努利试验](https://en.wikipedia.org/wiki/Bernoulli_trial)，以 Jacob Bernoulli 命名。伯努利试验中的比例具有称为[二项分布](https://en.wikipedia.org/wiki/Binomial_distribution)的特定分布。值得庆幸的是，对于大样本量（例如超过 30 个），我们可以使用高斯近似分布。

> 在统计学中，一系列成功或失败的独立事件称为伯努利过程。 [...]对于大 N，该随机变量的分布接近正态分布。

- 第 148 页，[数据挖掘：实用机器学习工具和技术](http://amzn.to/2G7sxhP)，第二版，2005 年。

我们可以使用比例的高斯分布（即分类精度或误差）的假设来容易地计算置信区间。

在分类错误的情况下，间隔的半径可以计算为：

```py
interval = z * sqrt( (error * (1 - error)) / n)
```

在分类精度的情况下，间隔的半径可以计算为：

```py
interval = z * sqrt( (accuracy * (1 - accuracy)) / n)
```

其中，间隔是置信区间的半径，误差和准确度分别是分类误差和分类精度，n 是样本的大小，sqrt 是平方根函数，z 是高斯分布的临界值。从技术上讲，这称为二项式比例置信区间。

常用的高斯分布临界值及其相应的显着性水平如下：

*   1.64（90％）
*   1.96（95％）
*   2.33（98％）
*   2.58（99％）

在具有 50 个示例（n = 50）的验证数据集上考虑具有 20％或 0.2（误差= 0.2）的误差的模型。我们可以如下计算 95％置信区间（z = 1.96）：

```py
# binomial confidence interval
from math import sqrt
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 50)
print('%.3f' % interval)
```

运行该示例，我们看到计算和打印的置信区间的计算半径。

```py
0.111
```

然后我们可以提出如下声明：

*   模型的分类误差为 20％+ / - 11％
*   模型的真实分类误差可能在 9％到 31％之间。

我们可以看到样本大小对置信区间半径的估计精度的影响。

```py
# binomial confidence interval
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 100)
print('%.3f' % interval)
```

运行该示例显示置信区间降至约 7％，从而提高了模型技能估计的精度。

```py
0.078
```

请记住，置信区间是一个范围内的可能性。真正的模型技能可能超出范围。

> 事实上，如果我们一遍又一遍地重复这个实验，每次绘制一个包含新例子的新样本 S，我们会发现，对于大约 95％的这些实验，计算的间隔将包含真实误差。出于这个原因，我们将此区间称为 95％置信区间估计

- 第 131 页，[机器学习](http://amzn.to/2tr32Wb)，1997。

[proportion_confint（）statsmodels 函数](http://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html)是二项式比例置信区间的实现。

默认情况下，它为二项分布做出高斯假设，但支持其他更复杂的计算变量。该函数将成功（或失败）计数，试验总数和显着性水平作为参数，并返回置信区间的下限和上限。

下面的例子在一个假设的案例中证明了这个函数，其中模型从具有 100 个实例的数据集中做出了 88 个正确的预测，并且我们对 95％置信区间感兴趣（作为 0.05 的显着性提供给函数）。

```py
from statsmodels.stats.proportion import proportion_confint
lower, upper = proportion_confint(88, 100, 0.05)
print('lower=%.3f, upper=%.3f' % (lower, upper))
```

运行该示例将打印模型分类精度的下限和上限。

```py
lower=0.816, upper=0.944
```

## 非参数置信区间

我们通常不知道所选绩效指标的分布。或者，我们可能不知道计算技能分数的置信区间的分析方法。

> 通常会违反作为参数置信区间基础的假设。预测变量有时不是正态分布的，即使是预测变量，正态分布的方差在预测变量的所有级别上也可能不相等。

- 第 326 页，[人工智能的经验方法](http://amzn.to/2FrFJgg)，1995。

在这些情况下，自举重采样方法可以用作计算置信区间的非参数方法，名义上称为自举置信区间。

自举是一种模拟蒙特卡罗方法，其中样本是从具有替换的固定有限数据集中提取的，并且对每个样本估计参数。该过程通过采样导致对真实总体参数的稳健估计。

我们可以使用以下伪代码来证明这一点。

```py
statistics = []
for i in bootstraps:
	sample = select_sample_with_replacement(data)
	stat = calculate_statistic(sample)
	statistics.append(stat)
```

该程序可用于通过在每个样品上拟合模型并评估模型在未包括在样品中的样品的技能来估计预测模型的技能。然后，当对看不见的数据进行评估时，可以将模型的均值或中值技能表示为模型技能的估计值。

通过从特定百分位数的技能分数样本中选择观察，可以将置信区间添加到该估计中。

回想一下百分位数是从分类样本中抽取的观察值，其中样本中观察值的百分比下降。例如，样本的第 70 个百分位表示 70％的样本低于该值。第 50 百分位数是分布的中位数或中间位数。

首先，我们必须选择置信水平的显着性水平，例如 95％，表示为 5.0％（例如 100-95）。因为置信区间在中位数附近是对称的，所以我们必须选择第 2.5 百分位数和第 97.5 百分位数的观察值来给出全范围。

我们可以用一个有效的例子来计算 bootstrap 置信区间。

假设我们有一个数据集，其中 1000 个观测值的值介于 0.5 和 1.0 之间，均匀分布。

```py
# generate dataset
dataset = 0.5 + rand(1000) * 0.5
```

我们将执行 100 次自举程序，并从替换的数据集中抽取 1,000 个观察样本。我们将估计人口的平均值作为我们将在自助样本上计算的统计量。这可以很容易地成为模型评估。

```py
# bootstrap
scores = list()
for _ in range(100):
	# bootstrap sample
	indices = randint(0, 1000, 1000)
	sample = dataset[indices]
	# calculate and store statistic
	statistic = mean(sample)
	scores.append(statistic)
```

一旦我们获得了 bootstrap 统计量的样本，我们就可以计算出集中趋势。我们将使用中位数或第 50 百分位，因为我们不假设任何分布。

```py
print('median=%.3f' % median(scores))
```

然后，我们可以将置信区间计算为以中位数为中心的中间 95％的观察统计值。

```py
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
```

首先，基于所选择的置信区间计算期望的较低百分位数。然后从引导统计量样本中检索此百分位数的观察结果。

```py
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
lower = max(0.0, percentile(scores, lower_p))
```

我们对置信区间的上边界做同样的事情。

```py
# calculate upper percentile (e.g. 97.5)
upper_p = (100 - alpha) + (alpha / 2.0)
# retrieve observation at upper percentile
upper = min(1.0, percentile(scores, upper_p))
```

下面列出了完整的示例。

```py
# bootstrap confidence intervals
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile
# seed the random number generator
seed(1)
# generate dataset
dataset = 0.5 + rand(1000) * 0.5
# bootstrap
scores = list()
for _ in range(100):
	# bootstrap sample
	indices = randint(0, 1000, 1000)
	sample = dataset[indices]
	# calculate and store statistic
	statistic = mean(sample)
	scores.append(statistic)
print('50th percentile (median) = %.3f' % median(scores))
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
lower = max(0.0, percentile(scores, lower_p))
print('%.1fth percentile = %.3f' % (lower_p, lower))
# calculate upper percentile (e.g. 97.5)
upper_p = (100 - alpha) + (alpha / 2.0)
# retrieve observation at upper percentile
upper = min(1.0, percentile(scores, upper_p))
print('%.1fth percentile = %.3f' % (upper_p, upper))
```

运行该示例总结了 bootstrap 样本统计量的分布，包括 2.5th，50th（中位数）和 97.5th 百分位数。

```py
50th percentile (median) = 0.750
2.5th percentile = 0.741
97.5th percentile = 0.757
```

然后，我们可以使用这些观察结果来对样本分布做出声明，例如：

_ 范围 0.741 至 0.757 的可能性为 95％，涵盖了真实的统计平均值。_

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   在您自己的小型人工测试数据集上测试每个置信区间方法。
*   找到 3 篇研究论文，证明每种置信区间法的使用。
*   开发一个函数来计算机器学习技能分数的给定样本的自举置信区间。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何以置信区间报告分类器表现](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)
*   [如何计算 Python 中机器学习结果的 Bootstrap 置信区间](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
*   [使用 Python 的置信区间理解时间序列预测不确定性](https://machinelearningmastery.com/time-series-forecast-uncertainty-using-confidence-intervals-python/)

### 图书

*   [了解新统计：影响大小，置信区间和元分析](http://amzn.to/2oQW6No)，2011。
*   [新统计学概论：估计，开放科学及其他](http://amzn.to/2HhrT0w)，2016 年。
*   [统计间隔：从业者和研究人员指南](http://amzn.to/2G8w3IL)，2017 年。
*   [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。
*   [机器学习](http://amzn.to/2tr32Wb)，1997。
*   [数据挖掘：实用机器学习工具和技术](http://amzn.to/2G7sxhP)，第二版，2005 年。
*   [Bootstrap 简介](http://amzn.to/2p2zUPl)，1996。
*   [人工智能的经验方法](http://amzn.to/2FrFJgg)，1995。

### 文件

*   [估算统计量应取代 2016 年的显着性检验](https://www.nature.com/articles/nmeth.3729)。
*   [Bootstrap 置信区间，统计科学](https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214)，1996。

### API

*   [statsmodels.stats.proportion.proportion_confint（）API](http://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html)
*   [numpy.random.rand（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html)
*   [numpy.random.randint（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html)
*   [numpy.random.seed（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)
*   [numpy.percentile（）API](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)
*   [numpy.median（）API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html)

### 用品

*   [维基百科的间隔估计](https://en.wikipedia.org/wiki/Interval_estimation)
*   [维基百科上的置信区间](https://en.wikipedia.org/wiki/Confidence_interval)
*   [维基百科上的二项式比例置信区间](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
*   [交叉验证的 RMSE 置信区间](https://stats.stackexchange.com/questions/78079/confidence-interval-of-rmse)
*   [维基百科上的引导](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

## 摘要

在本教程中，您发现了置信区间以及如何在实践中计算置信区间。

具体来说，你学到了：

*   置信区间是对总体参数的估计的界限。
*   可以直接计算分类方法的估计技能的置信区间。
*   可以使用引导程序以无分布的方式估计任意任意总体统计量的置信区间。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。