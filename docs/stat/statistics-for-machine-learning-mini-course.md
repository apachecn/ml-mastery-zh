# 机器学习统计（7天迷你课程）

> 原文： [https://machinelearningmastery.com/statistics-for-machine-learning-mini-course/](https://machinelearningmastery.com/statistics-for-machine-learning-mini-course/)

### 机器学习速成课程统计。

#### _获取7天机器学习中使用的统计数据。_

统计学是一门数学领域，普遍认为这是更深入理解机器学习的先决条件。

虽然统计数据是一个具有许多深奥理论和发现的大型领域，但机器学习从业者需要从该领域获取的螺母和螺栓工具和符号。凭借统计数据的坚实基础，可以专注于好的或相关的部分。

在本速成课程中，您将了解如何在七天内开始并自信地阅读和实现使用Python进行机器学习的统计方法。

这是一个重要且重要的帖子。您可能想要将其加入书签。

让我们开始吧。

![Statistics for Machine Learning (7-Day Mini-Course)](img/61c2f760ae3a13b773943cbb352dcc64.jpg)

机器学习统计（7天迷你课程）
摄影： [Graham Cook](https://www.flickr.com/photos/grazza123/14076525468/) ，保留一些权利。

## 谁是这个崩溃课程？

在我们开始之前，让我们确保您在正确的位置。

本课程适用于可能了解某些应用机器学习的开发人员。也许你知道如何使用流行的工具来完成预测性建模问题的端到端，或者至少是大多数主要步骤。

本课程的课程会假设您的一些事情，例如：

*   你知道你的基本Python编程方式。
*   你可能知道一些基本的NumPy用于数组操作。
*   您希望学习统计数据，以加深您对机器学习的理解和应用。

你不需要知道：

*   你不需要成为一个数学家！
*   您不需要成为机器学习专家！

这个速成课程将带您从了解机器学习的开发人员到可以浏览统计方法基础知识的开发人员。

注意：此速成课程假设您有一个至少安装了NumPy的Python3 SciPy环境。如果您需要有关环境的帮助，可以按照此处的分步教程进行操作：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 速成课程概述

这个速成课程分为七个课程。

您可以每天完成一节课（推荐）或在一天内完成所有课程（硬核）。这取决于你有空的时间和你的热情程度。

下面列出了七个课程，这些课程将帮助您开始并提高Python中机器学习的统计数据：

*   **第01课**：统计和机器学习
*   **第02课**：统计学概论
*   **第03课**：高斯分布和描述性统计
*   **第04课**：变量之间的相关性
*   **第05课**：统计假设检验
*   **第06课**：估算统计
*   **第07课**：非参数统计

每节课可能需要60秒或30分钟。花点时间，按照自己的进度完成课程。在下面的评论中提出问题甚至发布结果。

课程期望你去学习如何做事。我会给你提示，但每节课的部分内容是强迫你学习去哪里寻求帮助以及统计方法和NumPy API以及Python中最好的工具（提示：我直接在这个博客上得到了所有的答案;使用搜索框）。

在评论中发布您的结果;我会为你加油！

挂在那里;不要放弃。

注意：这只是一个速成课程。有关更多详细信息和充实的教程，请参阅我的书，题为“[机器学习统计方法](https://machinelearningmastery.com/statistics_for_machine_learning/)”。

## 第01课：统计和机器学习

在本课程中，您将了解机器学习从业者应该加深对统计学的理解的五个原因。

### 1.数据准备统计

在为您的机器学习模型准备训练和测试数据时需要统计方法。

这包括以下技术：

*   异常值检测。
*   缺少价值归责。
*   数据采样。
*   数据扩展。
*   变量编码。

以及更多。

需要对数据分布，描述性统计和数据可视化有基本的了解，以帮助您确定执行这些任务时要选择的方法。

### 2.模型评估统计

在评估机器学习模型对训练期间未见的数据的技能时，需要统计方法。

This includes techniques for:

*   数据采样。
*   数据重采样。
*   实验设计。

机器学习从业者通常很好地理解诸如k折交叉验证之类的重采样技术，但是为什么需要这种方法的理由却不是。

### 3.模型选择中的统计

在选择用于预测性建模问题的最终模型或模型配置时，需要统计方法。

这些包括以下技术：

*   检查结果之间的显着差异。
*   量化结果之间差异的大小。

这可能包括使用统计假设检验。

### 4.模型演示中的统计

在向利益相关者介绍最终模型的技能时，需要统计方法。

This includes techniques for:

*   总结模型的预期技能平均值。
*   在实践中量化模型技能的预期变化。

这可能包括估计统计数据，如置信区间。

### 5.预测统计

在使用新数据的最终模型做出预测时，需要统计方法。

This includes techniques for:

*   量化预测的预期可变性。

这可能包括估计统计数据，如预测间隔。

### 你的任务

在本课程中，您必须列出您个人想要学习统计信息的三个原因。

在下面的评论中发表您的答案。我很乐意看到你想出了什么。

在下一课中，您将发现统计数据的简明定义。

## 第02课：统计学概论

在本课程中，您将发现统计信息的简明定义。

统计数据是大多数应用机器学习书籍和课程的必备先决条件。但究竟什么是统计数据？

统计学是数学的一个子领域。它指的是一组处理数据和使用数据来回答问题的方法。

这是因为该领域包含一系列用于处理数据的方法，对于初学者而言，这些方法看起来很大且无定形。很难看到属于统计的方法和属于其他研究领域的方法之间的界限。

当谈到我们在实践中使用的统计工具时，将统计领域划分为两大类方法可能会有所帮助：用于汇总数据的描述性统计数据和用于从数据样本中得出结论的推论统计数据。

*   **描述性统计**：描述性统计是指将原始观察汇总为我们可以理解和分享的信息的方法。
*   **推论统计**：推论统计是一种奇特的名称，有助于从一小组获得的观察结果（称为样本）中量化域或种群的属性。

### Your Task

在本课程中，您必须列出可用于每个描述性和推理统计信息的三种方法。

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

在下一课中，您将发现高斯分布以及如何计算摘要统计。

## 第03课：高斯分布和描述性统计

在本课程中，您将发现数据的高斯分布以及如何计算简单的描述性统计数据。

数据样本是来自更广泛群体的快照，可以从域中获取或由流程生成。

有趣的是，许多观察结果符合称为正态分布的常见模式或分布，或者更正式地说，符合高斯分布。这是您可能熟悉的钟形分布。

关于高斯分布的知识很多，因此，存在可以与高斯数据一起使用的统计和统计方法的整个子场。

任何高斯分布，以及从高斯分布中提取的任何数据样本，都可以用两个参数进行汇总：

*   **平均值**。分布中的中心趋势或最可能的价值（钟的顶部）。
*   **方差**。观察值与分布中的平均值（差值）之间的平均差异。

均值的单位与分布的单位相同，尽管方差的单位是平方的，因此难以解释。方差参数的一个流行替代方案是**标准差**，它只是方差的平方根，返回的单位与分布的单位相同。

可以直接在NumPy中的数据样本上计算均值，方差和标准差。

下面的示例生成从高斯分布绘制的100个随机数的样本，其已知均值为50，标准差为5，并计算汇总统计量。

```py
# calculate summary stats
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import var
from numpy import std
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate statistics
print('Mean: %.3f' % mean(data))
print('Variance: %.3f' % var(data))
print('Standard Deviation: %.3f' % std(data))
```

运行该示例并将估计的平均值和标准偏差与预期值进行比较。

### Your Task

在本课程中，您必须在Python中从零开始计算一个描述性统计信息，例如计算样本均值。

Post your answer in the comments below. I would love to see what you discover.

在下一课中，您将了解如何量化两个变量之间的关系。

## 课04：变量之间的相关性

在本课程中，您将了解如何计算相关系数以量化两个变量之间的关系。

数据集中的变量可能由于许多原因而相关。

它可用于数据分析和建模，以更好地理解变量之间的关系。两个变量之间的统计关系称为它们的相关性。

相关性可能是正的，意味着两个变量在相同的方向上移动，或者是负的，这意味着当一个变量的值增加时，其他变量的值会减少。

*   **正相关**：两个变量在同一方向上变化。
*   **中性相关**：变量变化没有关系。
*   **负相关**：变量方向相反。

如果两个或多个变量紧密相关，某些算法的表现可能会恶化，称为多重共线性。一个例子是线性回归，其中应删除一个违规的相关变量，以提高模型的技能。

我们可以使用称为Pearson相关系数的统计方法量化两个变量样本之间的关系，该方法以该方法的开发者Karl Pearson命名。

`pearsonr()`NumPy函数可用于计算两个变量样本的Pearson相关系数。

下面列出了完整的示例，显示了一个变量依赖于第二个变量的计算。

```py
# calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# calculate Pearson's correlation
corr, p = pearsonr(data1, data2)
# display the correlation
print('Pearsons correlation: %.3f' % corr)
```

运行示例并查看计算的相关系数。

### Your Task

在本课程中，您必须加载标准机器学习数据集并计算每对数值变量之间的相关性。

Post your answer in the comments below. I would love to see what you discover.

在下一课中，您将发现统计假设检验。

## 第05课：统计假设检验

在本课程中，您将发现统计假设检验以及如何比较两个样本。

必须解释数据以增加含义。我们可以通过假设特定结构来解释数据，并使用统计方法来确认或拒绝假设。

该假设称为假设，用于此目的的统计检验称为统计假设检验。

统计检验的假设称为零假设，或假设为零（简称H0）。它通常被称为默认假设，或者假设没有任何变化。违反测试假设通常被称为第一个假设，假设为1，或简称为H1。

*   **假设0（H0）**：测试的假设成立并且未被拒绝。
*   **假设1（H1）**：测试的假设不成立并且在某种程度上被拒绝。

我们可以使用p值来解释统计假设检验的结果。

在零假设为真的情况下，p值是观察数据的概率。

概率很大意味着可能存在H0或默认假设。一个较小的值，例如低于5％（o.05）表明它不太可能并且我们可以拒绝H0而支持H1，或者某些东西可能不同（例如显着结果）。

广泛使用的统计假设检验是Student's t检验，用于比较两个独立样本的平均值。

默认假设是样本之间没有差异，而拒绝此假设表明存在一些显着差异。测试假设两个样本均来自高斯分布并具有相同的方差。

Student's t-test可以通过`ttest_ind()`SciPy函数在Python中实现。

下面是计算和解释已知不同的两个数据样本的学生t检验的示例。

```py
# student's t-test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = ttest_ind(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
```

运行代码并查看计算的统计值和p值的解释。

### Your Task

在本课程中，您必须列出三个其他统计假设检验，可用于检查样本之间的差异。

Post your answer in the comments below. I would love to see what you discover.

在下一课中，您将发现估计统计数据作为统计假设检验的替代方法。

## 第06课：估算统计

在本课程中，您将发现可用作统计假设检验替代方法的估算统计数据。

统计假设检验可用于指示两个样本之间的差异是否是由于随机机会，但不能评论差异的大小。

被称为“_新统计_”的一组方法正在增加使用而不是p值或者除了p值之外，以便量化效应的大小和估计值的不确定性的量。这组统计方法称为估计统计。

估算统计是描述三种主要方法类别的术语。三种主要
类方法包括：

*   **效果大小**。用于量化治疗或干预的效果大小的方法。
*   **区间估计**。量化值的不确定性的方法。
*   **Meta分析**。在多个类似研究中量化结果的方法。

在这三种中，应用机器学习中最有用的方法可能是区间估计。

间隔有三种主要类型。他们是：

*   **容差区间**：具有特定置信水平的分布的一定比例的界限或覆盖范围。
*   **置信区间**：总体参数估计的界限。
*   **预测区间**：单次观察的界限。

计算分类算法的置信区间的简单方法是计算二项式比例置信区间，其可以提供围绕模型的估计精度或误差的区间。

这可以使用`confint()`Statsmodels函数在Python中实现。

该函数将成功（或失败）计数，试验总数和显着性水平作为参数，并返回置信区间的下限和上限。

下面的例子在一个假设的案例中证明了这个函数，其中模型从具有100个实例的数据集中做出了88个正确的预测，并且我们对95％置信区间感兴趣（作为0.05的显着性提供给函数）。

```py
# calculate the confidence interval
from statsmodels.stats.proportion import proportion_confint
# calculate the interval
lower, upper = proportion_confint(88, 100, 0.05)
print('lower=%.3f, upper=%.3f' % (lower, upper))
```

运行示例并查看估计准确度的置信区间。

### Your Task

在本课程中，您必须列出两种方法，用于计算应用机器学习中的效果大小以及它们何时有用。

作为提示，考虑一个用于变量之间的关系，一个用于样本之间的差异。

Post your answer in the comments below. I would love to see what you discover.

在下一课中，您将发现非参数统计方法。

## 第07课：非参数统计

在本课程中，您将发现当数据不是来自高斯分布时可能使用的统计方法。

统计和统计方法领域的很大一部分专用于已知分布的数据。

分布未知或不易识别的数据称为非参数。

在使用非参数数据的情况下，可以使用专门的非参数统计方法来丢弃有关分布的所有信息。因此，这些方法通常被称为无分秘籍法。

在可以应用非参数统计方法之前，必须将数据转换为等级格式。因此，期望排名格式的数据的统计方法有时被称为排名统计，例如排名相关和排名统计假设检验。排名数据正如其名称所示。

程序如下：

*   按升序对样本中的所有数据进行排序。
*   为数据样本中的每个唯一值分配1到N的整数等级。

用于检查两个独立样本之间差异的广泛使用的非参数统计假设检验是Mann-Whitney U检验，以Henry Mann和Donald Whitney命名。

它是学生t检验的非参数等价物，但不假设数据是从高斯分布中提取的。

该测试可以通过`mannwhitneyu()`SciPy函数在Python中实现。

下面的例子演示了从已知不同的均匀分布中抽取的两个数据样本的测试。

```py
# example of the mann-whitney u test
from numpy.random import seed
from numpy.random import rand
from scipy.stats import mannwhitneyu
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
# compare samples
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
```

运行该示例并查看计算的统计数据和p值的解释。

### Your Task

在本课程中，您必须列出另外三种非参数统计方法。

Post your answer in the comments below. I would love to see what you discover.

这是迷你课程的最后一课。

## 结束！
（看你有多远）

你做到了。做得好！

花点时间回顾一下你到底有多远。

你发现：

*   统计学在应用机器学习中的重要性。
*   统计的简明定义和方法划分为两种主要类型。
*   高斯分布以及如何使用统计信息来描述具有此分布的数据。
*   如何量化两个变量的样本之间的关系。
*   如何使用统计假设检验检查两个样本之间的差异。
*   统计假设检验的替代方法称为估计统计。
*   不从高斯分布中提取数据时可以使用的非参数方法。

这只是您的机器学习统计数据的开始。继续练习和发展你的技能。

下一步，查看我的书[机器学习统计方法](https://machinelearningmastery.com/statistics_for_machine_learning/)。

## 摘要

你是怎么做迷你课程的？
你喜欢这个速成班吗？

你有任何问题吗？有没有任何问题？
让我知道。在下面发表评论。