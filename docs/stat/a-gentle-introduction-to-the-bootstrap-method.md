# 浅谈自举法

> 原文： [https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)

引导方法是一种重采样技术，用于通过对替换的数据集进行采样来估计总体的统计量。

它可用于估计汇总统计量，例如平均值或标准差。它用于应用机器学习，以在对未包括在训练数据中的数据做出预测时估计机器学习模型的技能。

估计机器学习模型技能的结果的期望特性是估计技能可以用置信区间来呈现，该特征区间是诸如交叉验证之类的其他方法不易获得的特征。

在本教程中，您将发现用于估计机器学习模型在未见数据上的技能的引导程序重采样方法。

完成本教程后，您将了解：

*   引导方法涉及使用替换迭代地重采样数据集。
*   在使用引导程序时，您必须选择样本的大小和重复次数。
*   scikit-learn提供了一个函数，您可以使用该函数重采样引导程序方法的数据集。

让我们开始吧。

![A Gentle Introduction to the Bootstrap Method](img/24057e8549daa5cb03ff20448f643257.jpg)

一个关于自举法的温和介绍
照片由 [john mcsporran](https://www.flickr.com/photos/127130111@N06/16789909464/) 拍摄，保留一些权利。

## 教程概述

本教程分为4个部分;他们是：

1.  自举法
2.  Bootstrap的配置
3.  工作示例
4.  Bootstrap API

## 自举法

自举方法是一种统计技术，用于通过平均来自多个小数据样本的估计来估计关于群体的量。

重要的是，通过一次一个地从大数据样本中绘制观察结果并在选择它们之后将它们返回到数据样本来构建样本。这允许给定的观察不止一次地包括在给定的小样本中。这种采样方法称为采样替换。

构建一个样本的过程可归纳如下：

1.  选择样本的大小。
2.  虽然样本的大小小于所选的大小
    1.  从数据集中随机选择一个观测值
    2.  将其添加到样本中

引导方法可用于估计总体数量。这是通过重复采样小样本，计算统计量和计算统计量的平均值来完成的。我们可以总结这个程序如下：

1.  选择要执行的许多引导程序示例
2.  选择样本大小
3.  对于每个bootstrap样本
    1.  使用所选尺寸替换样品
    2.  计算样本的统计量
4.  计算计算的样本统计量的平均值。

该过程还可用于估计机器学习模型的技能。

> 引导程序是一种广泛适用且功能非常强大的统计工具，可用于量化与给定估计器或统计学习方法相关的不确定性。

- 第187页，[统计学习导论](http://amzn.to/2FkHqvW)，2013。

这是通过在样品上训练模型并评估模型的技能来完成的，这些样品不包括在样品中。这些未包含在给定样品中的样品称为袋外样品，或简称OOB。

使用自举法估计模型技能的这个过程可以总结如下：

1.  选择要执行的许多引导程序示例
2.  选择样本大小
3.  对于每个bootstrap样本
    1.  使用所选尺寸替换样品
    2.  在数据样本上拟合模型
    3.  估计模型在袋外样品上的技巧。
4.  计算模型技能估计样本的平均值。

> 未选择的样品通常称为“袋外”样品。对于给定的bootstrap重采样迭代，在所选样本上建立模型，并用于预测袋外样本。

- 第72页， [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。

重要的是，在拟合模型或调整模型的超参数之前的任何数据准备必须在数据样本的for循环内进行。这是为了避免数据泄漏，其中使用测试数据集的知识来改进模型。反过来，这可以导致对模型技能的乐观估计。

自举方法的一个有用特征是所得到的估计样本通常形成高斯分布。除了总结这种具有集中趋势的分布之外，还可以给出方差测量，例如标准偏差和标准误差。此外，可以计算置信区间并用于约束所呈现的估计。这在展示机器学习模型的估计技能时很有用。

## Bootstrap的配置

执行引导时必须选择两个参数：样本的大小和要执行的过程的重复次数。

### 样本量

在机器学习中，通常使用与原始数据集相同的样​​本大小。

> 引导样本与原始数据集的大小相同。因此，一些样本将在bootstrap样本中多次表示，而其他样本则根本不会被选中。

— Page 72, [Applied Predictive Modeling](http://amzn.to/2Fmrbib), 2013.

如果数据集很大并且计算效率是个问题，则可以使用较小的样本，例如数据集大小的50％或80％。

### 重复

重复次数必须足够大，以确保可以在样本上计算有意义的统计量，例如平均值，标准偏差和标准误差。

最小值可能是20或30次重复。可以使用较小的值将进一步增加对估计值样本计算的统计量的方差。

理想情况下，在给定时间资源的情况下，估计样本将尽可能大，具有数百或数千个重复。

## 工作示例

我们可以通过一个小的工作示例来使引导程序具体化。我们将完成该过程的一次迭代。

想象一下，我们有一个包含6个观测值的数据集：

```py
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
```

第一步是选择样本的大小。在这里，我们将使用4。

接下来，我们必须从数据集中随机选择第一个观测值。我们选择0.2。

```py
sample = [0.2]
```

该观察结果返回到数据集，我们再重复此步骤3次。

```py
sample = [0.2, 0.1, 0.2, 0.6]
```

我们现在有我们的数据样本。该示例有目的地证明了相同的值在样本中可以显示为零，一次或多次。这里观察0.2出现两次。

然后可以对绘制的样本计算估计值。

```py
statistic = calculation([0.2, 0.1, 0.2, 0.6])
```

未为样品选择的那些观察结果可用作样品外观察结果。

```py
oob = [0.3, 0.4, 0.5]
```

在评估机器学习模型的情况下，模型适合于绘制的样本并在袋外样本上进行评估。

```py
train = [0.2, 0.1, 0.2, 0.6]
test = [0.3, 0.4, 0.5]
model = fit(train)
statistic = evaluate(model, test)
```

结束了该程序的一个重复。它可以重复30次或更多次，以提供计算统计量的样本。

```py
statistics = [...]
```

然后，可以通过计算平均值，标准偏差或其他汇总值来汇总该统计样本，以给出统计量的最终可用估计值。

```py
estimate = mean([...])
```

## Bootstrap API

我们不必手动实现自举法。 scikit-learn库提供了一个实现，它将创建数据集的单个引导样本。

可以使用 [resample（）scikit-learn函数](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)。它采用数据数组作为参数，是否采样替换，采样的大小，以及采样之前使用的伪随机数生成器的种子。

例如，我们可以创建一个引导程序来创建一个替换为4个观察值的样本，并为伪随机数生成器使用值1。

```py
boot = resample(data, replace=True, n_samples=4, random_state=1)
```

不幸的是，API没有包含任何机制来轻松收集可用作评估拟合模型的测试集的袋外观察。

至少在单变量情况下，我们可以使用简单的Python列表理解来收集袋外观察。

```py
# out of bag observations
oob = [x for x in data if x not in boot]
```

我们可以将所有这些与我们在前一部分的工作示例中使用的小数据集结合在一起。

```py
# scikit-learn bootstrap
from sklearn.utils import resample
# data sample
data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# prepare bootstrap sample
boot = resample(data, replace=True, n_samples=4, random_state=1)
print('Bootstrap Sample: %s' % boot)
# out of bag observations
oob = [x for x in data if x not in boot]
print('OOB Sample: %s' % oob)
```

运行该示例打印自举样本中的观察结果以及袋外样本中的观察结果

```py
Bootstrap Sample: [0.6, 0.4, 0.5, 0.1]
OOB Sample: [0.2, 0.3]
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   列出3个可以使用自举法估计的摘要统计量。
*   找到3篇使用自举法评估机器学习模型表现的研究论文。
*   使用自举法实现您自己的函数以创建样本和袋外样本。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何计算Python中机器学习结果的Bootstrap置信区间](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)

### 图书

*   [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。
*   [统计学习导论](http://amzn.to/2FkHqvW)，2013年。
*   [引导程序简介](http://amzn.to/2G0Yatr)，1994。

### API

*   [sklearn.utils.resample（）API](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)
*   [sklearn.model_selection：模型选择API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

### 用品

*   [维基百科上的重新取样（统计）](https://en.wikipedia.org/wiki/Resampling_(statistics))
*   [维基百科上的引导（统计）](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
*   [经验样本数量](https://stats.stackexchange.com/questions/86040/rule-of-thumb-for-number-of-bootstrap-samples)的经验法则，CrossValiated。

## 摘要

在本教程中，您发现了用于估计机器学习模型在未见数据上的技能的引导程序重采样方法。

具体来说，你学到了：

*   引导方法涉及使用替换迭代地重采样数据集。
*   在使用引导程序时，您必须选择样本的大小和重复次数。
*   scikit-learn提供了一个函数，您可以使用该函数重采样引导程序方法的数据集。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。