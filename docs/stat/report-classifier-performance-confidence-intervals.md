# 如何使用置信区间报告分类器表现

> 原文： [https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)

为分类问题选择机器学习算法后，需要向利益相关者报告模型的表现。

这很重要，因此您可以设置模型对新数据的期望。

常见的错误是仅报告模型的分类准确率。

在这篇文章中，您将了解如何计算模型表现的置信区间，以提供模型技能的校准和稳健指示。

让我们开始吧。

![How to Report Classifier Performance with Confidence Intervals](img/2e92d75ca53eaf79d5297bc78a6bfa10.jpg)

如何使用置信区间报告分类器表现
[Andrew](https://www.flickr.com/photos/arg_flickr/15966084776/) 的照片，保留一些权利。

## 分类准确率

分类机器学习算法的技能通常被报告为分类准确率。

这是所有预测的正确预测百分比。计算方法如下：

```py
classification accuracy = correct predictions / total predictions * 100.0
```

分类器可以具有诸如 60％或 90％的准确度，并且这仅在问题域的上下文中具有意义。

## 分类错误

在向利益相关者讨论模型时，谈论分类错误或只是错误可能更为相关。

这是因为利益相关者认为模型表现良好，他们可能真的想知道模型是否容易出错。

您可以将分类错误计算为对预测数量的错误预测百分比，表示为 0 到 1 之间的值。

```py
classification error = incorrect predictions / total predictions
```

分类器可能具有 0.25 或 0.02 的误差。

该值也可以通过乘以 100 转换为百分比。例如，0.02 将变为（0.02 * 100.0）或 2％分类错误。

## 验证数据集

您使用什么数据集来计算模型技能？

从建模过程中提取验证数据集是一种很好的做法。

这意味着随机选择可用数据的样本并从可用数据中删除，以便在模型选择或配置期间不使用它。

在针对训练数据准备最终模型之后，可以使用它来对验证数据集做出预测。这些预测用于计算分类准确度或分类错误。

## 置信区间

不是仅呈现单个错误分数，而是可以计算置信区间并将其呈现为模型技能的一部分。

置信区间由两部分组成：

*   **范围**。这是模型上可以预期的技能的下限和上限。
*   **概率**。这是模型技能落在范围内的概率。

通常，分类错误的置信区间可以如下计算：

```py
error +/- const * sqrt( (error * (1 - error)) / n)
```

如果 error 是分类错误，const 是定义所选概率的常数值，sqrt 是平方根函数，n 是用于评估模型的观察（行）数。从技术上讲，这被称为 [Wilson 评分区间](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)。

const 的值由统计提供，常用值为：

*   1.64（90％）
*   1.96（95％）
*   2.33（98％）
*   2.58（99％）

使用这些置信区间会产生一些您需要确保可以满足的假设。他们是：

*   验证数据集中的观察结果独立地从域中提取（例如它们是[独立且相同分布的](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)）。
*   至少使用 30 个观察值来评估模型。

这是基于采样理论的一些统计量，它将分类器的误差计算为二项分布，我们有足够的观测值来逼近二项分布的正态分布，并且通过中心极限定理我们分类的观察结果越多，我们越接近真实但未知的模型技能。

## 置信区间示例

在具有 50 个示例（n = 50）的验证数据集上考虑具有 0.02（错误= 0.02）的错误的模型。

我们可以如下计算 95％置信区间（const = 1.96）：

```py
error +/- const * sqrt( (error * (1 - error)) / n)
0.02 +/- 1.96 * sqrt( (0.02 * (1 - 0.02)) / 50)
0.02 +/- 1.96 * sqrt(0.0196 / 50)
0.02 +/- 1.96 * 0.0197
0.02 +/- 0.0388
```

或者，换句话说：

置信区间[0.0,0.0588]有 95％的可能性涵盖模型对未见数据的真实分类误差。

请注意，分类错误的置信区间必须剪切为值 0.0 和 1.0。不可能有负误差（例如小于 0.0）或误差大于 1.0。

## 进一步阅读

*   第 5 章，[机器学习](http://www.amazon.com/dp/1259096955?tag=inspiredalgor-20)，1997
*   维基百科上的[二项式比例置信区间](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
*   维基百科上的[置信区间](https://en.wikipedia.org/wiki/Confidence_interval)

## 摘要

在这篇文章中，您了解了如何计算分类器的置信区间。

具体来说，你学到了：

*   报告结果时如何计算分类准确度和分类错误。
*   在计算要报告的模型技能时要使用的数据集。
*   如何计算选定可能性水平的分类误差的下限和上限。

您对分类器置信区间有任何疑问吗？
在下面的评论中提出您的问题。