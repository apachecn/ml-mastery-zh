# 使用随机森林：在121个数据集上测试179个分类器

> 原文： [https://machinelearningmastery.com/use-random-forest-testing-179-classifiers-121-datasets/](https://machinelearningmastery.com/use-random-forest-testing-179-classifiers-121-datasets/)

如果您不知道在问题上使用什么算法，请尝试一些。

或者，您可以尝试**随机森林**和可能**高斯SVM** 。

在最近的一项研究中，这两种算法在与超过100种数据集平均的近200种其他算法竞争时被证明是最有效的。

在这篇文章中，我们将回顾这项研究，并考虑在我们自己应用的机器学习问题上测试算法的一些含义。

[![Do We Need Hundreds of Classifiers](img/9d8c9a4e94f79734e175a188b54806c9.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/12/Do-We-Need-Hundreds-of-Classifiers.jpg)

我们需要数百个分类器
照片来自 [Thomas Leth-Olsen](http://www.flickr.com/photos/thomasletholsen/8064127235) ，保留一些权利

## 我们需要数百种分类器吗？

这篇论文的标题是“[我们需要数百个分类器来解决现实世界的分类问题吗？](http://jmlr.csail.mit.edu/papers/v15/delgado14a.html) “并于2014年10月发表于[机器学习研究期刊](http://www.jmlr.org/)。

*   [在此下载PDF](http://jmlr.csail.mit.edu/papers/volume15/delgado14a/delgado14a.pdf) 。

在本文中，作者评估了来自 [UCI机器学习库](http://archive.ics.uci.edu/ml/)的121个标准数据集中17个家族的179个分类器。

作为一种品味，这里列出了所研究的算法系列和每个系列中的算法数量。

*   判别分析（DA）：20个分类器
*   贝叶斯（BY）方法：6个分类器
*   神经网络（NNET）：21个分类器
*   支持向量机（SVM）：10个分类器
*   决策树（DT）：14个分类器。
*   基于规则的方法（RL）：12个分类器。
*   提升（BST）：20个分类器
*   套袋（BAG）：24种分类器
*   堆叠（STC）：2个分类器。
*   随机森林（RF）：8个分类器。
*   其他合奏（OEN）：11个分类。
*   广义线性模型（GLM）：5个分类器。
*   最近邻方法（NN）：5个分类器。
*   偏最小二乘和主成分回归（PLSR）：6
*   Logistic和多项式回归（LMR）：3个分类器。
*   多元自适应回归样条（MARS）：2个分类器
*   其他方法（OM）：10个分类器。

这是一项庞大的研究。

在贡献最终得分之前调整了一些算法，并且使用4倍交叉验证来评估算法。

切入追逐他们发现随机森林（特别是R中的平行随机森林）和高斯支持向量机（特别是来自 [libSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) ）的整体表现最佳。

从论文摘要：

> 最有可能成为最佳分类器的是随机森林（RF）版本，其中最好的（在R中实现并通过插入符访问）在84.3％的数据集中达到最大准确度的94.1％，超过90％。

在[讨论HackerNews关于论文](https://news.ycombinator.com/item?id=8719723)的讨论中，来自Kaggle的 [Ben Hamner](https://www.linkedin.com/pub/ben-hamner/12/597/987) 对袋装决策树的深刻表现作出了一个确凿的评论：

> 这与我们运行数百个Kaggle比赛的经验一致：对于大多数分类问题，合奏决策树（随机森林，梯度提升机器等）的一些变化表现最佳。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 非常小心准备数据

某些算法仅适用于分类数据，而其他算法则需要数字数据。少数人可以处理你扔给他们的任何东西。 UCI机器中的数据集通常是标准化的，但不足以在其原始状态下用于此类研究。

这已在“[关于为分类器准备数据的评论](http://www.win-vector.com/blog/2014/12/a-comment-on-preparing-data-for-classifiers/)”中指出。

在这篇评论中，作者指出，测试的相关数据集中的分类数据被系统地转换为数值，但这可能会阻碍某些算法的测试。

高斯SVM可能表现良好，因为分类属性转换为数字和所执行数据集的标准化。

尽管如此，我赞扬作者在应对这一挑战方面所具有的勇气，而那些愿意接受后续研究的人可以解决这些问题。

作者还注意到 [OpenML项目](http://openml.org)看起来像是一项公民科学工作，可以承担同样的挑战。

## 为什么研究这样？

根据[无免费午餐定理](http://en.wikipedia.org/wiki/No_free_lunch_in_search_and_optimization)（NFLT）的论点，很容易在本研究中嗤之以鼻。在对所有问题进行平均时，所有算法的表现都是相同的。

我不喜欢这个论点。 NFLT要求您没有先验知识。你不知道你正在做什么问题或者你正在尝试什么算法。这些条件不实用。

在论文中，作者列出了该项目的四个目标：

*   为所选数据集集合选择全局最佳分类器
*   根据其准确性对每个分类器和家庭进行排名
*   为每个分类器确定其达到最佳准确度的概率，以及其准确性与最佳准确度之间的差异
*   评估分类器行为改变数据集属性（复杂性，模式数量，类数和输入数）

该研究的作者承认，我们想要解决的实际问题是所有可能问题的一个子集，有效算法的数量不是无限的，而是可管理的。

本文是一个陈述，事实上我们可能对一套已知（但很小）问题的最常用（或实现）算法套件的能力有所说明。 （很像20世纪90年代中期的 [STATLOG项目](http://www.tandfonline.com/doi/abs/10.1080/08839519508945477#.VIi8ZVQW05E)）

## 在实践中：选择一个中间地带

在开始之前，您无法知道哪种算法（或算法配置）在您的问题上表现良好甚至最佳。

你必须尝试多种算法，并在那些证明能够挑选问题结构的少数人身上加倍努力。

我称之为[现场检查](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")，强烈建议采用[数据驱动方法来应用机器学习](http://machinelearningmastery.com/a-data-driven-approach-to-machine-learning/ "A Data-Driven Approach to Machine Learning")。

在这项研究的背景下，现场检查是一方面与您最喜欢的算法和另一方面测试所有已知算法之间的中间地带。

1.  **选择你最喜欢的算法**。快速但限于你最喜欢的算法或库。
2.  **现货检查十几个算法**。一种平衡的方法，允许更好的表现算法上升到顶部，让您专注。
3.  **测试所有已知/实现的算法**。耗时的详尽方法有时会产生令人惊讶的结果。

您在此范围内着陆取决于您拥有的时间和资源。请记住，问题的试验算法只是[处理问题](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/ "Process for working through Machine Learning Problems")过程中的一步。

测试所有算法需要强大的测试工具。这不容小觑。

当我[过去尝试过这个](http://machinelearningmastery.com/the-seductive-trap-of-black-box-machine-learning/ "The Seductive Trap of Black-Box Machine Learning")时，我发现大多数算法挑选出问题中的大部分结构。这是一个结果分布的结果，一个胖头长尾，脂肪头的差异往往非常小。

你想要有意义的是这个微小的差异。因此，您需要投入大量的前期时间来设计强大的测试工具（交叉验证，大量折叠，可能是单独的验证数据集），而不会出现数据泄漏（交叉验证折叠内的数据缩放/转换等）

我现在认为应用问题是理所当然的。我甚至不关心哪些算法上升。我专注于数据准备和集合各种足够好的模型的结果。

## 下一步

在处理机器学习问题时，你在哪里？

你坚持使用最喜欢或最喜欢的算法吗？您是否发现了检查，或者您是否试图详尽无遗地测试您最喜欢的库提供的所有内容？