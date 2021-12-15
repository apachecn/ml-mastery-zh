# 如何获得基线结果及其重要性

> 原文： [https://machinelearningmastery.com/how-to-get-baseline-results-and-why-they-matter/](https://machinelearningmastery.com/how-to-get-baseline-results-and-why-they-matter/)

在我的课程和指南中，我在深入研究现场检查算法之前教授基线结果的准备。

我的一名学生最近问：

> 如果没有针对问题计算基线，是否会使其他算法的结果有问题？

他继续问：

> 如果其他算法没有提供比基线更好的准确度，我们应该从中得到什么教训？它是否表明数据集没有预测能力？

这些都是很好的问题，它们解释了为什么我们首先创建基线以及它提供的过滤能力。

在本文中，您将了解为什么我们创建基线预测结果，如何创建基线以及特定问题类型，以及如何使用它来通知您可用的数据和您正在使用的算法。

[![Baseline Machine Learning Results](img/0f07eecc60c827c4570c9722d470020f.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/11/Baseline-Machine-Learning-Results.jpg)

基线机器学习结果
照片由[特雷西惊人](http://www.flickr.com/photos/tracy_the_astonishing/7226371136)，保留一些权利

## 查找可以建模的数据

当您练习机器学习时，每个问题都是独一无二的。您很可能以前没有看到它，您无法知道要使用哪些算法，哪些数据属性将是有用的，甚至是否可以有效地建模问题。

我个人认为这是最激动人心的时刻。

如果您处于这种情况，您很可能自己从不同的来源收集数据并选择您认为可能有价值的属性。 [将需要特征选择](http://machinelearningmastery.com/an-introduction-to-feature-selection/ "An Introduction to Feature Selection")和[特征工程](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/ "Discover Feature Engineering, How to Engineer Features and How to Get Good at It")。

在此过程中，您需要了解您迭代尝试定义和收集数据的问题为做出预测提供了有用的基础。

## 一个有用的比较点

您需要针对问题的[抽样检查算法](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")，看看您是否有一个有用的基础来建模您的预测问题。但是你怎么知道结果有什么好处呢？

您需要一个比较结果的基础。您需要一个有意义的参考点来进行比较。

一旦开始从不同的机器学习算法中收集结果，基线结果可以告诉您更改是否正在增加值。

它如此简单，却如此强大。获得基线后，您可以添加或更改数据属性，正在尝试的算法或算法参数，并了解您是否已改进了问题的方法或解决方案。

## 计算基线结果

您可以使用常用方法计算基线结果。

基线结果是最简单的预测。对于某些问题，这可能是随机结果，而在其他问题中可能是最常见的预测。

*   **分类**：如果您有分类问题，可以选择观察次数最多的类，并将该类用作所有预测的结果。在 [Weka](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/ "How to Run Your First Classifier in Weka") 中，这被称为 [ZeroR](http://weka.sourceforge.net/doc.dev/weka/classifiers/rules/ZeroR.html) 。如果训练数据集中所有类的观察数相等，则可以选择特定类或枚举每个类，并查看哪个类在测试工具中提供了更好的结果。
*   **回归**：如果您正在处理回归问题，您可以使用集中趋势度量作为所有预测的结果，例如均值或中位数。
*   **优化**：如果您正在处理优化问题，则可以在域中使用固定数量的随机样本。

您可以将宝贵的时间用于集体讨论可以测试问题的所有最简单的结果，然后继续评估它们。结果可以是非常有效的过滤方法。如果更高级的建模方法不能胜过简单的中心趋势，那么您就知道自己有工作要做，最有可能更好地定义或重构问题。

您使用的[准确度分数](http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/ "Classification Accuracy is Not Enough: More Performance Measures You Can Use")很重要。在计算基线之前，您必须选择计划使用的准确度分数。分数必须是相关的，并通过首先处理问题来告知您要回答的问题。

如果您正在处理分类问题，您可能需要查看 [Kappa统计量](http://en.wikipedia.org/wiki/Cohen's_kappa)，它会为您提供基线标准化的准确度分数。基线准确度为0，高于零的分数显示基线的改善。

## 将结果与基线进行比较

如果您的基线结果不佳，则可以。它可能表明问题特别困难，或者可能意味着您的算法有很大的改进空间。

如果您无法获得比基线更好的准确度，那么这很重要。这表明问题可能很难。

您可能需要收集更多或不同的数据来进行建模。您可能需要研究使用不同的，可能更强大的机器学习算法或算法配置。最终，经过这些类型的更改，您可能会遇到一个对预测有抵抗力的问题，可能需要重新构建。

## 行动步骤

此帖子的操作步骤是开始使用基准调查下一个数据问题，您可以从中比较所有结果。

如果您已经在处理问题，请包含基线结果并使用该结果来解释所有其他结果。

分享您的结果，您的问题是什么以及您使用的基线是什么？