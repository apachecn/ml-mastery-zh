# 使用来自 UCI 机器学习库的数据集练习机器学习

> 原文： [https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/](https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)

在哪里可以获得良好的数据集来练习机器学习？

数据集是真实的，因此它们很有趣且相关，虽然足够小，您可以在 Excel 中查看并在桌面上完成工作。

在这篇文章中，您将发现一个高质量，真实世界且易于理解的机器学习数据集的数据库，您可以使用它来练习应用的机器学习。

该数据库称为 UCI 机器学习库，您可以使用它来构建自学程序并为机器学习奠定坚实的基础。

![Practice Practice Practice](img/1b9695b3c816e91aceb719bc1dfd95f5.jpg)

实践练习
摄影： [Phil Roeder](https://www.flickr.com/photos/tabor-roeder/16760089648/) ，保留一些权利。

## 我们为什么需要练习数据集？

如果您对应用机器学习感兴趣，则需要练习数据集。

这个问题可以阻止你死。

*   你应该使用哪个数据集？
*   你应该自己收集还是使用现成的？
*   哪一个和为什么？

我教授一种自上而下的机器学习方法，我鼓励您学习端到端解决问题的过程，将该过程映射到工具上，并以有针对性的方式对数据进行处理。有关更多信息，请参阅我的文章“[程序员机器学习：从开发人员到机器学习从业者的跳跃](http://machinelearningmastery.com/machine-learning-for-programmers/)”。

### 那你如何以有针对性的方式练习？

我教导说，最好的入门方法是练习具有特定特征的数据集。

我建议您选择在遇到自己的问题时遇到并需要解决的特征，例如：

*   不同类型的监督学习，如分类和回归。
*   来自数十，数百，数千和数百万个实例的不同大小的数据集。
*   来自少于十个，几十个，几百个和几千个属性的不同数量的属性
*   来自实数，整数，分类，序数和混合的不同属性类型
*   不同的域名会迫使您快速了解和描述您之前没有经验的新问题。

您可以通过设计一个测试问题数据集程序来创建一个学习和学习的特征程序以及解决它们所需的算法。

这样的程序有许多实际要求，例如：

*   **真实世界**：数据集应该来自现实世界（而不是设计）。这将使他们感兴趣并介绍真实数据带来的挑战。
*   **小**：数据集需要很小，以便您可以检查和理解它们，并且可以快速运行多个模型以加快学习周期。
*   **很好理解**：应该清楚地知道数据包含什么，为什么收集数据，需要解决的问题是什么，以便您可以构建调查框架。
*   **基线**：了解已知哪些算法表现良好以及获得的分数以便您有一个有用的比较点也很重要。当您开始学习时，这很重要，因为您需要快速反馈您的表现（接近最新技术或某些内容已被破坏）。
*   **丰富**：您需要选择许多数据集，以满足您想要调查的特征和（如果可能的话）您的天生好奇心和兴趣。

对于初学者，您可以从 UCI 机器学习库中获取所需的所有内容以及更多数据集。

## 什么是 UCI 机器学习库？

[UCI 机器学习库](http://archive.ics.uci.edu/ml/)是一个机器学习问题的数据库，您可以免费访问。

它由位于加州大学欧文分校的[机器学习和智能系统中心](http://cml.ics.uci.edu/)托管和维护。它最初由 [David Aha](http://home.earthlink.net/~dwaha/) 创建，作为加州大学欧文分校的研究生。

25 年来，它一直是需要数据集的机器学习研究人员和机器学习从业者的首选。

[![UCI Machine Learning Repository](img/c37b3b5aff40c6a329e7e93b31dbb937.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/08/UCI-Machine-Learning-Repository.png)

UCI 机器学习库

每个数据集都有自己的网页，列出了所有已知的详细信息，包括调查它的任何相关出版物。数据集本身可以作为 ASCII 文件下载，通常是有用的 CSV 格式。

例如，这里是[鲍鱼数据集](http://archive.ics.uci.edu/ml/datasets/Abalone)的网页，需要从物理测量中预测鲍鱼的年龄。

### 存储库的好处

该库的一些有益功能包括：

*   几乎所有数据集都是从域中提取的（而不是合成的），这意味着它们具有真实世界的品质。
*   数据集涵盖了从生物学到粒子物理学的广泛主题。
*   数据集的详细信息通过属性类型，实例数，属性数和可以排序和搜索的已发布年份等方面进行汇总。
*   对数据集进行了充分研究，这意味着它们在有趣的属性和预期的“好”结果方面是众所周知的。这可以为比较提供有用的基线。
*   大多数数据集都很小（数百到数千个实例），这意味着您可以在文本编辑器或 MS Excel 中轻松加载它们并查看它们，您也可以在工作站上快速建模它们。

使用[这个支持排序和搜索的方便表](http://archive.ics.uci.edu/ml/datasets.html)浏览 300 多个数据集。

### 对存储库的批评

对存储库的一些批评包括：

*   清理数据集，这意味着准备它们的研究人员通常已经根据属性和实例的选择进行了一些预处理。
*   数据集很小，如果您对调查更大规模的问题和技术感兴趣，这没有用。
*   有很多可供选择，你可以通过犹豫不决和过度分析来冻结。当您不确定它是否是您正在调查的“_ 良好数据集 _”时，可能很难选择数据集并开始使用。
*   数据集仅限于表格数据，主要用于分类（尽管列出了聚类和回归数据集）。这对于那些对自然语言，计算机视觉，推荐器和其他数据感兴趣的人来说是有限的。

看一下[存储库主页](http://archive.ics.uci.edu/ml/)，因为它显示了特色数据集，最新的数据集以及当前最受欢迎的数据集。

## 自学课程

那么，如何充分利用 UCI 机器学习库？

我建议你考虑一下你想要了解的问题数据集中的特征。

这些可能是您想要建模的特征（如回归），或者是您希望在使用时更熟练的这些特征的模型算法（如随机森林用于多分类）。

示例程序可能如下所示：

*   二分类：[皮马印第安人糖尿病数据集](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
*   多分类：[虹膜数据集](http://archive.ics.uci.edu/ml/datasets/Iris)
*   回归：[葡萄酒质量数据集](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
*   分类属性：[乳腺癌数据集](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer)
*   整数属性：[计算机硬件数据集](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware)
*   分类成本函数：[德国信贷数据](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
*   缺失数据：[马绞痛数据集](https://archive.ics.uci.edu/ml/datasets/Horse+Colic)

这只是一个特征列表，可以挑选和选择自己的特征进行调查。

我列出了每个特征的一个数据集，但是您可以选择 2-3 个不同的数据集并完成一些小项目以提高您的理解并进行更多练习。

对于每个问题，我建议您从端到端系统地进行操作，例如，在应用的机器学习过程中执行以下步骤：

1.  定义问题
2.  准备数据
3.  评估算法
4.  改善结果
5.  写作结果

[![Machine Learning for Programmers - Select a Systematic Process](img/9808919901691497af468a6cf9a89d8d.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/08/Machine-Learning-for-Programmers-Select-a-Systematic-Process-e1439699783406.png)

选择一个系统且可重复的流程，您可以使用该流程始终如一地提供结果。

有关系统学习机器学习问题的更多信息，请参阅我的帖子“[处理机器学习问题的过程](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)”。

写作是关键部分。

它允许您构建一系列项目，您可以将这些项目作为未来项目的参考并获得快速启动，以及用作公共简历或您在应用机器学习中不断增长的技能和能力。

有关构建项目组合的更多信息，请参阅我的文章“[构建机器学习组合：完成小型项目并展示您的技能](http://machinelearningmastery.com/build-a-machine-learning-portfolio/)”。

## 但是，如果......

**我不知道机器学习工具。**
选择一个工具或平台（如 Weka，R 或 scikit-learn）并使用此过程学习工具。完成机器学习和同时擅长工具的工作。

**我不知道如何编程（或代码非常好）。**
[使用 Weka](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/) 。它具有图形用户界面，无需编程。我会向初学者推荐这个，无论他们是否可以编程，因为工作机器学习问题的过程很好地映射到平台上。

**我没有时间。**
凭借强大的系统流程和涵盖整个流程的优秀工具，我认为您可以在一两个小时内解决问题。这意味着您可以在一个晚上或两个晚上完成一个项目。

您可以选择要调查的详细程度，最好在刚开始时保持简洁明了。

**我在我正在建模的领域没有背景知识。**
数据集页面提供了有关数据集的一些背景知识。通常，您可以通过查看主数据集附带的出版物或信息文件来深入了解。

**我几乎没有经验来解决机器学习问题。**
现在是时候开始了。选择一个[系统过程](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)，选择一个简单的数据集和像 [Weka](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/) 这样的工具，解决你的第一个问题。把第一块石头放在你的机器学习基础上。

**我没有数据分析经验。**
无需数据分析经验。数据集简单易懂，易于理解。您只需要使用数据集主页并通过查看数据文件本身来阅读它们。

## 行动步骤

选择一个数据集并开始使用。

如果您认真对待自学，请考虑设计一个适度的特征列表和相应的数据集进行调查。

您将学到很多东西，并为潜入更复杂和有趣的问题奠定宝贵的基础。

你觉得这篇文章有用吗？发表评论并告诉我。