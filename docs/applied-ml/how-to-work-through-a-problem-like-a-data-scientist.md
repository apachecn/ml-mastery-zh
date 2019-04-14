# 如何解决像数据科学家这样的问题

> 原文： [https://machinelearningmastery.com/how-to-work-through-a-problem-like-a-data-scientist/](https://machinelearningmastery.com/how-to-work-through-a-problem-like-a-data-scientist/)

在2010年的一篇文章中，Hilary Mason和Chris Wiggins将OSEMN流程描述为数据科学家应该感到舒服的任务分类。

该帖子的标题是“[数据科学分类](http://www.dataists.com/2010/09/a-taxonomy-of-data-science/)”，现已解散的数据库博客。这个过程也被用作最近一本书的结构，特别是“命令行的[数据科学：面向未来的经过时间测试的工具](http://www.amazon.com/dp/1491947853?tag=inspiredalgor-20)”，作者是由O'Reilly出版的Jeroen Janssens。

在这篇文章中，我们仔细研究了解决数据问题的OSEMN流程。

[![Work Through A Problem Like A Data Scientist](img/ee1f6a04ed3c8462c1a89b2df5c85be1.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/12/Work-Through-A-Problem-Like-A-Data-Scientist.jpg)

像数据科学家一样解决问题
照片来自[美国陆军RDECOM](http://www.flickr.com/photos/rdecom/7336886600) ，保留一些权利

## OSEMN流程

OSEMN是与“负鼠”或“令人敬畏”押韵的缩写，代表获取，磨砂，探索，模型和iNterpret。

这是数据科学家应该熟悉和熟悉的任务列表。尽管如此，作者指出，没有数据科学家会成为所有这些人的专家。

除了任务列表之外，OSEMN还可以用作使用机器学习工具处理数据问题的蓝图。

从这个过程中，作者指出数据黑客符合“ _O_ ”和“ _S_ ”任务，机器学习符合“ _E_ ”和“ _M_ ”任务，而数据科学需要所有元素的组合。

## 1.获取数据

作者指出，数据收集的手动过程不会扩展，您必须学习如何自动获取给定问题所需的数据。

他们指向手动过程，如使用鼠标指向和单击，并从文档中复制和粘贴数据。

作者建议您采用一系列工具并使用最适合手头工作的工具。他们指向unix命令行工具，数据库中的SQL，使用Python和shell脚本进行Web抓取和脚本编写。

最后，作者指出了使用API​​访问数据的重要性，其中API可能是公共的，也可能是组织内部的。数据通常以JSON格式呈现，而像Python这样的脚本语言可以使数据检索变得更加容易。

## 2.磨砂数据

您获得的数据将是混乱的。

真实数据可能存在不一致，缺失值和各种其他形式的损坏。如果从困难的数据源中删除它，可能需要跳闸和清理。即使是干净的数据也可能需要进行后期处理才能使其统一和一致。

数据清理或清理需要“命令行fu”和简单的脚本。

作者指出，数据清理是处理数据问题最不性感的部分，但良好的数据清理可以为您实现的结果提供最大的好处。

> 对干净数据的简单分析比对噪声和不规则数据的复杂分析更有成效。

作者指出了简单的命令行工具，如sed，awk，grep和脚本语言，如Python和Perl。

有关更多信息，请查看[数据准备过程](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/ "How to Prepare Data For Machine Learning")。

## 3.探索数据

在这种情况下探索是指探索性数据分析。

这是没有正在测试的假设，也没有正在评估的预测。

数据探索对于了解您的数据，构建对其形式的直觉以及获取数据转换的想法以及甚至在此过程中使用的预测模型非常有用。

作者列出了许多可能有助于此任务的方法：

*   命令行工具，用于检查更多，更少，头部，尾部或其他任何数据。
*   直方图总结了各个数据属性的分布。
*   成对直方图可以相互绘制属性并突出显示关系和异常值
*   维度减少方法，用于创建较低维度的图和数据模型
*   聚类以暴露数据中的自然分组

有关更多信息，请查看[探索性数据分析](http://machinelearningmastery.com/understand-problem-get-better-results-using-exploratory-data-analysis/ "Understand Your Problem and Get Better Results Using Exploratory Data Analysis")。

## 4.模型数据

模型精度通常是给定数据问题的最终目标。这意味着最具预测性的模型是选择模型的过滤器。

> 通常，“最佳”模型是最具预测性的模型

通常，目标是使用模型预测和解释。可以定量地评估预测，而解释更柔和和定性。

模型的预测准确性可以通过它在看不见的数据上的表现来评估。可以使用诸如交叉验证之类的方法来估计。

您尝试的算法以及您对可以为问题构建的可能模型的假设空间的偏差和减少。做出明智的选择。

有关更多信息，请查看[如何评估模型](http://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/ "How to Evaluate Machine Learning Algorithms")和[如何进行抽样检查算法](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")。

## 5.解释结果

> 计算的目的是洞察力，而不是数字

- 理查德汉明

作者使用手写数字识别的例子。他们指出，这个问题的模型没有每个数字的理论，而是一种区分数字的机制。

此示例强调预测的关注点可能与模型解释不同。事实上，他们可能会发生冲突。复杂模型可以是高度预测的，但是执行的术语或数据变换的数量可以使得理解为什么在域的上下文中进行特定预测几乎是不可能的。

模型的预测能力取决于其推广的能力。作者认为，模型的解释力是它能够建议接下来要进行的最有趣的实验。它提供了对问题和领域的见解。

作者在选择模型以平衡模型的预测性和可解释性时指出了三个关键问题：

*   选择一个好的表示形式，您获得的数据形式，大多数数据都是混乱的。
*   选择好的功能，您选择建模的数据的属性
*   选择一个良好的假设空间，受您选择的模型和数据转换的约束。

有关更多信息，请查看[如何使用机器学习结果](http://machinelearningmastery.com/how-to-use-machine-learning-results/ "How to Use Machine Learning Results")。

## 摘要

在这篇文章中，你发现了Hilary Mason和Chris Wiggins提出的OSEMN。

OSEMN代表Obtain，Scrub，Explore，Model和iNterpret。

与[数据库中的知识发现](http://machinelearningmastery.com/what-is-data-mining-and-kdd/ "What is Data Mining and KDD")和[应用的机器学习过程](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/ "Process for working through Machine Learning Problems")类似，您可以使用此过程来解决机器学习问题。