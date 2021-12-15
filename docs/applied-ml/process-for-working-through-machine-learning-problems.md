# 应用机器学习的过程

> 原文： [https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/](https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)

## 
_
_提供高于平均结果的_
_预测模型问题的系统过程__

随着时间的推移，在应用机器学习问题时，您需要开发一种模式或流程，以便快速获得良好的稳健结果。

开发完成后，您可以在项目之后一次又一次地使用此过程。您的流程越健壮，越发展，您获得可靠结果的速度就越快。

在这篇文章中，我想与您分享我处理机器学习问题的过程的骨架。

您可以将其用作下一个项目的起点或模板。

## 5步系统过程

我喜欢使用5个步骤：

1.  定义问题
2.  准备数据
3.  采样检查算法
4.  改善结果
5.  目前的结果

这个过程有很多灵活性。例如，“准备数据”步骤通常分解为分析数据（汇总和图表）并准备数据（为实验准备样品）。 “抽查”步骤可能涉及多个正式实验。

这是一条伟大的大型生产线，我尝试以线性方式进行。使用自动化工具的好处在于您可以返回几个步骤（例如从“改进结果”回到“准备数据”）并插入数据集的新变换并在中间步骤中重新运行实验以查看有趣的结果出来了，它们与你之前执行的实验相比如何。

[![Production Line](img/bbbe602095664e39a3e2ce489b0521d0.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/02/production-line.jpg)

生产线
摄影： [East Capital](http://www.flickr.com/photos/eastcapital/4554220770/sizes/o/) ，部分版权所有

我使用的过程是从数据库（或KDD）中的知识发现的标准数据挖掘过程改编而来的，有关详细信息，请参阅文章[什么是数据挖掘和KDD](http://machinelearningmastery.com/what-is-data-mining-and-kdd/ "What is Data Mining and KDD") 。

## 1.定义问题

我喜欢使用三步过程来定义问题。我喜欢快速行动，我使用这个迷你流程从几个不同的角度很快看到问题：

*   **第1步：有什么问题？** 非正式地和正式地描述问题并列出假设和类似问题。
*   **第2步：为什么问题需要解决？** 列出解决问题的动机，解决方案提供的好处以及解决方案的使用方法。
*   **第3步：我该如何解决这个问题？** 描述如何手动解决问题以刷新领域知识。

您可以在帖子中了解有关此过程的更多信息：

*   [如何定义机器学习问题](http://machinelearningmastery.com/how-to-define-your-machine-learning-problem/ "How to Define Your Machine Learning Problem")

## 2.准备数据

我将数据准备与数据分析阶段相结合，该阶段涉及总结属性并使用散点图和直方图对其进行可视化。我还想详细描述属性和属性之间的关系。这种笨拙的工作迫使我在问题丢失之前考虑问题上下文中的数据

实际的数据准备过程分为以下三个步骤：

*   **步骤1：数据选择**：考虑可用的数据，缺少的数据以及可以删除的数据。
*   **步骤2：数据预处理**：通过格式化，清理和采样来组织您选择的数据。
*   **步骤3：数据转换**：通过使用缩放，属性分解和属性聚合的工程特征，转换为机器学习做好准备的预处理数据。

您可以在帖子中了解有关准备数据的此过程的更多信息：

*   [如何为机器学习准备数据](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/ "How to Prepare Data For Machine Learning")

## 3.抽查算法

我默认在测试工具中使用10倍交叉验证。所有实验（算法和数据集组合）重复10次，并收集和报告准确度的均值和标准偏差。我还使用统计显着性检验从噪声中清除有意义的结果。箱形图对于总结每个算法和数据集对的准确度结果的分布非常有用。

我发现了检查算法，这意味着将一堆标准机器学习算法加载到我的测试工具中并执行正式实验。我通常在我准备好的数据集的所有转换和缩放版本中运行来自所有[主要算法系列](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/ "A Tour of Machine Learning Algorithms")的10-20个标准算法。

采样检查的目标是清除擅长挑选问题结构的算法类型和数据集组合，以便通过重点实验更详细地研究它们。

可以在该步骤中执行具有良好表现算法族的更集中的实验，但是算法调整留待下一步骤。

您可以在帖子中发现有关定义测试工具的更多信息：

*   [如何评估机器学习算法](http://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/ "How to Evaluate Machine Learning Algorithms")

您可以在帖子中发现抽查算法的重要性：

*   [为什么你应该在机器学习问题上进行采样检查算法](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")

## 4.改善结果

在现场检查之后，是时候从钻机中挤出最好的结果了。我这样做是通过对最佳表现算法的参数进行自动灵敏度分析。我还使用顶级执行算法的标准集合方法设计和运行实验。我花了很多时间思考如何从数据集或已经证明表现良好的算法族中获得更多。

同样，结果的统计显着性在这里至关重要。很容易关注方法并使用算法配置。结果只有在它们很重要且所有配置都已经过考虑并且实验是批量执行时才有意义。我也想在问题上保持自己的个人排行榜。

总之，改进结果的过程包括：

*   **算法调整**：通过模型参数空间将发现最佳模型视为搜索问题。
*   **集合方法**：将多个模型的预测结合起来。
*   **极限特征工程**：数据准备中看到的属性分解和聚合被推到极限。

您可以在帖子中发现有关此过程的更多信息：

*   [如何提高机器学习效果](http://machinelearningmastery.com/how-to-improve-machine-learning-results/ "How to Improve Machine Learning Results")

## 5.目前的结果

复杂的机器学习问题的结果除非付诸实现，否则毫无意义。这通常意味着向利益相关者展示。即使这是我为自己工作的竞争或问题，我仍然会经历呈现结果的过程。这是一个很好的练习，给了我明确的学习，我可以在下次建立。

我用来呈现结果的模板如下，可以采用文本文档，正式报告或演示幻灯片的形式。

*   **上下文（为什么）**：定义问题所在的环境并设置研究问题的动机。
*   **问题（问题）**：简单地将问题描述为你出去回答的问题。
*   **解决方案（答案）**：简要描述解决方案，作为您在上一节中提出的问题的答案。请明确点。
*   **调查结果**：您在观众感兴趣的路上发现的项目符号列表。它们可能是数据中的发现，已完成或未起作用的方法，或者您在旅程中获得的模型表现优势。
*   **限制**：考虑模型不起作用的地方或模型未回答的问题。不要回避这些问题，如果你可以定义它不擅长的地方，那么定义模型擅长的地方会更加可信。
*   **结论（为什么+问题+答案）**：重新审视“为什么”，研究问题以及您在一个易于记忆并为自己和他人重复的紧凑小包装中发现的答案。

您可以在帖子中发现有关使用机器学习项目结果的更多信息：

*   [如何使用机器学习结果](http://machinelearningmastery.com/how-to-use-machine-learning-results/ "How to Use Machine Learning Results")

## 摘要

在这篇文章中，您已经学习了我处理机器学习问题的通用模板。

我使用这个过程几乎没有失败，我使用它跨越平台，从 [Weka](http://machinelearningmastery.com/what-is-the-weka-machine-learning-workbench/ "What is the Weka Machine Learning Workbench") ，R和scikit-learn甚至新平台我一直在玩像pylearn2。

您的流程是什么，发表评论并分享？

您是否会复制此流程，如果是，您将对其进行哪些更改？