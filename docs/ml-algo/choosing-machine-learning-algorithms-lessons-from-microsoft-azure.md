# 选择机器学习算法：Microsoft Azure的经验教训

> 原文： [https://machinelearningmastery.com/choosing-machine-learning-algorithms-lessons-from-microsoft-azure/](https://machinelearningmastery.com/choosing-machine-learning-algorithms-lessons-from-microsoft-azure/)

微软最近在其Azure云计算平台上推出了对机器学习的支持。

在平台的一些技术文档中埋藏了一些资源，您可能会发现这些资源可用于考虑在不同情况下使用的机器学习算法。

在这篇文章中，我们将了解微软对机器学习算法的建议以及我们在任何平台上解决机器学习问题时可以使用的经验教训。

![Choosing Machine Learning Algorithms](img/0f70e0a9d081f8f231777e0d9bd70dd3.jpg)

选择机器学习算法。
照片由 [USDA](https://www.flickr.com/photos/usdagov/14195226791) 拍摄，部分版权所有。

## 机器学习算法Cheatsheet

微软发布了一份PDF备忘单，列出了什么样的机器学习算法。

one-pager将各种问题类型列为组，以及Azure在每个组中支持的算法。

这些群体是：

*   **回归**：用于预测值。
*   **异常检测**：用于查找异常数据点。
*   **聚类**：用于发现结构。
*   **两级分类**：用于预测两类。
*   **多级分类**：用于预测三个或更多类别。

这种方法的第一个问题是算法名称似乎映射到Azure API文档，并不是标准的。一些常见名称跳出来但其他只是标准算法的名称，为简单起见而旋转（或者我怀疑避免某种名称侵权）。

除了算法名称之外，您还可以选择一个给定算法的原因。这是一个好主意，鉴于它是一个备忘单，它简洁而简洁。

从标题为“ [Microsoft Azure机器学习工作室](http://aka.ms/MLCheatSheet)的机器学习算法备忘单”的配套博客文章中下载备忘单（PDF）。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 如何选择机器学习算法

备忘单的目标是帮助您快速选择问题的算法。

是吗？也许不是。

原因是，您可能永远不应该分析为您的问题选择一种算法。您应该检查一些算法并使用您对问题的任何要求进行评估。

有关抽样检查算法的更多信息，请参阅文章“[为什么您应该在机器学习问题](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/)上进行抽样检查算法”。

我认为cheatsheet最适合用来了解哪些算法投入到您的抽查中，从您的问题要求的角度来看。

在同一个Azure文档中的姐妹博客中，我们获得了更多符合这些想法的上下文，标题为“[如何选择Microsoft Azure机器学习算法](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-choice/)”。

该帖子首先提出了一个问题：“_我应该使用哪些机器学习算法？_ “并正确回答”_取决于_“。他们评论说：

> 即使是最有经验的数据科学家也无法确定哪种算法在尝试之前表现最佳。

对吧！

这篇文章的宝贵意义在于它们在您的需求背景下考虑算法选择时所考虑的因素。这些算法选择考虑因素是：

*   **准确度**：获得最佳分数是目标还是近似（“足够好”）解决方案，同时权衡过拟合。
*   **训练时间**：训练模型的可用时间（我猜，验证和调整）。
*   **线性**：模型问题如何建模的一个方面。与非线性模型相比，非线性模型通常更难以理解和调整。
*   **参数数量**：模型复杂性的另一个方面影响调整和灵敏度的时间和专业知识。
*   **特征数量**：实际上，具有比实例更多属性的问题， _p＆gt;＆gt; n_ 问题。这通常需要专门的处理或专门技术。

该帖子还提供了Azure支持的算法的可爱表格，以及它们对上面列出的一些注意事项的映射。

![Table of Algorithm Considerations](img/0eb8a221186d12c93587d8eb384049a0.jpg)

算法注意事项表
[Microsoft博客文章](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-choice/)的部分截图。

我认为这很好。

我还认为创建（需要专家）非常昂贵，不能扩展到数百（数千？）的机器学习算法，并且随着新的和更强大的算法的开发和发布需要不断更新。

## 我们如何有效地选择算法？

通常，预测性建模的目标是在给定合理时间和资源的情况下创建最准确的模型。

如果模型仅用于描述目的而不是用于实际做出预测，那么在模型的线性度和参数数量方面对算法复杂性的关注通常仅是一个问题。

通过精心设计的问题测试工具，选择哪种算法和要设置的参数值成为计算机要弄清楚的组合问题，而不是数据科学家。事实上，就像在A / B测试中的直觉一样，算法选择是有偏见的，可能会使表现严重下降。

这是机器学习的现场检查方法，并且由于强大的系统测试方法（如交叉验证）以及廉价和丰富的计算，因此只能实现大量算法。

您最喜欢的机器学习算法在您之前没有参与的问题上表现良好的几率是多少？不是很好（[，除非你使用随机森林！](http://machinelearningmastery.com/use-random-forest-testing-179-classifiers-121-datasets/) - 我在开玩笑）。

我要说的是，我们可以研究机器学习算法，并了解它们的工作方式和适合的方法，但我认为这种选择水平会在以后发生。当您尝试在3到4个高表现型号之间进行选择时。当你取得好成绩时，你需要深入研究以获得更好的结果。

## 行动步骤

查看备忘单并做一些笔记，并考虑如何在自己的过程中使用这些想法。

看看我的[机器学习算法](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)之旅，深入了解最流行的机器学习算法以及它们之间的相互关系。

如果您想了解更多关于如何从端到端系统地处理机器学习问题的信息，请查看我的帖子“[处理机器学习问题的过程](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)”。