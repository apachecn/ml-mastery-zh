# 如何充分利用机器学习数据

> 原文： [https://machinelearningmastery.com/how-to-get-the-most-from-your-machine-learning-data/](https://machinelearningmastery.com/how-to-get-the-most-from-your-machine-learning-data/)

您使用的数据以及使用方式可能会定义预测性建模问题的成功。

数据和问题框架可能是您项目最大的杠杆点。

为您的问题选择错误的数据或错误的框架可能会导致模型表现不佳，或者最糟糕的是，模型无法收敛。

无法分析地计算要使用的数据或如何使用它，但可以使用反复试验过程来发现如何最好地使用您拥有的数据。

在这篇文章中，您将发现从机器学习项目中的数据中获得最大收益。

阅读这篇文章后，你会知道：

*   探索预测性建模问题的替代框架的重要性。
*   需要在输入数据上开发一套“_视图_”并对每个视图进行系统测试。
*   功能选择，工程和准备的概念是为您的问题创建更多视图的方法。

让我们开始吧。

![How to Get the Most From Your Machine Learning Data](img/32b72c8e8215ba0a790b600e84a371ae.jpg)

如何充分利用机器学习数据
[Jean-Marc Bolfing](https://www.flickr.com/photos/bolfingyamauchi/34340279286/) 的照片，保留一些权利。

## 概观

这篇文章分为8个部分;他们是：

1.  问题框架
2.  收集更多数据
3.  研究你的数据
4.  训练数据样本量
5.  特征选择
6.  特色工程
7.  数据准备
8.  走得更远

## 1.问题框架

集思广益，以多种方式构建预测性建模问题。

问题的框架意味着以下组合：

*   输入
*   输出
*   问题类型

例如：

*   您可以使用更多或更少的数据作为模型的输入吗？
*   你能预测别的东西吗？
*   你能把问题改成回归/分类/序列等吗？

你获得的创意越多越好。

使用来自其他项目，论文和域本身的想法。

头脑风暴。写下所有的想法，即使它们是疯了。

我有一些框架可以帮助在这里集思广益：

*   [如何定义机器学习问题](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)

我在这篇文章中谈到改变问题类型：

*   [机器学习中分类和回归之间的差异](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

## 2.收集更多数据

获得比您需要的更多数据，甚至是与预测结果相切的数据。

我们无法知道[需要多少数据](https://machinelearningmastery.com/much-training-data-required-machine-learning/)。

数据是模型开发过程中花费的货币。这是项目呼吸所需的氧气。每次使用某些数据时，其他任务的可用数据就越少。

您需要在以下任务上花费数据：

*   模范训练。
*   模型评估。
*   模型调整。
*   模型验证。

此外，该项目是新的。之前没有人完成您的特定项目，建模您的特定数据。你真的不知道哪些功能还有用。你可能有想法，但你不知道。全部收集;在这个阶段让它们全部可用。

## 3.研究你的数据

使用您可以想到的每个数据可视化从各个角度查看您的数据。

*   查看原始数据有帮助。你会发现事情。
*   查看摘要统计量有帮助。再一次，你会发现事情。
*   数据可视化就像是这两种学习方式的完美结合。你会发现更多的东西。

花费很长时间来处理原始数据和摘要统计量。然后继续进行可视化，因为它们可能需要更多时间来准备。

使用您能想到的每个数据可视化，并从您的数据的书籍和论文中收集。

*   查看图表。
*   保存情节。
*   注释图。
*   向领域专家显示图表。

您正在寻求更深入地了解数据。您可以使用的想法，以帮助更好地选择，设计和准备建模数据。它会得到回报。

## 4.训练数据样本量

使用数据样本执行灵敏度分析，以查看实际需要的数据量（或很少）。

你没有所有的观察结果。如果您这样做，则无需对新数据做出预测。

相反，您正在处理数据样本。因此，对于需要多少数据来拟合模型存在一个悬而未决的问题。

不要以为越多越好。测试。

*   设计实验，了解模型技能如何随样本量而变化。
*   使用统计量查看趋势和趋势随样本量变化的重要程度。

如果没有这些知识，您将无法充分了解您的测试工具，以便明智地评论模型技能。

在此帖子中了解有关样本量的更多信息：

*   [机器学习需要多少训练数据？](https://machinelearningmastery.com/much-training-data-required-machine-learning/)

## 5.特征选择

创建输入功能的许多不同视图并测试每个视图。

您不知道哪些变量对预测性建模问题有帮助或最有帮助。

*   你可以猜猜看。
*   您可以使用领域专家的建议。
*   您甚至可以使用功能选择方法中的建议。

但他们都只是猜测。

每组建议的输入功能都是您的问题的“视图”。了解哪些特性可能对建模和预测输出变量有用。

尽可能多地集思广益，计算和收集输入数据的不同视图。

设计实验并仔细测试并比较每个视图。使用数据通知您哪些功能和哪个视图最具预测性。

有关功能选择的更多信息，请参阅此帖子：

*   [特征选择介绍](https://machinelearningmastery.com/an-introduction-to-feature-selection/)

## 6.特征工程

使用要素工程为预测性建模问题创建其他功能和视图。

有时您拥有可以获得的所有数据，但是给定的功能或功能集会锁定对于机器学习方法学习和映射到结果变量而言过于密集的知识。

例子包括：

*   日期/时间。
*   交易。
*   说明。

将这些数据细分为更简单的附加组件功能，例如计数，标志和其他元素。

尽可能简化建模过程。

有关特征工程的更多信息，请参阅帖子：

*   [发现特征工程，如何设计特征以及如何获得它](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

## 7.数据准备

您可以通过各种方式预处理数据，以满足算法的期望等。

预处理数据（如特征选择和特征工程）可在输入要素上创建其他视图。

一些算法具有关于预处理的偏好，例如：

*   标准化输入功能。
*   标准化输入功能。
*   使输入功能固定。

准备好预期这些预期的数据，然后再进一步。

应用您可以想到的每种数据预处理方法。继续为您的问题创建新视图，并使用一个或一组模型测试它们，看看什么效果最好。

您的目标是发现有关数据的视图，该数据最好地将映射问题的未知底层结构暴露给学习算法。

## 8.走得更远

你可以随时走得更远。

您可以收集更多数据，可以在数据上创建更多视图。

头脑风暴。

一旦您感觉自己走在路的尽头，一个简单的胜利就是开始研究从建模问题的不同视角创建的模型的集合。

它简单而高效，特别是如果视图暴露了底层映射问题的不同结构（例如模型具有不相关的错误）。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [为什么应用机器学习很难](https://machinelearningmastery.com/applied-machine-learning-is-hard/)
*   [应用机器学习作为搜索问题的温和介绍](https://machinelearningmastery.com/applied-machine-learning-as-a-search-problem/)
*   [如何定义机器学习问题](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)
*   [机器学习表现改进备忘单](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)
*   [机器学习需要多少训练数据？](https://machinelearningmastery.com/much-training-data-required-machine-learning/)
*   [特征选择介绍](https://machinelearningmastery.com/an-introduction-to-feature-selection/)
*   [发现特征工程，如何设计特征以及如何获得它](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

## 摘要

在这篇文章中，您发现了可用于充分利用预测性建模问题数据的技术。

具体来说，你学到了：

*   探索预测性建模问题的替代框架的重要性。
*   需要在输入数据上开发一套“视图”并系统地测试每个视图。
*   功能选择，工程和准备的概念是为您的问题创建更多视图的方法。

您是否有更多想法可以充分利用您的数据？
你通常在一个项目上做什么？
请在下面的评论中告诉我。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。