# 为什么你应该在你的机器学习问题上采样检查算法

> 原文： [https://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/](https://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/)

点检算法是关于在机器学习问题上快速评估一堆不同的算法，以便您了解要关注的算法和丢弃的内容。

[![spot check machine learning algorithm](img/c3f5ebc44bf8c02b929d6b9f6524f5bb.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/02/spot-check-machine-learning-algorithm.jpg)

照片来自 [withassociates](http://www.flickr.com/photos/withassociates/4385364607/sizes/l/) ，保留一些权利

在这篇文章中，您将发现斑点检查算法的3个好处，5个用于对您的下一个问题进行抽查的技巧，以及您可以在算法套件中进行抽查的十大最流行的数据挖掘算法。

## 采样检查算法

[采样检查算法](http://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/)是应用机器学习过程的一部分。在一个新问题上，您需要快速确定哪种类型或类别的算法擅长选择问题中的结构，哪些不是。

现场检查的替代方案是，您可能会尝试使用大量的算法和算法类型，使您最终尝试的很少或者过去使用过的功能。这会导致浪费时间和低于标准的结果。

## 点检算法的好处

在您的机器学习问题上，采样算法有三个主要优点：

*   **速度**：您可以花费大量时间来使用不同的算法，调整参数并考虑哪些算法可以很好地解决您的问题。我一直在那里，最后一遍又一遍地测试相同的算法因为我没有系统化。一次采样实验可以节省数小时，数天甚至数周的涂鸦时间。
*   **目标**：有一种趋势可以追溯到以前对你有用的东西。我们选择我们最喜欢的算法（或算法）并将它们应用于我们看到的每个问题。机器学习的力量在于有很多不同的方法可以解决特定问题。通过采样实验，您可以自动客观地发现那些最佳选择问题结构的算法，以便您可以集中注意力。
*   **结果**：现场检查算法可以快速获得可用的结果。您可以在第一个现场实验中发现一个足够好的解决方案。或者，您可以快速了解到您的数据集没有为任何主流算法提供足够的结构以使其表现良好。采样检查为您提供了决定是否继续前进并优化给定模型或向后重新访问问题表示所需的结果。

我认为现场检查主流算法对你的问题是一个简单的第一步。

## 点检算法提示

当您进行采样检查算法时，您可以做一些事情，以确保您获得有用且可操作的结果。

[![Tips for Spot-Checking Algorithms](img/523a73ee541937510210f41453302406.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/02/Tips-for-Spot-Checking-Algorithms.jpg)

点检算法提示
照 [vintagedept](http://www.flickr.com/photos/vintagedept/6358537847/sizes/l/) ，保留一些权利。

以下是5个提示，以确保您从问题的现场检查机器学习算法中获得最大收益。

*   **算法多样性**：您需要良好的算法类型组合。我喜欢包括基于实例的方法（实时LVQ和knn），函数和内核（如神经网络，回归和SVM），规则系统（如决策表和RIPPER）和决策树（如CART，ID3和C4.5）。
*   **最佳足部前锋**：每个算法都需要有机会将其发挥得最好。这并不意味着对每个算法的参数进行灵敏度分析，而是使用实验和启发式方法为每个算法提供公平的机会。例如，如果kNN在混合中，则给它3次机会，k值为1,5和7。
*   **正式实验**：不要玩。以非正式的方式尝试许多不同的东西，以解决问题的算法是一种巨大的诱惑。现场检查的想法是快速找到能够很好地解决问题的方法。设计实验，运行它，然后分析结果。有条不紊。我喜欢通过统计显着的胜利（在成对比较中）对算法进行排名，并将前3-5作为调整的基础。
*   **跳跃点**：表现最佳的算法是解决问题的起点。显示有效的算法可能不是该作业的最佳算法。它们最有可能是指向在问题上表现良好的算法类型的有用指针。例如，如果kNN表现良好，请考虑对您可以想到的所有基于实例的方法和kNN变体的后续实验。
*   **构建您的短名单**：当您学习并尝试许多不同的算法时，您可以将新算法添加到您在采样实验中使用的算法套件中。当我发现一个特别强大的算法配置时，我喜欢将它概括并将其包含在我的套件中，使我的套件对下一个问题更加健壮。

开始构建用于抽查实验的算法套件。

## 十大算法

2008年发表了一篇题为“[数据挖掘前十大算法](http://scholar.google.com/scholar?q=Top+10+algorithms+in+data+mining)”的论文。谁可以通过这样的头衔？它也变成了一本书“[数据挖掘中的十大算法](http://www.amazon.com/dp/1420089641?tag=inspiredalgor-20)”，并启发了另一个“机器学习在行动”的结构。

[![Amazon Image](img/6ecaee515e4ac4c1906474d65ec2907e.jpg)](http://www.amazon.com/dp/1420089641?tag=inspiredalgor-20)

这可能是一篇很好的论文，可以帮助您快速启动算法的简短列表，以便对您的下一次机器学习问题进行抽查。本文列出的前10个数据挖掘算法是。

*   C4.5这是一种决策树算法，包括着名的C5.0和ID3算法等后代方法。
*   K均值。转向聚类算法。
*   支持向量机。这真是一个巨大的研究领域。
*   先验。这是规则提取的首选算法。
*   EM。随着k-means，go-to聚类算法。
*   网页排名。我很少接触基于图形的问题。
*   AdaBoost的。这实际上是推动整体方法的一族。
*   knn（k-最近邻居）。简单有效的基于实例的方法。
*   朴素贝叶斯。在数据上简单而稳健地使用贝叶斯定理。
*   CART（分类和回归树）另一种基于树的方法。

关于这个主题还有一个[伟大的Quora问题，你可以挖掘算法的想法来试试你的问题。](http://www.quora.com/Machine-Learning/What-are-some-Machine-Learning-algorithms-that-you-should-always-have-a-strong-understanding-of-and-why)

## 资源

*   [数据挖掘中的十大算法](http://scholar.google.com/scholar?q=Top+10+algorithms+in+data+mining)（2008）
*   Quora：[你应该对这些机器学习算法有什么了解，为什么？](http://www.quora.com/Machine-Learning/What-are-some-Machine-Learning-algorithms-that-you-should-always-have-a-strong-understanding-of-and-why)

你喜欢哪种算法来检查问题？你有最喜欢的吗？