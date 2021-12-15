# 机器学习 Q＆amp; A：概念漂移，更好的结果和学习更快

> 原文： [https://machinelearningmastery.com/machine-learning-qa-concept-drift-better-results-and-learning-faster/](https://machinelearningmastery.com/machine-learning-qa-concept-drift-better-results-and-learning-faster/)

我通过电子邮件得到了很多关于机器学习的问题，我喜欢回答它们。

我会看到真正的人在做什么，并帮助改变现状。 （你对机器学习有疑问吗？[联系我](http://machinelearningmastery.com/contact/ "Contact")）。

在这篇文章中，我重点介绍了我最近收到的一些有趣的问题并总结了我的答案。

[![machine learning q&a](img/462c374e497fef634b835d40bc0ecd2a.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/11/machine-learning-qa.jpg)

机器学习 Q＆amp; A
摄影： [Angelo Amboldi](http://www.flickr.com/photos/angelocesare/8141688142) ，保留一些权利

## 为什么我的垃圾邮件分类器会在所有旧电子邮件上进行训练时变得更糟？

这是一个很好的问题，因为它突出了机器学习中一个称为概念漂移的重要概念。

电子邮件的内容随时间而变化。用户将更改他们与谁交谈以及在哪些主题上。电子邮件垃圾邮件发送者将发送不同的优惠，并将积极改变他们在电子邮件中的策略，以避免电子邮件垃

这些更改会影响建模。

关于哪些电子邮件是垃圾邮件和哪些不是垃圾邮件的最佳信息来源是最近收到的电子邮件。回到过去，电子邮件对建模问题的用处越少。

在模型中捕获了什么是垃圾邮件和非垃圾邮件的概念，它基于您用于训练该模型的数据。如果垃圾邮件的概念或概念发生变化，那么您需要收集更多示例并更新模型。

这是问题的重要属性，可以影响您对问题建模所做的决策。例如，您可能希望选择一个可以轻松更新的模型，而不是从头开始重建。

## 如何在机器学习问题上获得更好的结果？

就像一件软件或一件艺术品，它永远不会完成。有一天你会停止工作。

你可以尝试很多东西，一些广泛的领域包括：

*   **处理数据**：查看特征工程，试图将更多有用的问题结构暴露给建模算法。看看您是否可以收集可以解决问题的其他数据。调查[数据准备](http://machinelearningmastery.com/improve-model-accuracy-with-data-pre-processing/ "Improve Model Accuracy with Data Pre-Processing")，例如缩放和其他数据转换，可以更好地揭示问题中的结构。
*   **使用其他算法**：是否有其他[算法可以检查](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")？总是有更多的算法，并且通常有非常强大的算法可供您查找和尝试。
*   **使用算法**：你从你尝试过的算法中获得了最大的收益吗？ [使用网格或随机搜索调整算法参数](http://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/ "How to Tune Algorithm Parameters with Scikit-Learn")。
*   **结合预测**：尝试结合多个表现良好但不同算法的预测。使用[整体方法](http://machinelearningmastery.com/improve-machine-learning-results-with-boosting-bagging-and-blending-ensemble-methods-in-weka/ "Improve Machine Learning Results with Boosting, Bagging and Blending Ensemble Methods in Weka")，如装袋，加强和混合。

您推动准确度越高，您将模型过拟合到训练数据的可能性就越高，并且限制了对看不见的数据的适用性。

重新访问[问题定义](http://machinelearningmastery.com/how-to-define-your-machine-learning-problem/ "How to Define Your Machine Learning Problem")并设置最低精度阈值。通常，“_ 足够好 _”模型比精细调整（和脆弱）模型更实用。

请参阅这篇题为“[模型预测准确率与机器学习中的解释](http://machinelearningmastery.com/model-prediction-versus-interpretation-in-machine-learning/ "Model Prediction Accuracy Versus Interpretation in Machine Learning")”的文章。

## 如何更快地学习机器学习？

实践。很多。

1.  阅读书籍，学习课程，学习并利用其他人的想法。
2.  擅长端到端的[工作问题](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/ "Process for working through Machine Learning Problems")。
3.  [学习机器学习算法](http://machinelearningmastery.com/how-to-study-machine-learning-algorithms/ "How to Study Machine Learning Algorithms")。
4.  工作问题，重现论文和竞赛的结果。
5.  设计并执行小型[自学项目](http://machinelearningmastery.com/self-study-machine-learning-projects/ "4 Self-Study Machine Learning Projects")并建立一系列结果。

学习新事物还不够好。

要更快地学习，您需要更加努力。你需要将你正在学习的东西付诸行动。你需要工作和返工问题。

## 有什么问题可以解决？

从 [UCI 机器学习库](http://archive.ics.uci.edu/ml/)中的数据集开始。它们很小，它们适合内存，学术界使用它们来演示算法属性行为，因此它们有点被很好地理解。

最受欢迎的数据集列表将是一个很好的起点。

转向竞争数据集。获得足够好的结果，然后尝试在竞赛获胜者上重现结果（粗略地说，通常没有足够的信息）。

来自最新 [KDDCup](http://www.sigkdd.org/kddcup/index.php) 和 [Kaggle](https://www.kaggle.com/) 比赛的数据集将是一个很好的起点。

最后，提出自己的问题（或接受他人）并定义自己的问题，收集数据，并通常端到端地解决问题。

更多信息：

*   [处理对您来说很重要的问题](http://machinelearningmastery.com/work-on-machine-learning-problems-that-matter-to-you/ "Work on Machine Learning Problems That Matter To You")
*   [处理与钱有关的问题](http://machinelearningmastery.com/machine-learning-for-money/ "Machine Learning for Money")

## 如何超越驾驶机器学习工具？

我建议初学者学习如何驱动机器学习工具和库，并擅长端到端的[工作机器学习问题](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/ "Process for working through Machine Learning Problems")。

我这样做是因为这是应用机器学习的基础，在这个过程中需要学习很多东西，从数据准备到算法，再到沟通结果。

更深层次的涉及专业化。例如，您可以深入了解机器学习算法。你可以[研究它们](http://machinelearningmastery.com/how-to-study-machine-learning-algorithms/ "How to Study Machine Learning Algorithms")，[制作列表](http://machinelearningmastery.com/create-lists-of-machine-learning-algorithms/ "Take Control By Creating Targeted Lists of Machine Learning Algorithms")，[描述它们](http://machinelearningmastery.com/how-to-learn-a-machine-learning-algorithm/ "How to Learn a Machine Learning Algorithm")和[从头开始实现它们](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ "Tutorial To Implement k-Nearest Neighbors in Python From Scratch")。事实上，您可以潜水的深度没有限制，但您确实想要选择一个您觉得引人注目的区域。

我建议通过自学更深入的一般框架是我的[小项目方法](http://machinelearningmastery.com/self-study-machine-learning-projects/ "4 Self-Study Machine Learning Projects")。这是您定义一个小项目（5 到 10 个小时的努力），执行它们并共享结果，然后重复的地方。

我建议使用四类项目：研究工具，研究算法，研究问题并实现算法。如果您渴望超越驾驭机器学习工具或库，后三个项目可能会有吸引力。

## 问一个问题

如果您有机器学习问题，[请与我联系](http://machinelearningmastery.com/contact/ "Contact")。

如果您对我的机器学习方法感兴趣，请查看我的 [start-here 页面](http://machinelearningmastery.com/start-here/ "Start Here")。它链接到许多有用的博客文章和资源。