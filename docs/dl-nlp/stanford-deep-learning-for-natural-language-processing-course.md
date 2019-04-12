# 斯坦福自然语言处理深度学习课程述评

> 原文： [https://machinelearningmastery.com/stanford-deep-learning-for-natural-language-processing-course/](https://machinelearningmastery.com/stanford-deep-learning-for-natural-language-processing-course/)

自然语言处理（NLP）是机器学习的一个子领域，涉及理解语音和文本数据。

统计方法和统计机器学习在该领域占主导地位，并且最近深度学习方法已被证明在挑战语音识别和文本翻译等 NLP 问题方面非常有效。

在这篇文章中，您将发现有关深度学习方法的自然语言处理主题的斯坦福课程。

本课程是免费的，我鼓励您使用这个优秀的资源。

完成这篇文章后，你会知道：

*   本课程的目标和先决条件。
*   课程讲座细分以及如何访问幻灯片，笔记和视频。
*   如何充分利用这种材料。

让我们开始吧。

## 概观

这篇文章分为 5 部分;他们是：

1.  课程摘要
2.  先决条件
3.  讲座
4.  项目
5.  如何最好地使用这种材料

## 课程摘要

该课程由 Chris Manning 和 Richard Socher 教授。

[Chris Manning](https://nlp.stanford.edu/manning/) 是至少两本关于自然语言处理的顶级教科书的作者：

*   [统计自然语言处理基础](http://amzn.to/2gVBX7j)
*   [信息检索简介](http://amzn.to/2gVU9gZ)

[Richard Socher](http://www.socher.org/) 是 [MetaMind](http://www.metamind.io/) 背后的人，也是 Salesforce 的首席科学家。

自然语言处理是研究处理语音和文本数据的计算方法。

> 目标：计算机处理或“理解”自然语言以执行有用的任务

自 20 世纪 90 年代以来，该领域一直专注于统计方法。最近，该领域正在转向深度学习方法，因为它们提供了明显改进的功能。

本课程的重点是用深度学习方法教授统计自然语言处理。从网站上的课程描述：

> 最近，深度学习方法在许多不同的 NLP 任务中获得了非常高的性能。这些模型通常可以使用单个端到端模型进行培训，而不需要传统的，针对任务的特征工程。

![Reasons for Exploring Deep Learning, from the Stanford Deep Learning for NLP course](img/d1d51794b031ea6872e8e23ed3ab7056.jpg)

从斯坦福深度学习 NLP 课程探索深度学习的原因

课程目标

*   了解和使用有效的现代方法进行深度学习的能力
*   对人类语言的一些全局了解以及理解和产生它们的困难
*   了解和建立 NLP 中一些主要问题的系统的能力

![Goals of the Stanford Deep Learning for NLP Course](img/6dea3fa93089e7a1b9bcee6f089d52f8.jpg)

斯坦福深度学习 NLP 课程的目标

本课程在斯坦福大学讲授，虽然课程中使用的讲座已被记录并公布，我们将专注于这些免费提供的材料。

## 先决条件

该课程假设一些数学和编程技巧。

然而，如果必要的技能生锈，则提供进修材料。

特别：

*   大学微积分
    *   [线性代数综述](http://cs229.stanford.edu/section/cs229-linalg.pdf)
*   统计与概率
    *   [概率审查](http://cs229.stanford.edu/section/cs229-prob.pdf)
*   机器学习
    *   [凸优化评论](http://cs229.stanford.edu/section/cs229-cvxopt.pdf)
    *   [随机梯度下降评论](http://cs231n.github.io/optimization-1/)
*   Python 编程
    *   [Python 评论](http://cs231n.github.io/python-numpy-tutorial/)

代码示例在 Python 中，并使用 [NumPy](http://www.numpy.org/) 和 [TensorFlow](https://www.tensorflow.org/) Python 库。

## 讲座

每次讲授课程时，讲座和材料似乎都会有所改变。考虑到事情正在改变领域的速度，这并不奇怪。

在这里，我们将看一下 [CS224n 2017 年冬季课程大纲](http://web.stanford.edu/class/cs224n/syllabus.html)以及公开发表的讲座。

我建议观看讲座的 [YouTube 视频](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)，并仅在需要时访问幻灯片，论文和进一步阅读课程提纲。

该课程分为以下 18 个讲座和一个评论：

*   第 1 讲：深度学习的自然语言处理
*   第 2 讲：单词矢量表示：word2vec
*   第 3 讲：GloVe：Word 表示的全局向量
*   第 4 讲：词窗分类和神经网络
*   第 5 讲：反向传播和项目建议
*   第 6 讲：依赖性解析
*   第 7 讲：TensorFlow 简介
*   第 8 讲：回归神经网络和语言模型
*   第 9 讲：机器翻译和高级复现 LSTM 和 GRU
*   审查会议：中期审查
*   第 10 讲：神经机器翻译和注意模型
*   第 11 讲：门控复发单元和 NMT 的其他主题
*   第 12 讲：语音处理的端到端模型
*   第 13 讲：卷积神经网络
*   第 14 讲：树递归神经网络和选区解析
*   第 15 讲：共同决议
*   第 16 讲：用于问答的动态神经网络
*   第 17 讲：NLP 中的问题和 NLP 的可能架构
*   第 18 讲：解决 NLP 深度学习的局限性

我在 YouTube 上以双倍播放速度观看了所有内容，并在记笔记时打开了幻灯片。

## 项目

预计该课程的学生将完成作业。

您可能希望自己完成评估，以便通过讲座来测试您的知识。

你可以在这里看到作业： [CS224n 作业](http://web.stanford.edu/class/cs224n/assignments)

重要的是，学生必须使用深度学习自然语言处理问题提交最终项目报告。

如果您正在寻找如何测试新发现技能的想法，这些项目可以很有趣。

提交的学生报告目录可在此处获得：

*   [2015 学生项目报告](http://cs224d.stanford.edu/reports_2015.html)
*   [2016 学生项目报告](http://cs224d.stanford.edu/reports_2016.html)

如果您发现了一些很棒的报告，请在评论中发布您的发现。

## 如何最好地使用这种材料

本课程专为学生设计，目的是教授足够的 NLP 和深度学习理论，让学生开始开发自己的方法。

这可能不是你的目标。

您可能是开发人员。您可能只对使用 NLP 问题的深度学习工具感兴趣，以获得当前项目的结果。

事实上，这是我的大多数读者的情况。如果这听起来像你，我会提醒你在处理材料时要非常小心。

*   **跳过数学**。不要关注方法的工作原理。相反，请关注方法如何工作的摘要，并跳过方程的大部分。您可以随时回来加深理解，以获得更好的结果。
*   **专注于流程**。从讲座中学习，并将您可以在自己的项目中使用的过程组合在一起。这些方法是分段教授的，关于如何将它们实际联系在一起的信息很少。
*   **工具不变量**。我不建议您自己编写方法，甚至不建议使用讲座中演示的 TensorFlow。学习原理并使用像 Keras 这样的高效工具来实际实现项目中的方法。

对于从业者来说，这种材料中有很多金币，但你必须保持智慧，而不是落入“_ 我必须了解所有 _”陷阱。作为一名从业者，你的目标是非常不同的，你必须无情地坚持目标。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [CS224d：自然语言处理的深度学习](http://cs224d.stanford.edu/index.html)
*   [CS224n：深度学习的自然语言处理](http://web.stanford.edu/class/cs224n/)
*   [CS224n 教学大纲（2017 年冬季）](http://web.stanford.edu/class/cs224n/syllabus.html)
*   [CS224n 视频讲座（2017 年冬季）](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
*   [CS22d Sub-Reddit](https://www.reddit.com/r/CS224d/)
*   CS224d 学生项目报告（ [2015](http://cs224d.stanford.edu/reports_2015.html) ， [2016](http://cs224d.stanford.edu/reports_2016.html) ）
*   [CS224n 分配](http://web.stanford.edu/class/cs224n/assignments)

### 较旧的相关材料

*   [CS 224N / Ling 284 - 自然语言处理](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/)
*   [2015 CS224d 讲座](https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q)（2016 年新讲座弃用）
*   [2016 CS224D 讲座视频](https://www.youtube.com/playlist?list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam)
*   [深度学习自然语言处理（无魔法）2013](https://nlp.stanford.edu/courses/NAACL2013/)

## 摘要

在这篇文章中，您发现了斯坦福自然语言处理深度学习课程。

具体来说，你学到了：

*   本课程的目标和先决条件。
*   课程讲座细分以及如何访问幻灯片，笔记和视频。
*   如何充分利用这种材料。

您是否完成了部分或全部课程材料？
请在下面的评论中告诉我。