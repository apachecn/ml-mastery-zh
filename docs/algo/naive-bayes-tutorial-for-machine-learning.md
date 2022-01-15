# 机器学习中的朴素贝叶斯教程

> 原文： [https://machinelearningmastery.com/naive-bayes-tutorial-for-machine-learning/](https://machinelearningmastery.com/naive-bayes-tutorial-for-machine-learning/)

Naive Bayes 是一种非常简单的分类算法，它对每个输入变量的独立性做出了一些强有力的假设。

然而，它已被证明在许多问题领域都是有效的。在这篇文章中，您将发现用于分类数据的朴素贝叶斯算法。阅读这篇文章后，你会知道的。

*   如何使用 Naive Bayes 的分类数据。
*   如何为朴素贝叶斯模型准备类和条件概率。
*   如何使用学习的朴素贝叶斯模型做出预测。

这篇文章是为开发人员编写的，不承担统计或概率的背景。打开电子表格并按照说明进行操作。如果您对 Naive Bayes 有任何疑问，请在评论中提出，我会尽力回答。

让我们开始吧。

![Naive Bayes Tutorial for Machine Learning](img/bb83b3ac3664bbd3af14ba5e8680a73e.jpg)

朴素贝叶斯机器学习教程
照片由 [Beshef](https://www.flickr.com/photos/sharif/2515894536) ，保留一些权利。

## 教程数据集

数据集是人为设计的。它描述了两个分类输入变量和一个具有两个输出的类变量。

```py
Weather	Car	Class
sunny	working	go-out
rainy	broken	go-out
sunny	working	go-out
sunny	working	go-out
sunny	working	go-out
rainy	broken	stay-home
rainy	broken	stay-home
sunny	working	stay-home
sunny	broken	stay-home
rainy	broken	stay-home
```

我们可以将其转换为数字。每个输入只有两个值，输出类变量有两个值。我们可以将每个变量转换为二进制如下：

**变量：天气**

*   晴天= 1
*   下雨= 0

**变量：汽车**

*   工作= 1
*   破碎= 0

**变量：类**

*   外出= 1
*   stay-home = 0

因此，我们可以将数据集重新表示为：

```py
Weather	Car	Class
1	1	1
0	0	1
1	1	1
1	1	1
1	1	1
0	0	0
0	0	0
1	1	0
1	0	0
0	0	0
```

如果您跟进，这可以使数据更容易在电子表格或代码中使用。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的 60 多种算法的方便思维导图。

下载，打印并使用它。

## 学习朴素贝叶斯模型

需要从朴素贝叶斯模型的数据集中计算出两种类型的数量：

*   类概率。
*   条件概率。

让我们从类概率开始。

### 计算类概率

数据集是一个两类问题，我们已经知道每个类的概率，因为我们设计了数据集。

不过，我们可以计算出 0 级和 1 级的类概率，如下所示：

*   P（class = 1）= count（class = 1）/（count（class = 0）+ count（class = 1））
*   P（class = 0）= count（class = 0）/（count（class = 0）+ count（class = 1））

要么

*   P（class = 1）= 5 /（5 + 5）
*   P（class = 0）= 5 /（5 + 5）

对于属于 0 级或 1 级的任何给定数据实例，这可能是 0.5 的概率。

### 计算条件概率

条件概率是给定每个类值的每个输入值的概率。

数据集的条件概率可以如下计算：

#### 天气输入变量

*   P（天气=晴天|上课=外出）=计数（天气=晴天和上课=外出）/计数（上课=外出）
*   P（天气=雨天|上课=外出）=伯爵（天气=下雨和上课=外出）/伯爵（上课=外出）
*   P（天气=晴天|等级=住宿）=计数（天气=晴天和等级=住宿）/计数（等级=住宿 - 住宿）
*   P（天气=多雨|等级=住宿）=计数（天气=多雨，等级=住宿）/计数（等级=住宿）

插入我们得到的数字：

*   P（天气=晴天|等级=外出）= 0.8
*   P（天气= rainy | class = go-out）= 0.2
*   P（天气=晴天|等级=住宿）= 0.4
*   P（天气= rainy | class = stay-home）= 0.6

#### 汽车输入变量

*   P（car = working | class = go-out）= count（car = working and class = go-out）/ count（class = go-out）
*   P（car = broken | class = go-out）= count（car = brokenrainy 和 class = go-out）/ count（class = go-out）
*   P（car = working | class = stay-home）= count（car = working and class = stay-home）/ count（class = stay-home）
*   P（car = broken | class = stay-home）= count（car = brokenrainy and class = stay-home）/ count（class = stay-home）

Plugging in the numbers we get:

*   P（car = working | class = go-out）= 0.8
*   P（car = broken | class = go-out）= 0.2
*   P（car = working | class = stay-home）= 0.2
*   P（car = broken | class = stay-home）= 0.8

我们现在拥有使用朴素贝叶斯模型做出预测所需的一切。

## 用朴素贝叶斯做出预测

我们可以使用贝叶斯定理做出预测。

P（h | d）=（P（d | h）* P（h））/ P（d）

哪里：

*   **P（h | d）**是给定数据 d 的假设 h 的概率。这称为后验概率。
*   **P（d | h）**是假设 h 为真的数据 d 的概率。
*   **P（h）**是假设 h 为真的概率（无论数据如何）。这被称为 h 的先验概率。
*   **P（d）**是数据的概率（不论假设）。

实际上，我们不需要概率来预测新数据实例的最可能类。我们只需要得到最大响应的分子和类，这将是预测输出。

MAP（h）= max（P（d | h）* P（h））

让我们从我们的数据集中获取第一条记录，并使用我们的学习模型来预测我们认为它属于哪个类。

天气=晴天，车=工作

我们为两个类插入模型的概率并计算响应。从输出“go-out”的响应开始。我们将条件概率相乘，并将其乘以属于该类的任何实例的概率。

*   go-out = P（weather = sunny | class = go-out）* P（car = working | class = go-out）* P（class = go-out）
*   外出= 0.8 * 0.8 * 0.5
*   外出= 0.32

我们可以为住宿情况执行相同的计算：

*   stay-home = P（天气=阳光|等级=住宿）* P（汽车=工作|等级=住宿）* P（等级=住宿）
*   住宿= 0.4 * 0.2 * 0.5
*   住宿= 0.04

我们可以看到 0.32 大于 0.04，因此我们预测此实例的“走出去”，这是正确的。

我们可以对整个数据集重复此操作，如下所示：

```py
Weather	Car	Class		out?	home?	Prediction
sunny	working	go-out		0.32	0.04	go-out
rainy	broken	go-out		0.02	0.24	stay-home
sunny	working	go-out		0.32	0.04	go-out
sunny	working	go-out		0.32	0.04	go-out
sunny	working	go-out		0.32	0.04	go-out
rainy	broken	stay-home	0.02	0.24	stay-home
rainy	broken	stay-home	0.02	0.24	stay-home
sunny	working	stay-home	0.32	0.04	go-out
sunny	broken	stay-home	0.08	0.16	stay-home
rainy	broken	stay-home	0.02	0.24	stay-home
```

如果我们将预测与实际类值进行比较，我们得到 80％的准确度，鉴于数据集中存在冲突的示例，这是非常好的。

## 摘要

在这篇文章中，您发现了如何从零开始实现 Naive Bayes。你了解到：

*   如何使用 Naive Bayes 处理分类数据。
*   如何根据训练数据计算班级概率。
*   如何从训练数据计算条件概率。
*   如何使用学习的朴素贝叶斯模型对新数据做出预测。

您对 Naive Bayes 或这篇文章有任何疑问吗？
发表评论提出您的问题，我会尽力回答。