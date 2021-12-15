# 机器学习中的朴素贝叶斯

> 原文： [https://machinelearningmastery.com/naive-bayes-for-machine-learning/](https://machinelearningmastery.com/naive-bayes-for-machine-learning/)

Naive Bayes是一种简单但令人惊讶的强大的预测性建模算法。

在这篇文章中，您将发现朴素贝叶斯算法的分类。阅读这篇文章后，你会知道：

*   朴素贝叶斯使用的表示，当模型写入文件时实际存储。
*   如何使用学习模型做出预测。
*   如何从训练数据中学习朴素的贝叶斯模型。
*   如何最好地为朴素贝叶斯算法准备数据。
*   哪里可以获得有关朴素贝叶斯的更多信息。

这篇文章是为开发人员编写的，不承担统计或概率的任何背景，尽管知道一点概率不会受到伤害。

让我们开始吧。

![Naive Bayes for Machine Learning](img/4b403ace39c6d94edae6ae4094ccca20.jpg)

朴素贝叶斯机器学习
摄影：[约翰摩根](https://www.flickr.com/photos/aidanmorgan/3249101355/)，保留一些权利。

## 贝叶斯定理的快速入门

在机器学习中，我们常常对给定数据（d）选择最佳假设（h）感兴趣。

在分类问题中，我们的假设（h）可以是为新数据实例分配的类（d）。

选择最可能的假设的最简单方法之一是给定我们可以使用的数据作为我们关于该问题的先验知识。贝叶斯定理提供了一种方法，我们可以根据我们的先验知识计算假设的概率。

贝叶斯定理的陈述如下：

P（h | d）=（P（d | h）* P（h））/ P（d）

哪里

*   **P（h | d）**是给定数据d的假设h的概率。这称为后验概率。
*   **P（d | h）**是假设h为真的数据d的概率。
*   **P（h）**是假设h为真的概率（无论数据如何）。这被称为h的先验概率。
*   **P（d）**是数据的概率（不论假设）。

你可以看到我们感兴趣的是用P（D）和P（d | h）从先验概率p（h）计算P（h | d）的后验概率。

在计算了许多不同假设的后验概率后，您可以选择概率最高的假设。这是最大可能假设，并且可以正式称为[最大后验](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)（MAP）假设。

这可以写成：

MAP（h）= max（P（h | d））

要么

MAP（h）= max（（P（d | h）* P（h））/ P（d））

or

MAP（h）= max（P（d | h）* P（h））

P（d）是归一化项，它允许我们计算概率。当我们对最可能的假设感兴趣时，我们可以放弃它，因为它是常数并且仅用于标准化。

回到分类，如果我们的训练数据中每个类都有偶数个实例，那么每个类的概率（例如P（h））将是相等的。同样，这将是我们等式中的一个常数项，我们可以放弃它，以便最终得到：

MAP（h）= max（P（d | h））

这是一个有用的练习，因为当您在Naive Bayes上进一步阅读时，您可能会看到所有这些形式的定理。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 朴素贝叶斯分类器

朴素贝叶斯是一种二元（两类）和多分类问题的分类算法。当使用二进制或分类输入值描述时，该技术最容易理解。

它被称为_朴素贝叶斯_或_白痴贝叶斯_因为每个假设的概率计算被简化以使其计算易于处理。不是试图计算每个属性值P（d1，d2，d3 | h）的值，而是假定它们在给定目标值的情况下是条件独立的并且计算为P（d1 | h）* P（d2 | H）并且等等。

这是一个非常强大的假设，在实际数据中最不可能，即属性不相互作用。尽管如此，该方法在这种假设不成立的数据上表现出色。

### 朴素贝叶斯模型使用的表示法

朴素贝叶斯的表示是概率。

概率列表存储到文件中，用于学习的朴素贝叶斯模型。这包括：

*   **类概率**：训练数据集中每个类的概率。
*   **条件概率**：给定每个类值的每个输入值的条件概率。

### 从数据中学习朴素贝叶斯模型

从训练数据中学习朴素的贝叶斯模型很快。

训练很快，因为只需要计算每个类的概率和给定不同输入（x）值的每个类的概率。优化程序不需要拟合系数。

#### 计算类概率

类概率只是属于每个类的实例的频率除以实例的总数。

例如，在二分类中，属于类1的实例的概率将计算为：

P（class = 1）= count（class = 1）/（count（class = 0）+ count（class = 1））

在最简单的情况下，对于具有相同数量的实例的二分类问题，每个类的概率为0.5或50％。

#### 计算条件概率

条件概率是给定类值的每个属性值的频率除以具有该类值的实例的频率。

例如，如果“`weather`”属性的值为“`sunny`”和“`rainy`”，并且class属性的类值为“_” -out_ “和” _stay-home_ “，然后每个类别值的每个天气值的条件概率可以计算为：

*   P（weather = sunny | class = go-out）= count（天气=晴天和class = go-out的实例）/ count（class = go-out的实例）
*   P（天气=晴天| class = stay-home）=计数（天气=阳光和班级=住宿的实例）/计数（班级=住宿的情况）
*   P（weather = rainy | class = go-out）= count（天气= rainy和class = go-out的实例）/ count（class = go-out的实例）
*   P（天气= rainy | class = stay-home）= count（天气= rainy和class = stay-home的实例）/ count（class = stay-home）

### 用朴素贝叶斯模型做出预测

给定一个朴素的贝叶斯模型，您可以使用贝叶斯定理对新数据做出预测。

MAP(h) = max(P(d|h) * P(h))

使用上面的例子，如果我们有_晴天_的_天气_的新实例，我们可以计算：

go-out = P（weather = sunny | class = go-out）* P（class = go-out）
stay-home = P（天气=晴天| class = stay-home）* P（class = stay -家）

我们可以选择具有最大计算值的类。我们可以通过如下标准化它们将这些值转换为概率：

P（go-out | weather = sunny）= go-out /（go-out + stay-home）
P（stay-home | weather = sunny）= stay-home /（go-out + stay-home ）

如果我们有更多的输入变量，我们可以扩展上面的例子。例如，假装我们有一个“`car`”属性，其值为“_正在工作_”和“_打破_”。我们可以将这个概率乘以等式。

例如，下面是“go-out”类标签的计算，添加了car input变量设置为“working”：

go-out = P（weather = sunny | class = go-out）* P（car = working | class = go-out）* P（class = go-out）

## 高斯朴素贝叶斯

朴素贝叶斯可以扩展到实值属性，最常见的是假设高斯分布。

朴素贝叶斯的这种延伸被称为高斯朴素贝叶斯。其他函数可用于估计数据的分布，但高斯（或正态分布）是最容易使用的，因为您只需要估计训练数据的均值和标准差。

### 高斯朴素贝叶斯的表示

在上面，我们使用频率计算每个类的输入值的概率。通过实值输入，我们可以计算每个类的输入值（x）的均值和标准差，以总结分布。

这意味着除了每个类的概率之外，我们还必须为每个类存储每个输入变量的均值和标准偏差。

### 从数据中学习高斯朴素贝叶斯模型

这就像计算每个类值的每个输入变量（x）的[平均值](https://en.wikipedia.org/wiki/Mean)和[标准偏差](https://en.wikipedia.org/wiki/Standard_deviation)值一样简单。

mean（x）= 1 / n * sum（x）

其中n是实例数，x是训练数据中输入变量的值。

我们可以使用以下等式计算标准偏差：

标准差（x）= sqrt（1 / n * sum（xi-mean（x）^ 2））

这是x的每个值与x的平均值的平均平方差的平方根，其中n是实例数，sqrt（）是平方根函数，sum（）是sum函数，xi是a第i个实例的x变量的特定值和上述的均值（x），^ 2是正方形。

### 用高斯朴素贝叶斯模型做出预测

使用[高斯概率密度函数](https://en.wikipedia.org/wiki/Normal_distribution)（PDF）计算新x值的概率。

在做出预测时，可以将这些参数插入到具有变量的新输入的高斯PDF中，作为回报，高斯PDF将提供该类的新输入值的概率的估计。

pdf（x，mean，sd）=（1 /（sqrt（2 * PI）* sd））* exp（ - （（x-mean ^ 2）/（2 * sd ^ 2）））

其中pdf（x）是高斯PDF，sqrt（）是平方根，mean和sd是上面计算的平均值和标准差， [PI](https://en.wikipedia.org/wiki/Pi) 是数值常数，exp（）是数值常数e或[欧拉数](https://en.wikipedia.org/wiki/E_(mathematical_constant))上升到幂，x是输入变量的输入值。

然后我们可以将概率插入上面的等式中，以使用实值输入做出预测。

例如，使用天气和汽车的数值调整上述计算之一：

go-out = P（pdf（天气）| class = go-out）* P（pdf（car）| class = go-out）* P（class = go-out）

## 为朴素贝叶斯准备最佳数据

*   **分类输入**：朴素贝叶斯假设标签属性，如二进制，分类或名义。
*   **高斯输入**：如果输入变量是实值，则假定为高斯分布。在这种情况下，如果数据的单变量分布是高斯分布或接近高斯分布，则算法将表现得更好。这可能需要去除异常值（例如，与平均值相差超过3或4个标准偏差的值）。
*   **分类问题**：朴素贝叶斯是一种适用于二元和多分类的分类算法。
*   **对数概率**：计算不同类别值的可能性涉及将许多小数字相乘。这可能导致数值精度下降。因此，优良作法是使用概率的对数变换来避免这种下溢。
*   **核函数**：不是假设数值输入值的高斯分布，而是可以使用更复杂的分布，例如各种核密度函数。
*   **更新概率**：当新数据可用时，您只需更新模型的概率即可。如果数据经常更改，这可能会有所帮助。

## 进一步阅读

关于Naive Bayes的另外两篇你可能会感兴趣的帖子是：

*   [如何在Python中从零开始实现朴素贝叶斯](http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)
*   [更好的朴素贝叶斯：从朴素贝叶斯算法中获取最多的12个技巧](http://machinelearningmastery.com/better-naive-bayes/)

我喜欢书。下面是一些很好的通用机器学习书籍，供开发人员使用，包括朴素的贝叶斯：

*   [数据挖掘：实用机器学习工具和技术](http://www.amazon.com/dp/0123748569?tag=inspiredalgor-20)，第88页
*   [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，第353页
*   [人工智能：现代方法](http://www.amazon.com/dp/0136042597?tag=inspiredalgor-20)，第808页
*   [机器学习](http://www.amazon.com/dp/0070428077?tag=inspiredalgor-20)，第6章

## 摘要

在这篇文章中，您发现了Naive Bayes算法进行分类。你了解到：

*   贝叶斯定理以及如何在实践中计算它。
*   朴素贝叶斯算法包括表示，做出预测和学习模型。
*   朴素贝叶斯对实值输入数据的改编称为高斯朴素贝叶斯。
*   如何为朴素贝叶斯准备数据。

您对朴素贝叶斯或这篇文章有任何疑问吗？发表评论并提出问题，我会尽力回答。