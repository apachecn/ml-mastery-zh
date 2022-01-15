# 什么是机器学习中的 Argmax？

> 原文:[https://machinelearning master . com/arg max-in-machine-learning/](https://machinelearningmastery.com/argmax-in-machine-learning/)

最后更新于 2020 年 8 月 19 日

Argmax 是你在应用机器学习中可能会遇到的一个数学函数。

例如，您可能会在一篇用于描述算法的研究论文中看到“ *argmax* ”或“ *arg max* ”。您可能还会被指示在算法实现中使用 argmax 函数。

这可能是您第一次遇到 argmax 函数，您可能想知道它是什么以及它是如何工作的。

在本教程中，您将发现 argmax 函数及其在机器学习中的应用。

完成本教程后，您将知道:

*   Argmax 是从目标函数中找到给出最大值的参数的操作。
*   Argmax 在机器学习中最常用于寻找预测概率最大的类。
*   Argmax 可以手动实现，尽管实践中首选 argmax() NumPy 函数。

**用我的新书[机器学习线性代数](https://machinelearningmastery.com/linear_algebra_for_machine_learning/)启动你的项目**，包括*循序渐进教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![What Is argmax in Machine Learning?](img/fc66144154f4656f5883c469ea6f01d2.png)

机器学习中的 argmax 是什么？
伯纳德·斯普拉格摄。新西兰，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  什么是 Argmax？
2.  Argmax 在机器学习中是如何使用的？
3.  如何在 Python 中实现 Argmax

## 什么是 Argmax？

[Argmax](https://en.wikipedia.org/wiki/Arg_max) 是一个数学函数。

它通常应用于另一个接受参数的函数。例如，给定一个采用参数 *x* 的函数 *g()* ，该函数的 *argmax* 操作将描述如下:

*   结果= argmax(g(x))

*argmax* 函数返回目标函数的一个或多个参数( *arg* ，目标函数从目标函数返回最大值( *max* )。

考虑这样的例子，其中 *g(x)* 被计算为 *x* 值的平方，并且输入值的域或范围( *x* )被限制为从 1 到 5 的整数:

*   g(1) = 1^2 = 1
*   g(2) = 2^2 = 4
*   g(3) = 3^2 = 9
*   g(4) = 4^2 = 16
*   g(5) = 5^2 = 25

我们可以直观地看到函数 *g(x)* 的 argmax 为 5。

也就是说，从目标函数(25)得到最大值的目标函数 *g()* 的自变量( *x* )是 5。Argmax 提供了一种简写方式，可以在不知道特定情况下该值可能是什么的情况下，以抽象方式指定该参数。

*   argmax（g（x）） = 5

请注意，这不是从函数返回的值的 *max()* 。这将是 25。

它也不是参数的 *max()* ，尽管在这种情况下参数的 argmax 和 max 是相同的，例如 5。 *argmax()* 是 5，因为当提供 5 时，g 返回最大值(25)，而不是因为 5 是最大的参数。

通常，“ *argmax* ”被写成两个独立的单词，例如“ *arg max* ”。例如:

*   结果= arg 最大值(g(x))

将 arg max 函数用作目标函数周围没有括号的操作也很常见。这通常是你在研究论文或教科书中看到的操作。例如:

*   结果= arg 最大 g(x)

您也可以使用类似的操作来查找目标函数的参数，这些参数导致目标函数的最小值，称为 *argmin* 或“ *arg min* ”

## Argmax 在机器学习中是如何使用的？

argmax 函数用于整个数学和机器学习领域。

然而，在一些特定的情况下，您会看到 argmax 被用于应用机器学习，并且可能需要自己实现它。

在应用机器学习中，使用 argmax 最常见的情况是找到导致最大值的数组的索引。

回想一下，数组是数字的列表或向量。

多类分类模型通常预测一个概率向量(或类似概率的值)，每个类标签有一个概率。概率表示样本属于每个类别标签的可能性。

对预测概率进行排序，使得索引 0 处的预测概率属于第一类，索引 1 处的预测概率属于第二类，依此类推。

通常，对于多类分类问题，需要从一组预测概率中进行单类标签预测。

这种从预测概率向量到类标签的转换最常用 argmax 操作来描述，最常用 argmax 函数来实现。

让我们用一个例子来具体说明。

考虑一个三类的多类分类问题:“*红色*”、“*蓝色*”和“*绿色*”类标签被映射为用于建模的整数值，如下所示:

*   红色= 0
*   蓝色= 1
*   绿色= 2

每个类标签整数值映射到一个 3 元素向量的索引，该索引可以通过一个模型来预测，该模型指定了一个示例属于每个类的可能性。

假设一个模型对输入样本进行了一次预测，并预测了以下概率向量:

*   yhat = [0.4，0.5，0.1]

我们可以看到，这个例子有 40%的概率属于红色，50%的概率属于蓝色，10%的概率属于绿色。

我们可以将 argmax 函数应用于概率向量。向量是函数，函数的输出是概率，函数的输入是向量元素索引或数组索引。

*   arg max yhat

我们可以直观地看到，在这种情况下，预测概率向量(yhat)的 argmax 为 1，因为数组索引 1 处的概率是最大值。

请注意，这不是概率的最大值()，可能是 0.5。还要注意，这不是参数的最大值，应该是 2。取而代之的是导致最大值的参数，例如 1 导致 0.5。

*   arg max yhat = 1

然后，我们可以将这个整数值映射回一个类标签，它将是蓝色的

*   arg max yhat = "blue "

## 如何在 Python 中实现 Argmax

对于给定的数字向量，argmax 函数可以在 Python 中实现。

### 从头开始的 Argmax

首先，我们可以定义一个名为 *argmax()* 的函数，该函数枚举一个提供的向量并返回具有最大值的索引。

下面列出了完整的示例。

```py
# argmax function
def argmax(vector):
	index, value = 0, vector[0]
	for i,v in enumerate(vector):
		if v > value:
			index, value = i,v
	return index

# define vector
vector = [0.4, 0.5, 0.1]
# get argmax
result = argmax(vector)
print('arg max of %s: %d' % (vector, result))
```

运行该示例会打印上一节中使用的测试数据的 argmax，在本例中，该数据的索引为 1。

```py
arg max of [0.4, 0.5, 0.1]: 1
```

### 带 NumPy 的 Argmax

谢天谢地，NumPy 库提供了一个内置版本的 [argmax()函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)。

这是你应该在实践中使用的版本。

下面的例子演示了同一个概率向量上的 *argmax()* NumPy 函数。

```py
# numpy implementation of argmax
from numpy import argmax
# define vector
vector = [0.4, 0.5, 0.1]
# get argmax
result = argmax(vector)
print('arg max of %s: %d' % (vector, result))
```

如预期的那样，运行该示例将打印索引 1。

```py
arg max of [0.4, 0.5, 0.1]: 1
```

更有可能的是，你会有多个样本的预测概率的集合。

这将被存储为一个矩阵，其中包含预测概率行，每列代表一个类标签。这个矩阵上 argmax 的期望结果是一个向量，每行预测有一个索引(或类标签整数)。

这可以通过 *argmax()* NumPy 函数通过设置“*轴*参数来实现。默认情况下，将为整个矩阵计算 argmax，返回一个数字。相反，我们可以将轴值设置为 1，并计算每一行数据跨列的 argmax。

下面的例子用三个类标签的四行预测概率矩阵来演示这一点。

```py
# numpy implementation of argmax
from numpy import argmax
from numpy import asarray
# define vector
probs = asarray([[0.4, 0.5, 0.1], [0.0, 0.0, 1.0], [0.9, 0.0, 0.1], [0.3, 0.3, 0.4]])
print(probs.shape)
# get argmax
result = argmax(probs, axis=1)
print(result)
```

运行该示例首先打印预测概率矩阵的形状，确认我们有四行，每行三列。

然后计算矩阵的 argmax 并打印为向量，显示四个值。这就是我们所期望的，每一行都以最大的概率产生一个 argmax 值或索引。

```py
(4, 3)
[1 2 0 2]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   num py . argmax API。
*   [愤怒的马克斯，维基百科](https://en.wikipedia.org/wiki/Arg_max)。

## 摘要

在本教程中，您发现了 argmax 函数以及它是如何在机器学习中使用的。

具体来说，您了解到:

*   Argmax 是从目标函数中找到给出最大值的参数的操作。
*   Argmax 在机器学习中最常用于寻找预测概率最大的类。
*   Argmax 可以手动实现，尽管实践中首选 argmax() NumPy 函数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。