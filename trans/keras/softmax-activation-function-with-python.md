# Python 的 Softmax 激活函数

> 原文：<https://machinelearningmastery.com/softmax-activation-function-with-python/>

**Softmax** 是一个数学函数，它将一个数字向量转换成一个概率向量，其中每个值的概率与向量中每个值的相对比例成正比。

softmax 函数在应用机器学习中最常见的用途是用作神经网络模型中的激活函数。具体而言，网络被配置为输出 N 个值，分类任务中的每个类一个值，并且 softmax 函数被用于归一化输出，将它们从加权和值转换成总和等于 1 的概率。softmax 函数输出中的每个值都被解释为每个类的成员概率。

在本教程中，您将发现神经网络模型中使用的 softmax 激活函数。

完成本教程后，您将知道:

*   线性和 Sigmoid 激活函数不适用于多类分类任务。
*   Softmax 可以被认为是 argmax 函数的软化版本，它返回列表中最大值的索引。
*   如何在 Python 中从头实现 softmax 函数，如何将输出转换成类标签。

我们开始吧。

![Softmax Activation Function with Python](img/fc4a52fda0460cac8c396b7a80ad2c29.png)

带 Python 的 Softmax 激活函数
图片由[伊恩·d·基廷](https://flickr.com/photos/ian-arlett/36340268755/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  用神经网络预测概率
2.  最大值、最大值和最大值
3.  软最大激活函数

## 用神经网络预测概率

神经网络模型可用于对分类预测建模问题进行建模。

分类问题是那些涉及预测给定输入的类标签的问题。建模分类问题的标准方法是使用模型来预测类成员的概率。也就是说，举个例子，它属于每个已知类别标签的概率是多少？

*   对于二元分类问题，使用[二项概率分布](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)。这是通过使用在输出层具有单个节点的网络来实现的，该网络预测属于类别 1 的示例的概率。
*   对于多类分类问题，使用[多项式概率](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)。这是通过使用一个网络来实现的，该网络在输出层中每个类都有一个节点，并且预测概率的总和等于 1。

神经网络模型需要模型输出层的激活函数来进行预测。

有不同的激活函数可供选择；我们来看几个。

### 线性激活函数

预测类成员概率的一种方法是使用线性激活。

线性激活函数只是节点的加权输入之和，任何激活函数都需要它作为输入。因此，它通常被称为“无激活函数”*，因为不执行额外的转换。*

 *回想一下[概率](https://machinelearningmastery.com/what-is-probability/)或可能性是一个介于 0 和 1 之间的数值。

假设没有对输入的加权和执行任何变换，则线性激活函数可以输出任何数值。这使得线性激活函数不适用于预测二项式或多项式情况下的概率。

### Sigmoid激活函数

预测类成员概率的另一种方法是使用 sigmoid 激活函数。

这个函数也被称为逻辑函数。不管输入是什么，函数总是输出一个介于 0 和 1 之间的值。函数的形式是 0 到 1 之间的 S 形，垂直或中间的“ *S* ”为 0.5。

这允许作为输入加权和给出的非常大的值输出为 1.0，非常小的或负值映射为 0.0。

sigmoid 激活是二元分类问题的理想激活函数，其中输出被解释为二项式概率分布。

sigmoid 激活函数也可以用作多类分类问题的激活函数，其中类是非互斥的。这些通常被称为多标签分类，而不是多类别分类。

sigmoid 激活函数不适用于需要多项式概率分布的互斥类的多类分类问题。

相反，需要一个称为**软最大功能**的替代激活。

## 最大值、最大值和最大值

### 最大函数

最大值或“ *max* ”数学函数返回数值列表中最大的数值。

我们可以使用 *max()* Python 函数来实现；例如:

```py
# example of the max of a list of numbers
# define data
data = [1, 3, 2]
# calculate the max of the list
result = max(data)
print(result)
```

运行该示例将从数字列表中返回最大值“3”。

```py
3
```

### Argmax 函数

argmax 或“ *arg max* ”数学函数返回列表中包含最大值的索引。

把它想象成 max 的元版本:max 之上的一级间接，指向列表中具有 max 值的位置，而不是值本身。

我们可以使用 [argmax() NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)来实现；例如:

```py
# example of the argmax of a list of numbers
from numpy import argmax
# define data
data = [1, 3, 2]
# calculate the argmax of the list
result = argmax(data)
print(result)
```

运行该示例将返回列表索引值“1”，该值指向包含列表“3”中最大值的数组索引[1]。

```py
1
```

### 软最大函数

softmax，或“T0”soft max，数学函数可以被认为是 argmax 函数的概率或“T2”更软的版本。

> 之所以使用 softmax 这个术语，是因为这个激活函数代表了赢者通吃激活模型的平滑版本，其中输入最大的单元具有输出+1，而所有其他单元具有输出 0。

—第 238 页，[用于模式识别的神经网络](https://amzn.to/2TQYuDo)，1995。

从概率的角度来看，如果 *argmax()* 函数在前一节中返回 1，那么对于其他两个数组索引，它将返回 0，对于列表[1，3，2]中最大的值，给索引 1 完全的权重，而不给索引 0 和索引 2 任何权重。

```py
[0, 1, 0]
```

如果我们不太确定，想用概率来表达 argmax，有可能吗？

这可以通过缩放列表中的值并将其转换为概率来实现，以便返回的列表中的所有值的总和为 1.0。

这可以通过计算列表中每个值的指数并将其除以指数值的总和来实现。

*   概率= exp(值)/列表中的总和 v exp(v)

例如，我们可以将列表[1，3，2]中的第一个值“1”转换为概率，如下所示:

*   概率= exp(1) / (exp(1) + exp(3) + exp(2))
*   概率= exp(1) / (exp(1) + exp(3) + exp(2))
*   概率= 2.718281828459045/30。58865 . 8888888886
*   概率= 0.09003057317038046

我们可以用 Python 为列表[1，3，2]中的每个值演示如下:

```py
# transform values into probabilities
from math import exp
# calculate each probability
p1 = exp(1) / (exp(1) + exp(3) + exp(2))
p2 = exp(3) / (exp(1) + exp(3) + exp(2))
p3 = exp(2) / (exp(1) + exp(3) + exp(2))
# report probabilities
print(p1, p2, p3)
# report sum of probabilities
print(p1 + p2 + p3)
```

运行该示例将列表中的每个值转换为概率并报告这些值，然后确认所有概率的总和为值 1.0。

我们可以看到，大部分权重被放在指数 1 上(67%)，指数 2 上的权重较小(24%)，指数 0 上的权重甚至更小(9%)。

```py
0.09003057317038046 0.6652409557748219 0.24472847105479767
1.0
```

这是 softmax 功能。

我们可以将其实现为一个函数，该函数接受一个数字列表，并返回该列表的软最大值或多项式概率分布。

下面的例子实现了这个函数，并在我们的小数字列表中演示了它。

```py
# example of a function for calculating softmax for a list of numbers
from numpy import exp

# calculate the softmax of a vector
def softmax(vector):
	e = exp(vector)
	return e / e.sum()

# define data
data = [1, 3, 2]
# convert list of numbers to a list of probabilities
result = softmax(data)
# report the probabilities
print(result)
# report the sum of the probabilities
print(sum(result))
```

运行该示例会报告大致相同的数字，但精确率略有差异。

```py
[0.09003057 0.66524096 0.24472847]
1.0
```

最后，我们可以使用内置的 [softmax() NumPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html)来计算一个数组或数字列表的 softmax，如下所示:

```py
# example of calculating the softmax for a list of numbers
from scipy.special import softmax
# define data
data = [1, 3, 2]
# calculate softmax
result = softmax(data)
# report the probabilities
print(result)
# report the sum of the probabilities
print(sum(result))
```

再次运行该示例，我们得到了非常相似的结果，但精确率差异很小。

```py
[0.09003057 0.66524096 0.24472847]
0.9999999999999997
```

现在我们已经熟悉了 softmax 函数，让我们看看它是如何在神经网络模型中使用的。

## 软最大激活函数

softmax 函数用作预测多项式概率分布的神经网络模型的输出层中的激活函数。

也就是说，softmax 被用作多类分类问题的激活函数，其中在两个以上的类标签上需要类成员资格。

> 任何时候我们希望用 n 个可能值来表示一个离散变量的概率分布，我们都可以使用 softmax 函数。这可以看作是 sigmoid 函数的推广，sigmoid 函数用于表示二元变量的概率分布。

—第 184 页，[深度学习](https://amzn.to/33iMC06)，2016。

该函数可以用作神经网络中隐藏层的激活函数，尽管这不太常见。当模型内部需要在瓶颈层或连接层选择或加权多个不同的输入时，可以使用它。

> Softmax 单位自然代表一个具有 k 个可能值的离散变量的概率分布，因此它们可以用作一种开关。

—第 196 页，[深度学习](https://amzn.to/33iMC06)，2016。

在具有三类分类任务的 Keras 深度学习库中，输出层中 softmax 的使用可能如下所示:

```py
...
model.add(Dense(3, activation='softmax'))
```

根据定义，softmax 激活将为输出层中的每个节点输出一个值。输出值将表示(或者可以解释为)概率，并且这些值的总和为 1.0。

在对多类分类问题建模时，必须准备好数据。包含类标签的目标变量首先被标签编码，这意味着一个整数被应用于从 0 到 N-1 的每个类标签，其中 N 是类标签的数量。

然后对标签编码(或整数编码)的目标变量进行一次热编码。这是类标签的概率表示，很像 softmax 输出。为每个类标签和位置创建一个带有位置的向量。所有值都标记为 0(不可能)，1(确定)用于标记类别标签的位置。

例如，三个类标签将被整数编码为 0、1 和 2。然后编码成矢量，如下所示:

*   类别 0: [1，0，0]
*   类别 1: [0，1，0]
*   类别 2: [0，0，1]

这叫做[一热编码](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)。

它表示用于在监督学习下校正模型的每个类的期望多项式概率分布。

softmax 函数将为每个类别标签输出类别成员的概率，并尝试为给定的输入最好地近似预期目标。

例如，如果在一个示例中需要整数编码的类 1，那么目标向量将是:

*   [0, 1, 0]

softmax 输出可能如下所示，它将最大的权重放在类 1 上，而将较小的权重放在其他类上。

*   [0.09003057 0.66524096 0.24472847]

预期的和预测的多项式概率分布之间的误差通常使用交叉熵来计算，然后使用该误差来更新模型。这被称为交叉熵损失函数。

有关计算概率分布差异的交叉熵的更多信息，请参见教程:

*   [机器学习交叉熵的温和介绍](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)

我们可能希望将概率转换回整数编码的类标签。

这可以使用 *argmax()* 函数来实现，该函数返回具有最大值的列表的索引。假设类标签是从 0 到 N-1 的整数编码，概率的 argmax 将总是整数编码的类标签。

*   class integer = arg max([0.09003057 0.66524096 0.24472847])
*   类整数= 1

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [用于模式识别的神经网络](https://amzn.to/2TQYuDo)，1995。
*   [神经网络:交易的诀窍:交易的诀窍](https://amzn.to/2U6fPYc)，第二版，2012 年。
*   [深度学习](https://amzn.to/33iMC06)，2016 年。

### 蜜蜂

*   num py . argmax API。
*   [scipy.special.softmax API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html) 。

### 文章

*   [Softmax 功能，维基百科](https://en.wikipedia.org/wiki/Softmax_function)。

## 摘要

在本教程中，您发现了神经网络模型中使用的 softmax 激活函数。

具体来说，您了解到:

*   线性和 Sigmoid 激活函数不适用于多类分类任务。
*   Softmax 可以被认为是 argmax 函数的软化版本，它返回列表中最大值的索引。
*   如何在 Python 中从头实现 softmax 函数，如何将输出转换成类标签。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。*