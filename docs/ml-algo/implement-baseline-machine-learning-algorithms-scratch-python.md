# 如何用Python从头开始实现基线机器学习算法

> 原文： [https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/](https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/)

在预测建模问题上建立基线表现非常重要。

基线为您稍后评估的更高级方法提供了比较点。

在本教程中，您将了解如何在Python中从头开始实现基线机器学习算法。

完成本教程后，您将了解：

*   如何实现随机预测算法。
*   如何实现零规则预测算法。

让我们开始吧。

*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement Baseline Machine Learning Algorithms From Scratch With Python](img/a6eef0a4a1a40fdd7d4ea80ebe51c609.jpg)

如何使用Python从头开始实施基线机器学习算法
照片由 [Vanesser III](https://www.flickr.com/photos/hapinachu/13767713/) ，保留一些权利。

## 描述

有许多机器学习算法可供选择。事实上数以百计。

您必须知道给定算法的预测是否良好。但你怎么知道的？

答案是使用基线预测算法。基线预测算法提供了一组预测，您可以像对问题的任何预测一样进行评估，例如分类准确度或RMSE。

在评估问题的所有其他机器学习算法时，这些算法的分数提供了所需的比较点。

一旦建立，您可以评论给定算法与朴素基线算法相比有多好，提供给定方法实际有多好的背景。

两种最常用的基线算法是：

*   随机预测算法。
*   零规则算法。

当开始一个比传统分类或回归问题更具粘性的新问题时，首先设计一个特定于您的预测问题的随机预测算法是个好主意。稍后您可以对此进行改进并设计零规则算法。

让我们实现这些算法，看看它们是如何工作的。

## 教程

本教程分为两部分：

1.  随机预测算法。
2.  零规则算法。

这些步骤将为您的机器学习算法实现和计算基线表现提供所需的基础。

### 1.随机预测算法

随机预测算法预测在训练数据中观察到的随机结果。

它可能是最简单的算法。

它要求您将所有不同的结果值存储在训练数据中，这可能对具有许多不同值的回归问题很大。

因为随机数用于做出决策，所以在使用算法之前修复随机数种子是个好主意。这是为了确保我们获得相同的随机数集，并且每次运行算法时都会得到相同的决策。

下面是名为 **random_algorithm（）**的函数中的随机预测算法的实现。

该函数既包含包含输出值的训练数据集，也包含必须预测输出值的测试数据集。

该功能适用​​于分类和回归问题。它假定训练数据中的输出值是每行的最后一列。

首先，从训练数据中收集该组唯一输出值。然后，为测试集中的每一行选择随机选择的输出值。

```py
# Generate random predictions
def random_algorithm(train, test):
	output_values = [row[-1] for row in train]
	unique = list(set(output_values))
	predicted = list()
	for row in test:
		index = randrange(len(unique))
		predicted.append(unique[index])
	return predicted
```

为简单起见，我们可以使用仅包含输出列的小数据集来测试此函数。

训练数据集中的输出值为“0”或“1”，表示算法将从中选择的预测集是{0,1}。测试集还包含单个列，没有数据，因为预测未知。

```py
from random import seed
from random import randrange

# Generate random predictions
def random_algorithm(train, test):
	output_values = [row[-1] for row in train]
	unique = list(set(output_values))
	predicted = list()
	for row in test:
		index = randrange(len(unique))
		predicted.append(unique[index])
	return predicted

seed(1)
train = [[0], [1], [0], [1], [0], [1]]
test = [[None], [None], [None], [None]]
predictions = random_algorithm(train, test)
print(predictions)
```

运行该示例计算测试数据集的随机预测并打印这些预测。

```py
[0, 0, 1, 0]
```

随机预测算法易于实现且运行速度快，但我们可以做得更好作为基线。

### 2.零规则算法

零规则算法是比随机算法更好的基线。

它使用有关给定问题的更多信息来创建一个规则以进行预测。此规则因问题类型而异。

让我们从分类问题开始，预测一个类标签。

#### 分类

对于分类问题，一条规则是预测训练数据集中最常见的类值。这意味着如果训练数据集具有90个类“0”的实例和10个类“1”的实例，则它将预测“0”并且实现90/100或90％的基线准确度。

这比随机预测算法要好得多，后者平均只能达到82％的准确率。有关如何计算随机搜索估计值的详细信息，请参阅以下内容：

```py
= ((0.9 * 0.9) + (0.1 * 0.1)) * 100
= 82%
```

下面是一个名为 **zero_rule_algorithm_classification（）**的函数，它为分类情况实现了这个功能。

```py
# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted
```

该函数利用 **max（）**函数和key属性，这有点聪明。

给定训练数据中观察到的类值列表， **max（）**函数采用一组唯一的类值，并调用集合中每个类值的类值列表上的计数。

结果是它返回在训练数据集中观察到的类值列表中具有最高观察值计数的类值。

如果所有类值具有相同的计数，那么我们将选择在数据集中观察到的第一个类值。

一旦我们选择了一个类值，它就会用于对测试数据集中的每一行进行预测。

下面是一个包含设计数据集的工作示例，其中包含4个类“0”的示例和2个类“1”的示例。我们希望算法选择类值“0”作为测试数据集中每行的预测。

```py
from random import seed
from random import randrange

# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(train))]
	return predicted

seed(1)
train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
test = [[None], [None], [None], [None]]
predictions = zero_rule_algorithm_classification(train, test)
print(predictions)
```

运行此示例进行预测并将其打印到屏幕。正如所料，选择并预测了类值“0”。

```py
['0', '0', '0', '0', '0', '0']
```

现在，让我们看一下回归问题的零规则算法。

#### 回归

回归问题需要预测实际价值。

对实际值的良好默认预测是预测集中趋势。这可以是均值或中位数。

一个好的默认值是使用训练数据中观察到的输出值的平均值（也称为平均值）。

这可能比随机预测具有更低的误差，随机预测将返回任何观察到的输出值。

下面是一个名为 **zero_rule_algorithm_regression（）**的函数。它的工作原理是计算观察到的输出值的平均值。

```py
mean = sum(value) / total values
```

一旦计算出，则对训练数据中的每一行预测平均值。

```py
from random import randrange

# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
	output_values = [row[-1] for row in train]
	prediction = sum(output_values) / float(len(output_values))
	predicted = [prediction for i in range(len(test))]
	return predicted
```

可以使用一个简单的示例测试此功能。

我们可以设计一个小数据集，其中已知平均值为15。

```py
10
15
12
15
18
20

mean = (10 + 15 + 12 + 15 + 18 + 20) / 6
mean = 90 / 6
mean = 15
```

以下是完整的示例。我们期望测试数据集中的4行中的每一行都预测平均值15。

```py
from random import seed
from random import randrange

# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
	output_values = [row[-1] for row in train]
	prediction = sum(output_values) / float(len(output_values))
	predicted = [prediction for i in range(len(test))]
	return predicted

seed(1)
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_algorithm_regression(train, test)
print(predictions)
```

运行该示例计算打印的预测输出值。正如所料，测试数据集中每行的平均值为15。

```py
[15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
```

## 扩展

以下是基线算法的一些扩展，您可能希望将工具作为本教程的扩展进行研究。

*   替代中心趋势，其中预测中位数，模式或其他集中趋势计算而不是平均值。
*   移动平均值用于预测最后n个记录的平均值的时间序列问题。

## 评论

在本教程中，您发现了计算机器学习问题的表现基线的重要性。

你现在知道了：

*   如何实现分类和回归问题的随机预测算法。
*   如何实现分类和回归问题的零规则算法。

**你有什么问题吗？**
在评论中提出您的问题，我会尽力回答。