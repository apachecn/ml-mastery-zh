# 5学习LSTM循环神经网络的简单序列预测问题的例子

> 原文： [https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/](https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/)

序列预测不同于传统的分类和回归问题。

它要求您考虑观察的顺序，并使用具有记忆的长短期记忆（LSTM）循环神经网络等模型，并且可以学习观察之间的任何时间依赖性。

应用LSTM来学习如何在序列预测问题上使用它们是至关重要的，为此，您需要一套明确定义的问题，使您能够专注于不同的问题类型和框架。至关重要的是，您可以建立您对序列预测问题如何不同的直觉，以及如何使用像LSTM这样复杂的模型来解决它们。

在本教程中，您将发现一套5个狭义定义和可扩展的序列预测问题，您可以使用这些问题来应用和了解有关LSTM循环神经网络的更多信息。

完成本教程后，您将了解：

*   简单的记忆任务，用于测试LSTM的学习记忆能力。
*   简单的回声任务，用于测试LSTM的学习时间依赖表现力。
*   用于测试LSTM解释能力的简单算术任务。

让我们开始吧。

![5 Examples of Simple Sequence Prediction Problems for Learning LSTM Recurrent Neural Networks](img/b45c5a73c43228fa44ebdd42dcf41db3.jpg)

5用于学习LSTM循环神经网络的简单序列预测问题示例
照片由 [Geraint Otis Warlow](https://www.flickr.com/photos/gpwarlow/850611221/) ，保留一些权利。

## 教程概述

本教程分为5个部分;他们是：

1.  序列学习问题
2.  价值记忆
3.  回声随机整数
4.  回声随机子序列
5.  序列分类

## 问题的属性

序列问题的设计考虑了一些属性：

*   **缩小**。集中于序列预测的一个方面，例如记忆或函数近似。
*   **可扩展**。在选择的狭隘焦点上或多或少地变得困难。
*   **重新定型**。提出每个问题的两个或更多个框架以支持不同算法学习能力的探索。

我试图提供狭隘的焦点，问题困难和所需的网络架构。

如果您有进一步扩展的想法或类似的精心设计的问题，请在下面的评论中告诉我。

## 1.序列学习问题

在该问题中，生成0.0和1.0之间的连续实数值序列。给定过去值的一个或多个时间步长，模型必须预测序列中的下一个项目。

我们可以直接生成这个序列，如下所示：

```py
from numpy import array

# generate a sequence of real values between 0 and 1.
def generate_sequence(length=10):
	return array([i/float(length) for i in range(length)])

print(generate_sequence())
```

运行此示例将打印生成的序列：

```py
[ 0\. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

这可以被视为记忆挑战，如果在前一时间步骤观察，模型必须预测下一个值：

```py
X (samples),	y
0.0,			0.1
0.1,			0.2
0.2,			0.3
...
```

网络可以记住输入 - 输出对，这很无聊，但会展示网络的功能近似能力。

该问题可以被构造为随机选择的连续子序列作为输入时间步骤和序列中的下一个值作为输出。

```py
X (timesteps),		y
0.4, 0.5, 0.6,		0.7
0.0, 0.2, 0.3,		0.4
0.3, 0.4, 0.5,		0.6
...
```

这将要求网络学习向最后看到的观察添加固定值或记住所生成问题的所有可能子序列。

该问题的框架将被建模为多对一序列预测问题。

这是测试序列学习的原始特征的简单问题。这个问题可以通过多层感知器网络来解决。

## 2.价值记忆

问题是要记住序列中的第一个值，并在序列的末尾重复它。

该问题基于用于在1997年论文[长期短期记忆](http://www.bioinf.jku.at/publications/older/2604.pdf)中证明LSTM的“实验2”。

这可以被视为一步预测问题。

给定序列中的一个值，模型必须预测序列中的下一个值。例如，给定值“0”作为输入，模型必须预测值“1”。

考虑以下两个5个整数的序列：

```py
3, 0, 1, 2, 3
4, 0, 1, 2, 4
```

Python代码将生成两个任意长度的序列。如果您愿意，可以进一步概括。

```py
def generate_sequence(length=5):
	return [i for i in range(length)]

# sequence 1
seq1 = generate_sequence()
seq1[0] = seq1[-1] = seq1[-2]
print(seq1)
# sequence 2
seq1 = generate_sequence()
seq1[0] = seq1[-1]
print(seq1)
```

运行该示例生成并打印上述两个序列。

```py
[3, 1, 2, 3, 3]
[4, 1, 2, 3, 4]
```

可以对整数进行归一化，或者更优选地对一个热编码进行归一化。

这些模式引入了皱纹，因为两个序列之间存在冲突的信息，并且模型必须知道每个一步预测的上下文（例如，它当前正在预测的序列），以便正确地预测每个完整序列。

我们可以看到序列的第一个值重复作为序列的最后一个值。这是指示器为模型提供关于它正在处理的序列的上下文。

冲突是从每个序列中的第二个项目到最后一个项目的过渡。在序列1中，给出“2”作为输入并且必须预测“3”，而在序列2中，给出“2”作为输入并且必须预测“4”。

```py
Sequence 1:

X (samples),	y
...
1,				2
2,				3

Sequence 2:

X (samples),	y
...
1,				2
2,				4
```

这种皱纹对于防止模型记忆每个序列中的每个单步输入 - 输出值对非常重要，因为序列未知模型可能倾向于这样做。

该成帧将被建模为一对一的序列预测问题。

这是多层感知器和其他非循环神经网络无法学习的问题。必须记住多个样本中序列中的第一个值。

这个问题可以被定义为提供除最后一个值之外的整个序列作为输入时间步长并预测最终值。

```py
X (timesteps),		y
3, 0, 1, 2, 		3
4, 0, 1, 2, 		4
```

每个时间步仍然一次显示给网络，但网络必须记住第一个时间步的值。不同的是，网络可以通过时间反向传播更好地了解序列之间和长序列之间的差异。

This framing of the problem would be modeled as a many-to-one sequence prediction problem.

同样，多层感知器无法学习这个问题。

## 3.回声随机整数

在这个问题中，生成随机的整数序列。模型必须记住特定滞后时间的整数，并在序列结束时回显它。

例如，10个整数的随机序列可以是：

```py
5, 3, 2, 1, 9, 9, 2, 7, 1, 6
```

该问题可能被设置为在第5个时间步骤回显该值，在这种情况下为9。

下面的代码将生成随机的整数序列。

```py
from random import randint

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=10):
	return [randint(0, 99) for _ in range(length)]

print(generate_sequence())
```

运行该示例将生成并打印随机序列，例如：

```py
[47, 69, 76, 9, 71, 87, 8, 16, 32, 81]
```

可以对整数进行归一化，但更优选地，可以使用一个热编码。

该问题的简单框架是回显当前输入值。

```py
yhat(t) = f(X(t))
```

例如：

```py
X (timesteps),		y
5, 3, 2, 1, 9,		9
```

这个微不足道的问题可以通过多层感知器轻松解决，并可用于测试线束的校准或诊断。

更具挑战性的问题框架是回显前一时间步的值。

```py
yhat(t) = f(X(t-1))
```

For example:

```py
X (timesteps),		y
5, 3, 2, 1, 9,		1
```

这是多层感知器无法解决的问题。

echo的索引可以进一步推迟，从而对LSTM内存产生更多需求。

与上面的“值记忆”问题不同，每个训练时期都会产生一个新的序列。这将要求模型学习泛化回声解，而不是记忆特定序列或随机数序列。

在这两种情况下，问题都将被建模为多对一序列预测问题。

## 4.回声随机子序列

该问题还涉及生成随机的整数序列。

这个问题要求模型记住并输出输入序列的部分子序列，而不是像前一个问题那样回显单个前一个时间步骤。

最简单的框架将是前一节中的回声问题。相反，我们将专注于序列输出，其中最简单的框架是模型记住并输出整个输入序列。

For example:

```py
X (timesteps),		y
5, 3, 2, 4, 1,		5, 3, 2, 4, 1
```

这可以被建模为多对一序列预测问题，其中输出序列直接在输入序列中的最后一个值的末尾输出。

这也可以被建模为网络为每个输入时间步长输出一个值，例如，一对一的模式。

更具挑战性的框架是输出输入序列的部分连续子序列。

For example:

```py
X (timesteps),		y
5, 3, 2, 4, 1,		5, 3, 2
```

这更具挑战性，因为输入数量与输出数量不匹配。这个问题的多对多模型需要更高级的架构，例如编解码器LSTM。

同样，一个热编码将是优选的，尽管该问题可以被建模为标准化整数值。

## 5.序列分类

该问题被定义为0和1之间的随机值序列。该序列被作为问题的输入，每个时间步提供一个数字。

二进制标签（0或1）与每个输入相关联。输出值均为0.一旦序列中输入值的累积和超过阈值，则输出值从0翻转为1。

使用序列长度的1/4的阈值。

例如，下面是10个输入时间步长（X）的序列：

```py
0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514
```

相应的分类输出（y）将是：

```py
0 0 0 1 1 1 1 1 1 1
```

我们可以用Python实现它。

```py
from random import random
from numpy import array
from numpy import cumsum

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y

X, y = get_sequence(10)
print(X)
print(y)
```

运行该示例会生成随机输入序列，并计算二进制值的相应输出序列。

```py
[ 0.31102339  0.66591885  0.7211718   0.78159441  0.50496384  0.56941485
  0.60775583  0.36833139  0.180908    0.80614878]
[0 0 0 0 1 1 1 1 1 1]
```

这是一个序列分类问题，可以建模为一对一。状态需要解释过去的时间步骤，以正确预测输出序列何时从0翻转到1。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [长期短期记忆](http://www.bioinf.jku.at/publications/older/2604.pdf)，1997年
*   [如何使用Keras](http://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/) 在Python中使用不同的批量大小进行训练和预测
*   [用Python中的长短期内存网络演示内存](http://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/)
*   [如何通过长短期记忆循环神经网络学习回声随机整数](http://machinelearningmastery.com/learn-echo-random-integers-long-short-term-memory-recurrent-neural-networks/)
*   [如何将编解码器LSTM用于随机整数的回波序列](http://machinelearningmastery.com/learn-echo-random-integers-long-short-term-memory-recurrent-neural-networks/)
*   [如何使用Keras开发用于Python序列分类的双向LSTM](http://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)

## 摘要

在本教程中，您发现了一套精心设计的人工序列预测问题，可用于探索LSTM循环神经网络的学习和记忆功能。

具体来说，你学到了：

*   简单的记忆任务，用于测试LSTM的学习记忆能力。
*   简单的回声任务，用于测试LSTM的学习时间依赖表现力。
*   用于测试LSTM解释能力的简单算术任务。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。