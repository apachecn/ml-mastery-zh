# 如何在 Python 中从零开始实现反向传播算法

> 原文： [https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

反向传播算法是经典的前馈人工神经网络。

这种技术仍然用于训练大型[深度学习](http://machinelearningmastery.com/what-is-deep-learning/)网络。

在本教程中，您将了解如何使用 Python 从零开始实现反向传播算法。

完成本教程后，您将了解：

*   如何向前传播输入以计算输出。
*   如何反向传播错误并训练网络。
*   如何将反向传播算法应用于实际预测性建模问题。

让我们开始吧。

*   **2016 年 11 月更新**：修复了 activate（）函数中的错误。谢谢 Alex！
*   **2017 年 1 月更新**：将 cross_validation_split（）中的 fold_size 计算更改为始终为整数。修复了 Python 3 的问题。
*   **2017 年 1 月更新**：更新了 update_weights（）中的小错误。谢谢 Tomasz！
*   **Update Apr / 2018** ：添加了直接链接到 CSV 数据集。
*   **更新 Aug / 2018** ：经过测试和更新，可与 Python 3.6 配合使用。

![How to Implement the Backpropagation Algorithm From Scratch In Python](img/d97e39d3b77378dff36499a1877a9015.jpg)

如何在 Python 中从零开始实现反向传播算法
照片由 [NICHD](https://www.flickr.com/photos/nichd/21086425615/) ，保留一些权利。

## 描述

本节简要介绍了我们将在本教程中使用的反向传播算法和小麦种子数据集。

### 反向传播算法

反向传播算法是一种来自人工神经网络领域的多层前馈网络的监督学习方法。

前馈神经网络受到一个或多个神经细胞（称为神经元）的信息处理的启发。神经元通过其树突接受输入信号，树突将电信号传递到细胞体。轴突将信号传递给突触，突触是细胞轴突与其他细胞树突的连接。

反向传播方法的原理是通过修改输入信号的内部权重来模拟给定函数，以产生预期的输出信号。使用监督学习方法训练系统，其中系统输出和已知预期输出之间的误差被呈现给系统并用于修改其内部状态。

从技术上讲，反向传播算法是一种在多层前馈神经网络中训练权重的方法。因此，它要求网络结构由一个或多个层定义，其中一个层完全连接到下一层。标准网络结构是一个输入层，一个隐藏层和一个输出层。

反向传播可用于分类和回归问题，但我们将重点关注本教程中的分类。

在分类问题中，当网络在每个类值的输出层中具有一个神经元时，实现最佳结果。例如，具有 A 和 B 类值的 2 类或二分类问题。这些预期输出必须转换为二进制向量，每个类值具有一列。例如分别为 A 和 B 的[1,0]和[0,1]。这称为单热编码。

### 小麦种子数据集

种子数据集涉及从不同品种的小麦给出测量种子的物种的预测。

有 201 条记录和 7 个数字输入变量。这是一个有 3 个输出类的分类问题。每个数字输入值的比例变化，因此可能需要一些数据标准化以用于加权输入的算法，如反向传播算法。

下面是数据集的前 5 行的示例。

```py
15.26,14.84,0.871,5.763,3.312,2.221,5.22,1
14.88,14.57,0.8811,5.554,3.333,1.018,4.956,1
14.29,14.09,0.905,5.291,3.337,2.699,4.825,1
13.84,13.94,0.8955,5.324,3.379,2.259,4.805,1
16.14,14.99,0.9034,5.658,3.562,1.355,5.175,1
```

使用预测最常见类值的零规则算法，问题的基线准确度为 28.095％。

您可以从 [UCI 机器学习库](http://archive.ics.uci.edu/ml/datasets/seeds)了解更多信息并下载种子数据集。

下载种子数据集并将其放入当前工作目录，文件名为 **seeds_dataset.csv** 。

数据集采用制表符分隔格式，因此您必须使用文本编辑器或电子表格程序将其转换为 CSV。

更新，直接下载 CSV 格式的数据集：

*   [下载小麦种子数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)

## 教程

本教程分为 6 个部分：

1.  初始化网络。
2.  向前传播。
3.  返回传播错误。
4.  训练网络。
5.  预测。
6.  种子数据集案例研究。

这些步骤将为您提供从零开始实现反向传播算法所需的基础，并将其应用于您自己的预测性建模问题。

### 1.初始化网络

让我们从简单的事情开始，创建一个可供训练的新网络。

每个神经元都有一组需要维持的权重。每个输入连接一个重量和偏置的额外重量。我们需要在训练期间为神经元存储其他属性，因此我们将使用字典来表示每个神经元并通过名称存储属性，例如权重的'**权重**'。

网络按层组织。输入层实际上只是我们训练数据集的一行。第一个真实层是隐藏层。接下来是输出层，每个类值都有一个神经元。

我们将层组织为字典数组，并将整个网络视为一个层数组。

最好将网络权重初始化为小的随机数。在这种情况下，我们将使用 0 到 1 范围内的随机数。

下面是一个名为 **initialize_network（）**的函数，它创建了一个可供训练的新神经网络。它接受三个参数，输入数量，隐藏层中的神经元数量和输出数量。

您可以看到，对于隐藏层，我们创建 **n_hidden** 神经元，隐藏层中的每个神经元都有 **n_inputs + 1** 权重，一个用于数据集中的每个输入列，另一个用于偏见。

您还可以看到连接到隐藏层的输出层具有 **n_outputs** 神经元，每个神经元具有 **n_hidden + 1** 权重。这意味着输出层中的每个神经元都连接到隐藏层中每个神经元（具有权重）。

```py
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
```

让我们测试一下这个功能。下面是一个创建小型网络的完整示例。

```py
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)
```

运行该示例，您可以看到代码逐个打印出每个层。您可以看到隐藏层有一个具有 2 个输入权重和偏差的神经元。输出层有 2 个神经元，每个神经元有 1 个权重加上偏差。

```py
[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
```

现在我们知道了如何创建和初始化网络，让我们看看如何使用它来计算输出。

### 2.前向传播

我们可以通过在每层传播输入信号直到输出层输出其值来计算神经网络的输出。

我们将此称为前向传播。

我们需要在训练期间生成需要纠正的预测技术，这是我们在训练网络以对新数据做出预测后需要的方法。

我们可以将传播分解为三个部分：

1.  神经元激活。
2.  神经元转移。
3.  前向传播。

#### 2.1。神经元激活

第一步是计算给定输入的一个神经元的激活。

输入可以是我们的训练数据集中的一行，如隐藏层的情况。在输出层的情况下，它也可以是隐藏层中每个神经元的输出。

神经元激活计算为输入的加权和。很像线性回归。

```py
activation = sum(weight_i * input_i) + bias
```

**权重**是网络权重，**输入**是输入， **i** 是权重或输入的指标，**偏差**是没有输入的特殊权重（或者你可以认为输入总是为 1.0）。

下面是一个名为 **activate（）**的函数的实现。您可以看到该函数假定偏差是权重列表中的最后一个权重。这有助于此处以及稍后使代码更易于阅读。

```py
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```

现在，让我们看看如何使用神经元激活。

#### 2.2。神经元转移

一旦神经元被激活，我们需要转移激活以查看神经元输出实际是什么。

可以使用不同的传递函数。传统上使用 [sigmoid 激活函数](https://en.wikipedia.org/wiki/Sigmoid_function)，但您也可以使用 tanh（[双曲正切](https://en.wikipedia.org/wiki/Hyperbolic_function)）函数来传输输出。最近，[整流器传递函数](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))已经在大型深度学习网络中流行。

S 形激活函数看起来像 S 形，它也称为逻辑函数。它可以取任何输入值并在 S 曲线上产生 0 到 1 之间的数字。它也是一个函数，我们可以很容易地计算出反向传播误差后我们将需要的导数（斜率）。

我们可以使用 sigmoid 函数传递激活函数，如下所示：

```py
output = 1 / (1 + e^(-activation))
```

其中 **e** 是自然对数的基数（[欧拉数](https://en.wikipedia.org/wiki/E_(mathematical_constant))）。

下面是一个名为 **transfer（）**的函数，它实现了 sigmoid 方程。

```py
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```

现在我们已经有了它们，让我们看看它们是如何被使用的。

#### 2.3。前向传播

向前传播输入很简单。

我们通过网络的每一层计算每个神经元的输出。来自一层的所有输出都成为下一层神经元的输入。

下面是一个名为 **forward_propagate（）**的函数，它使用我们的神经网络实现数据集中一行数据的前向传播。

您可以看到神经元的输出值存储在神经元中，名称为“**输出**”。您还可以看到我们收集名为 **new_inputs** 的数组中的层的输出，该数组成为数组**输入**，并用作后续层的输入。

该函数返回最后一层的输出，也称为输出层。

```py
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```

让我们将所有这些部分放在一起，测试我们网络的前向传播。

我们定义我们的网络内联一个隐藏的神经元，需要 2 个输入值和一个带有两个神经元的输出层。

```py
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)
```

运行该示例会传播输入模式[1,0]并生成打印的输出值。因为输出层有两个神经元，所以我们得到两个数字的列表作为输出。

实际输出值现在只是无意义，但接下来，我们将开始学习如何使神经元中的权重更有用。

```py
[0.6629970129852887, 0.7253160725279748]
```

### 3.返回传播错误

反向传播算法以训练权重的方式命名。

在预期输出和从网络传播的输出之间计算误差。然后，这些错误通过网络从输出层向后传播到隐藏层，为错误分配责任并随时更新权重。

反向传播误差的数学基础是微积分，但我们将在本节中保持高水平，并关注计算的内容以及计算采用这种特定形式的方式而不是为什么。

这部分分为两部分。

1.  转移衍生品。
2.  错误反向传播。

#### 3.1。转移衍生品

给定神经元的输出值，我们需要计算它的斜率。

我们使用 sigmoid 传递函数，其导数可以计算如下：

```py
derivative = output * (1.0 - output)
```

下面是一个名为 **transfer_derivative（）**的函数，它实现了这个等式。

```py
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
```

现在，让我们看看如何使用它。

#### 3.2。错误反向传播

第一步是计算每个输出神经元的误差，这将使我们的误差信号（输入）向后传播通过网络。

给定神经元的误差可以如下计算：

```py
error = (expected - output) * transfer_derivative(output)
```

**预期**是神经元的预期输出值，**输出**是神经元的输出值， **transfer_derivative（）**计算神经元输出值的斜率，如上所示。

此错误计算用于输出层中的神经元。期望值是类值本身。在隐藏层中，事情有点复杂。

隐藏层中神经元的误差信号被计算为输出层中每个神经元的加权误差。想象一下错误沿输出层的权重返回到隐藏层中的神经元。

累积反向传播的误差信号，然后用于确定隐藏层中神经元的误差，如下所示：

```py
error = (weight_k * error_j) * transfer_derivative(output)
```

**error_j** 是输出层中 **j** 神经元的误差信号， **weight_k** 是连接 **k** 神经元的权重到当前的神经元和输出是当前神经元的输出。

下面是一个名为 **backward_propagate_error（）**的函数，它实现了这个过程。

您可以看到为每个神经元计算的误差信号以名称“delta”存储。您可以看到网络层以相反的顺序迭代，从输出开始并向后工作。这确保了输出层中的神经元具有首先计算的“delta”值，隐藏层中的神经元可以在随后的迭代中使用。我选择名称“delta”来反映错误对神经元的改变（例如，权重增量）。

您可以看到隐藏层中神经元的误差信号是从输出层中的神经元累积的，其中隐藏的神经元数 **j** 也是输出层**神经元中神经元重量的指数[ '权重'] [j]** 。

```py
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```

让我们将所有部分放在一起，看看它是如何工作的。

我们定义一个具有输出值的固定神经网络，并反向传播预期的输出模式。下面列出了完整的示例。

```py
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)
```

运行该示例在错误的反向传播完成后打印网络。您可以看到计算错误值并将其存储在输出层和隐藏层的神经元中。

```py
[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]
```

现在让我们使用错误的反向传播来训练网络。

### 4.训练网络

使用随机梯度下降训练网络。

这涉及将训练数据集暴露给网络以及向前传播输入的每行数据的多次迭代，反向传播错误并更新网络权重。

这部分分为两部分：

1.  更新权重。
2.  训练网络。

#### 4.1。更新权重

一旦通过上述反向传播方法计算网络中每个神经元的误差，就可以使用它们来更新权重。

网络权重更新如下：

```py
weight = weight + learning_rate * error * input
```

当**权重**是给定权重时， **learning_rate** 是您必须指定的参数，**错误**是由神经元和**的反向传播程序计算的误差 input** 是导致错误的输入值。

除了没有输入项，或者输入是 1.0 的固定值之外，可以使用相同的程序来更新偏差权重。

学习率控制改变重量以校正错误的程度。例如，值为 0.1 将更新可能更新量的 10％的权重。较小的学习率是优选的，导致在大量训练迭代中学习较慢。这增加了网络在所有层上找到一组良好权重的可能性，而不是最小化误差的最快权重集（称为早熟收敛）。

下面是一个名为 **update_weights（）**的函数，它在给定输入数据行，学习率的情况下更新网络的权重，并假设已经执行了前向和后向传播。

请记住，输出层的输入是隐藏层的输出集合。

```py
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
```

现在我们知道如何更新网络权重，让我们看看我们如何重复这样做。

#### 4.2。训练网络

如上所述，使用随机梯度下降来更新网络。

这涉及首先循环固定数量的时期并且在每个时期内更新训练数据集中的每一行的网络。

由于针对每种训练模式进行了更新，因此这种类型的学习称为在线学习。如果在更新权重之前在迭代中累积了错误，则称为批量学习或批量梯度下降。

下面是一个函数，它利用给定的训练数据集，学习率，固定的迭代数和预期的输出值数来实现已经初始化的神经网络的训练。

预期的输出值数量用于将训练数据中的类值转换为单热编码。这是一个二进制向量，每个类值有一列，以匹配网络的输出。这是计算输出层的误差所必需的。

您还可以看到预期输出和网络输出之间的总和平方误差在每个时期累积并打印。这有助于创建网络学习和改进每个时代的痕迹。

```py
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```

我们现在拥有训练网络的所有部分。我们可以汇总一个示例，其中包括我们目前所见的所有内容，包括网络初始化和在小型数据集上训练网络。

下面是一个小型人为的数据集，我们可以用它来测试我们的神经网络的训练。

```py
X1			X2			Y
2.7810836		2.550537003		0
1.465489372		2.362125076		0
3.396561688		4.400293529		0
1.38807019		1.850220317		0
3.06407232		3.005305973		0
7.627531214		2.759262235		1
5.332441248		2.088626775		1
6.922596716		1.77106367		1
8.675418651		-0.242068655		1
7.673756466		3.508563011		1
```

以下是完整的示例。我们将在隐藏层中使用 2 个神经元。这是一个二分类问题（2 个类），因此输出层中将有两个神经元。该网络将被训练 20 个时代，学习率为 0.5，这很高，因为我们正在训练如此少的迭代。

```py
from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)
```

运行该示例首先打印每个训练时期的总和平方误差。我们可以看到这个错误的趋势随着每个时期而减少。

一旦经过训练，就会打印网络，显示学习的重量。网络中还有输出和 delta 值，可以忽略。如果需要，我们可以更新我们的训练功能以删除这些数据。

```py
>epoch=0, lrate=0.500, error=6.350
>epoch=1, lrate=0.500, error=5.531
>epoch=2, lrate=0.500, error=5.221
>epoch=3, lrate=0.500, error=4.951
>epoch=4, lrate=0.500, error=4.519
>epoch=5, lrate=0.500, error=4.173
>epoch=6, lrate=0.500, error=3.835
>epoch=7, lrate=0.500, error=3.506
>epoch=8, lrate=0.500, error=3.192
>epoch=9, lrate=0.500, error=2.898
>epoch=10, lrate=0.500, error=2.626
>epoch=11, lrate=0.500, error=2.377
>epoch=12, lrate=0.500, error=2.153
>epoch=13, lrate=0.500, error=1.953
>epoch=14, lrate=0.500, error=1.774
>epoch=15, lrate=0.500, error=1.614
>epoch=16, lrate=0.500, error=1.472
>epoch=17, lrate=0.500, error=1.346
>epoch=18, lrate=0.500, error=1.233
>epoch=19, lrate=0.500, error=1.132
[{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': -0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': 0.0026279652850863837}]
[{'weights': [2.515394649397849, -0.3391927502445985, -0.9671565426390275], 'output': 0.23648794202357587, 'delta': -0.04270059278364587}, {'weights': [-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 'output': 0.7790535202438367, 'delta': 0.03803132596437354}]
```

一旦网络被训练，我们需要使用它来做出预测。

### 5.预测

使用训练有素的神经网络做出预测很容易。

我们已经看到了如何向前传播输入模式以获得输出。这就是我们做出预测所需要做的。我们可以直接使用输出值本身作为属于每个输出类的模式的概率。

将此输出转换为清晰的类预测可能更有用。我们可以通过选择具有更大概率的类值来做到这一点。这也称为 [arg max 函数](https://en.wikipedia.org/wiki/Arg_max)。

下面是一个名为 **predict（）**的函数，它实现了这个过程。它返回网络输出中具有最大概率的索引。它假定类值已从 0 开始转换为整数。

```py
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
```

我们可以将它与上面的代码一起用于前向传播输入，并使用我们的小型设计数据集来测试使用已经训练过的网络做出预测。该示例对从上一步骤训练的网络进行硬编码。

下面列出了完整的示例。

```py
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
```

运行该示例将打印训练数据集中每条记录的预期输出，然后是网络进行的清晰预测。

它表明网络在这个小数据集上达到了 100％的准确率。

```py
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
```

现在我们准备将反向传播算法应用于现实世界数据集。

### 6.小麦种子数据集

本节将 Backpropagation 算法应用于小麦种子数据集。

第一步是加载数据集并将加载的数据转换为我们可以在神经网络中使用的数字。为此我们将使用辅助函数 **load_csv（）**来加载文件， **str_column_to_float（）**将字符串数转换为浮点数， **str_column_to_int（）**转换​​为 class 列到整数值。

输入值的比例不同，需要归一化到 0 和 1 的范围。通常的做法是将输入值标准化为所选传递函数的范围，在这种情况下，输出 0 到 1 之间的值的 sigmoid 函数。 **dataset_minmax（）**和 **normalize_dataset（）**辅助函数用于标准化输入值。

我们将使用 5 倍折叠交叉验证来评估算法。这意味着每个折叠中将有 201/5 = 40.2 或 40 个记录。我们将使用辅助函数 **evaluate_algorithm（）**来评估具有交叉验证的算法和 **accuracy_metric（）**来计算预测的准确率。

开发了一个名为 **back_propagation（）**的新功能来管理反向传播算法的应用，首先初始化网络，在训练数据集上训练它，然后使用训练好的网络对测试数据集做出预测。

The complete example is listed below.

```py
# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

构建了隐藏层中有 5 个神经元，输出层中有 3 个神经元的网络。该网络训练了 500 个时代，学习率为 0.3。通过一些试验和错误发现了这些参数，但您可以做得更好。

运行该示例打印每个折叠的平均分类准确度以及所有折叠的平均表现。

您可以看到反向传播和所选配置实现了大约 93％的平均分类精度，这明显优于零精度算法，其精确度略高于 28％。

```py
Scores: [92.85714285714286, 92.85714285714286, 97.61904761904762, 92.85714285714286, 90.47619047619048]
Mean Accuracy: 93.333%
```

## 扩展

本节列出了您可能希望探索的教程的扩展。

*   **调谐算法参数**。尝试更长或更短的训练更大或更小的网络。看看你是否可以在种子数据集上获得更好的表现。
*   **其他方法**。尝试不同的权重初始化技术（如小随机数）和不同的传递函数（如 tanh）。
*   **更多层**。添加对更多隐藏层的支持，其训练方式与本教程中使用的一个隐藏层相同。
*   **回归**。更改网络，使输出层中只有一个神经元，并预测实际值。选择回归数据集进行练习。线性传递函数可以用于输出层中的神经元，或者所选数据集的输出值可以缩放到 0 和 1 之间的值。
*   **批量梯度下降**。将训练程序从在线更改为批量梯度下降，并仅在每个时期结束时更新权重。

**你有没有试过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何从零开始实现 Backpropagation 算法。

具体来说，你学到了：

*   如何转发传播输入以计算网络输出。
*   如何反向传播错误并更新网络权重。
*   如何将反向传播算法应用于现实世界数据集。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。