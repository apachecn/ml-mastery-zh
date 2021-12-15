# 如何在Python中从零开始实现感知机算法

> 原文： [https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)

Perceptron算法是最简单的人工神经网络。

它是单个神经元的模型，可用于两类分类问题，并为以后开发更大的网络提供基础。

在本教程中，您将了解如何使用Python从零开始实现Perceptron算法。

完成本教程后，您将了解：

*   如何训练Perceptron的网络权重。
*   如何使用Perceptron做出预测。
*   如何针对真实世界的分类问题实现Perceptron算法。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement The Perceptron Algorithm From Scratch In Python](img/99556d3de6e9574ec7ac5641fa5490cb.jpg)

如何在Python中从零开始实现感知机算法
照片由 [Les Haines](https://www.flickr.com/photos/leshaines123/5835538829/) ，保留一些权利。

## 描述

本节简要介绍Perceptron算法和我们稍后将应用它的Sonar数据集。

### 感知机算法

Perceptron的灵感来自于称为神经元的单个神经细胞的信息处理。

神经元通过其树突接受输入信号，树突将电信号传递到细胞体。

以类似的方式，Perceptron从训练数据的示例接收输入信号，我们对这些训练数据进行加权并组合成称为激活的线性方程。

```py
activation = sum(weight_i * x_i) + bias
```

然后使用诸如步进传递函数的传递函数将激活变换为输出值或预测。

```py
prediction = 1.0 if activation >= 0.0 else 0.0
```

通过这种方式，Perceptron是一个分类算法，用于解决两个类（0和1）的问题，其中线性方程（如超平面）可以用来分离这两个类。

它与线性回归和逻辑回归密切相关，以类似的方式做出预测（例如输入的加权和）。

必须使用随机梯度下降从训练数据中估计Perceptron算法的权重。

### 随机梯度下降

梯度下降是通过遵循成本函数的梯度来最小化函数的过程。

这包括了解成本的形式以及衍生物，以便从给定的点知道梯度并且可以在该方向上移动，例如，向下走向最小值。

在机器学习中，我们可以使用一种技术来评估和更新称为随机梯度下降的每次迭代的权重，以最小化模型对我们的训练数据的误差。

此优化算法的工作方式是每个训练实例一次显示给模型一个。该模型对训练实例做出预测，计算误差并更新模型以减少下一次预测的误差。

此过程可用于在模型中查找权重集，从而导致训练数据上模型的最小误差。

对于Perceptron算法，每次迭代使用以下等式更新权重（ **w** ）：

```py
w = w + learning_rate * (expected - predicted) * x
```

**w** 的权重被优化， **learning_rate** 是你必须配置的学习率（例如0.01），**（预期 - 预测）**是预测误差归因于重量和 **x** 的训练数据的模型是输入值。

### 声纳数据集

我们将在本教程中使用的数据集是Sonar数据集。

这是一个描述声纳啁啾返回弹跳不同服务的数据集。 60个输入变量是不同角度的回报强度。这是一个二分类问题，需要一个模型来区分岩石和金属圆柱。

这是一个众所周知的数据集。所有变量都是连续的，通常在0到1的范围内。因此我们不必对输入数据进行标准化，这通常是Perceptron算法的一个好习惯。输出变量是我的字符串“M”和摇滚的“R”，需要将其转换为整数1和0。

通过预测数据集（M或矿）中具有最多观测值的类，零规则算法可以实现53％的准确度。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))中了解有关此数据集的更多信息。您可以免费下载数据集并将其放在工作目录中，文件名为 **sonar.all-data.csv** 。

## 教程

本教程分为3个部分：

1.  做出预测。
2.  训练网络权重。
3.  声纳数据集建模。

这些步骤将为您提供实施Perceptron算法并将其应用于您自己的分类预测建模问题的基础。

### 1.做出预测

第一步是开发一个可以做出预测的功能。

这在随机梯度下降中的候选权重值的评估中以及在模型完成之后并且我们希望开始对测试数据或新数据做出预测时都需要。

下面是一个名为 **predict（）**的函数，它预测给定一组权重的行的输出值。

第一个权重始终是偏差，因为它是独立的，不负责特定的输入值。

```py
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
```

我们可以设计一个小数据集来测试我们的预测函数。

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

我们还可以使用先前准备的权重来对此数据集做出预测。

综合这些，我们可以测试下面的 **predict（）**函数。

```py
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# test predictions
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
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
for row in dataset:
	prediction = predict(row, weights)
	print("Expected=%d, Predicted=%d" % (row[-1], prediction))
```

有两个输入值（ **X1** 和 **X2** ）和三个权重值（**偏差**， **w1** 和 **w2** ）。我们为此问题建模的激活方程是：

```py
activation = (w1 * X1) + (w2 * X2) + bias
```

或者，我们手动选择具体的重量值：

```py
activation = (0.206 * X1) + (-0.234 * X2) + -0.1
```

运行此函数，我们得到与预期输出（ **y** ）值匹配的预测。

```py
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=0, Predicted=0
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
Expected=1, Predicted=1
```

现在我们准备实施随机梯度下降来优化我们的重量值。

### 2.训练网络权重

我们可以使用随机梯度下降来估计训练数据的权重值。

随机梯度下降需要两个参数：

*   **学习率**：用于限制每次更新时每个重量的校正量。
*   **Epochs** ：更新体重时运行训练数据的次数。

这些以及训练数据将是该函数的参数。

我们需要在函数中执行3个循环：

1.  循环每个时代。
2.  循环遍历训练数据中的每一行以获得一个迭代。
3.  循环遍历每个权重并将其更新为一个迭代中的一行。

如您所见，我们更新训练数据中每一行的每个权重，每个时期。

权重根据模型产生的错误进行更新。该误差被计算为预期输出值与用候选权重进行的预测之间的差异。

每个输入属性都有一个权重，这些权重以一致的方式更新，例如：

```py
w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)
```

偏差以类似的方式更新，除非没有输入，因为它与特定输入值无关：

```py
bias(t+1) = bias(t) + learning_rate * (expected(t) - predicted(t))
```

现在我们可以将所有这些放在一起。下面是一个名为 **train_weights（）**的函数，它使用随机梯度下降计算训练数据集的权重值。

```py
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
```

您可以看到我们还跟踪每个时期的平方误差（正值）的总和，以便我们可以在每个外部循环中打印出一条好消息。

我们可以在上面的同样小的人为数据集上测试这个函数。

```py
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

# Calculate weights
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
l_rate = 0.1
n_epoch = 5
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
```

我们使用0.1的学习率并且仅将模型训练5个时期，或者将权重的5次暴露训练到整个训练数据集。

运行该示例在每个迭代打印一条消息，其中包含该迭代和最终权重集的总和平方误差。

```py
>epoch=0, lrate=0.100, error=2.000
>epoch=1, lrate=0.100, error=1.000
>epoch=2, lrate=0.100, error=0.000
>epoch=3, lrate=0.100, error=0.000
>epoch=4, lrate=0.100, error=0.000
[-0.1, 0.20653640140000007, -0.23418117710000003]
```

您可以通过算法快速了解问题是如何学习的。

现在，让我们将这个算法应用于真实数据集。

### 3.对声纳数据集进行建模

在本节中，我们将使用Sonar数据集上的随机梯度下降来训练Perceptron模型。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **sonar.all-data.csv** 。

首先加载数据集，将字符串值转换为数字，并将输出列从字符串转换为0到1的整数值。这可以通过辅助函数 **load_csv（）**， **str_column_to_float（ ）**和 **str_column_to_int（）**加载和准备数据集。

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型误差。分类精度将用于评估每个模型。这些行为在 **cross_validation_split（）**， **accuracy_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们将使用上面创建的 **predict（）和** **train_weights（）**函数来训练模型和新的 **perceptron（）**函数将它们绑定在一起。

以下是完整的示例。

```py
# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader

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

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

k值为3用于交叉验证，每次迭代时评估每个折叠208/3 = 69.3或略低于70的记录。通过一些实验选择了0.1和500个训练时期的学习率。

您可以尝试自己的配置，看看是否可以打败我的分数。

运行此示例将打印3个交叉验证折叠中每个折叠的分数，然后打印平均分类精度。

我们可以看到，如果我们仅使用零规则算法预测大多数类，则准确度约为72％，高于仅超过50％的基线值。

```py
Scores: [76.81159420289855, 69.56521739130434, 72.46376811594203]
Mean Accuracy: 72.947%
```

## 扩展

本节列出了您可能希望考虑探索的本教程的扩展。

*   **调整示例**。调整学习率，时期数甚至数据准备方法以获得数据集上的改进分数。
*   **批随机梯度下降**。更改随机梯度下降算法以在每个时期累积更新，并且仅在时期结束时批量更新权重。
*   **其他回归问题**。将该技术应用于UCI机器学习库中的其他分类问题。

**你有没有探索过这些扩展？**
请在下面的评论中告诉我。

## 评论

在本教程中，您了解了如何使用Python从零开始使用随机梯度下降来实现Perceptron算法。

你学到了

*   如何预测二分类问题。
*   如何使用随机梯度下降来优化一组权重。
*   如何将该技术应用于真实的分类预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。