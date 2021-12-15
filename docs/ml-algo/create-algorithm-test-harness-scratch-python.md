# 如何使用Python从零开始创建算法测试工具

> 原文： [https://machinelearningmastery.com/create-algorithm-test-harness-scratch-python/](https://machinelearningmastery.com/create-algorithm-test-harness-scratch-python/)

我们无法知道哪种算法最适合给定的问题。

因此，我们需要设计一个可用于评估不同机器学习算法的测试工具。

在本教程中，您将了解如何在Python中从零开始开发机器学习算法测试工具。

完成本教程后，您将了解：

*   如何实现训练测试算法测试工具。
*   如何实现k-fold交叉验证算法测试工具。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Create an Algorithm Test Harness From Scratch With Python](img/c4506f436c1db5a203009d0db6942e81.jpg)

如何使用Python从[Scragch]创建算法测试工具
照片由 [Chris Meller](https://www.flickr.com/photos/mellertime/5688957140/) ，保留一些权利。

## 描述

测试工具提供了一种一致的方法来评估数据集上的机器学习算法。

它涉及3个要素：

1.  拆分数据集的重采样方法。
2.  机器学习算法进行评估。
3.  用于评估预测的表现度量。

数据集的加载和准备是必须在使用测试工具之前完成的先决条件步骤。

测试工具必须允许评估不同的机器学习算法，同时数据集，重采样方法和表现测量保持不变。

在本教程中，我们将使用真实数据集演示测试工具。

使用的数据集是Pima Indians糖尿病数据集。它包含768行和9列。文件中的所有值都是数字，特别是浮点值。您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上了解有关此数据集的更多信息（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

零规则算法将作为本教程的一部分进行评估。零规则算法始终预测训练数据集中观察次数最多的类。

## 教程

本教程分为两个主要部分：

1.  训练 - 测试算法测试线束。
2.  交叉验证算法测试工具。

这些测试工具将为您提供在给定的预测建模问题上评估一套机器学习算法所需的基础。

### 1.训练 - 测试算法测试线束

训练测试拆分是一种简单的重采样方法，可用于评估机器学习算法。

因此，它是开发测试工具的良好起点。

我们可以假设先前开发了一个函数来将数据集拆分为训练集和测试集，以及一个函数来评估一组预测的准确率。

我们需要一个可以采用数据集和算法并返回表现分数的函数。

下面是一个名为 **evaluate_algorithm（）**的函数来实现这一点。它需要3个固定的参数，包括数据集，算法函数和训练测试拆分的拆分百分比。

首先，数据集分为训练和测试元素。接下来，制作测试集的副本，并通过将其设置为**无**值来清除每个输出值，以防止算法意外作弊。

作为参数提供的算法是一种函数，该函数期望准备的训练和测试数据集然后做出预测。该算法可能需要额外的配置参数。这是通过在 **evaluate_algorithm（）**函数中使用变量参数 *** args** 并将它们传递给算法函数来处理的。

期望算法函数返回预测列表，一个针对训练数据集中的每一行。通过 **accuracy_metric（）**函数将这些与未修改的测试数据集的实际输出值进行比较。

最后，返回准确率。

```py
# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	accuracy = accuracy_metric(actual, predicted)
	return accuracy
```

评估函数确实做了一些强有力的假设，但如果需要，它们可以很容易地改变。

具体来说，它假定数据集中的最后一行始终是输出值。可以使用不同的列。使用 **accuracy_metric（）**假设问题是一个分类问题，但这可能会改变为回归问题的均方误差。

让我们将这个与一个有效的例子结合起来。

我们将使用Pima Indians糖尿病数据集并评估零规则算法。您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上下载并了解有关糖尿病数据集的更多信息（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

```py
# Train-Test Test Harness
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset into a train and test set
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	accuracy = accuracy_metric(actual, predicted)
	return accuracy

# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

# Test the zero rule algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, split)
print('Accuracy: %.3f%%' % (accuracy))
```

数据集分为60％用于训练模型，40％用于评估模型。

注意零规则算法 **zero_rule_algorithm_classification** 的名称是如何作为参数传递给 **evaluate_algorithm（）**函数的。您可以看到如何使用不同的算法一次又一次地使用此测试工具。

运行上面的示例会打印出模型的准确率。

```py
Accuracy: 67.427%
```

### 2.交叉验证算法测试工具

交叉验证是一种重采样技术，可以对看不见的数据提供更可靠的算法表现估计。

它需要在数据的不同子集上创建和评估k模型，这样计算成本更高。然而，它是评估机器学习算法的黄金标准。

与前一节一样，我们需要创建一个将重采样方法，算法对数据集的评估和表现计算方法联系在一起的函数。

与上述不同，必须多次在数据集的不同子集上评估算法。这意味着我们需要在 **evaluate_algorithm（）**函数中添加其他循环。

下面是一个使用交叉验证实现算法评估的函数。

首先，数据集被分成称为折叠的 **n_folds** 组。

接下来，我们循环给每个折叠提供一个不受训练的机会并用于评估算法。创建折叠列表的副本，并从该列表中删除保留折叠。然后将折叠列表展平为一个长行列表，以匹配训练数据集的算法期望。这是使用 **sum（）**功能完成的。

一旦准备好训练数据集，该循环内的其余功能如上所述。制作测试数据集（折叠）的副本并清除输出值以避免算法意外欺骗。该算法是在训练数据集上准备的，并对测试数据集做出预测。评估预测并将其存储在列表中。

与训练测试算法测试工具不同，返回分数列表，每个交叉验证折叠一个。

```py
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
```

尽管代码稍微复杂且运行速度较慢，但​​此函数可提供更稳健的算法表现估计。

我们可以通过零规则算法将所有这些与糖尿病数据集的完整示例结合在一起。

```py
# Cross Validation Test Harness
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

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

# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

# Test the zero rule algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/len(scores)))
```

总共5个交叉验证折叠被用于评估零规则算法。因此，从 **evaluate_algorithm（）**算法返回5个分数。

运行此示例都会打印这些计算得分列表并打印平均得分。

```py
Scores: [62.091503267973856, 64.70588235294117, 64.70588235294117, 64.70588235294117, 69.28104575163398]
Mean Accuracy: 65.098%
```

您现在可以使用两种不同的测试工具来评估自己的机器学习算法。

## 扩展

本节列出了您可能希望考虑的本教程的扩展。

*   **参数化评估**。传入用于评估预测的函数，使您可以无缝地处理回归问题。
*   **参数化重采样**。传递用于计算重采样分割的函数，允许您在训练测试和交叉验证方法之间轻松切换。
*   **标准偏差分数**。计算标准偏差，以便在使用交叉验证评估算法时了解分数的传播。

**你有没有试过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何从零开始创建测试工具以评估您的机器学习算法。

具体来说，您现在知道：

*   如何实现和使用训练测试算法测试工具。
*   如何实现和使用交叉验证算法测试工具。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。