# 如何用Python从头开始实现堆栈泛化（Stacking）

> 原文： [https://machinelearningmastery.com/implementing-stacking-scratch-python/](https://machinelearningmastery.com/implementing-stacking-scratch-python/)

#### 使用Python从头开始编写堆栈集合，循序渐进。

集合方法是提高机器学习问题预测表现的绝佳方法。

堆叠泛化或堆叠是一种集合技术，它使用新模型来学习如何最佳地组合来自数据集上训练的两个或更多模型的预测。

在本教程中，您将了解如何在Python中从头开始实现堆叠。

完成本教程后，您将了解：

*   如何学习在数据集上组合多个模型的预测。
*   如何将叠加泛化应用于实际预测建模问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Implementing Stacking From Scratch With Python](img/2960d608a0641f5cd292c91a7669c364.jpg)

如何使用Python从头开始实现堆叠
[Kiran Foster](https://www.flickr.com/photos/rueful/7885846128/) 的照片，保留一些权利。

## 描述

本节简要概述了本教程中使用的Stacked Generalization算法和Sonar数据集。

### 叠加泛化算法

堆叠泛化或堆叠是一种集合算法，其中训练新模型以组合来自已经训练的两个或更多模型或您的数据集的预测。

来自现有模型或子模型的预测使用新模型组合，并且因为这种堆叠通常被称为混合，因为来自子模型的预测被混合在一起。

通常使用简单的线性方法将子模型的预测（例如简单平均或投票）与使用线性回归或逻辑回归的加权和相结合。

将预测结合起来的模型必须掌握问题的技巧，但不需要是最好的模型。这意味着您不需要专心调整子模型，只要模型显示出优于基线预测的优势。

子模型产生不同的预测非常重要，即所谓的不相关预测。当组合的预测都是熟练的，但以不同的方式熟练时，堆叠效果最佳。这可以通过使用使用非常不同的内部表示（与实例相比的树）和/或在训练数据的不同表示或投影上训练的模型的算法来实现。

在本教程中，我们将考虑采用两个非常不同且未调整的子模型，并将它们的预测与简单的逻辑回归算法相结合。

### 声纳数据集

我们将在本教程中使用的数据集是Sonar数据集。

这是一个描述声纳啁啾返回从不同表面反弹的数据集。 60个输入变量是不同角度的回报强度。这是一个二元分类问题，需要一个模型来区分岩石和金属圆柱。共有208个观测结果。

这是一个众所周知的数据集。所有变量都是连续的，通常在0到1的范围内。输出变量是我的字符串“M”和摇滚的“R”，需要将其转换为整数1和0。

通过预测数据集（M或矿）中具有最多观测值的类，零规则算法可以实现约53％的准确度。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))中了解有关此数据集的更多信息。

免费下载数据集并将其放在工作目录中，文件名为 **sonar.all-data.csv** 。

## 教程

本教程分为3个步骤：

1.  子模型和聚合器。
2.  结合预测。
3.  声纳数据集案例研究。

这些步骤为您在自己的预测建模问题上理解和实现堆叠提供了基础。

### 1.子模型和聚合器

我们将使用两个模型作为子模型进行堆叠，使用线性模型作为聚合器模型。

这部分分为3个部分：

1.  子模型＃1：k-Nearest Neighbors。
2.  子模型＃2：感知器。
3.  聚合器模型：Logistic回归。

每个模型将根据用于训练模型的函数和用于做出预测的函数来描述。

#### 1.1子模型＃1：k-最近邻居

k-Nearest Neighbors算法或kNN使用整个训练数据集作为模型。

因此，训练模型涉及保留训练数据集。下面是一个名为 **knn_model（）**的函数。

```py
# Prepare the kNN model
def knn_model(train):
	return train
```

做出预测涉及在训练数据集中查找k个最相似的记录并选择最常见的类值。欧几里德距离函数用于计算训练数据集中新行数据和行之间的相似性。

以下是涉及对kNN模型做出预测的这些辅助函数。函数 **euclidean_distance（）**计算两行数据之间的距离， **get_neighbors（）**定位训练数据集中所有邻居的新行数据和 **knn_predict（）** 从邻居做出新行数据的预测。

```py
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with kNN
def knn_predict(model, test_row, num_neighbors=2):
	neighbors = get_neighbors(model, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
```

您可以看到邻居数（k）设置为2作为 **knn_predict（）**函数的默认参数。这个数字是经过一些试验和错误选择的，没有调整。

现在我们已经有了kNN模型的构建块，让我们来看看Perceptron算法。

#### 1.2子模型＃2：感知器

Perceptron算法的模型是从训练数据中学习的一组权重。

为了训练权重，需要对训练数据进行许多预测以便计算误差值。因此，模型训练和预测都需要预测功能。

下面是实现Perceptron算法的辅助函数。 **perceptron_model（）**函数在训练数据集上训练Perceptron模型， **perceptron_predict（）**用于对一行数据做出预测。

```py
# Make a prediction with weights
def perceptron_predict(model, row):
	activation = model[0]
	for i in range(len(row)-1):
		activation += model[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = perceptron_predict(weights, row)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
```

**perceptron_model（）**模型将学习率和训练时期数指定为默认参数。同样，这些参数是通过一些试验和错误选择的，但没有在数据集上进行调整。

我们现在已经实现了两个子模型，让我们看一下实现聚合器模型。

#### 1.3聚合器模型：Logistic回归

与Perceptron算法一样，Logistic回归使用一组称为系数的权重作为模型的表示。

与Perceptron算法一样，通过迭代地对训练数据做出预测并更新它们来学习系数。

以下是用于实现逻辑回归算法的辅助函数。 **logistic_regression_model（）**函数用于训练训练数据集上的系数， **logistic_regression_predict（）**用于对一行数据做出预测。

```py
# Make a prediction with coefficients
def logistic_regression_predict(model, row):
	yhat = model[0]
	for i in range(len(row)-1):
		yhat += model[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = logistic_regression_predict(coef, row)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
```

**logistic_regression_model（）**将学习率和时期数定义为默认参数，并且与其他算法一样，这些参数在一些试验和错误中被发现并且未被优化。

现在我们已经实现了子模型和聚合器模型，让我们看看如何组合多个模型的预测。

### 2.结合预测

对于机器学习算法，学习如何组合预测与从训练数据集学习非常相似。

可以根据子模型的预测构建新的训练数据集，如下所示：

*   每行代表训练数据集中的一行。
*   第一列包含由第一个子模型制作的训练数据集中每一行的预测，例如k-Nearest Neighbors。
*   第二列包含由第二个子模型制作的训练数据集中每一行的预测，例如Perceptron算法。
*   第三列包含训练数据集中行的预期输出值。

下面是构造的堆叠数据集的外观设计示例：

```py
kNN,	Per,	Y
0,	0	0
1,	0	1
0,	1	0
1,	1	1
0,	1	0
```

然后可以在该新数据集上训练机器学习算法，例如逻辑回归。实质上，这种新的元算法学习如何最好地组合来自多个子模型的预测。

下面是一个名为 **to_stacked_row（）**的函数，它实现了为此堆叠数据集创建新行的过程。

该函数将模型列表作为输入，这些用于做出预测。该函数还将函数列表作为输入，一个函数用于对每个模型做出预测。最后，包括训练数据集中的单行。

一行一列构造一个新行。使用每个模型和训练数据行计算预测。然后将训练数据集行的预期输出值添加为行的最后一列。

```py
# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	stacked_row.append(row[-1])
	return stacked_row
```

在一些预测建模问题上，通过在训练行和子模型做出的预测上训练聚合模型，可以获得更大的提升。

这种改进为聚合器模型提供了训练行中所有数据的上下文，以帮助确定如何以及何时最佳地组合子模型的预测。

我们可以通过聚合训练行（减去最后一列）和上面创建的堆叠行来更新我们的 **to_stacked_row（）**函数以包含它。

以下是实现此改进的 **to_stacked_row（）**函数的更新版本。

```py
# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	stacked_row.append(row[-1])
	return row[0:len(row)-1] + stacked_row
```

在您的问题上尝试两种方法以查看哪种方法效果最好是个好主意。

既然我们拥有堆叠泛化的所有部分，我们就可以将它应用于现实世界的问题。

### 3.声纳数据集案例研究

在本节中，我们将堆叠算法应用于Sonar数据集。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **sonar.all-data.csv** 。

首先加载数据集，将字符串值转换为数字，并将输出列从字符串转换为0到1的整数值。这可以通过辅助函数 **load_csv（）**， **str_column_to_float（ ）**和 **str_column_to_int（）**加载和准备数据集。

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型误差。分类精度将用于评估模型。这些行为在 **cross_validation_split（）**， **accuracy_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们将使用上面实现的k-Nearest Neighbors，Perceptron和Logistic回归算法。我们还将使用我们的技术来创建上一步中定义的新堆叠数据集。

开发了新的函数名 **stacking（）**。这个功能做了4件事：

1.  它首先训练一个模型列表（kNN和Perceptron）。
2.  然后，它使用模型做出预测并创建新的堆叠数据集。
3.  然后，它在堆叠数据集上训练聚合器模型（逻辑回归）。
4.  然后，它使用子模型和聚合器模型对测试数据集做出预测。

下面列出了完整的示例。

```py
# Test stacking on the sonar dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
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

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with kNN
def knn_predict(model, test_row, num_neighbors=2):
	neighbors = get_neighbors(model, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Prepare the kNN model
def knn_model(train):
	return train

# Make a prediction with weights
def perceptron_predict(model, row):
	activation = model[0]
	for i in range(len(row)-1):
		activation += model[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = perceptron_predict(weights, row)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Make a prediction with coefficients
def logistic_regression_predict(model, row):
	yhat = model[0]
	for i in range(len(row)-1):
		yhat += model[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = logistic_regression_predict(coef, row)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	stacked_row.append(row[-1])
	return row[0:len(row)-1] + stacked_row

# Stacked Generalization Algorithm
def stacking(train, test):
	model_list = [knn_model, perceptron_model]
	predict_list = [knn_predict, perceptron_predict]
	models = list()
	for i in range(len(model_list)):
		model = model_list[i](train)
		models.append(model)
	stacked_dataset = list()
	for row in train:
		stacked_row = to_stacked_row(models, predict_list, row)
		stacked_dataset.append(stacked_row)
	stacked_model = logistic_regression_model(stacked_dataset)
	predictions = list()
	for row in test:
		stacked_row = to_stacked_row(models, predict_list, row)
		stacked_dataset.append(stacked_row)
		prediction = logistic_regression_predict(stacked_model, stacked_row)
		prediction = round(prediction)
		predictions.append(prediction)
	return predictions

# Test stacking on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

k值为3用于交叉验证，每次迭代时评估每个折叠208/3 = 69.3或略低于70的记录。

运行该示例将打印最终配置的分数和分数平均值。

```py
Scores: [78.26086956521739, 76.81159420289855, 69.56521739130434]
Mean Accuracy: 74.879%
```

## 扩展

本节列出了您可能有兴趣探索的本教程的扩展。

*   **调谐算法**。本教程中用于子模型和聚合模型的算法未经过调整。探索备用配置，看看是否可以进一步提升表现。
*   **预测相关性**。如果子模型的预测相关性较弱，则堆叠效果更好。实施计算以估计子模型的预测之间的相关性。
*   **不同的子模型**。使用堆叠过程实现更多和不同的子模型。
*   **不同的聚合模型**。尝试使用更简单的模型（如平均和投票）和更复杂的聚合模型来查看是否可以提高表现。
*   **更多数据集**。将堆叠应用于UCI机器学习库中的更多数据集。

**你有没有探索过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何在Python中从头开始实现堆叠算法。

具体来说，你学到了：

*   如何组合多个模型的预测。
*   如何将堆叠应用于现实世界的预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。