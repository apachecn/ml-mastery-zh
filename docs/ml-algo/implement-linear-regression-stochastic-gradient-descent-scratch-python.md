# 如何利用Python从零开始实现线性回归

> 原文： [https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/](https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/)

许多机器学习算法的核心是优化。

机器学习算法使用优化算法在给定训练数据集的情况下找到一组好的模型参数。

机器学习中最常用的优化算法是随机梯度下降。

在本教程中，您将了解如何实现随机梯度下降，以便从零开始使用Python优化线性回归算法。

完成本教程后，您将了解：

*   如何使用随机梯度下降估计线性回归系数。
*   如何预测多元线性回归。
*   如何利用随机梯度下降实现线性回归来预测新数据。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Implement Linear Regression With Stochastic Gradient Descent From Scratch With Python](img/bed6bc92622a00f766eb37d7fd8d9439.jpg)

如何使用Python随机梯度下降进行线性回归
照片 [star5112](https://www.flickr.com/photos/johnjoh/6981384675/) ，保留一些权利。

## 描述

在本节中，我们将描述线性回归，随机梯度下降技术和本教程中使用的葡萄酒质量数据集。

### 多元线性回归

线性回归是一种预测实际价值的技术。

令人困惑的是，这些预测真实价值的问题被称为回归问题。

线性回归是一种使用直线来模拟输入和输出值之间关系的技术。在两个以上的维度中，该直线可以被认为是平面或超平面。

预测作为输入值的组合来预测输出值。

使用系数（b）对每个输入属性（x）进行加权，并且学习算法的目标是发现导致良好预测（y）的一组系数。

```py
y = b0 + b1 * x1 + b2 * x2 + ...
```

使用随机梯度下降可以找到系数。

### 随机梯度下降

梯度下降是通过遵循成本函数的梯度来最小化函数的过程。

这包括了解成本的形式以及衍生物，以便从给定的点知道梯度并且可以在该方向上移动，例如，向下走向最小值。

在机器学习中，我们可以使用一种技术来评估和更新称为随机梯度下降的每次迭代的系数，以最小化模型对我们的训练数据的误差。

此优化算法的工作方式是每个训练实例一次显示给模型一个。该模型对训练实例做出预测，计算误差并更新模型以减少下一次预测的误差。该过程重复固定次数的迭代。

该过程可用于在模型中找到导致训练数据上模型的最小误差的系数集。每次迭代，机器学习语言中的系数（b）使用以下等式更新：

```py
b = b - learning_rate * error * x
```

其中 **b** 是被优化的系数或权重， **learning_rate** 是您必须配置的学习率（例如0.01），**错误**是模型的预测误差关于归因于重量的训练数据， **x** 是输入值。

### 葡萄酒质量数据集

在我们开发具有随机梯度下降的线性回归算法之后，我们将使用它来模拟葡萄酒质量数据集。

该数据集由4,898种白葡萄酒的细节组成，包括酸度和pH值等测量值。目标是使用这些客观测量来预测0到10之间的葡萄酒质量。

以下是此数据集中前5个记录的示例。

```py
7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8,6
6.3,0.3,0.34,1.6,0.049,14,132,0.994,3.3,0.49,9.5,6
8.1,0.28,0.4,6.9,0.05,30,97,0.9951,3.26,0.44,10.1,6
7.2,0.23,0.32,8.5,0.058,47,186,0.9956,3.19,0.4,9.9,6
7.2,0.23,0.32,8.5,0.058,47,186,0.9956,3.19,0.4,9.9,6
```

必须将数据集标准化为0到1之间的值，因为每个属性具有不同的单位，并且反过来具有不同的比例。

通过预测归一化数据集上的平均值（零规则算法），可以实现0.148的基线均方根误差（RMSE）。

您可以在 [UCI机器学习库](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)上了解有关数据集的更多信息。

您可以下载数据集并将其保存在当前工作目录中，名称为 **winequality-white.csv** 。您必须从文件的开头删除标头信息，并将“;”值分隔符转换为“，”以符合CSV格式。

## 教程

本教程分为3个部分：

1.  做出预测。
2.  估计系数。
3.  葡萄酒质量预测。

这将为您自己的预测建模问题提供实现和应用具有随机梯度下降的线性回归所需的基础。

### 1.做出预测

第一步是开发一个可以做出预测的功能。

在随机梯度下降中的候选系数值的评估中以及在模型完成之后，我们希望开始对测试数据或新数据做出预测。

下面是一个名为 **predict（）**的函数，它预测给定一组系数的行的输出值。

第一个系数in总是截距，也称为偏差或b0，因为它是独立的，不负责特定的输入值。

```py
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat
```

我们可以设计一个小数据集来测试我们的预测函数。

```py
x, y
1, 1
2, 3
4, 3
3, 2
5, 5
```

下面是该数据集的图表。

![Small Contrived Dataset For Simple Linear Regression](img/e644f57958bd1d11e6f2b3211166ea1c.jpg)

线性回归的小受控数据集

我们还可以使用先前准备的系数来对该数据集做出预测。

综合这些，我们可以测试下面的 **predict（）**函数。

```py
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
coef = [0.4, 0.8]
for row in dataset:
	yhat = predict(row, coef)
	print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))
```

有一个输入值（x）和两个系数值（b0和b1）。我们为这个问题建模的预测方程是：

```py
y = b0 + b1 * x
```

或者，我们手动选择的具体系数值为：

```py
y = 0.4 + 0.8 * x
```

运行此函数，我们得到的结果与预期的输出（y）值相当接近。

```py
Expected=1.000, Predicted=1.200
Expected=3.000, Predicted=2.000
Expected=3.000, Predicted=3.600
Expected=2.000, Predicted=2.800
Expected=5.000, Predicted=4.400
```

现在我们准备实现随机梯度下降来优化我们的系数值。

### 2.估计系数

我们可以使用随机梯度下降来估计训练数据的系数值。

随机梯度下降需要两个参数：

*   **学习率**：用于限制每次更新时每个系数的校正量。
*   **时期**：更新系数时运行训练数据的次数。

这些以及训练数据将是该函数的参数。

我们需要在函数中执行3个循环：

1.  循环每个时代。
2.  循环遍历训练数据中的每一行以获得一个迭代。
3.  循环遍历每个系数并将其更新为一个迭代中的一行。

如您所见，我们更新训练数据中每一行的每个系数，每个时期。

系数根据模型产生的误差进行更新。该误差被计算为用候选系数进行的预测与预期输出值之间的差异。

```py
error = prediction - expected
```

有一个系数可以对每个输入属性进行加权，并且这些系数以一致的方式更新，例如：

```py
b1(t+1) = b1(t) - learning_rate * error(t) * x1(t)
```

列表开头的特殊系数（也称为截距或偏差）以类似的方式更新，除非没有输入，因为它与特定输入值无关：

```py
b0(t+1) = b0(t) - learning_rate * error(t)
```

现在我们可以将所有这些放在一起。下面是一个名为 **coefficients_sgd（）**的函数，它使用随机梯度下降计算训练数据集的系数值。

```py
# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += error**2
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef
```

您可以看到，此外，我们会跟踪每个时期的平方误差（正值）的总和，以便我们可以在外部循环中打印出一条好消息。

我们可以在上面的同样小的人为数据集上测试这个函数。

```py
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += error**2
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

# Calculate coefficients
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
l_rate = 0.001
n_epoch = 50
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)
```

我们使用0.001的小学习率并且将模型训练50个时期，或者将系数的50次曝光训练到整个训练数据集。

运行该示例在每个时期打印一条消息，该消息包含该时期的总和平方误差和最后一组系数。

```py
>epoch=45, lrate=0.001, error=2.650
>epoch=46, lrate=0.001, error=2.627
>epoch=47, lrate=0.001, error=2.607
>epoch=48, lrate=0.001, error=2.589
>epoch=49, lrate=0.001, error=2.573
[0.22998234937311363, 0.8017220304137576]
```

你可以看到即使在最后一个时代，错误仍会继续下降。我们可以训练更长时间（更多迭代）或增加每个时期更新系数的量（更高的学习率）。

试验并看看你提出了什么。

现在，让我们将这个算法应用于真实数据集。

### 3.葡萄酒质量预测

在本节中，我们将使用随机梯度下降在葡萄酒质量数据集上训练线性回归模型。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **winequality-white.csv** 。

首先加载数据集，将字符串值转换为数字，并将每列标准化为0到1范围内的值。这可以通过辅助函数 **load_csv（）**和 **str_column_to_float（）[来实现]。 HTG3]加载并准备数据集和 **dataset_minmax（）**和 **normalize_dataset（）**来规范化它。**

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型误差。均方根误差将用于评估每个模型。这些行为在 **cross_validation_split（）**， **rmse_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们将使用上面创建的 **predict（）**， **coefficient_sgd（）**和 **linear_regression_sgd（）**函数来训练模型。

以下是完整的示例。

```py
# Linear Regression With Stochastic Gradient Descent for Wine Quality
from random import seed
from random import randrange
from csv import reader
from math import sqrt

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

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
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

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

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
		rmse = rmse_metric(actual, predicted)
		scores.append(rmse)
	return scores

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
			# print(l_rate, n_epoch, error)
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)

# Linear Regression on wine quality dataset
seed(1)
# load and prepare data
filename = 'winequality-white.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.01
n_epoch = 50
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
```

k值为5用于交叉验证，每次迭代时评估每个折叠4,898 / 5 = 979.6或略低于1000个记录。通过一些实验选择了0.01和50个训练时期的学习率。

您可以尝试自己的配置，看看是否可以打败我的分数。

运行此示例将打印5个交叉验证折叠中每个折叠的分数，然后打印平均RMSE。

我们可以看到RMSE（在标准化数据集上）是0.126，如果我们只是预测平均值（使用零规则算法），则低于0.148的基线值。

```py
Scores: [0.12248058224159092, 0.13034017509167112, 0.12620370547483578, 0.12897687952843237, 0.12446990678682233]
Mean RMSE: 0.126
```

## 扩展

本节列出了本教程的一些扩展，您可能希望考虑这些扩展。

*   **调整示例**。调整学习率，时期数，甚至数据准备方法，以获得葡萄酒质量数据集的改进分数。
*   **批随机梯度下降**。改变随机梯度下降算法以在每个时期累积更新，并且仅在时期结束时批量更新系数。
*   **其他回归问题**。将该技术应用于UCI机器学习库中的其他回归问题。

**你有没有探索过这些扩展？**
请在下面的评论中告诉我。

## 评论

在本教程中，您了解了如何使用Python从零开始使用随机梯度下降来实现线性回归。

你学到了

*   如何预测多元线性回归问题。
*   如何使用随机梯度下降来优化一组系数。
*   如何将该技术应用于真实的回归预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。