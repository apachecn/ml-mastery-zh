# 如何利用Python从零开始实现逻辑回归

> 原文： [https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)

逻辑回归是两类问题的首选线性分类算法。

它易于实现，易于理解，并且可以在各种各样的问题上获得很好的结果，即使这些方法对您的数据的期望受到侵犯也是如此。

在本教程中，您将了解如何使用Python从零开始随机梯度下降实现逻辑回归。

完成本教程后，您将了解：

*   如何使用逻辑回归模型做出预测。
*   如何使用随机梯度下降估计系数。
*   如何将逻辑回归应用于实际预测问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement Logistic Regression With Stochastic Gradient Descent From Scratch With Python](img/5bf94c3190b2f31398104e303c49125d.jpg)

如何使用随机梯度下降实现逻辑回归从Python [照片](https://www.flickr.com/photos/31246066@N04/15171955576/) [Ian Sane](https://www.flickr.com/photos/31246066@N04/15171955576/) ，保留一些权利。

## 描述

本节将简要介绍逻辑回归技术，随机梯度下降和我们将在本教程中使用的Pima Indians糖尿病数据集。

### 逻辑回归

逻辑回归以在该方法的核心使用的函数命名，即逻辑函数。

逻辑回归使用方程作为表示，非常类似于线性回归。使用权重或系数值线性组合输入值（ **X** ）以预测输出值（ **y** ）。

与线性回归的主要区别在于，建模的输出值是二进制值（0或1）而不是数值。

```py
yhat = e^(b0 + b1 * x1) / (1 + e^(b0 + b1 * x1))
```

这可以简化为：

```py
yhat = 1.0 / (1.0 + e^(-(b0 + b1 * x1)))
```

其中 **e** 是自然对数（欧拉数）的基础， **yhat** 是预测输出， **b0** 是偏差或截距项和 **b1** 是单输入值的系数（ **x1** ）。

yhat预测是0到1之间的实数值，需要舍入为整数值并映射到预测的类值。

输入数据中的每一列都有一个相关的b系数（一个恒定的实际值），必须从训练数据中学习。您将存储在存储器或文件中的模型的实际表示是等式中的系数（β值或b）。

逻辑回归算法的系数必须根据您的训练数据进行估算。

### 随机梯度下降

梯度下降是通过遵循成本函数的梯度来最小化函数的过程。

这包括了解成本的形式以及衍生物，以便从给定的点知道梯度并且可以在该方向上移动，例如，向下走向最小值。

在机器学习中，我们可以使用一种技术来评估和更新称为随机梯度下降的每次迭代的系数，以最小化模型对我们的训练数据的误差。

此优化算法的工作方式是每个训练实例一次显示给模型一个。该模型对训练实例做出预测，计算误差并更新模型以减少下一次预测的误差。

该过程可用于在模型中找到导致训练数据上模型的最小误差的系数集。每次迭代，机器学习语言中的系数（b）使用以下等式更新：

```py
b = b + learning_rate * (y - yhat) * yhat * (1 - yhat) * x
```

**b** 是被优化的系数或权重， **learning_rate** 是你必须配置的学习率（例如0.01），**（y - yhat）**是预测对于归因于权重的训练数据的模型的误差， **yhat** 是由系数做出的预测， **x** 是输入值。

### 皮马印第安人糖尿病数据集

Pima Indians数据集涉及在皮马印第安人中预测5年内糖尿病的发病情况，并提供基本医疗细节。

这是一个二分类问题，其中预测是0（无糖尿病）或1（糖尿病）。

它包含768行和9列。文件中的所有值都是数字，特别是浮点值。下面是问题前几行的一小部分样本。

```py
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
...
```

预测多数类（零规则算法），该问题的基线表现为65.098％分类精度。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上了解有关此数据集的更多信息（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

下载数据集并使用文件名 **pima-indians-diabetes.csv** 将其保存到当前工作目录。

## 教程

本教程分为3个部分。

1.  做出预测。
2.  估计系数。
3.  糖尿病预测。

这将为您自己的预测性建模问题提供实现和应用具有随机梯度下降的逻辑回归所需的基础。

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
	return 1.0 / (1.0 + exp(-yhat))
```

我们可以设计一个小数据集来测试我们的 **predict（）**函数。

```py
X1		X2		Y
2.7810836	2.550537003	0
1.465489372	2.362125076	0
3.396561688	4.400293529	0
1.38807019	1.850220317	0
3.06407232	3.005305973	0
7.627531214	2.759262235	1
5.332441248	2.088626775	1
6.922596716	1.77106367	1
8.675418651	-0.242068655	1
7.673756466	3.508563011	1
```

下面是使用不同颜色的数据集的图，以显示每个点的不同类。

![Small Contrived Classification Dataset](img/1185be78011e6ed0a7abd2f441ce26dc.jpg)

小型受控分类数据集

我们还可以使用先前准备的系数来对该数据集做出预测。

综合这些，我们可以测试下面的 **predict（）**函数。

```py
# Make a prediction
from math import exp

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

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
coef = [-0.406605464, 0.852573316, -1.104746259]
for row in dataset:
	yhat = predict(row, coef)
	print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
```

有两个输入值（X1和X2）和三个系数值（b0，b1和b2）。我们为这个问题建模的预测方程是：

```py
y = 1.0 / (1.0 + e^(-(b0 + b1 * X1 + b2 * X2)))
```

或者，我们手动选择的具体系数值为：

```py
y = 1.0 / (1.0 + e^(-(-0.406605464 + 0.852573316 * X1 + -1.104746259 * X2)))
```

运行此函数，我们可以得到与预期输出（y）值相当接近的预测，并在舍入时对类进行正确的预测。

```py
Expected=0.000, Predicted=0.299 [0]
Expected=0.000, Predicted=0.146 [0]
Expected=0.000, Predicted=0.085 [0]
Expected=0.000, Predicted=0.220 [0]
Expected=0.000, Predicted=0.247 [0]
Expected=1.000, Predicted=0.955 [1]
Expected=1.000, Predicted=0.862 [1]
Expected=1.000, Predicted=0.972 [1]
Expected=1.000, Predicted=0.999 [1]
Expected=1.000, Predicted=0.905 [1]
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

系数根据模型产生的误差进行更新。该误差被计算为预期输出值与利用候选系数进行的预测之间的差异。

有一个系数可以对每个输入属性进行加权，并且这些系数以一致的方式更新，例如：

```py
b1(t+1) = b1(t) + learning_rate * (y(t) - yhat(t)) * yhat(t) * (1 - yhat(t)) * x1(t)
```

列表开头的特殊系数（也称为截距）以类似的方式更新，除非没有输入，因为它与特定输入值无关：

```py
b0(t+1) = b0(t) + learning_rate * (y(t) - yhat(t)) * yhat(t) * (1 - yhat(t))
```

现在我们可以将所有这些放在一起。下面是一个名为 **coefficients_sgd（）**的函数，它使用随机梯度下降计算训练数据集的系数值。

```py
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef
```

你可以看到，此外，我们跟踪每个时期的平方误差（正值）的总和，这样我们就可以在每个外环中打印出一条好消息。

我们可以在上面的同样小的人为数据集上测试这个函数。

```py
from math import exp

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

# Calculate coefficients
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
l_rate = 0.3
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)
```

我们使用更大的学习率0.3并且将模型训练100个时期，或者将系数的100次曝光训练到整个训练数据集。

运行该示例在每个时期打印一条消息，该消息包含该时期的总和平方误差和最后一组系数。

```py
>epoch=95, lrate=0.300, error=0.023
>epoch=96, lrate=0.300, error=0.023
>epoch=97, lrate=0.300, error=0.023
>epoch=98, lrate=0.300, error=0.023
>epoch=99, lrate=0.300, error=0.022
[-0.8596443546618897, 1.5223825112460005, -2.218700210565016]
```

你可以看到即使在最后一个时代，错误仍会继续下降。我们可以训练更长时间（更多迭代）或增加每个时期更新系数的量（更高的学习率）。

试验并看看你提出了什么。

现在，让我们将这个算法应用于真实数据集。

### 3.糖尿病预测

在本节中，我们将使用糖尿病数据集上的随机梯度下降来训练逻辑回归模型。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **pima-indians-diabetes.csv** 。

首先加载数据集，将字符串值转换为数字，并将每列标准化为0到1范围内的值。这是通过辅助函数 **load_csv（）**和 **str_column_to_float（）实现的。** 加载并准备数据集和 **dataset_minmax（）**和 **normalize_dataset（）**来规范化它。

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型表现。分类精度将用于评估每个模型。这些行为在 **cross_validation_split（）**， **accuracy_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们将使用上面创建的 **predict（）**， **coefficient_sgd（）**函数和新的 **logistic_regression（）**函数来训练模型。

以下是完整的示例。

```py
# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
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

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

k值为5用于交叉验证，每次迭代时评估每个折叠768/5 = 153.6或仅超过150个记录。通过一些实验选择了0.1和100个训练时期的学习率。

您可以尝试自己的配置，看看是否可以打败我的分数。

运行此示例将打印5个交叉验证折叠中每个折叠的分数，然后打印平均分类精度。

我们可以看到，如果我们使用零规则算法预测大多数类，精度约为77％，高于基线值65％。

```py
Scores: [73.8562091503268, 78.43137254901961, 81.69934640522875, 75.81699346405229, 75.81699346405229]
Mean Accuracy: 77.124%
```

## 扩展

本节列出了本教程的一些扩展，您可能希望考虑这些扩展。

*   **调整示例**。调整学习率，时期数甚至数据准备方法以获得数据集上的改进分数。
*   **批随机梯度下降**。改变随机梯度下降算法以在每个时期累积更新，并且仅在时期结束时批量更新系数。
*   **其他分类问题**。将该技术应用于UCI机器学习库中的其他二进制（2类）分类问题。

**你有没有探索过这些扩展？**
请在下面的评论中告诉我。

## 评论

在本教程中，您了解了如何使用Python从零开始使用随机梯度下降来实现逻辑回归。

你学到了

*   如何预测多变量分类问题。
*   如何使用随机梯度下降来优化一组系数。
*   如何将该技术应用于真实的分类预测性建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。