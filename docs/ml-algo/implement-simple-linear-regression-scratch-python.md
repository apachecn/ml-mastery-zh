# 如何用Python从零开始实现简单线性回归

> 原文： [https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/](https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/)

线性回归是一种超过200年的预测方法。

[简单线性回归](http://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning/)是一个很好的第一个机器学习算法，因为它需要你从训练数据集中估计属性，但是对于初学者来说很简单。

在本教程中，您将了解如何在Python中从零开始实现简单的线性回归算法。

完成本教程后，您将了解：

*   如何从训练数据中估算统计量。
*   如何从数据中估计线性回归系数。
*   如何使用线性回归对新数据做出预测。

让我们开始吧。

*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。
*   **2002年2月更新**：对保险数据集的预期默认RMSE进行小幅更新。

![How To Implement Simple Linear Regression From Scratch With Python](img/8d21d8bbe9560bec279e6b57fc9cde68.jpg)

如何使用Python实现简单的线性回归
照片由 [Kamyar Adl](https://www.flickr.com/photos/kamshots/456696484/) ，保留一些权利。

## 描述

本节分为两部分，简单线性回归技术的描述和我们稍后将应用它的数据集的描述。

### 简单线性回归

线性回归假设输入变量（X）和单个输出变量（y）之间存在线性或直线关系。

更具体地，可以从输入变量（X）的线性组合计算输出（y）。当存在单个输入变量时，该方法称为简单线性回归。

在简单线性回归中，我们可以使用训练数据的统计量来估计模型所需的系数，以对新数据做出预测。

简单线性回归模型的行可以写成：

```py
y = b0 + b1 * x
```

其中b0和b1是我们必须根据训练数据估计的系数。

一旦系数已知，我们可以使用此公式来估计给定x的新输入示例的y的输出值。

它要求您根据数据计算统计特性，例如均值，方差和协方差。

我们已经处理了所有的代数，我们留下了一些算法来实现估计简单的线性回归系数。

简而言之，我们可以估算系数如下：

```py
B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )
B0 = mean(y) - B1 * mean(x)
```

其中i指的是输入x或输出y的第i个值。

如果现在还不清楚，请不要担心，这些功能将在教程中实现。

### 瑞典保险数据集

我们将使用真实数据集来演示简单的线性回归。

该数据集被称为“瑞典汽车保险”数据集，并涉及根据索赔总数（x）预测数千瑞典克朗（y）中所有索赔的总付款额。

这意味着对于新的索赔（x），我们将能够预测索赔的总支付额（y）。

以下是数据集前5个记录的一小部分样本。

```py
108,392.5
19,46.2
13,15.7
124,422.2
40,119.4
```

使用零规则算法（预测平均值）预期的均方根误差或RMSE约为81（千克朗）。

下面是整个数据集的散点图。

![Swedish Insurance Dataset](img/4f9b3ca41a4899476ef6d85f6edffa23.jpg)

瑞典保险数据集

您可以在 或 [](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr06.html) 下载[的原始数据集。](https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt)

将其保存到本地工作目录中的CSV文件，名称为“ **insurance.csv** ”。

注意，您可能需要将欧洲“，”转换为小数“。”。您还需要将文件从空格分隔的变量更改为CSV格式。

## 教程

本教程分为五个部分：

1.  计算均值和方差。
2.  计算协方差。
3.  估算系数。
4.  作出预测。
5.  预测保险。

这些步骤将为您提供实现和训练简单线性回归模型所需的基础，以满足您自己的预测问题。

### 1.计算均值和方差

第一步是从训练数据中估计输入和输出变量的均值和方差。

数字列表的平均值可以计算为：

```py
mean(x) = sum(x) / count(x)
```

下面是一个名为 **mean（）**的函数，它为数字列表实现了这种行为。

```py
# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))
```

方差是每个值与平均值的总和平方差。

数字列表的差异可以计算为：

```py
variance = sum( (x - mean(x))^2 )
```

下面是一个名为 **variance（）**的函数，它计算数字列表的方差。它要求将列表的均值作为参数提供，这样我们就不必多次计算它。

```py
# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
```

我们可以将这两个函数放在一起，并在一个小的设计数据集上进行测试。

下面是x和y值的小数据集。

**注意**：如果将其保存到.CSV文件以与最终代码示例一起使用，则从该数据中删除列标题。

```py
x, y
1, 1
2, 3
4, 3
3, 2
5, 5
```

我们可以在散点图上绘制此数据集，如下所示：

![Small Contrived Dataset For Simple Linear Regression](img/e644f57958bd1d11e6f2b3211166ea1c.jpg)

简单线性回归的小受控数据集

我们可以在下面的例子中计算x和y值的均值和方差。

```py
# Estimate Mean and Variance

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# calculate mean and variance
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
```

运行此示例会打印出两列的均值和方差。

```py
x stats: mean=3.000 variance=10.000
y stats: mean=2.800 variance=8.800
```

这是我们的第一步，接下来我们需要将这些值用于计算协方差。

### 2.计算协方差

两组数字的协方差描述了这些数字如何一起变化。

协方差是相关性的推广。相关性描述了两组数字之间的关系，而协方差可以描述两组或更多组数字之间的关系。

另外，可以对协方差进行归一化以产生相关值。

不过，我们可以计算两个变量之间的协方差如下：

```py
covariance = sum((x(i) - mean(x)) * (y(i) - mean(y)))
```

下面是一个名为 **covariance（）**的函数，它实现了这个统计量。它建立在前一步骤的基础上，并将x和y值的列表以及这些值的平均值作为参数。

```py
# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
```

我们可以在与前一节相同的小型设计数据集上测试协方差的计算。

总而言之，我们得到以下示例。

```py
# Calculate Covariance

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# calculate covariance
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
covar = covariance(x, mean_x, y, mean_y)
print('Covariance: %.3f' % (covar))
```

运行此示例将打印x和y变量的协方差。

```py
Covariance: 8.000
```

我们现在已经准备好所有部件来计算模型的系数。

### 3.估算系数

我们必须在简单线性回归中估计两个系数的值。

第一个是B1，可以估算为：

```py
B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )
```

我们已经学到了上面的一些东西，可以简化这个算法：

```py
B1 = covariance(x, y) / variance(x)
```

我们已经有了计算**协方差（）**和**方差（）**的函数。

接下来，我们需要估计B0的值，也称为截距，因为它控制与y轴相交的直线的起点。

```py
B0 = mean(y) - B1 * mean(x)
```

同样，我们知道如何估计B1，我们有一个函数来估计 **mean（）**。

我们可以将所有这些放在一个名为 **coefficients（）**的函数中，该函数将数据集作为参数并返回系数。

```py
# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
```

我们可以将它与前两个步骤中的所有函数放在一起，并测试系数的计算。

```py
# Calculate Coefficients

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# calculate coefficients
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
b0, b1 = coefficients(dataset)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
```

运行此示例计算并打印系数。

```py
Coefficients: B0=0.400, B1=0.800
```

现在我们知道如何估计系数，下一步就是使用它们。

### 4.做出预测

简单线性回归模型是由训练数据估计的系数定义的线。

一旦估计了系数，我们就可以使用它们做出预测。

使用简单线性回归模型做出预测的等式如下：

```py
y = b0 + b1 * x
```

下面是一个名为 **simple_linear_regression（）**的函数，它实现预测方程以对测试数据集做出预测。它还将来自上述步骤的训练数据的系数估计联系在一起。

从训练数据准备的系数用于对测试数据做出预测，然后返回。

```py
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions
```

让我们将我们学到的所有内容汇集在一起​​，并为我们简单的人为数据集做出预测。

作为此示例的一部分，我们还将添加一个函数来管理名为 **evaluate_algorithm（）**的预测评估，并添加另一个函数来估计名为 **rmse_metric（）的预测的均方根误差**。

下面列出了完整的示例。

```py
# Standalone simple linear regression example
from math import sqrt

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Test simple linear regression
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))
```

运行此示例将显示以下输出，该输出首先列出这些预测的预测和RMSE。

```py
[1.1999999999999995, 1.9999999999999996, 3.5999999999999996, 2.8, 4.3999999999999995]
RMSE: 0.693
```

最后，我们可以将预测绘制为一条线并将其与原始数据集进行比较。

![Predictions For Small Contrived Dataset For Simple Linear Regression](img/62e745ccf1b6c3e4984f001c59e3e90f.jpg)

简单线性回归的小参数数据集预测

### 5.预测保险

我们现在知道如何实现简单的线性回归模型。

我们将它应用于瑞典保险数据集。

本节假定您已将数据集下载到文件 **insurance.csv** ，并且它在当前工作目录中可用。

我们将为前面步骤中的简单线性回归添加一些便利函数。

特别是加载称为 **load_csv（）**的CSV文件的函数，该函数将加载的数据集转换为称为 **str_column_to_float（）**的数字，这是一个使用训练和测试来评估算法的函数设置调用 **train_test_split（）**一个函数来计算称为 **rmse_metric（）**的RMSE和一个函数来评估一个叫做 **evaluate_algorithm（）**的算法。

下面列出了完整的示例。

使用60％数据的训练数据集来准备模型，并对剩余的40％做出预测。

```py
# Simple Linear Regression on the Swedish Insurance Dataset
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

# Split a dataset into a train and test set
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

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
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Simple linear regression on insurance dataset
seed(1)
# load and prepare data
filename = 'insurance.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: %.3f' % (rmse))
```

运行算法会在训练数据集上打印训练模型的RMSE。

获得了大约33（千克朗）的分数，这比在相同问题上实现大约81（数千克朗）的零规则算法好得多。

```py
RMSE: 33.630
```

## 扩展

本教程的最佳扩展是在更多问题上尝试该算法。

只有输入（x）和输出（y）列的小数据集很受欢迎，可用于统计书籍和课程的演示。其中许多数据集都可在线获取。

寻找更多小型数据集并使用简单的线性回归做出预测。

**您是否将简单线性回归应用于其他数据集？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何在Python中从零开始实现简单的线性回归算法。

具体来说，你学到了：

*   如何从训练数据集中估计统计量，如均值，方差和协方差。
*   如何估计模型系数并使用它们做出预测。
*   如何使用简单线性回归对真实数据集做出预测。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。