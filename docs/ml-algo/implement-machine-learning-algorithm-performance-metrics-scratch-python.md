# 如何用Python从头开始实现机器学习算法表现指标

> 原文： [https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/](https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/)

在做出预测之后，你需要知道它们是否有用。

我们可以使用标准度量来总结一组预测实际上有多好。

了解一组预测有多好，可以让您估算出问题的给定机器学习模型的好坏，

在本教程中，您将了解如何在Python中从头开始实现四个标准预测评估指标。

阅读本教程后，您将了解：

*   如何实现分类准确性。
*   如何实现和解释混淆矩阵。
*   如何实现回归的均值绝对误差。
*   如何实现回归的均方根误差。

让我们开始吧。

*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement Machine Learning Algorithm Performance Metrics From Scratch In Python](img/3bb0770ef9a693bf253daed634235bc5.jpg)

如何使用Python从头开始实现机器学习算法表现指标
照片由[HernánPiñera](https://www.flickr.com/photos/hernanpc/8407944523/)，保留一些权利。

## 描述

在训练机器学习模型时，您必须估计一组预测的质量。

分类准确度和均方根误差等表现指标可以让您清楚地了解一组预测的好坏，以及生成它们的模型有多好。

这很重要，因为它可以让您区分并选择：

*   用于训练相同机器学习模型的数据的不同变换。
*   不同的机器学习模型训练相同的数据。
*   针对相同数据训练的机器学习模型的不同配置。

因此，表现指标是从头开始实现机器学习算法的必要构建块。

## 教程

本教程分为4个部分：

*   1.分类准确性。
*   2.混淆矩阵。
*   3.平均绝对误差。
*   4.均方根误差。

这些步骤将为您提供处理机器学习算法预测评估所需的基础。

### 1.分类准确性

评估分类问题的一组预测的快速方法是使用准确性。

分类准确性是所有预测中正确预测数量的比率。

它通常以最差可能精度的0％和最佳精度的100％之间的百分比表示。

```py
accuracy = correct predictions / total predictions * 100
```

我们可以在一个将预期结果和预测作为参数的函数中实现它。

下面是名为 **accuracy_metric（）**的函数，它以百分比形式返回分类精度。请注意，我们使用“==”来比较实际值与预测值的相等性。这允许我们比较整数或字符串，我们在加载分类数据时可能选择使用的两种主要数据类型。

```py
# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
```

我们可以设计一个小数据集来测试这个功能。下面是一组10个实际和预测的整数值。预测集中有两个错误。

```py
actual          predicted
0		0
0		1
0		0
0		0
0		0
1		1
1		0
1		1
1		1
1		1
```

下面是该数据集的完整示例，用于测试 **accuracy_metric（）**函数。

```py
# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Test accuracy
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)
```

运行此示例可以产生80％或8/10的预期准确度。

```py
80.0
```

当您具有少量类值（例如2）时，准确度是一个很好的度量标准，也称为二元分类问题。

当你有更多的类值时，准确性开始失去意义，你可能需要审查结果的不同观点，例如混淆矩阵。

### 2.混淆矩阵

混淆矩阵提供了与预期实际值相比所做的所有预测的摘要。

结果以矩阵形式呈现，每个细胞中有计数。水平汇总实际类值的计数，而垂直显示每个类值的预测计数。

一组完美的预测显示为矩阵左上角到右下角的对角线。

分类问题的混淆矩阵的值是，您可以清楚地看到哪些预测是错误的，以及所犯的错误类型。

让我们创建一个计算混淆矩阵的函数。

我们可以从定义函数开始，在给定实际类值列表和预测列表的情况下计算混淆矩阵。

该功能如下所示，名为 **confusion_matrix（）**。它首先列出所有唯一类值，并将每个类值分配给混淆矩阵中的唯一整数或索引。

混淆矩阵始终是正方形，类值的数量表示所需的行数和列数。

这里，矩阵的第一个索引是实际值的行，第二个是预测值的列。在创建方形混淆矩阵并在每个单元格中初始化为零计数之后，循环所有预测并递增每个单元格中的计数。

该函数返回两个对象。第一个是唯一类值的集合，以便在绘制混淆矩阵时显示它们。第二个是混淆矩阵本身与每个单元格中的计数。

```py
# calculate a confusion matrix
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix
```

让我们以一个例子来具体化。

下面是另一个人为的数据集，这次有3个错误。

```py
actual     	predicted
0		0
0		1
0		1
0		0
0		0
1		1
1		0
1		1
1		1
1		1
```

我们可以计算并打印此数据集的混淆矩阵，如下所示：

```py
# Example of Calculating a Confusion Matrix

# calculate a confusion matrix
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix

# Test confusion matrix with integers
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print(unique)
print(matrix)
```

运行该示例将生成以下输出。该示例首先打印唯一值列表，然后打印混淆矩阵。

```py
{0, 1}
[[3, 1], [2, 4]]
```

用这种方式解释结果很难。如果我们可以按行和列显示矩阵，将会有所帮助。

以下是正确显示矩阵的功能。

该函数名为 **print_confusion_matrix（）**。它将列命名为P表示预测，行指定为A表示实际。每个列和行都以其对应的类值命名。

矩阵的布局期望每个类标签是单个字符或单个数字整数，并且计数也是单个数字整数。您可以将其扩展为处理大类标签或预测计数作为练习。

```py
# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
	print('(A)' + ' '.join(str(x) for x in unique))
	print('(P)---')
	for i, x in enumerate(unique):
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))
```

我们可以拼凑所有功能并显示人类可读的混淆矩阵。

```py
# Example of Calculating and Displaying a Pretty Confusion Matrix

# calculate a confusion matrix
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix

# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
	print('(A)' + ' '.join(str(x) for x in unique))
	print('(P)---')
	for i, x in enumerate(unique):
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))

# Test confusion matrix with integers
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print_confusion_matrix(unique, matrix)
```

运行该示例将生成以下输出。我们可以在顶部和底部看到0和1的类标签。从左上角到右下角向下看矩阵的对角线，我们可以看到3个0的预测是正确的，4个预测的1个是正确的。

查看其他单元格，我们可以看到2 + 1或3个预测错误。我们可以看到2个预测是1，实际上实际上是0类值。我们可以看到1个预测是0，实际上实际上是1。

```py
(A)0 1
(P)---
0| 3 1
1| 2 4
```

除了分类准确性之外，混淆矩阵始终是一个好主意，以帮助解释预测。

### 3.平均绝对误差

回归问题是预测实际价值的问题。

要考虑的一个简单指标是预测值与预期值相比的误差。

平均绝对误差或简称MAE是一个很好的第一个误差度量标准。

它被计算为绝对误差值的平均值，其中“绝对值”表示“为正”，以便它们可以加在一起。

```py
MAE = sum( abs(predicted_i - actual_i) ) / total predictions
```

下面是一个名为 **mae_metric（）**的函数，它实现了这个指标。如上所述，它需要一个实际结果值列表和一个预测列表。我们使用内置的 **abs（）** Python函数来计算求和的绝对误差值。

```py
def mae_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
```

我们可以设计一个小的回归数据集来测试这个函数。

```py
actual 		predicted
0.1		0.11
0.2		0.19
0.3		0.29
0.4		0.41
0.5		0.5
```

只有一个预测（0.5）是正确的，而所有其他预测都是错误的0.01。因此，我们预计这些预测的平均绝对误差（或平均正误差）略小于0.01。

下面是一个用设想的数据集测试 **mae_metric（）**函数的例子。

```py
# Calculate mean absolute error
def mae_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
	return sum_error / float(len(actual))

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)
```

运行此示例将打印下面的输出。我们可以看到，正如预期的那样，MAE约为0.008，小值略低于0.01。

```py
0.007999999999999993
```

### 4.均方根误差

在一组回归预测中计算误差的另一种流行方法是使用均方根误差。

缩写为RMSE，该度量有时称为均方误差或MSE，从计算和名称中删除根部分。

RMSE计算为实际结果和预测之间的平方差异的平均值的平方根。

平方每个错误会强制值为正，并且均方误差的平方根将误差度量返回到原始单位以进行比较。

```py
RMSE = sqrt( sum( (predicted_i - actual_i)^2 ) / total predictions)
```

下面是一个名为 **rmse_metric（）**的函数的实现。它使用数学模块中的 **sqrt（）**函数，并使用**运算符将误差提高到2次幂。

```py
# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
```

我们可以在用于测试上面的平均绝对误差计算的相同数据集上测试该度量。

以下是一个完整的例子。同样，我们希望误差值通常接近0.01。

```py
from math import sqrt

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
rmse = rmse_metric(actual, predicted)
print(rmse)
```

运行该示例，我们看到下面的结果。结果略高于0.0089。

RMSE值总是略高于MSE值，随着预测误差的增加，MSE值变得更加明显。这是使用RMSE而不是MSE的好处，因为它会以较差的分数惩罚较大的错误。

```py
0.00894427190999915
```

## 扩展

您只看到了最广泛使用的表现指标的一小部分样本。

您可能还需要许多其他表现指标。

下面列出了5个额外的表现指标，您可能希望实现这些指标以扩展本教程

*   分类精度。
*   回想一下分类。
*   F1进行分类。
*   ROC曲线下的面积或分类的AUC。
*   拟合优度或R ^ 2（R平方）用于回归。

**您是否实施了这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何在Python中从头开始实现算法预测表现指标。

具体来说，你学到了：

*   如何实现和解释分类准确性。
*   如何实现和解释分类问题的混淆矩阵。
*   如何实现和解释回归的平均绝对误差。
*   如何实现和解释回归的均方根误差。

**你有什么问题吗？**
在评论中提出您的问题，我会尽力回答。