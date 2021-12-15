# 如何使用Python从零开始扩展机器学习数据

> 原文： [https://machinelearningmastery.com/scale-machine-learning-data-scratch-python/](https://machinelearningmastery.com/scale-machine-learning-data-scratch-python/)

许多机器学习算法都希望数据能够一致地进行缩放。

在为机器学习扩展数据时，您应该考虑两种常用方法。

在本教程中，您将了解如何重新调整数据以进行机器学习。阅读本教程后，您将了解：

*   如何从零开始标准化您的数据。
*   如何从零开始标准化您的数据。
*   何时进行标准化而不是标准化数据。

让我们开始吧。

*   **更新Feb / 2018** ：修复了最小/最大代码示例中的小错字。
*   **更新Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Prepare Machine Learning Data From Scratch With Python](img/32fa1f42f072869aeab7b8fa797cd3db.jpg)

如何使用Python从零开始准备机器学习数据
照片由 [Ondra Chotovinsky](https://www.flickr.com/photos/-chetta-/5634966046/) ，保留一些权利。

## 描述

许多机器学习算法期望输入的规模甚至输出数据都是等效的。

它可以帮助重量输入以做出预测的方法，例如线性回归和逻辑回归。

在以复杂方式组合加权输入的方法中实际上需要它，例如在人工神经网络和深度学习中。

在本教程中，我们将练习以CSV格式重新缩放一个标准机器学习数据集。

具体来说，Pima Indians数据集。它包含768行和9列。文件中的所有值都是数字，特别是浮点值。我们将首先学习如何加载文件，然后学习如何将加载的字符串转换为数值。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上了解有关此数据集的更多信息（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

## 教程

本教程分为3个部分：

1.  规范化数据。
2.  标准化数据。
3.  何时标准化和标准化。

这些步骤将为您处理扩展自己的数据提供所需的基础。

### 1.规范化数据

归一化可以根据上下文引用不同的技术。

这里，我们使用规范化来指示将输入变量重新缩放到0到1之间的范围。

规范化要求您知道每个属性的最小值和最大值。

如果您对问题域有深入了解，可以从训练数据中估算或直接指定。

您可以通过枚举值轻松估算数据集中每个属性的最小值和最大值。

下面的代码片段定义了 **dataset_minmax（）**函数，该函数计算数据集中每个属性的最小值和最大值，然后返回这些最小值和最大值的数组。

```py
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
```

我们可以设计一个小数据集进行测试，如下所示：

```py
x1	x2
50	30
20	90
```

有了这个设计的数据集，我们可以测试我们的函数来计算每列的最小值和最大值。

```py
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Contrive small dataset
dataset = [[50, 30], [20, 90]]
print(dataset)
# Calculate min and max for each column
minmax = dataset_minmax(dataset)
print(minmax)
```

运行该示例将生成以下输出。

首先，数据集以列表格式列表打印，然后每列的最小值和最大值以 **column1：min，max和column2：min，max** 的格式打印。

例如：

```py
[[50, 30], [20, 90]]
[[20, 50], [30, 90]]
```

一旦我们估计了每列的最大和最小允许值，我们现在可以将原始数据标准化为0和1的范围。

规范化列的单个值的计算是：

```py
scaled_value = (value - min) / (max - min)
```

下面是一个名为 **normalize_dataset（）**的函数的实现，它对提供的数据集的每一列中的值进行标准化。

```py
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
```

我们可以将此函数与 **dataset_minmax（）**函数结合使用，并对设计的数据集进行规范化。

```py
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

# Contrive small dataset
dataset = [[50, 30], [20, 90]]
print(dataset)
# Calculate min and max for each column
minmax = dataset_minmax(dataset)
print(minmax)
# Normalize columns
normalize_dataset(dataset, minmax)
print(dataset)
```

运行此示例将打印下面的输出，包括规范化数据集。

```py
[[50, 30], [20, 90]]
[[20, 50], [30, 90]]
[[1, 0], [0, 1]]
```

我们可以将此代码与用于加载CSV数据集的代码相结合，并加载和规范化Pima Indians糖尿病数据集。

从 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)下载Pima Indians数据集，并将其放在当前目录中，名称为 **pima-indians-diabetes.csv** （更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) **）**。打开文件并删除底部的任何空行。

该示例首先加载数据集，并将每列的值从字符串转换为浮点值。从数据集估计每列的最小值和最大值，最后，对数据集中的值进行标准化。

```py
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

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
# Calculate min and max for each column
minmax = dataset_minmax(dataset)
# Normalize columns
normalize_dataset(dataset, minmax)
print(dataset[0])
```

运行该示例将生成以下输出。

在标准化之前和之后打印数据集中的第一条记录，显示缩放的效果。

```py
Loaded data file pima-indians-diabetes.csv with 768 rows and 9 columns
[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]
[0.35294117647058826, 0.7437185929648241, 0.5901639344262295, 0.35353535353535354, 0.0, 0.5007451564828614, 0.23441502988898377, 0.48333333333333334, 1.0]
```

### 2.标准化数据

标准化是一种重新缩放技术，它指的是将数据分布的中心值0和标准偏差定义为值1。

总之，平均值和标准偏差可用于总结正态分布，也称为高斯分布或钟形曲线。

它要求在缩放之前知道每列的值的平均值和标准偏差。与上面的规范化一样，我们可以根据训练数据估算这些值，或使用领域知识来指定它们的值。

让我们从创建函数开始，估算数据集中每列的均值和标准差统计量。

均值描述了数字集合的中间或中心趋势。列的平均值计算为列的所有值之和除以值的总数。

```py
mean = sum(values) / total_values
```

以下名为 **column_means（）**的函数计算数据集中每列的平均值。

```py
# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
```

标准偏差描述了平均值的平均值。它可以计算为每个值与平均值之间的平方差之和的平方根，并除以值的数量减1。

```py
standard deviation = sqrt( (value_i - mean)^2 / (total_values-1))
```

以下名为 **column_stdevs（）**的函数计算数据集中每列的值的标准偏差，并假设已经计算了均值。

```py
# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs
```

同样，我们可以设计一个小数据集来演示数据集的均值和标准差的估计值。

```py
x1	x2
50	30
20	90
30	50
```

使用Excel电子表格，我们可以估算每列的平均值和标准差，如下所示：

```py
 	x1	x2
mean 	33.3	56.6
stdev 	15.27	30.55
```

使用人为的数据集，我们可以估计汇总统计数据。

```py
from math import sqrt

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means

# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# Standardize dataset
dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)
# Estimate mean and standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
print(means)
print(stdevs)
```

执行该示例提供以下输出，与电子表格中计算的数字相匹配。

```py
[[50, 30], [20, 90], [30, 50]]
[33.333333333333336, 56.666666666666664]
[15.275252316519467, 30.550504633038933]
```

计算摘要统计信息后，我们可以轻松地标准化每列中的值。

标准化给定值的计算如下：

```py
standardized_value = (value - mean) / stdev
```

下面是一个名为 **standardize_dataset（）**的函数，它实现了这个等式

```py
# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]
```

将此与用于估计均值和标准差汇总统计量的函数相结合，我们可以标准化我们设计的数据集。

```py
from math import sqrt

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means

# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

# Standardize dataset
dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)
# Estimate mean and standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
print(means)
print(stdevs)
# standardize dataset
standardize_dataset(dataset, means, stdevs)
print(dataset)
```

执行此示例将生成以下输出，显示设计数据集的标准化值。

```py
[[50, 30], [20, 90], [30, 50]]
[33.333333333333336, 56.666666666666664]
[15.275252316519467, 30.550504633038933]
[[1.0910894511799618, -0.8728715609439694], [-0.8728715609439697, 1.091089451179962], [-0.21821789023599253, -0.2182178902359923]]
```

同样，我们可以演示机器学习数据集的标准化。

下面的示例演示了如何加载和标准化Pima Indians糖尿病数据集，假设它在当前工作目录中，如前面的标准化示例中所示。

```py
from csv import reader
from math import sqrt

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

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means

# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
# Estimate mean and standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
# standardize dataset
standardize_dataset(dataset, means, stdevs)
print(dataset[0])
```

运行该示例将打印数据集的第一行，首先以加载的原始格式打印，然后标准化，这样我们就可以看到差异以进行比较。

```py
Loaded data file pima-indians-diabetes.csv with 768 rows and 9 columns
[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]
[0.6395304921176576, 0.8477713205896718, 0.14954329852954296, 0.9066790623472505, -0.692439324724129, 0.2038799072674717, 0.468186870229798, 1.4250667195933604, 1.3650063669598067]
```

### 3.何时标准化和标准化

标准化是一种缩放技术，假设您的数据符合正态分布。

如果给定的数据属性正常或接近正常，则这可能是要使用的缩放方法。

记录标准化过程中使用的摘要统计信息是一种很好的做法，这样您可以在将来可能希望与模型一起使用的数据标准化时应用它们。

规范化是一种不采用任何特定分布的缩放技术。

如果您的数据不是正态分布的，请在应用机器学习算法之前考虑对其进行标准化。

优良作法是记录标准化过程中使用的每个列的最小值和最大值，如果您需要在将来规范化新数据以与模型一起使用。

## 扩展

您可以应用许多其他数据转换。

数据转换的想法是最好地将数据中的问题结构暴露给学习算法。

可能尚不清楚需要提前进行哪些转换。试验和错误以及探索性数据分析（图表和统计数据）的组合可以帮助梳理出可行的方法。

以下是您可能需要考虑研究和实现的一些其他变换：

*   允许可配置范围的标准化，例如-1到1和更多。
*   标准化，允许可配置的传播，例如平均值的1,2或更多标准偏差。
*   指数变换，如对数，平方根和指数。
*   功率变换（例如box-cox）用于修复正态分布数据中的偏斜。

## 评论

在本教程中，您了解了如何从零开始重新调整数据以进行机器学习。

具体来说，你学到了：

*   如何从零开始标准化数据。
*   如何从零开始标准化数据。
*   何时对数据使用规范化或标准化。

您对扩展数据或此帖子有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。