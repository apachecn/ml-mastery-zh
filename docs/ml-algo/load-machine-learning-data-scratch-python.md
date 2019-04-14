# 如何在Python中从头开始加载机器学习数据

> 原文： [https://machinelearningmastery.com/load-machine-learning-data-scratch-python/](https://machinelearningmastery.com/load-machine-learning-data-scratch-python/)

您必须知道如何加载数据，然后才能使用它来训练机器学习模型。

在开始时，最好使用标准文件格式（如逗号分隔值（.csv））来坚持使用小型内存数据集。

在本教程中，您将了解如何从头开始在Python中加载数据，包括：

*   如何加载CSV文件。
*   如何将字符串从文件转换为浮点数。
*   如何将类值从文件转换为整数。

让我们开始吧。

*   **2016年11月更新**：增加了改进的数据加载功能，以跳过空行。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Load Machine Learning Data From Scratch In Python](img/2fe1327d264ccbc63c79513f4fa3fd1c.jpg)

如何在Python中从头开始加载机器学习数据
照片由 [Amanda B](https://www.flickr.com/photos/muddybones/5550623402/) ，保留一些权利。

## 描述

### 逗号分隔值

小数据集的标准文件格式是逗号分隔值或CSV。

最简单的形式是，CSV文件由数据行组成。每行使用逗号（“，”）分成列。

您可以在 [RFC 4180：逗号分隔值（CSV）文件](https://tools.ietf.org/html/rfc4180)的通用格式和MIME类型中了解有关CSV文件格式的更多信息。

在本教程中，我们将练习以CSV格式加载两个不同的标准机器学习数据集。

### 皮马印第安人糖尿病数据集

第一个是皮马印第安人糖尿病数据集。它包含768行和9列。

文件中的所有值都是数字，特别是浮点值。我们将首先学习如何加载文件，然后学习如何将加载的字符串转换为数值。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上了解有关此数据集的更多信息。

### 鸢尾花种类数据集

我们将使用的第二个数据集是虹膜花数据集。

它包含150行和4列。前3列是数字。不同之处在于，类值（最终列）是一个字符串，表示一种花。我们将学习如何将数字列从字符串转换为数字以及如何将花种字符串转换为我们可以一致使用的整数。

您可以在 [UCI机器学习库](http://archive.ics.uci.edu/ml/datasets/Iris)上了解有关此数据集的更多信息。

## 教程

本教程分为3个部分：

1.  加载文件。
2.  加载文件并将字符串转换为浮点数。
3.  加载文件并将字符串转换为整数。

这些步骤将为您处理加载自己的数据提供所需的基础。

### 1.加载CSV文件

第一步是加载CSV文件。

我们将使用作为标准库一部分的 [csv模块](https://docs.python.org/2/library/csv.html)。

csv模块中的 **reader（）**函数将文件作为参数。

我们将创建一个名为 **load_csv（）**的函数来包装此行为，该行为将采用文件名并返回我们的数据集。我们将加载的数据集表示为列表列表。第一个列表是观察或行的列表，第二个列表是给定行的列值列表。

以下是加载CSV文件的完整功能。

```py
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset
```

我们可以通过加载Pima Indians数据集来测试这个函数。下载数据集并将其放在当前工作目录中，名称为 **pima-indians-diabetes.csv** 。打开文件并删除底部的任何空行。

看一下原始数据文件的前5行，我们可以看到以下内容：

```py
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
```

数据是数字的，用逗号分隔，我们可以期望整个文件符合这个期望。

让我们使用新函数并加载数据集。加载后，我们可以报告一些简单的细节，例如加载的行数和列数。

将所有这些放在一起，我们得到以下结果：

```py
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Load dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
```

运行此示例我们看到：

```py
Loaded data file pima-indians-diabetes.csv with 768 rows and 9 columns
```

此函数的一个限制是它将从数据文件加载空行并将它们添加到我们的行列表中。我们可以通过向数据集一次添加一行数据并跳过空行来解决这个问题。

下面是 **load_csv（）**函数的这个新改进版本的更新示例。

```py
# Example of loading Pima Indians CSV dataset
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

# Load dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
```

Running this example we see:

```py
Loaded data file pima-indians-diabetes.csv with 768 rows and 9 columns
```

### 2.将字符串转换为浮点数

大多数（如果不是全部）机器学习算法都喜欢使用数字。

具体而言，优选浮点数。

我们用于加载CSV文件的代码将数据集作为列表列表返回，但每个值都是一个字符串。如果我们从数据集中打印出一条记录，我们可以看到这一点：

```py
print(dataset[0])
```

这会产生如下输出：

```py
['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1']
```

我们可以编写一个小函数来将我们加载的数据集的特定列转换为浮点值。

下面是这个函数叫做 **str_column_to_float（）**。它会将数据集中的给定列转换为浮点值，小心地在进行转换之前从值中去除任何空格。

```py
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
```

我们可以通过将它与上面的加载CSV函数相结合来测试此函数，并将Pima Indians数据集中的所有数值数据转换为浮点值。

完整的例子如下。

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

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
print(dataset[0])
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
```

运行此示例，我们看到在转换之前和之后打印的数据集的第一行。我们可以看到每列中的值已从字符串转换为数字。

```py
Loaded data file pima-indians-diabetes.csv with 768 rows and 9 columns
['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1']
[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]
```

### 3.将字符串转换为整数

虹膜花数据集与Pima Indians数据集类似，因为列包含数字数据。

差异是最后一列，传统上用于保存给定行的预测结果或值。虹膜花数据中的最后一列是鸢尾花种。

下载数据集并将其放在当前工作目录中，文件名为 **iris.csv** 。打开文件并删除底部的任何空行。

例如，下面是原始数据集的前5行。

```py
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
```

一些机器学习算法更喜欢所有值都是数字，包括结果或预测值。

我们可以通过创建地图将虹膜花数据集中的类值转换为整数。

1.  首先，我们找到所有独特的类值，恰好是：Iris-setosa，Iris-versicolor和Iris-virginica。
2.  接下来，我们为每个分配一个整数值，例如：0,1和2。
3.  最后，我们将所有出现的类字符串值替换为相应的整数值。

下面是一个只调用 **str_column_to_int（）**的函数。与之前介绍的 **str_column_to_float（）**一样，它在数据集中的单个列上运行。

```py
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
```

除了前两个函数之外，我们还可以测试这个新函数，以加载CSV文件并将列转换为浮点值。它还将类值的字典映射返回到整数值，以防下游任何用户希望再次将预测转换回字符串值。

下面的示例加载iris数据集，然后将前3列转换为浮点数，将最后一列转换为整数值。

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

# Load iris dataset
filename = 'iris.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
print(dataset[0])
# convert string columns to float
for i in range(4):
	str_column_to_float(dataset, i)
# convert class column to int
lookup = str_column_to_int(dataset, 4)
print(dataset[0])
print(lookup)
```

运行此示例将生成下面的输出。

我们可以在数据类型转换之前和之后看到数据集的第一行。我们还可以看到类值到整数的字典映射。

```py
Loaded data file iris.csv with 150 rows and 5 columns
['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
[5.1, 3.5, 1.4, 0.2, 1]
{'Iris-virginica': 0, 'Iris-setosa': 1, 'Iris-versicolor': 2}
```

## 扩展

您学习了如何加载CSV文件和执行基本数据转换。

考虑到从问题到问题可能需要的各种数据清理和转换，数据加载可能是一项困难的任务。

您可以进行许多扩展，以使这些示例对新的和不同的数据文件更加健壮。以下是您可以考虑自己研究和实施的一些想法：

*   检测并删除文件顶部或底部的空行。
*   检测并处理列中的缺失值。
*   检测并处理与文件其余部分的期望不匹配的行。
*   支持其他分隔符，例如“|”（管道）或空格。
*   支持更高效的数据结构，如数组。

您可能希望在实践中用于加载CSV数据的两个库是NumPy和Pandas。

NumPy提供 [loadtxt（）](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html)函数，用于将数据文件作为NumPy数组加载。 Pandas提供 [read_csv（）](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)功能，它在数据类型，文件头等方面提供了很大的灵活性。

## 评论

在本教程中，您了解了如何在Python中从头开始加载机器学习数据。

具体来说，你学到了：

*   如何将CSV文件加载到内存中。
*   如何将字符串值转换为浮点值。
*   如何将字符串类值转换为整数编码。

您对加载机器学习数据或此帖子有任何疑问吗？
在评论中提出您的问题，我会尽力回答。