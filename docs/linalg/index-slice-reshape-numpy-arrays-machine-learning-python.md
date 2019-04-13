# 如何在 Python 中为机器学习索引，切片和重塑 NumPy 数组

> 原文： [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)

机器学习数据表示为数组。

在 Python 中，数据几乎普遍表示为 NumPy 数组。

如果您是 Python 的新手，您可能会对某些 [pythonic](https://stackoverflow.com/questions/25011078/what-does-pythonic-mean) 访问数据的方式感到困惑，例如负索引和数组切片。

在本教程中，您将了解如何在 NumPy 阵列中正确操作和访问数据。

完成本教程后，您将了解：

*   如何将列表数据转换为 NumPy 数组。
*   如何使用 Pythonic 索引和切片访问数据。
*   如何调整数据大小以满足某些机器学习 API 的期望。

让我们开始吧。

![How to Index, Slice and Reshape NumPy Arrays for Machine Learning in Python](img/9c384b0ca667a815e91824f434e4dcac.jpg)

如何在 Python 中为机器学习索引，切片和重塑 NumPy 数组
[BjörnSöderqvist](https://www.flickr.com/photos/kapten/433809071/)的照片，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  从列表到数组
2.  数组索引
3.  阵列切片
4.  数组重塑

## 1.从列表到数组

通常，我建议使用 Pandas 甚至 NumPy 函数从文件加载数据。

例如，请看帖子：

*   [如何在 Python 中加载机器学习数据](http://machinelearningmastery.com/load-machine-learning-data-python/)

本节假定您已通过其他方式加载或生成数据，现在使用 Python 列表表示它。

我们来看看将列表中的数据转换为 NumPy 数组。

### 一维列表到数组

您可以加载数据或生成数据，并以列表形式访问它。

您可以通过调用 array（）NumPy 函数将一维数据列表转换为数组。

```
# one dimensional example
from numpy import array
# list of data
data = [11, 22, 33, 44, 55]
# array of data
data = array(data)
print(data)
print(type(data))
```

运行该示例将一维列表转换为 NumPy 数组。

```
[11 22 33 44 55]
<class 'numpy.ndarray'>
```

### 数组列表的二维列表

在机器学习中，你更有可能拥有二维数据。

这是一个数据表，其中每一行代表一个新观察，每一列代表一个新特征。

也许您使用自定义代码生成数据或加载数据，现在您有一个列表列表。每个列表代表一个新观察。

您可以通过调用 array（）函数将列表列表转换为 NumPy 数组，方法与上面相同。

```
# two dimensional example
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print(data)
print(type(data))
```

运行该示例显示已成功转换的数据。

```
[[11 22]
 [33 44]
 [55 66]]
<class 'numpy.ndarray'>
```

## 2.数组索引

使用 NumPy 阵列表示数据后，可以使用索引访问它。

我们来看一些通过索引访问数据的例子。

### 一维索引

通常，索引的工作方式与您对其他编程语言（如 Java，C＃和 C ++）的体验一样。

例如，您可以使用括号运算符[]访问元素，为要检索的值指定零偏移索引。

```
# simple indexing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[0])
print(data[4])
```

运行该示例将打印数组中的第一个和最后一个值。

```
11
55
```

指定对于数组边界而言太大的整数将导致错误。

```
# simple indexing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[5])
```

运行该示例将显示以下错误：

```
IndexError: index 5 is out of bounds for axis 0 with size 5
```

一个关键的区别是您可以使用负索引来检索从数组末尾偏移的值。

例如，索引-1 指的是数组中的最后一项。对于当前示例中的第一个项，索引-2 将第二个最后一项返回到-5。

```
# simple indexing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[-1])
print(data[-5])
```

运行该示例将打印数组中的最后一项和第一项。

```
55
11
```

### 二维索引

索引二维数据类似于索引一维数据，除了使用逗号分隔每个维度的索引。

```
data[0,0]
```

这与基于 C 的语言不同，其中每个维度使用单独的括号运算符。

```
data[0][0]
```

例如，我们可以访问第一行和第一列，如下所示：

```
# 2d indexing
from numpy import array
# define array
data = array([[11, 22], [33, 44], [55, 66]])
# index data
print(data[0,0])
```

运行该示例将打印数据集中的第一个项目。

```
11
```

如果我们对第一行中的所有项目感兴趣，我们可以将第二个维度索引留空，例如：

```
# 2d indexing
from numpy import array
# define array
data = array([[11, 22], [33, 44], [55, 66]])
# index data
print(data[0,])
```

这将打印第一行数据。

```
[11 22]
```

## 3.阵列切片

到现在为止还挺好;创建和索引数组看起来很熟悉。

现在我们来到数组切片，这是一个导致 Python 和 NumPy 数组初学者出现问题的功能。

像列表和 NumPy 数组这样的结构可以被切片。这意味着可以索引和检索结构的子序列。

在指定输入变量和输出变量或从测试行中分割训练行时，这在机器学习中最有用。

使用冒号运算符'：'指定切片，分别在列之前和之后使用来自'的'_ 和'_ 到 _'索引。切片从'from'索引延伸，并在'to'索引之前结束一个项目。_

```
data[from:to]
```

让我们通过一些例子来解决。

### 一维切片

您可以通过指定没有索引的切片'：'来访问数组维度中的所有数据。

```
# simple slicing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[:])
```

运行该示例将打印数组中的所有元素。

```
[11 22 33 44 55]
```

可以通过指定从索引 0 开始并在索引 1 结束的切片（“to”索引之前的一个项目）来切片数组的第一项。

```
# simple slicing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[0:1])
```

运行该示例将返回带有第一个元素的子数组。

```
[11]
```

我们也可以在切片中使用负索引。例如，我们可以通过在-2 处开始切片（第二个最后一项）并且不指定'to'索引来切片列表中的最后两个项目;将切片带到维度的末尾。

```
# simple slicing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[-2:])
```

运行该示例仅返回包含最后两个项的子数组。

```
[44 55]
```

### 二维切片

让我们看看你最有可能在机器学习中使用的二维切片的两个例子。

#### 分割输入和输出功能

通常将加载的数据拆分为输入变量（X）和输出变量（y）。

我们可以通过将所有行和所有列切片到最后一列，然后分别索引最后一列来完成此操作。

对于输入要素，我们可以通过在行索引中指定'：'来选择除最后一行之外的所有行和所有列，并且在列 index 中指定-1。

```
X = [:, :-1]
```

对于输出列，我们可以使用'：'再次选择所有行，并通过指定-1 索引仅索引最后一列。

```
y = [:, -1]
```

将所有这些放在一起，我们可以将 3 列 2D 数据集分成输入和输出数据，如下所示：

```
# split input and output
from numpy import array
# define array
data = array([[11, 22, 33],
		[44, 55, 66],
		[77, 88, 99]])
# separate data
X, y = data[:, :-1], data[:, -1]
print(X)
print(y)
```

运行该示例将打印分隔的 X 和 y 元素。注意，X 是 2D 阵列，y 是 1D 阵列。

```
[[11 22]
 [44 55]
 [77 88]]
[33 66 99]
```

#### 拆分列车和测试行

将加载的数据集拆分为单独的列车和测试集是很常见的。

这是行的分割，其中一些部分将用于训练模型，剩余部分将用于估计训练模型的技能。

这将涉及通过在第二个维度索引中指定“：”来切片所有列。训练数据集将是从开始到分割点的所有行。

```
dataset
train = data[:split, :]
```

测试数据集将是从拆分点到维度末尾的所有行。

```
test = data[split:, :]
```

将所有这些放在一起，我们可以将数据集拆分为 2 的设计分割点。

```
# split train and test
from numpy import array
# define array
data = array([[11, 22, 33],
		[44, 55, 66],
		[77, 88, 99]])
# separate data
split = 2
train,test = data[:split,:],data[split:,:]
print(train)
print(test)
```

运行该示例选择前两行进行训练，选择最后一行进行测试。

```
[[11 22 33]
[44 55 66]]
[[77 88 99]]
```

## 4.阵列重塑

切片数据后，您可能需要重新整形。

例如，某些库（例如 scikit-learn）可能要求将输出变量（y）的一维数组形成为具有一列的二维数组，并且每列的结果。

一些算法，如 Keras 中的长短期记忆递归神经网络，需要将输入指定为由样本，时间步长和特征组成的三维阵列。

了解如何重塑 NumPy 数组以使您的数据满足特定 Python 库的期望非常重要。我们将看看这两个例子。

### 数据形状

NumPy 数组具有 shape 属性，该属性返回数组每个维度长度的元组。

例如：

```
# array shape
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
```

运行该示例会打印一维的元组。

```
(5,)
```

对于二维数组，返回具有两个长度的元组。

```
# array shape
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print(data.shape)
```

运行该示例将返回一个包含行数和列数的元组。

```
(3, 2)
```

您可以在形状维度中使用数组维度的大小，例如指定参数。

可以像数组一样访问元组的元素，行数为第 0 个索引，列数为第 1 个索引。例如：

```
# array shape
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])
```

运行该示例可访问每个维度的特定大小。

```
Rows: 3
Cols: 2
```

### 重塑 1D 到 2D 阵列

通常需要将一维阵列重塑为具有一列和多个阵列的二维阵列。

NumPy 在 NumPy 数组对象上提供 reshape（）函数，可用于重塑数据。

reshape（）函数采用一个参数来指定数组的新形状。在将一维数组重新整形为具有一列的二维数组的情况下，元组将是第一维（data.shape [0]）的数组形状和第二维的 1。

```
data = data.reshape((data.shape[0], 1))
```

综上所述，我们得到以下工作示例。

```
# reshape 1D array
from numpy import array
from numpy import reshape
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)
```

运行该示例将打印一维数组的形状，将数组重新整形为包含 1 列的 5 行，然后打印此新形状。

```
(5,)
(5, 1)
```

### 重塑 2D 到 3D 阵列

通常需要重新形成二维数据，其中每行表示序列为三维阵列，用于期望一个或多个时间步长和一个或多个特征的多个样本的算法。

一个很好的例子是 Keras 深度学习库中的 [LSTM 递归神经网络](https://keras.io/layers/recurrent/#lstm)模型。

可以直接使用重塑功能，指定新的维度。这是清楚的，其中每个序列具有多个时间步长，每个时间步长具有一个观察（特征）。

我们可以使用数组上 shape 属性的大小来指定样本（行）和列的数量（时间步长），并将要素数量固定为 1。

```
data.reshape((data.shape[0], data.shape[1], 1))
```

Putting this all together, we get the following worked example.

```
# reshape 2D array
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print(data.shape)
# reshape
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)
```

首先运行示例打印 2D 阵列中每个维度的大小，重新整形数组，然后总结新 3D 阵列的形状。

```
(3, 2)
(3, 2, 1)
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [Python 非正式介绍](https://docs.python.org/3/tutorial/introduction.html)
*   [在 NumPy API 中创建数组](https://docs.scipy.org/doc/numpy/user/basics.creation.html)
*   [NumPy API 中的索引和切片](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
*   [NumPy API 中的基本索引](https://docs.scipy.org/doc/numpy/user/basics.indexing.html)
*   [NumPy 形状属性](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)
*   [NumPy reshape（）函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)

## 摘要

在本教程中，您了解了如何使用 Python 访问和重塑 NumPy 数组中的数据。

具体来说，你学到了：

*   如何将列表数据转换为 NumPy 数组。
*   如何使用 Pythonic 索引和切片访问数据。
*   如何调整数据大小以满足某些机器学习 API 的期望。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。