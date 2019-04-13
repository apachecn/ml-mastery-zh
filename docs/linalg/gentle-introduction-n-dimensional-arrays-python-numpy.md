# 用 NumPy 轻松介绍 Python 中的 N 维数组

> 原文： [https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)

数组是机器学习中使用的主要数据结构。

在 Python 中，来自 NumPy 库的数组（称为 N 维数组或 ndarray）被用作表示数据的主要数据结构。

在本教程中，您将发现 NumPy 中的 N 维数组，用于表示 Python 中的数值和操作数据。

完成本教程后，您将了解：

*   什么是 ndarray 以及如何在 Python 中创建和检查数组。
*   用于创建具有默认值的新空数组和数组的关键函数。
*   如何组合现有数组以创建新数组。

让我们开始吧。

![A Gentle Introduction to N-Dimensional Arrays in Python with NumPy](img/dc322aace9030533fafbac6855cb7a83.jpg)

使用 NumPy
在 Python 中对 N 维数组的简要介绍 [patrickkavanagh](https://www.flickr.com/photos/patrick_k59/9216134592/) ，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  NumPy N 维数组
2.  创建数组的函数
3.  结合数组

## NumPy N 维数组

NumPy 是一个 Python 库，可用于科学和数字应用程序，是用于线性代数运算的工具。

NumPy 中的主要数据结构是 ndarray，它是 N 维数组的简写名称。使用 NumPy 时，ndarray 中的数据简称为数组。

它是内存中固定大小的数组，包含相同类型的数据，例如整数或浮点值。

可以通过阵列上的“dtype”属性访问数组支持的数据类型。可以通过“shape”属性访问数组的维度，该属性返回描述每个维度长度的元组。还有许多其他属性。在这里了解更多：

*   [N 维数组](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html)

从数据或简单的 Python 数据结构（如列表）创建数组的简单方法是使用 array（）函数。

下面的示例创建一个包含 3 个浮点值的 Python 列表，然后从列表中创建一个 ndarray 并访问数组的形状和数据类型。

```
# create array
from numpy import array
l = [1.0, 2.0, 3.0]
a = array(l)
print(a)
print(a.shape)
print(a.dtype)
```

运行该示例打印 ndarray 的内容，形状是具有 3 个元素的一维数组，以及数据类型，即 64 位浮点。

```
[ 1\. 2\. 3.]
(3,)
float64
```

## 创建数组的函数

有更多便利功能可用于创建您可能遇到或需要使用的固定大小的阵列。

我们来看几个。您可以在此处查看完整列表：

*   [数组创建例程](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-creation.html)

### 空

empty（）函数将创建指定形状的新数组。

函数的参数是一个数组或元组，它指定要创建的数组的每个维度的长度。创建的数组的值或内容将是随机的，需要在使用前进行分配。

下面的示例创建一个空的 3×3 二维数组。

```
# create empty array
from numpy import empty
a = empty([3,3])
print(a)
```

运行该示例将打印空数组的内容。您的具体数组内容会有所不同。

```
[[ 0.00000000e+000 0.00000000e+000 2.20802703e-314]
[ 2.20803350e-314 2.20803353e-314 2.20803356e-314]
[ 2.20803359e-314 2.20803362e-314 2.20803366e-314]]
```

### 零

zeros（）函数将创建一个指定大小的新数组，其内容填充零值。

函数的参数是一个数组或元组，它指定要创建的数组的每个维度的长度。

下面的示例创建一个 3×5 零二维数组。

```
# create zero array
from numpy import zeros
a = zeros([3,5])
print(a)
```

运行该示例将打印创建的零数组的内容。

```
[[ 0\. 0\. 0\. 0\. 0.]
[ 0\. 0\. 0\. 0\. 0.]
[ 0\. 0\. 0\. 0\. 0.]]
```

### 那些

ones（）函数将创建一个指定大小的新数组，其内容填充一个值。

The argument to the function is an array or tuple that specifies the length of each dimension of the array to create.

下面的示例创建一个 5 元素的一维数组。

```
# create one array
from numpy import ones
a = ones([5])
print(a)
```

运行该示例将打印创建的 one 数组的内容。

```
[ 1\. 1\. 1\. 1\. 1.]
```

## 结合数组

NumPy 提供了许多函数来从现有数组创建新数组。

让我们来看看您可能需要或遇到的两个最流行的功能。

### 垂直堆栈

给定两个或更多现有数组，您可以使用 vstack（）函数垂直堆叠它们。

例如，给定两个一维数组，您可以通过垂直堆叠它们来创建一个包含两行的新二维数组。

这在以下示例中进行了演示。

```
# vstack
from numpy import array
from numpy import vstack
a1 = array([1,2,3])
print(a1)
a2 = array([4,5,6])
print(a2)
a3 = vstack((a1, a2))
print(a3)
print(a3.shape)
```

首先运行该示例打印两个单独定义的一维数组。阵列垂直堆叠，形成一个新的 2×3 阵列，其内容和形状被打印出来。

```
[1 2 3]

[4 5 6]

[[1 2 3]
[4 5 6]]

(2, 3)
```

### 水平堆栈

给定两个或更多现有数组，您可以使用 hstack（）函数水平堆栈它们。

例如，给定两个一维数组，您可以创建一个新的一维数组或一行，第一个和第二个数组的列连接在一起。

This is demonstrated in the example below.

```
# hstack
from numpy import array
from numpy import hstack
a1 = array([1,2,3])
print(a1)
a2 = array([4,5,6])
print(a2)
a3 = hstack((a1, a2))
print(a3)
print(a3.shape)
```

首先运行该示例打印两个单独定义的一维数组。然后水平堆叠阵列，产生具有 6 个元素的新的一维阵列，其内容和形状被打印。

```
[1 2 3]

[4 5 6]

[1 2 3 4 5 6]

(6,)
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   尝试使用不同的方法创建数组到您自己的大小或新数据。
*   找到并开发另外 3 个用于创建数组的 NumPy 函数的示例。
*   找到并开发另外 3 个用于组合数组的 NumPy 函数的示例。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [Python for Data Analysis](http://amzn.to/2B1sfXi) ，2017。
*   [优雅的 SciPy](http://amzn.to/2yujXnT) ，2017。
*   [NumPy 指南](http://amzn.to/2j3kEzd)，2015 年。

### 参考

*   [NumPy 参考](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
*   [N 维数组](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html)
*   [数组创建例程](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-creation.html)

### API

*   [numpy.array（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html)
*   [numpy.empty（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.empty.html)
*   [numpy.zeros（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.zeros.html)
*   [numpy.ones（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ones.html)
*   [numpy.vstack（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.vstack.html)
*   [numpy.hstack（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.hstack.html)

## 摘要

在本教程中，您在 NumPy 中发现了 N 维数组，用于表示 Python 中的数值和操作数据。

具体来说，你学到了：

*   什么是 ndarray 以及如何在 Python 中创建和检查数组。
*   用于创建具有默认值的新空数组和数组的关键函数。
*   如何组合现有数组以创建新数组。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。