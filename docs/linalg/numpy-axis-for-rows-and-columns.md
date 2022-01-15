# 如何在 NumPy 中为行和列设置轴

> 原文：<https://machinelearningmastery.com/numpy-axis-for-rows-and-columns/>

NumPy 数组为在 Python 中存储和操作数据提供了一种快速有效的方法。

它们对于在机器学习中将数据表示为向量和矩阵特别有用。

NumPy 数组中的数据可以通过列和行索引直接访问，这相当简单。然而，有时我们必须对数据数组执行操作，例如按行或列对值求和或求平均值，这需要指定操作的轴。

不幸的是，NumPy 数组上的列和行操作与我们从行和列索引中获得的直觉不匹配，这可能会给初学者和经验丰富的机器学习从业者带来困惑。具体来说，像求和这样的操作可以使用轴=0 按列**执行，使用轴=1** 按行**执行。**

在本教程中，您将发现如何按行和按列访问和操作 NumPy 数组。

完成本教程后，您将知道:

*   如何用数据的行和列定义 NumPy 数组？
*   如何通过行索引和列索引访问 NumPy 数组中的值。
*   如何按行轴和列轴对 NumPy 数组执行操作？

我们开始吧。

![How to Set NumPy Axis for Rows and Columns in Python](img/964574d204735a1a32ef99a3a2eaf0f4.png)

如何在 Python 中为行和列设置 NumPy 轴
图片由[乔纳森·卡特勒](https://flickr.com/photos/joncutrer/43124145784/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  带行和列的 NumPy 数组
2.  NumPy 数组中的数据行和列
3.  按行和列的 NumPy 数组操作
    1.  轴=无阵列操作
    2.  轴=0 列式操作
    3.  轴=1 行操作

## 带行和列的 NumPy 数组

在深入讨论 NumPy 阵列轴之前，让我们刷新一下对 NumPy 阵列的了解。

通常在 Python 中，我们使用数字列表或数字列表。例如，我们可以将由两行三个数字组成的二维矩阵定义为数字列表，如下所示:

```py
...
# define data as a list
data = [[1,2,3], [4,5,6]]
```

NumPy 数组允许我们以有效的方式定义和操作向量和矩阵，例如，比简单的 Python 列表更有效。NumPy 阵列被称为 NDArrays，并且可以具有几乎任意数量的维度，尽管在机器学习中，我们最常用的是 1D 和 2D 阵列(或用于图像的 3D 阵列)。

例如，我们可以通过 [asarray()函数](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)将列表矩阵列表转换为 NumPy 数组:

```py
...
# convert to a numpy array
data = asarray(data)
```

我们可以直接打印数组，并期望看到两行数字，其中每行有三个数字或列。

```py
...
# summarize the array content
print(data)
```

我们可以通过打印“ *shape* ”属性来总结一个数组的维度，这个属性是一个元组，其中元组中值的个数定义维度的个数，每个位置的整数定义维度的大小。

例如，对于两行三列，我们期望我们的数组的形状是(2，3)。

```py
...
# summarize the array shape
print(data.shape)
```

将这些结合在一起，下面列出了一个完整的例子。

```py
# create and summarize a numpy array
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# summarize the array content
print(data)
# summarize the array shape
print(data.shape)
```

运行该示例将我们的数据定义为列表，将其转换为 NumPy 数组，然后打印数据和形状。

我们可以看到，当数组被打印时，它有两行三列的预期形状。然后我们可以看到印刷的形状符合我们的期望。

```py
[[1 2 3]
 [4 5 6]]
(2, 3)
```

有关 NumPy 数组基础知识的更多信息，请参见教程:

*   [Python 中 NumPy 数组的温和介绍](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)

目前为止，一切顺利。

但是我们如何按行或列访问数组中的数据呢？更重要的是，我们如何按行或按列对数组执行操作？

让我们仔细看看这些问题。

## NumPy 数组中的数据行和列

“*形状*”属性总结了我们数据的维度。

重要的是，第一维定义行数，第二维定义列数。例如(2，3)定义了一个两行三列的数组，正如我们在上一节中看到的。

我们可以通过从索引 0 枚举到数组形状的第一维，例如形状[0]，来枚举数组中的每一行数据。我们可以通过行和列索引来访问数组中的数据。

例如，数据[0，0]是第一行和第一列的值，而数据[0，]是第一行和所有列的值，例如我们矩阵中完整的第一行。

下面的示例枚举数据中的所有行，并依次打印每一行。

```py
# enumerate rows in a numpy array
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# step through rows
for row in range(data.shape[0]):
	print(data[row, :])
```

不出所料，结果显示了第一行数据，然后是第二行数据。

```py
[1 2 3]
[4 5 6]
```

我们可以为列实现同样的效果。

也就是说，我们可以按列枚举数据。例如，数据[:，0]访问第一列的所有行。我们可以枚举从第 0 列到最后一列的所有列，由“*形状*属性的第二维度定义，例如形状【1】。

下面的示例通过枚举矩阵中的所有列来演示这一点。

```py
# enumerate columns in a numpy array
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# step through columns
for col in range(data.shape[1]):
	print(data[:, col])
```

运行该示例枚举并打印矩阵中的每一列。

假设矩阵有三列，我们可以看到结果是我们打印了三列，每列都是一维向量。也就是说，第 1 列(索引 0)具有值 1 和 4，第 2 列(索引 1)具有值 2 和 5，第 3 列(索引 2)具有值 3 和 6。

它看起来很有趣，因为我们的专栏看起来不像专栏；它们是侧翻的，而不是垂直的。

```py
[1 4]
[2 5]
[3 6]
```

现在我们知道如何按列和行访问 numpy 数组中的数据了。

到目前为止，一切都很好，但是按列和数组对数组进行操作呢？那是下一个。

## 按行和列的 NumPy 数组操作

我们经常需要按列或按行对 NumPy 数组执行操作。

例如，我们可能需要按行或按列对数据矩阵的值求和或计算平均值。

这可以通过使用 *sum()* 或 *mean()* NumPy 函数并指定要在其上执行操作的“*轴*来实现。

我们可以将轴指定为要执行操作的维度，基于我们如何解释数组的“*形状*”以及如何索引数组中的数据，这个维度与我们的直觉不匹配。

**因此，这给初学者造成了最大的困惑**。

也就是说，**轴=0** 将按列执行操作，**轴=1** 将按行执行操作。我们还可以将轴指定为无，这将对整个数组执行操作。

总之:

*   **轴=无**:阵列式应用操作。
*   **轴=0** :对每列的所有行按列应用操作。
*   **轴=1** :逐行应用操作，跨越每行的所有列。

让我们用一个具体的例子来说明这一点。

我们将通过三个轴中的每一个轴对数组中的值求和。

### 轴=无阵列操作

在 NumPy 数组上执行操作时，设置**轴=无**将对整个数组执行操作。

这通常是大多数操作的默认值，如求和、求平均值、标准等。

```py
...
# sum data by array
result = data.sum(axis=None)
```

下面的例子演示了对一个数组中的所有值求和，例如按数组操作。

```py
# sum values array-wise
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# summarize the array content
print(data)
# sum data by array
result = data.sum(axis=None)
# summarize the result
print(result)
```

运行该示例首先打印数组，然后按数组执行求和操作并打印结果。

我们可以看到数组有六个值，如果我们手动将它们相加，它们的总和将达到 21，并且按数组方式执行的求和操作的结果与这个期望值相匹配。

```py
[[1 2 3]
 [4 5 6]]

21
```

### 轴=0 列式操作

在 NumPy 数组上执行操作时，将**轴设置为 0** 将按列执行操作，即跨每列的所有行执行操作。

```py
...
# sum data by column
result = data.sum(axis=0)
```

例如，假设我们的数据有两行三列:

```py
Data = [[1, 2, 3],
		 4, 5, 6]]
```

我们预计轴=0 的按列求和将产生三个值，每列一个，如下所示:

*   **第 1 列** : 1 + 4 = 5
*   **第 2 栏** : 2 + 5 = 7
*   **第 3 列** : 3 + 6 = 9

下面的示例演示了按列对数组中的值求和，例如按列操作。

```py
# sum values column-wise
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# summarize the array content
print(data)
# sum data by column
result = data.sum(axis=0)
# summarize the result
print(result)
```

运行该示例首先打印数组，然后逐列执行求和操作并打印结果。

我们可以看到数组有六个值，如预期的那样有两行三列；然后，我们可以看到列式运算的结果是一个有三个值的向量，其中一个值代表与我们的期望相匹配的每一列的总和。

```py
[[1 2 3]
 [4 5 6]]
[5 7 9]
```

### 轴=1 行操作

在 NumPy 数组上执行操作时，设置*轴=1* 将逐行执行操作，即每行的所有列。

```py
...
# sum data by row
result = data.sum(axis=1)
```

例如，假设我们的数据有两行三列:

```py
Data = [[1, 2, 3],
		 4, 5, 6]]
```

我们预计轴=1 的行方向求和将产生两个值，每行一个，如下所示:

*   **第 1 排** : 1 + 2 + 3 = 6
*   **第二排** : 4 + 5 + 6 = 15

下面的示例演示了逐行对数组中的值求和，例如逐行操作。

```py
# sum values row-wise
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# summarize the array content
print(data)
# sum data by row
result = data.sum(axis=1)
# summarize the result
print(result)
```

运行该示例首先打印数组，然后逐行执行求和操作并打印结果。

我们可以看到数组有六个值，如预期的那样有两行三列；然后，我们可以在一个向量中看到逐行操作的结果，该向量有两个值，一个值是与我们的期望相匹配的每一行的总和。

```py
[[1 2 3]
 [4 5 6]]
[ 6 15]
```

我们现在对如何在 NumPy 数组上执行操作时适当地设置 axis 有了具体的想法。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [Python 中 NumPy 数组的温和介绍](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)
*   [如何为机器学习对 NumPy 数组进行索引、切片和整形](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)
*   [用 NumPy 阵列温和介绍广播](https://machinelearningmastery.com/broadcasting-with-numpy-arrays/)

### 蜜蜂

*   num py . ndaarray API。
*   num py . sum API。
*   num py . asar ray API。
*   [NumPy 词汇表:沿轴](https://numpy.org/doc/stable/glossary.html)

## 摘要

在本教程中，您发现了如何按行和按列访问和操作 NumPy 数组。

具体来说，您了解到:

*   如何用数据的行和列定义 NumPy 数组？
*   如何通过行索引和列索引访问 NumPy 数组中的值。
*   如何按行轴和列轴对 NumPy 数组执行操作？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。