# 温和地介绍机器学习的特征分解，特征值和特征向量

> 原文： [https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/)

矩阵分解是用于将矩阵减少到其组成部分的有用工具，以简化一系列更复杂的操作。

也许最常用的矩阵分解类型是特征分解，它将矩阵分解为特征向量和特征值。这种分解也在机器学习中使用的方法中起作用，例如在主成分分析方法或 PCA 中。

在本教程中，您将发现线性代数中的特征分解，特征向量和特征值。

完成本教程后，您将了解：

*   特征分解是什么以及特征向量和特征值的作用。
*   如何使用 NumPy 在 Python 中计算特征分解。
*   如何确定向量是一个特征向量，以及如何从特征向量和特征值重建矩阵。

让我们开始吧。

![Gentle Introduction to Eigendecomposition, Eigenvalues, and Eigenvectors for Machine Learning](img/ee435b96d80b3e79ee37a0816aec4413.jpg)

用于机器学习的特征分解，特征值和特征向量的温和介绍
照片由 [Mathias Appel](https://www.flickr.com/photos/mathiasappel/26154953033/) 拍摄，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  矩阵的特征分解
2.  特征向量和特征值
3.  特征分解的计算
4.  确认特征向量和特征值
5.  重建原始矩阵

## 矩阵的特征分解

矩阵的特征分解是一种分解，其涉及将方阵分解为一组特征向量和特征值。

> 最广泛使用的一种矩阵分解称为特征分解，其中我们将矩阵分解为一组特征向量和特征值。

- 第 42 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

如果向量满足以下等式，则向量是矩阵的特征向量。

```
A . v = lambda . v
```

这称为特征值方程，其中 A 是我们正在分解的父方矩阵，v 是矩阵的特征向量，lambda 是小写的希腊字母，表示特征值标量。

或者没有点符号。

```
Av = lambdav
```

对于父矩阵的每个维度，矩阵可以具有一个特征向量和特征值。并非所有的方形矩阵都可以分解为特征向量和特征值，有些只能以需要复数的方式进行分解。可以将父矩阵显示为特征向量和特征值的乘积。

```
A = Q . diag(V) . Q^-1
```

或者，没有点符号。

```
A = Qdiag(V)Q^-1
```

其中 Q 是由特征向量组成的矩阵，diag（V）是由沿对角线的特征值（有时用大写λ表示）组成的对角矩阵，并且 Q ^ -1 是由特征向量组成的矩阵的逆。

> 但是，我们经常希望将矩阵分解为它们的特征值和特征向量。这样做可以帮助我们分析矩阵的某些属性，就像将整数分解为其素因子一样可以帮助我们理解该整数的行为。

- 第 43 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

Eigen 不是名称，例如该方法不以“Eigen”命名; eigen（发音为 eye-gan）是德语单词，意思是“拥有”或“先天”，如属于父矩阵。

分解操作不会导致矩阵的压缩;相反，它将其分解为组成部分，以使矩阵上的某些操作更容易执行。与其他矩阵分解方法一样，特征分解用作元素以简化其他更复杂的矩阵运算的计算。

> 当它们乘以 A 时，几乎所有向量都会改变方向。某些特殊向量 x 与 Ax 的方向相同。那些是“特征向量”。将特征向量乘以 A，向量 Ax 是原始 x 的λ倍。 [...]特征值 lambda 告诉特殊向量 x 是否被拉伸或收缩或反转或保持不变 - 当它乘以 A.

- 第 289 页，[线性代数导论](http://amzn.to/2AZ7R8j)，第五版，2016 年。

特征分解也可用于计算主成分分析方法或 PCA 中矩阵的主成分，可用于减少机器学习中数据的维数。

## 特征向量和特征值

特征向量是单位向量，这意味着它们的长度或幅度等于 1.0。它们通常被称为右向量，其仅表示列向量（与行向量或左向量相对）。右向量是我们理解它们的向量。

特征值是应用于特征向量的系数，其赋予向量其长度或幅度。例如，负特征值可以反转特征向量的方向作为缩放它的一部分。

仅具有正特征值的矩阵被称为正定矩阵，而如果特征值都是负的，则其被称为负定矩阵。

> 根据矩阵的特征值及其特征向量分解矩阵，可以对矩阵的属性进行有价值的分析。当我们使用矩阵的特征分解时，某些矩阵计算（如计算矩阵的幂）变得更加容易。

- 第 262 页，[无线性代数废话指南](http://amzn.to/2k76D4C)，2017 年

## 特征分解的计算

使用有效的迭代算法在方阵上计算特征分解，我们不会详细讨论。

通常首先找到特征值，然后找到特征向量以将该方程求解为一组系数。

可以使用 eig（）函数在 NumPy 中计算特征分解。

以下示例首先定义 3×3 方阵。在返回特征值和特征向量的矩阵上计算特征分解。

```
# eigendecomposition
from numpy import array
from numpy.linalg import eig
# define matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# calculate eigendecomposition
values, vectors = eig(A)
print(values)
print(vectors)
```

首先运行该示例打印定义的矩阵，然后是特征值和特征向量。更具体地，特征向量是右侧特征向量并且被归一化为单位长度。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

[  1.61168440e+01  -1.11684397e+00  -9.75918483e-16]

[[-0.23197069 -0.78583024  0.40824829]
 [-0.52532209 -0.08675134 -0.81649658]
 [-0.8186735   0.61232756  0.40824829]]
```

## 确认特征向量和特征值

我们可以确认向量确实是矩阵的特征向量。

我们通过将候选特征向量乘以特征向量并将结果与​​特征值进行比较来做到这一点。

首先，我们将定义一个矩阵，然后计算特征值和特征向量。然后，我们将测试第一个向量和值是否实际上是矩阵的特征值和特征向量。我们知道它们是，但这是一个很好的练习。

特征向量作为矩阵返回，其具有与父矩阵相同的维度，其中每列是特征向量，例如，第一个特征向量是向量[：，0]。特征值作为列表返回，其中返回数组中的值索引通过列索引与特征向量配对，例如，值[0]的第一个特征值与向量[：，0]处的第一个特征向量配对。

```
# confirm eigenvector
from numpy import array
from numpy.linalg import eig
# define matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# calculate eigendecomposition
values, vectors = eig(A)
# confirm first eigenvector
B = A.dot(vectors[:, 0])
print(B)
C = vectors[:, 0] * values[0]
print(C)
```

该示例将原始矩阵与第一个特征向量相乘，并将其与第一个特征向量乘以第一个特征值进行比较。

运行该示例将打印这两个乘法的结果，这些乘法显示相同的结果向量，正如我们所期望的那样。

```
[ -3.73863537  -8.46653421 -13.19443305]

[ -3.73863537  -8.46653421 -13.19443305]
```

## 重建原始矩阵

我们可以反转过程并仅在给定特征向量和特征值的情况下重建原始矩阵。

首先，必须将特征向量列表转换为矩阵，其中每个向量成为一行。特征值需要排列成对角矩阵。 NumPy diag（）函数可用于此目的。

接下来，我们需要计算特征向量矩阵的逆，我们可以使用 inv（）NumPy 函数来实现。最后，这些元素需要与 dot（）函数相乘。

```
# reconstruct matrix
from numpy import diag
from numpy import dot
from numpy.linalg import inv
from numpy import array
from numpy.linalg import eig
# define matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# calculate eigenvectors and eigenvalues
values, vectors = eig(A)
# create matrix from eigenvectors
Q = vectors
# create inverse of eigenvectors matrix
R = inv(Q)
# create diagonal matrix from eigenvalues
L = diag(values)
# reconstruct the original matrix
B = Q.dot(L).dot(R)
print(B)
```

该示例再次计算特征值和特征向量，并使用它们来重建原始矩阵。

首先运行该示例打印原始矩阵，然后从与原始矩阵匹配的特征值和特征向量重建矩阵。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

[[ 1\.  2\.  3.]
 [ 4\.  5\.  6.]
 [ 7\.  8\.  9.]]
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   使用您自己的数据使用每个操作创建 5 个示例。
*   为定义为列表列表的矩阵手动实现每个矩阵操作。
*   搜索机器学习论文并找到每个正在使用的操作的示例。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 6.1 节特征值和特征向量。 [线性代数无废话指南](http://amzn.to/2k76D4)，2017 年。
*   第 6 章特征值和特征向量，[线性代数导论](http://amzn.to/2AZ7R8j)，第 5 版，2016 年。
*   第 2.7 节特征分解，[深度学习](http://amzn.to/2B3MsuU)，2016 年。
*   第 5 章特征值，特征向量和不变子空间，[线性代数完成权](http://amzn.to/2BGuEqI)，第三版，2015 年。
*   第 24 讲，特征值问题，[数值线性代数](http://amzn.to/2BI9kRH)，1997。

### API

*   [numpy.linalg.eig（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eig.html)
*   [numpy.diag（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.diag.html)
*   [numpy.dot（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.dot.html)
*   [numpy.linalg.inv（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.inv.html)

### 用品

*   [eigen on Wiktionary](https://en.wiktionary.org/wiki/eigen)
*   [特征值和特征向量](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
*   [矩阵的特征分解](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
*   [特征值算法](https://en.wikipedia.org/wiki/Eigenvalue_algorithm)
*   [矩阵分解](https://en.wikipedia.org/wiki/Matrix_decomposition)

## 摘要

在本教程中，您发现了线性代数中的特征分解，特征向量和特征值。

具体来说，你学到了：

*   特征分解是什么以及特征向量和特征值的作用。
*   如何使用 NumPy 在 Python 中计算特征分解。
*   如何确定向量是一个特征向量，以及如何从特征向量和特征值重建矩阵。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。