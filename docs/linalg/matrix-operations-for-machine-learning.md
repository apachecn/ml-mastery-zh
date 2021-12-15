# 机器学习中的矩阵运算的温和介绍

> 原文： [https://machinelearningmastery.com/matrix-operations-for-machine-learning/](https://machinelearningmastery.com/matrix-operations-for-machine-learning/)

矩阵运算用于许多机器学习算法的描述中。

一些操作可以直接用于求解关键方程，而其他操作在描述和使用更复杂的矩阵运算中提供了有用的简写或基础。

在本教程中，您将发现在机器学习方法的描述中使用的重要线性代数矩阵运算。

完成本教程后，您将了解：

*   用于翻转矩阵尺寸的转置操作。
*   用于求解线性方程组的逆操作。
*   Trace 和 Determinant 操作在其他矩阵运算中用作简写符号。

让我们开始吧。

![A Gentle Introduction to Matrix Operations for Machine Learning](img/21b12f89460264157ed0bfb833eeb1e6.jpg)

机器学习矩阵操作的温和介绍
[Andrej](https://www.flickr.com/photos/adundovi/30638394675/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  颠倒
2.  逆温
3.  跟踪
4.  行列式
5.  秩

## 颠倒

可以转置定义的矩阵，这将创建一个新的矩阵，其中列和行的数量被翻转。

这由矩阵旁边的上标“T”表示。

```
C = A^T
```

可以通过矩阵从左上角到右下角绘制不可见的对角线，在该矩阵上可以翻转矩阵以给出转置。

```
     a11, a12
A = (a21, a22)
     a31, a32

       a11, a21, a31
A^T = (a12, a22, a32)
```

如果矩阵是对称的，则操作无效，例如，在不可见对角线两侧的相同位置具有相同数量的列和行以及相同的值。

> A ^ T 的列是 A 的行。

- 第 109 页，[线性代数简介](http://amzn.to/2AZ7R8j)，第五版，2016 年。

我们可以通过调用 T 属性在 NumPy 中转置矩阵。

```
# transpose matrix
from numpy import array
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
C = A.T
print(C)
```

运行该示例首先打印定义的矩阵，然后打印转置的版本。

```
[[1 2]
 [3 4]
 [5 6]]

[[1 3 5]
 [2 4 6]]
```

转置操作提供了用作许多矩阵运算中的元素的简短符号。

## 逆温

矩阵求逆是一个过程，它找到另一个矩阵，当与矩阵相乘时，得到一个单位矩阵。

给定矩阵 A，找到矩阵 B，使得 AB 或 BA = In。

```
AB = BA = In
```

反转矩阵的操作由矩阵旁边的-1 上标表示;例如，A ^ -1。该操作的结果称为原始矩阵的逆;例如，B 是 A 的倒数。

```
B = A^-1
```

如果存在导致单位矩阵的另一矩阵，则矩阵是可逆的，其中并非所有矩阵都是可逆的。不可逆的方阵被称为单数。

> 无论 A 做什么，A ^ -1 撤消。

- 第 83 页，[线性代数简介](http://amzn.to/2AZ7R8j)，第五版，2016 年。

矩阵求逆运算不是直接计算的，而是通过数值运算发现倒置矩阵，其中可以使用一套有效的方法，通常涉及矩阵分解的形式。

> 但是，A ^ -1 主要用作理论工具，实际上不应该在大多数软件应用程序中实际使用。

- 第 37 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

可以使用 inv（）函数在 NumPy 中反转矩阵。

```
# invert matrix
from numpy import array
from numpy.linalg import inv
# define matrix
A = array([[1.0, 2.0], [3.0, 4.0]])
print(A)
# invert matrix
B = inv(A)
print(B)
# multiply A and B
I = A.dot(B)
print(I)
```

首先，我们定义一个小的 2×2 矩阵，然后计算矩阵的逆，然后通过将其与原始矩阵相乘来确认逆，以给出单位矩阵。

运行该示例将打印原始，反向和标识矩阵。

```
[[ 1\.  2.]
 [ 3\.  4.]]

[[-2\.   1\. ]
 [ 1.5 -0.5]]

[[  1.00000000e+00   0.00000000e+00]
 [  8.88178420e-16   1.00000000e+00]]
```

矩阵求逆用作求解方程组的一个操作，该方程组成矩阵方程，我们有兴趣找到未知数的向量。一个很好的例子是在线性回归中找到系数值的向量。

## 跟踪

方形矩阵的轨迹是矩阵主对角线上的值的总和（从左上到右下）。

> 跟踪运算符给出矩阵的所有对角线条目的总和

- 第 46 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

使用符号“tr（A）”描述在方阵上计算轨迹的操作，其中 A 是在其上执行操作的方阵。

```
tr(A)
```

迹线计算为对角线值的总和;例如，在 3×3 矩阵的情况下：

```
tr(A) = a11 + a22 + a33
```

或者，使用数组表示法：

```
tr(A) = A[0,0] + A[1,1] + A[2,2]
```

我们可以使用 trace（）函数计算 NumPy 中矩阵的轨迹。

```
# trace
from numpy import array
from numpy import trace
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
B = trace(A)
print(B)
```

首先，创建 3×3 矩阵，然后计算轨迹。

运行该示例，首先打印数组，然后打印跟踪。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

15
```

单独地，跟踪操作并不令人感兴趣，但它提供了更简单的表示法，并且它在其他键矩阵运算中用作元素。

## 行列式

方阵的行列式是矩阵体积的标量表示。

> 行列式描述了构成矩阵行的向量的相对几何。更具体地说，矩阵 A 的行列式告诉你一个盒子的体积，其边由 A 行给出。

- 第 119 页，[无线性代数废话指南](http://amzn.to/2k76D4C)，2017 年

它由“det（A）”符号或| A |表示，其中 A 是我们计算行列式的矩阵。

```
det(A)
```

从矩阵的元素计算方阵的行列式。从技术上讲，行列式是矩阵的所有特征值的乘积。

行列式的直觉是它描述了矩阵在将它们相乘时对另一个矩阵进行缩放的方式。例如，1 的行列式保留了另一个矩阵的空间。行列式为 0 表示矩阵不能反转。

> 方阵的行列式是单个数。 [...]它立即告诉矩阵是否可逆。当矩阵没有逆时，行列式为零。

- 第 247 页，[线性代数简介](http://amzn.to/2AZ7R8j)，第五版，2016 年。

在 NumPy 中，可以使用 det（）函数计算矩阵的行列式。

```
# trace
from numpy import array
from numpy.linalg import det
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
B = det(A)
print(B)
```

首先，定义 3×3 矩阵，然后计算矩阵的行列式。

首先运行该示例打印定义的矩阵，然后打印矩阵的行列式。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

-9.51619735393e-16
```

与跟踪操作一样，单独使用行列式操作并不感兴趣，但它提供了更简单的表示法，并且它在其他键矩阵运算中用作元素。

## 矩阵排名

矩阵的秩是矩阵中线性独立的行或列的数量的估计。

矩阵 M 的秩通常表示为函数 rank（）。

```
rank(A)
```

排名的直觉是将其视为矩阵内所有向量所跨越的维数。例如，等级 0 表示所有向量跨越一个点，等级 1 表示所有向量跨越一条线，等级 2 表示所有向量跨越一个二维平面。

通过数字估计秩，通常使用矩阵分解方法。一种常见的方法是使用奇异值分解或简称 SVD。

NumPy 提供了 matrix_rank（）函数来计算数组的排名。它使用 SVD 方法来估计排名。

下面的示例演示了计算具有标量值的矩阵的秩和具有全零值的另一个向量。

```
# vector rank
from numpy import array
from numpy.linalg import matrix_rank
# rank
v1 = array([1,2,3])
print(v1)
vr1 = matrix_rank(v1)
print(vr1)
# zero rank
v2 = array([0,0,0,0,0])
print(v2)
vr2 = matrix_rank(v2)
print(vr2)
```

运行该示例将打印第一个向量及其等级 1，然后是第二个零向量，其等级为 0。

```
[1 2 3]

1

[0 0 0 0 0]

0
```

下一个例子清楚地表明，秩不是矩阵的维数，而是线性独立方向的数量。

提供了 2×2 矩阵的三个示例，其展示了具有等级 0,1 和 2 的矩阵。

```
# matrix rank
from numpy import array
from numpy.linalg import matrix_rank
# rank 0
M0 = array([[0,0],[0,0]])
print(M0)
mr0 = matrix_rank(M0)
print(mr0)
# rank 1
M1 = array([[1,2],[1,2]])
print(M1)
mr1 = matrix_rank(M1)
print(mr1)
# rank 2
M2 = array([[1,2],[3,4]])
print(M2)
mr2 = matrix_rank(M2)
print(mr2)
```

运行该示例首先打印 0 2×2 矩阵，然后是秩，然后是 2×2，其具有等级 1，最后是 2×2 矩阵，等级为 2。

```
[[0 0]
 [0 0]]

0

[[1 2]
 [1 2]]

1

[[1 2]
 [3 4]]

2
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

*   第 3.4 节决定因素。 [线性代数无废话指南](http://amzn.to/2k76D4)，2017 年。
*   第 3.5 节矩阵逆。 [线性代数无废话指南](http://amzn.to/2k76D4)，2017 年。
*   第 5.1 节行列式的性质，[线性代数导论](http://amzn.to/2AZ7R8j)，第五版，2016 年。
*   第 2.3 节身份和反向矩阵，[深度学习](http://amzn.to/2B3MsuU)，2016 年。
*   第 2.11 节“决定因素，[深度学习](http://amzn.to/2B3MsuU)，2016 年。
*   第 3.D 节可逆性和同构向量空间，[线性代数完成权](http://amzn.to/2BGuEqI)，第三版，2015 年。
*   第 10.A 节，[线性代数完成权](http://amzn.to/2BGuEqI)，第三版，2015 年。
*   第 10.B 节行列式，[线性代数完成权](http://amzn.to/2BGuEqI)，第三版，2015 年。

### API

*   [numpy.ndarray.T API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.T.html)
*   [numpy.linalg.inv（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.inv.html)
*   [numpy.trace（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.trace.html)
*   [numpy.linalg.det（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.det.html)
*   [numpy.linalg.matrix_rank（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.matrix_rank.html)

### 用品

*   [在维基百科上转置](https://en.wikipedia.org/wiki/Transpose)
*   [维基百科上的可逆矩阵](https://en.wikipedia.org/wiki/Invertible_matrix)
*   维基百科上的 [Trace（线性代数）](https://en.wikipedia.org/wiki/Trace_(linear_algebra))
*   维基百科上的[决定因素](https://en.wikipedia.org/wiki/Determinant)
*   维基百科上的 [Rank（线性代数）](https://en.wikipedia.org/wiki/Rank_(linear_algebra))

## 摘要

在本教程中，您发现了机器学习方法描述中使用的重要线性代数矩阵运算。

具体来说，你学到了：

*   用于翻转矩阵尺寸的转置操作。
*   用于求解线性方程组的逆操作。
*   Trace 和 Determinant 操作在其他矩阵运算中用作简写符号。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。