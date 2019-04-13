# 机器学习矩阵分解的温和介绍

> 原文： [https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)

使用有限的计算机精度无法有效地或稳定地解决许多复杂的矩阵运算。

矩阵分解是将矩阵简化为组成部分的方法，使得更容易计算更复杂的矩阵运算。矩阵分解方法，也称为矩阵分解方法，是计算机中线性代数的基础，甚至用于基本操作，如求解线性方程组，计算逆矩阵和计算矩阵的行列式。

在本教程中，您将发现矩阵分解以及如何在 Python 中计算它们。

完成本教程后，您将了解：

*   什么是矩阵分解以及为什么这些类型的操作很重要。
*   如何在 Python 中计算 LU 和 QR 矩阵分解。
*   如何在 Python 中计算 Cholesky 矩阵分解。

让我们开始吧。

*   **Update Mar / 2018** ：修正了 QR 分解描述中的小错字。

![A Gentle Introduction to Matrix Decompositions for Machine Learning](img/8fe573b3dc23c056885105cf983a9dd7.jpg)

机器学习矩阵分解的温和介绍
[mickey](https://www.flickr.com/photos/mc-pictures/7870241912/) 的照片，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  什么是矩阵分解？
2.  LU 矩阵分解
3.  QR 矩阵分解
4.  Cholesky 分解

## 什么是矩阵分解？

矩阵分解是一种将矩阵简化为其组成部分的方法。

这种方法可以简化更复杂的矩阵运算，这些运算可以在分解的矩阵上执行，而不是在原始矩阵本身上执行。

矩阵分解的一个常见类比是数字因式分解，例如将因子分解为 2×5。因此，矩阵分解也称为矩阵分解。与分解实数值一样，有许多方法可以分解矩阵，因此存在一系列不同的矩阵分解技术。

两种简单且广泛使用的矩阵分解方法是 LU 矩阵分解和 QR 矩阵分解。

接下来，我们将仔细研究这些方法。

## LU 矩阵分解

LU 分解用于方形矩阵，并将矩阵分解为 L 和 U 分量。

```
A = L . U
```

或者，没有点符号。

```
A = LU
```

其中 A 是我们希望分解的方阵，L 是下三角矩阵，U 是上三角矩阵。

> 因子 L 和 U 是三角矩阵。消除产生的因子分解是 A = LU。

- 第 97 页，[线性代数导论](http://amzn.to/2AZ7R8j)，第五版，2016 年。

LU 分解是使用迭代数值过程找到的，并且对于那些不易分解或分解的矩阵可能会失败。

在实践中数值上更稳定的这种分解的变化称为 LUP 分解，或具有部分枢转的 LU 分解。

```
A = P . L . U
```

重新排序父矩阵的行以简化分解过程，并且附加的 P 矩阵指定用于置换结果或将结果返回到原始顺序的方式。 LU 还有其他变体。

LU 分解通常用于简化线性方程组的求解，例如在线性回归中找到系数，以及计算矩阵的行列式和逆。

LU 分解可以使用 lu（）函数在 Python 中实现。更具体地说，该函数计算 LPU 分解。

以下示例首先定义 3×3 方阵。计算 LU 分解，然后从组件重建原始矩阵。

```
# LU decomposition
from numpy import array
from scipy.linalg import lu
# define a square matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# LU decomposition
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)
print(B)
```

首先运行该示例打印定义的 3×3 矩阵，然后打印分解的 P，L 和 U 分量，最后重建原始矩阵。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

[[ 0\.  1\.  0.]
 [ 0\.  0\.  1.]
 [ 1\.  0\.  0.]]

[[ 1\.          0\.          0\.        ]
 [ 0.14285714  1\.          0\.        ]
 [ 0.57142857  0.5         1\.        ]]

[[  7.00000000e+00   8.00000000e+00   9.00000000e+00]
 [  0.00000000e+00   8.57142857e-01   1.71428571e+00]
 [  0.00000000e+00   0.00000000e+00  -1.58603289e-16]]

[[ 1\.  2\.  3.]
 [ 4\.  5\.  6.]
 [ 7\.  8\.  9.]]
```

## QR 矩阵分解

QR 分解用于 m×n 矩阵（不限于方形矩阵）并将矩阵分解为 Q 和 R 分量。

```
A = Q . R
```

Or, without the dot notation.

```
A = QR
```

其中 A 是我们希望分解的矩阵，Q 是尺寸为 m×m 的矩阵，R 是尺寸为 m×n 的上三角矩阵。

使用迭代数值方法找到 QR 分解，该方法对于那些不能分解或易于分解的矩阵可能失败。

与 LU 分解一样，QR 分解通常用于求解线性方程组，但不限于方形矩阵。

可以使用 qr（）函数在 NumPy 中实现 QR 分解。默认情况下，该函数返回具有更小或“减小”维度的 Q 和 R 矩阵，这更经济。我们可以通过将 mode 参数指定为'complete'来更改此值以返回 Q 的 m x m 和 R 的预期大小，尽管大多数应用程序都不需要这样做。

下面的示例定义了 3×2 矩阵，计算 QR 分解，然后从分解的元素重建原始矩阵。

```
# QR decomposition
from numpy import array
from numpy.linalg import qr
# define a 3x2 matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# QR decomposition
Q, R = qr(A, 'complete')
print(Q)
print(R)
# reconstruct
B = Q.dot(R)
print(B)
```

运行该示例首先打印定义的 3×2 矩阵，然后打印 Q 和 R 元素，最后打印与我们开始时匹配的重建矩阵。

```
[[1 2]
 [3 4]
 [5 6]]

[[-0.16903085  0.89708523  0.40824829]
 [-0.50709255  0.27602622 -0.81649658]
 [-0.84515425 -0.34503278  0.40824829]]

[[-5.91607978 -7.43735744]
 [ 0\.          0.82807867]
 [ 0\.          0\.        ]]

[[ 1\.  2.]
 [ 3\.  4.]
 [ 5\.  6.]]
```

## Cholesky 分解

Cholesky 分解用于方形对称矩阵，其中所有值都大于零，即所谓的正定矩阵。

为了我们对机器学习的兴趣，我们将重点关注实值矩阵的 Cholesky 分解，并忽略处理复数时的情况。

分解定义如下：

```
A = L . L^T
```

或者没有点符号：

```
A = LL^T
```

其中 A 是被分解的矩阵，L 是下三角矩阵，L ^ T 是 L 的转置。

分解也可以写成上三角矩阵的乘积，例如：

```
A = U^T . U
```

其中 U 是上三角矩阵。

Cholesky 分解用于求解线性回归的线性最小二乘，以及模拟和优化方法。

当分解对称矩阵时，Cholesky 分解的效率几乎是 LU 分解的两倍，在这些情况下应该是首选。

> 虽然对称的正定矩阵相当特殊，但它们在某些应用中经常出现，因此它们的特殊分解（称为 Cholesky 分解）很容易理解。当你可以使用它时，Cholesky 分解比用于求解线性方程的替代方法快两倍。

- 第 100 页，[数字秘籍：科学计算的艺术](http://amzn.to/2BezVEE)，第三版，2007 年。

可以通过调用 cholesky（）函数在 NumPy 中实现 Cholesky 分解。该函数仅返回 L，因为我们可以根据需要轻松访问 L 转置。

下面的例子定义了一个 3×3 对称和正定矩阵并计算 Cholesky 分解，然后重建原始矩阵。

```
# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)
```

首先运行该示例打印对称矩阵，然后打印分解后的下三角矩阵，然后是重建矩阵。

```
[[2 1 1]
 [1 2 1]
 [1 1 2]]

[[ 1.41421356  0\.          0\.        ]
 [ 0.70710678  1.22474487  0\.        ]
 [ 0.70710678  0.40824829  1.15470054]]

[[ 2\.  1\.  1.]
 [ 1\.  2\.  1.]
 [ 1\.  1\.  2.]]
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   使用您自己的数据使用每个操作创建 5 个示例。
*   搜索机器学习论文并找到每个正在使用的操作的示例。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 6.6 节矩阵分解。 [线性代数无废话指南](http://amzn.to/2k76D4)，2017 年。
*   第 7 讲 QR 分解，[数值线性代数](http://amzn.to/2BI9kRH)，1997。
*   第 2.3 节 LU 分解及其应用，[数字秘籍：科学计算的艺术，第三版](http://amzn.to/2BezVEE)，2007。
*   第 2.10 节 QR 分解，[数字秘籍：科学计算的艺术](http://amzn.to/2BezVEE)，第三版，2007。
*   第 2.9 节 Cholesky 分解，[数字秘籍：科学计算的艺术](http://amzn.to/2BezVEE)，第三版，2007。
*   第 23 讲，Cholesky 分解，[数值线性代数](http://amzn.to/2BI9kRH)，1997。

### API

*   [scipy.linalg.lu（）API](https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.linalg.lu.html)
*   [numpy.linalg.qr（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.qr.html)
*   [numpy.linalg.cholesky（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.cholesky.html)

### 用品

*   维基百科上的[矩阵分解](https://en.wikipedia.org/wiki/Matrix_decomposition)
*   维基百科上的 [LU 分解](https://en.wikipedia.org/wiki/LU_decomposition)
*   维基百科上的 [QR 分解](https://en.wikipedia.org/wiki/QR_decomposition)
*   [维基百科上的 Cholesky 分解](https://en.wikipedia.org/wiki/Cholesky_decomposition)

## 摘要

在本教程中，您发现了矩阵分解以及如何在 Python 中计算它们。

具体来说，你学到了：

*   什么是矩阵分解以及为什么这些类型的操作很重要。
*   如何在 Python 中计算 LU 和 QR 矩阵分解。
*   如何在 Python 中计算 Cholesky 矩阵分解。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。