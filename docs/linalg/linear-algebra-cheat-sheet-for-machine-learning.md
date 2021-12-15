# 机器学习中的线性代数备忘单

> 原文： [https://machinelearningmastery.com/linear-algebra-cheat-sheet-for-machine-learning/](https://machinelearningmastery.com/linear-algebra-cheat-sheet-for-machine-learning/)

### NumPy 中用于机器学习的所有线性代数运算
。

名为 NumPy 的 Python 数值计算库提供了许多线性代数函数，可用作机器学习从业者。

在本教程中，您将发现使用向量和矩阵的关键函数，您可能会发现这些函数可用作机器学习从业者。

这是一个备忘单，所有示例都很简短，假设您熟悉正在执行的操作。

您可能希望为此页面添加书签以供将来参考。

![Linear Algebra Cheat Sheet for Machine Learning](img/0fdbdc2f5daee66450ef948387f7c772.jpg)

用于机器学习的线性代数备忘单
照片由 [Christoph Landers](https://www.flickr.com/photos/bewegtbildgestalter/1274219020/) 拍摄，保留一些权利。

## 概观

本教程分为 7 个部分;他们是：

1.  数组
2.  向量
3.  矩阵
4.  矩阵的类型
5.  矩阵运算
6.  矩阵分解
7.  统计

## 1.数组

有很多方法可以创建 NumPy 数组。

### 排列

```
from numpy import array
A = array([[1,2,3],[1,2,3],[1,2,3]])
```

### 空

```
from numpy import empty
A = empty([3,3])
```

### 零

```
from numpy import zeros
A = zeros([3,5])
```

### 那些

```
from numpy import ones
A = ones([5, 5])
```

## 2.向量

向量是标量的列表或列。

### 向量加法

```
c = a + b
```

### 向量减法

```
c = a - b
```

### 向量乘法

```
c = a * b
```

### 向量分部

```
c = a / b
```

### 向量点产品

```
c = a.dot(b)
```

### 向量标量乘法

```
c = a * 2.2
```

### 向量规范

```
from numpy.linalg import norm
l2 = norm(v)
```

## 3.矩阵

矩阵是标量的二维数组。

### 矩阵加法

```
C = A + B
```

### 矩阵减法

```
C = A - B
```

### 矩阵乘法（Hadamard 产品）

```
C = A * B
```

### 矩阵分部

```
C = A / B
```

### 矩阵 - 矩阵乘法（点积）

```
C = A.dot(B)
```

### 矩阵向量乘法（点积）

```
C = A.dot(b)
```

### 矩阵 - 标量乘法

```
C = A.dot(2.2)
```

## 4.矩阵的类型

不同类型的矩阵通常用作更广泛计算中的元素。

### 三角矩阵

```
# lower
from numpy import tril
lower = tril(M)
# upper
from numpy import triu
upper = triu(M)
```

### 对角矩阵

```
from numpy import diag
d = diag(M)
```

### 身份矩阵

```
from numpy import identity
I = identity(3)
```

## 5.矩阵运算

矩阵运算通常用作更广泛计算中的元素。

### 矩阵转置

```
B = A.T
```

### 矩阵反演

```
from numpy.linalg import inv
B = inv(A)
```

### 矩阵追踪

```
from numpy import trace
B = trace(A)
```

### 矩阵行列式

```
from numpy.linalg import det
B = det(A)
```

### 矩阵排名

```
from numpy.linalg import matrix_rank
r = matrix_rank(A)
```

## 6.矩阵分解

矩阵分解或矩阵分解将矩阵分解为其组成部分，以使其他操作更简单，数值更稳定。

### LU 分解

```
from scipy.linalg import lu
P, L, U = lu(A)
```

### QR 分解

```
from numpy.linalg import qr
Q, R = qr(A, 'complete')
```

### 特征分解

```
from numpy.linalg import eig
values, vectors = eig(A)
```

### 奇异值分解

```
from scipy.linalg import svd
U, s, V = svd(A)
```

## 7.统计

统计数据总结了向量或矩阵的内容，通常用作更广泛操作的组件。

### 意思

```
from numpy import mean
result = mean(v)
```

### 方差

```
from numpy import var
result = var(v, ddof=1)
```

### 标准偏差

```
from numpy import std
result = std(v, ddof=1)
```

### 协方差矩阵

```
from numpy import cov
sigma = cov(v1, v2)
```

### 线性最小二乘法

```
from numpy.linalg import lstsq
b = lstsq(X, y)
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### NumPy API

*   [线性代数](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)
*   [统计](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.statistics.html)

### 其他作弊表

*   [Python For Data Science 备忘单，DataCamp（PDF）](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_SciPy_Cheat_Sheet_Linear_Algebra.pdf)
*   [线性代数四页解释（PDF）](https://minireference.com/static/tutorials/linear_algebra_in_4_pages.pdf)
*   [线性代数备忘单](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet)

## 摘要

在本教程中，您发现了线性代数的关键函数，您可能会发现这些函数可用作机器学习从业者。

您是否使用或了解其他关键的线性代数函数？
请在下面的评论中告诉我。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。