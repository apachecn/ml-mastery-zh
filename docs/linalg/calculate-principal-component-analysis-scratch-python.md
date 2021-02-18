# 如何从 Python 中的 Scratch 计算主成分分析（PCA）

> 原文： [https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)

降维的重要机器学习方法称为主成分分析。

这是一种使用线性代数和统计学中的简单矩阵运算来计算原始数据到相同数量或更少维度的投影的方法。

在本教程中，您将发现用于降低维数的主成分分析机器学习方法以及如何在 Python 中从头开始实现它。

完成本教程后，您将了解：

*   计算主成分分析的过程以及如何选择主成分。
*   如何在 NumPy 中从头开始计算主成分分析。
*   如何计算主成分分析，以便在 scikit-learn 中使用更多数据。

让我们开始吧。

*   **Update Apr / 2018** ：修正了 sklearn PCA 属性解释中的拼写错误。由于克里斯。

![How to Calculate the Principal Component Analysis from Scratch in Python](img/c5e72b23e0130b0e1e4f7305b5f172ae.jpg)

如何在 Python 中从头开始计算主成分分析
照片由 [mickey](https://www.flickr.com/photos/mc-pictures/7870255710/) ，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  主成分分析
2.  手动计算主成分分析
3.  可重复使用的主成分分析

## 主成分分析

主成分分析（简称 PCA）是一种减少数据维数的方法。

它可以被认为是一种投影方法，其中具有 m 列（特征）的数据被投影到具有 m 个或更少列的子空间中，同时保留原始数据的本质。

可以使用线性代数工具来描述和实现 PCA 方法。

PCA 是应用于数据集的操作，由 n×m 矩阵 A 表示，其导致 A 的投影，我们将称之为 B.让我们逐步完成此操作的步骤。

```
     a11, a12
A = (a21, a22)
     a31, a32

B = PCA(A)
```

第一步是计算每列的平均值。

```
M = mean(A)
```

要么

```
              (a11 + a21 + a31) / 3
M(m11, m12) = (a12 + a22 + a32) / 3
```

接下来，我们需要通过减去平均列值来使每列中的值居中。

```
C = A - M
```

下一步是计算居中矩阵 C 的协方差矩阵。

相关性是两列一起变化的量和方向（正或负）的标准化度量。协方差是跨多列的相关的广义和非标准化版本。协方差矩阵是给定矩阵的协方差的计算，每个列与每个其他列的协方差分数，包括其自身。

```
V = cov(C)
```

最后，我们计算协方差矩阵 V 的特征分解。这导致特征值列表和特征向量列表。

```
values, vectors = eig(V)
```

特征向量表示 B 的缩小子空间的方向或分量，而特征值表示方向的大小。

特征向量可以按特征值按降序排序，以提供 A 的新子空间的分量或轴的等级。

如果所有特征值都具有相似的值，那么我们就知道现有的表示可能已经被合理地压缩或密集，并且投影可能提供的很少。如果存在接近零的特征值，则它们表示可以被丢弃的 B 的分量或轴。

必须选择总共 m 个或更少的组件来组成所选择的子空间。理想情况下，我们将选择具有 k 个最大特征值的 k 个本征向量，称为主成分。

```
B = select(values, vectors)
```

可以使用其他矩阵分解方法，例如奇异值分解或 SVD。因此，通常将这些值称为奇异值，并将子空间的向量称为主要分量。

一旦选择，可以通过矩阵乘法将数据投影到子空间中。

```
P = B^T . A
```

其中 A 是我们希望投影的原始数据，B ^ T 是所选主成分的转置，P 是 A 的投影。

这被称为计算 PCA 的协方差方法，尽管有其他方法可以计算它。

## 手动计算主成分分析

NumPy 中没有 pca（）函数，但我们可以使用 NumPy 函数轻松地逐步计算主成分分析。

下面的例子定义了一个小的 3×2 矩阵，将数据置于矩阵中心，计算中心数据的协方差矩阵，然后计算协方差矩阵的特征分解。特征向量和特征值作为主成分和奇异值，用于投影原始数据。

```
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)
```

运行该示例首先打印原始矩阵，然后打印中心协方差矩阵的特征向量和特征值，最后是原始矩阵的投影。

有趣的是，我们可以看到只需要第一个特征向量，这表明我们可以将 3×2 矩阵投影到 3×1 矩阵上而几乎没有损失。

```
[[1 2]
 [3 4]
 [5 6]]

[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]

[ 8\.  0.]

[[-2.82842712  0\.        ]
 [ 0\.          0\.        ]
 [ 2.82842712  0\.        ]]
```

## 可重复使用的主成分分析

我们可以使用 scikit-learn 库中的 PCA（）类计算数据集的主成分分析。这种方法的好处是，一旦计算出投影，它就可以很容易地一次又一次地应用于新数据。

创建类时，可以将组件数指定为参数。

首先通过调用 fit（）函数将类放在数据集上，然后通过调用 transform（）函数将原始数据集或其他数据投影到具有所选维数的子空间中。

一旦拟合，可以通过`explain_variance_`和`components_`属性在 PCA 类上访问特征值和主成分。

下面的示例演示了如何使用此类，首先创建一个实例，将其拟合到 3×2 矩阵上，访问投影的值和向量，以及转换原始数据。

```
# Principal Component Analysis
from numpy import array
from sklearn.decomposition import PCA
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)
```

运行该示例首先打印 3×2 数据矩阵，然后是主要分量和值，然后是原始矩阵的投影。

我们可以看到，通过一些非常小的浮点舍入，我们可以获得与前一个示例中相同的主成分，奇异值和投影。

```
[[1 2]
 [3 4]
 [5 6]]

[[ 0.70710678  0.70710678]
 [ 0.70710678 -0.70710678]]

[  8.00000000e+00   2.25080839e-33]

[[ -2.82842712e+00   2.22044605e-16]
 [  0.00000000e+00   0.00000000e+00]
 [  2.82842712e+00  -2.22044605e-16]]
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   使用您自己的小设计矩阵值重新运行示例。
*   加载数据集并计算其上的 PCA 并比较两种方法的结果。
*   搜索并找到 PCA 用于机器学习论文的 10 个例子。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 7.3 节主成分分析（SVD 的 PCA），[线性代数导论](http://amzn.to/2CZgTTB)，第 5 版，2016 年。
*   第 2.12 节示例：主成分分析，[深度学习](http://amzn.to/2B3MsuU)，2016。

### API

*   [numpy.mean（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.mean.html)
*   [numpy.cov（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.cov.html)
*   [numpy.linalg.eig（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eig.html)
*   [sklearn.decomposition.PCA API](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

### 用品

*   [维基百科上的主成分分析](https://en.wikipedia.org/wiki/Principal_component_analysis)
*   [协方差矩阵](https://en.wikipedia.org/wiki/Covariance_matrix)

### 教程

*   [主成分分析与 numpy](https://glowingpython.blogspot.com.au/2011/07/principal-component-analysis-with-numpy.html) ，2011。
*   [PCA 和图像压缩与 numpy](https://glowingpython.blogspot.com.au/2011/07/pca-and-image-compression-with-numpy.html) ，2011。
*   [实施主成分分析（PCA）](http://sebastianraschka.com/Articles/2014_pca_step_by_step.html)，2014 年。

## 摘要

在本教程中，您发现了降低维数的主成分分析机器学习方法。

具体来说，你学到了：

*   计算主成分分析的过程以及如何选择主成分。
*   如何在 NumPy 中从头开始计算主成分分析。
*   如何计算主成分分析，以便在 scikit-learn 中使用更多数据。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。