# NumPy 对预期价值，方差和协方差的简要介绍

> 原文： [https://machinelearningmastery.com/introduction-to-expected-value-variance-and-covariance/](https://machinelearningmastery.com/introduction-to-expected-value-variance-and-covariance/)

基础统计数据是应用机器学习中的有用工具，可以更好地理解您的数据。

它们也是为更先进的线性代数运算和机器学习方法提供基础的工具，例如协方差矩阵和主成分分析。因此，在线性代数符号的背景下强调基本统计是非常重要的。

在本教程中，您将了解基本统计操作如何工作以及如何使用 NumPy 以及线性代数中的符号和术语来实现它们。

完成本教程后，您将了解：

*   预期值，平均值和平均值是什么以及如何计算它们。
*   方差和标准偏差是什么以及如何计算它们。
*   协方差，相关性和协方差矩阵是什么以及如何计算它们。

让我们开始吧。

*   **更新于 Mar / 2018** ：修复了向量方差示例中的小错字。谢谢鲍勃。

![A Gentle Introduction to Expected Value, Variance, and Covariance with NumPy](img/66cef2cc9be1b9be684b0918ce2ce69f.jpg)

NumPy
照片由 [Robyn Jay](https://www.flickr.com/photos/learnscope/15866965009/) 对预期价值，方差和协方差的温和介绍，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  期望值
2.  方差
3.  协方差
4.  协方差矩阵

## 期望值

在概率上，某个随机变量 X 的平均值称为期望值或期望值。

期望值使用符号 E 和变量名称周围的方括号;例如：

```
E[X]
```

它被计算为可以绘制的值的概率加权和。

```
E[X] = sum(x1 * p1, x2 * p2, x3 * p3, ..., xn * pn)
```

在简单的情况下，例如掷硬币或掷骰子，每次事件的概率都是可能的。因此，可以将期望值计算为所有值的总和乘以值的倒数。

```
E[X] = sum(x1, x2, x3, ..., xn) . 1/n
```

在统计学中，可以从从域中抽取的示例的样本来估计平均值，或者更具技术性地，算术平均值或样本均值。这很混乱，因为平均值，平均值和期望值可以互换使用。

在摘要中，均值由小写希腊字母 mu 表示，并且是根据观察样本计算的，而不是所有可能的值。

```
mu = sum(x1, x2, x3, ..., xn) . 1/n
```

或者，写得更紧凑：

```
mu = sum(x . P(x))
```

其中 x 是观测向量，P（x）是每个值的计算概率。

当计算特定变量（例如 x）时，均值表示为小写变量名称，上面有一行，称为 x-bar。

```
_
x = sum from 1 to n (xi) . 1/n
```

可以使用 mean（）函数计算 NumPy 中的向量或矩阵的算术平均值。

下面的示例定义了 6 个元素的向量并计算平均值。

```
from numpy import array
from numpy import mean
v = array([1,2,3,4,5,6])
print(v)
result = mean(v)
print(result)
```

首先运行该示例打印定义的向量和向量中的值的平均值。

```
[1 2 3 4 5 6]

3.5
```

均值函数可以通过分别指定轴参数和值 0 或 1 来计算矩阵的行或列均值。

下面的示例定义了一个 2×6 矩阵，并计算列和行均值。

```
from numpy import array
from numpy import mean
M = array([[1,2,3,4,5,6],[1,2,3,4,5,6]])
print(M)
col_mean = mean(M, axis=0)
print(col_mean)
row_mean = mean(M, axis=1)
print(row_mean)
```

首先运行示例打印定义的矩阵，然后打印计算的列和行平均值。

```
[[1 2 3 4 5 6]
 [1 2 3 4 5 6]]

[ 1\.  2\.  3\.  4\.  5\.  6.]

[ 3.5  3.5]
```

## 方差

在概率上，一些随机变量 X 的方差是分布中平均值相对于平均值变化的量度。

方差表示为变量上的函数 Var（）。

```
Var[X]
```

方差计算为分布中每个值与预期值的平均平方差。或者与预期值的预期平方差异。

```
Var[X] = E[(X - E[X])^2]
```

假设已经计算了变量的期望值（E [X]），则可以将随机变量的方差计算为每个示例的平方差与期望值乘以该值的概率之和。

```
Var[X] = sum (p(x1) . (x1 - E[X])^2, p(x2) . (x2 - E[X])^2, ..., p(x1) . (xn - E[X])^2)
```

如果分布中每个示例的概率相等，则方差计算可以降低个体概率，并将平方差的总和乘以分布中的示例数的倒数。

```
Var[X] = sum ((x1 - E[X])^2, (x2 - E[X])^2, ...,(xn - E[X])^2) . 1/n
```

在统计中，可以从从域中抽取的示例的样本来估计方差。

在摘要中，样本方差由小写西格玛表示，其中 2 上标表示单位是平方的，而不是必须对最终值求平方。将平方差的总和乘以实例数的倒数减 1 以校正偏差。

```
sigma^2 = sum from 1 to n ( (xi - mu)^2 ) . 1 / (n - 1)
```

在 NumPy 中，可以使用 var（）函数计算向量或矩阵的方差。默认情况下，var（）函数计算总体方差。要计算样本方差，必须将 ddof 参数设置为值 1。

下面的示例定义了 6 个元素的向量并计算了样本方差。

```
from numpy import array
from numpy import var
v = array([1,2,3,4,5,6])
print(v)
result = var(v, ddof=1)
print(result)
```

首先运行该示例打印定义的向量，然后打印向量中值的计算样本方差。

```
[1 2 3 4 5 6]

3.5
```

var 函数可以通过分别指定 axis 参数和值 0 或 1 来计算矩阵的行或列方差，与上面的平均函数相同。

下面的示例定义了一个 2×6 矩阵，并计算了列和行样本方差。

```
from numpy import array
from numpy import var
M = array([[1,2,3,4,5,6],[1,2,3,4,5,6]])
print(M)
col_mean = var(M, ddof=1, axis=0)
print(col_mean)
row_mean = var(M, ddof=1, axis=1)
print(row_mean)
```

首先运行示例打印定义的矩阵，然后打印列和行样本方差值。

```
[[1 2 3 4 5 6]
 [1 2 3 4 5 6]]

[ 0\.  0\.  0\.  0\.  0\.  0.]

[ 3.5  3.5]
```

标准偏差计算为方差的平方根，并表示为小写“s”。

```
s = sqrt(sigma^2)
```

为了保持这种表示法，有时方差表示为 s ^ 2，其中 2 表示为上标，再次表明单位是平方的。

NumPy 还提供直接通过 std（）函数计算标准偏差的功能。与 var（）函数一样，ddof 参数必须设置为 1 才能计算无偏样本标准差，并且可以通过将 axis 参数分别设置为 0 和 1 来计算列和行标准偏差。

下面的示例演示了如何计算矩阵的行和列的样本标准偏差。

```
from numpy import array
from numpy import std
M = array([[1,2,3,4,5,6],[1,2,3,4,5,6]])
print(M)
col_mean = std(M, ddof=1, axis=0)
print(col_mean)
row_mean = std(M, ddof=1, axis=1)
print(row_mean)
```

首先运行示例打印定义的矩阵，然后打印列和行样本标准偏差值。

```
[[1 2 3 4 5 6]
 [1 2 3 4 5 6]]

[ 0\.  0\.  0\.  0\.  0\.  0.]

[ 1.87082869  1.87082869]
```

## 协方差

在概率上，协方差是两个随机变量的联合概率的度量。它描述了两个变量如何一起变化。

它表示为函数 cov（X，Y），其中 X 和 Y 是要考虑的两个随机变量。

```
cov(X,Y)
```

协方差计算为每个随机变量与其预期值的差异的乘积的预期值或平均值，其中 E [X]是 X 的期望值，E [Y]是 y 的期望值。

```
cov(X, Y) = E[(X - E[X] . (Y - E[Y])]
```

假设已计算出 X 和 Y 的预期值，则协方差可以计算为 x 值与其预期值的差值之和乘以 y 值与其预期值的差值乘以其数量的倒数。人口中的例子。

```
cov(X, Y) = sum (x - E[X]) * (y - E[Y]) * 1/n
```

在统计学中，样本协方差可以以相同的方式计算，尽管具有偏差校正，与方差相同。

```
cov(X, Y) = sum (x - E[X]) * (y - E[Y]) * 1/(n - 1)
```

协方差的符号可以解释为两个变量是一起增加（正）还是一起减少（负）。协方差的大小不容易解释。协方差值为零表示两个变量完全独立。

NumPy 没有直接计算两个变量之间协方差的函数。相反，它具有计算称为 cov（）的协方差矩阵的函数，我们可以使用它来检索协方差。默认情况下，cov（）函数将计算所提供的随机变量之间的无偏或样本协方差。

下面的例子定义了两个相等长度的向量，一个增加，一个减少。我们预计这些变量之间的协方差是负的。

我们只访问两个变量的协方差，因为返回了方差协方差矩阵的[0,1]元素。

```
from numpy import array
from numpy import cov
x = array([1,2,3,4,5,6,7,8,9])
print(x)
y = array([9,8,7,6,5,4,3,2,1])
print(y)
Sigma = cov(x,y)[0,1]
print(Sigma)
```

首先运行该示例打印两个向量，然后是两个向量中的值的协方差。正如我们预期的那样，价值是负的。

```
[1 2 3 4 5 6 7 8 9]
[9 8 7 6 5 4 3 2 1]

-7.5
```

可以将协方差归一化为-1 和 1 之间的分数，以通过将其除以 X 和 Y 的标准偏差来解释幅度。结果称为变量的相关性，也称为 Pearson 相关系数，命名为该方法的开发者。

```
r = cov(X, Y) / sX sY
```

其中 r 是 X 和 Y 的相关系数，cov（X，Y）是 X 和 Y 的样本协方差，sX​​和 sY 分别是 X 和 Y 的标准偏差。

NumPy 提供了 corrcoef（）函数，用于直接计算两个变量之间的相关性。与 cov（）一样，它返回一个矩阵，在本例中是一个相关矩阵。与 cov（）的结果一样，我们只能从返回的平方矩阵的[0,1]值中获取感兴趣的相关性。

```
from numpy import array
from numpy import corrcoef
x = array([1,2,3,4,5,6,7,8,9])
print(x)
y = array([9,8,7,6,5,4,3,2,1])
print(y)
Sigma = corrcoef(x,y)
print(Sigma)
```

首先运行该示例打印两个定义的向量，然后打印相关系数。我们可以看到，向量与我们设计的最大负相关。

```
[1 2 3 4 5 6 7 8 9]
[9 8 7 6 5 4 3 2 1]

-1.0
```

## 协方差矩阵

协方差矩阵是方形和对称矩阵，其描述两个或更多个随机变量之间的协方差。

协方差矩阵的对角线是每个随机变量的方差。

协方差矩阵是两个变量的协方差的推广，并且捕获数据集中的所有变量可以一起改变的方式。

协方差矩阵表示为大写希腊字母 Sigma。如上计算每对随机变量的协方差。

```
Sigma = E[(X - E[X] . (Y - E[Y])]
```

哪里：

```
Sigma(ij) = cov(Xi, Xj)
```

X 是一个矩阵，每列代表一个随机变量。

协方差矩阵为分离随机变量矩阵中的结构关系提供了有用的工具。这可以用于解相关变量或作为变换应用于其他变量。它是主成分分析数据简化方法中使用的关键元素，简称 PCA。

可以使用 cov（）函数在 NumPy 中计算协方差矩阵。默认情况下，此函数将计算样本协方差矩阵。

可以使用包含用于计算协方差矩阵的列的单个矩阵来调用 cov（）函数，或者使用两个数组（例如每个变量一个数组）来调用 cov（）函数。

下面是一个定义两个 9 元素向量并从中计算无偏协方差矩阵的示例。

```
from numpy import array
from numpy import cov
x = array([1,2,3,4,5,6,7,8,9])
print(x)
y = array([9,8,7,6,5,4,3,2,1])
print(y)
Sigma = cov(x,y)
print(Sigma)
```

首先运行该示例打印两个向量，然后打印计算的协方差矩阵。

设计数组的值使得当一个变量增加时，另一个变量减少。我们期望在这两个变量的协方差上看到负号，这就是我们在协方差矩阵中看到的。

```
[1 2 3 4 5 6 7 8 9]

[9 8 7 6 5 4 3 2 1]

[[ 7.5 -7.5]
 [-7.5  7.5]]
```

协方差矩阵广泛用于线性代数和线性代数与统计的交集，称为多变量分析。我们在这篇文章中只有一点点品味。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   使用您自己的小型人为数据探索每个示例。
*   从 CSV 文件加载数据并将每个操作应用于数据列。
*   编写自己的函数来实现每个统计操作。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [应用多变量统计分析](http://amzn.to/2AUcEc5)，2012。
*   [应用多变量统计分析](http://amzn.to/2AWIViz)，2015 年。
*   第 12 章概率中的线性代数＆amp;统计学，[线性代数导论](http://amzn.to/2AZ7R8j)，第五版，2016 年。
*   第 3 章，概率论和信息论，[深度学习](http://amzn.to/2j4oKuP)，2016 年。

### API

*   [NumPy 统计函数](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.statistics.html)
*   [numpy.mean（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.mean.html)
*   [numpy.var（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.var.html)
*   [numpy.std（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.std.html)
*   [numpy.cov（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.cov.html)
*   [numpy.corrcoef（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html)

### 用品

*   [维基百科的预期价值](https://en.wikipedia.org/wiki/Expected_value)
*   维基百科上的[意思](https://en.wikipedia.org/wiki/Mean)
*   [维基百科上的差异](https://en.wikipedia.org/wiki/Variance)
*   [维基百科的标准偏差](https://en.wikipedia.org/wiki/Standard_deviation)
*   [维基百科上的协方差](https://en.wikipedia.org/wiki/Covariance)
*   [样本均值和协方差](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)
*   [Pearson 相关系数](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
*   维基百科上的[协方差矩阵](https://en.wikipedia.org/wiki/Covariance_matrix)
*   [维基百科](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices)上协方差矩阵的估计

### 帖子

*   [什么是协方差矩阵？](http://fouryears.eu/2016/11/23/what-is-the-covariance-matrix/) ，2016 年
*   [协方差矩阵](http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/)的几何解释，2014。

## 摘要

在本教程中，您了解了基本的统计操作如何工作以及如何使用 NumPy 以及线性代数中的符号和术语来实现它们。

具体来说，你学到了：

*   那么期望值，平均值和平均值以及如何计算。
*   方差和标准偏差是什么以及如何计算它们。
*   协方差，相关性和协方差矩阵是什么以及如何计算它们。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。