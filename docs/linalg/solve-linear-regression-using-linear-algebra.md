# 如何用线性代数求解线性回归

> 原文： [https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/](https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/)

线性回归是一种用于对一个或多个自变量与因变量之间的关系进行建模的方法。

它是统计学的主要内容，通常被认为是一种很好的入门机器学习方法。它也是一种可以使用矩阵表示法重新构造并使用矩阵运算求解的方法。

在本教程中，您将发现线性回归的矩阵公式以及如何使用直接和矩阵分解方法来解决它。

完成本教程后，您将了解：

*   线性回归与正规方程的矩阵重构。
*   如何使用 QR 矩阵分解求解线性回归。
*   如何使用 SVD 和伪逆求解线性回归。

让我们开始吧。

![How to Solve Linear Regression Using Linear Algebra](img/390621d529d53c033bce38791fd2f38d.jpg)

如何使用线性代数解决线性回归
照片来自 [likeaduck](https://www.flickr.com/photos/thartz00/4855670046/) ，保留一些权利。

## 教程概述

本教程分为 6 个部分;他们是：

1.  线性回归
2.  线性回归的矩阵公式
3.  线性回归数据集
4.  直接解决
5.  通过 QR 分解解决
6.  通过奇异值分解求解

## 线性回归

线性回归是一种建模两个标量值之间关系的方法：输入变量 x 和输出变量 y。

该模型假设 y 是线性函数或输入变量的加权和。

```
y = f(x)
```

或者，用系数表示。

```
y = b0 + b1 . x1
```

在给定多个输入变量（称为多元线性回归）的情况下，该模型还可用于对输出变量进行建模（下面，为了便于阅读，添加了括号）。

```
y = b0 + (b1 . x1) + (b2 . x2) + ...
```

创建线性回归模型的目的是找到系数值（b）的值，以最小化输出变量 y 的预测误差。

线性回归的矩阵公式

线性回归可以使用 Matrix 符号表示;例如：

```
y = X . b
```

或者，没有点符号。

```
y = Xb
```

其中 X 是输入数据并且每列是数据特征，b 是系数的向量，y 是 X 中每行的输出变量的向量。

```
     x11, x12, x13
X = (x21, x22, x23)
     x31, x32, x33
     x41, x42, x43

     b1
b = (b2)
     b3

     y1
y = (y2)
     y3
     y4
```

重新制定后，问题就变成了一个线性方程组，其中 b 向量值是未知的。这种类型的系统被称为超定，因为存在比未知数更多的方程，即每个系数用于每行数据。

分析解决这个问题是一个具有挑战性的问题，因为存在多种不一致的解决方案，例如：系数的多个可能值。此外，所有解决方案都会有一些错误，因为没有任何线路几乎可以通过所有点，因此求解方程的方法必须能够处理。

通常实现这种方法的方法是找到一种解决方案，其中模型中 b 的值最小化平方误差。这称为线性最小二乘法。

```
||X . b - y||^2 = sum i=1 to m ( sum j=1 to n Xij . bj - yi)^2
```

只要输入列是独立的（例如不相关的），该秘籍就具有独特的解决方案。

> 我们不能总是得到错误 e = b - Ax 降到零。当 e 为零时，x 是 Ax = b 的精确解。当 e 的长度尽可能小时，xhat 是最小二乘解。

- 第 219 页，[线性代数简介](http://amzn.to/2AZ7R8j)，第五版，2016 年。

在矩阵表示法中，使用所谓的正规方程来表达这个问题：

```
X^T . X . b = X^T . y
```

这可以重新安排，以便为 b 指定解决方案：

```
b = (X^T . X)^-1 . X^T . y
```

这可以直接求解，但是假设存在矩阵逆可能在数值上具有挑战性或不稳定。

## 线性回归数据集

为了探索线性回归的矩阵公式，我们首先将数据集定义为上下文。

我们将使用一个简单的 2D 数据集，其中数据易于作为散点图可视化，并且模型很容易可视化为试图拟合数据点的线。

下面的示例定义了一个 5×2 矩阵数据集，将其拆分为 X 和 y 分量，并将数据集绘制为散点图。

```
from numpy import array
from matplotlib import pyplot
data = array([
	[0.05, 0.12],
	[0.18, 0.22],
	[0.31, 0.35],
	[0.42, 0.38],
	[0.5, 0.49],
	])
print(data)
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# plot dataset
pyplot.scatter(X, y)
pyplot.show()
```

首先运行该示例将打印定义的数据集。

```
[[ 0.05  0.12]
 [ 0.18  0.22]
 [ 0.31  0.35]
 [ 0.42  0.38]
 [ 0.5   0.49]]
```

然后创建数据集的散点图，显示直线不能精确拟合此数据。

![Scatter Plot of Linear Regression Dataset](img/3a83b3e74ce3afc9f877d33e7ac66270.jpg)

线性回归数据集的散点图

## 直接解决

第一种方法是尝试直接解决回归问题。

也就是说，给定 X，当乘以 X 时，系数 b 的集合将给出 y。正如我们在前一节中看到的那样，正规方程定义了如何直接计算 b。

```
b = (X^T . X)^-1 . X^T . y
```

这可以使用 inv（）函数直接在 NumPy 中计算，以计算矩阵求逆。

```
b = inv(X.T.dot(X)).dot(X.T).dot(y)
```

一旦计算出系数，我们就可以用它们来预测给定 X 的结果。

```
yhat = X.dot(b)
```

将其与上一节中定义的数据集放在一起，下面列出了完整的示例。

```
# solve directly
from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot
data = array([
	[0.05, 0.12],
	[0.18, 0.22],
	[0.31, 0.35],
	[0.42, 0.38],
	[0.5, 0.49],
	])
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# linear least squares
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
```

运行该示例执行计算并打印系数向量 b。

```
[ 1.00233226]
```

然后使用模型的线图创建数据集的散点图，显示数据的合理拟合。

![Scatter Plot of Direct Solution to the Linear Regression Problem](img/91b6afb9c67f31b0595aec4fd7e51e5c.jpg)

线性回归问题直接解的散点图

这种方法的一个问题是矩阵逆，它在计算上既昂贵又在数值上不稳定。另一种方法是使用矩阵分解来避免这种操作。我们将在以下部分中查看两个示例。

## 通过 QR 分解解决

QR 分解是将矩阵分解为其组成元素的方法。

```
A = Q . R
```

其中 A 是我们希望分解的矩阵，Q 是尺寸为 m×m 的矩阵，R 是尺寸为 m×n 的上三角矩阵。

QR 分解是解决线性最小二乘方程的常用方法。

跨越所有推导，可以使用 Q 和 R 元素找到系数，如下所示：

```
b = R^-1 . Q.T . y
```

该方法仍然涉及矩阵求逆，但在这种情况下仅在更简单的 R 矩阵上。

可以使用 NumPy 中的 qr（）函数找到 QR 分解。 NumPy 中系数的计算如下：

```
# QR decomposition
Q, R = qr(X)
b = inv(R).dot(Q.T).dot(y)
```

将其与数据集结合在一起，下面列出了完整的示例。

```
# least squares via QR decomposition
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49],
])
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# QR decomposition
Q, R = qr(X)
b = inv(R).dot(Q.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
```

首先运行该示例打印系数解决方案并使用模型绘制数据。

```
[ 1.00233226]
```

QR 分解方法比直接计算正规方程计算效率更高，数值更稳定，但不适用于所有数据矩阵。

![Scatter Plot of QR Decomposition Solution to the Linear Regression Problem](img/e5a01f60368022abf81f7728508c79bb.jpg)

QR 分解解对线性回归问题的散点图

## 通过奇异值分解求解

奇异值分解（简称 SVD）是一种像 QR 分解一样的矩阵分解方法。

```
X = U . Sigma . V^*
```

其中 A 是我们希望分解的真实 nxm 矩阵，U 是 amxm 矩阵，Sigma（通常由大写希腊字母 Sigma 表示）是 mxn 对角矩阵，V ^ *是 nxn 矩阵的共轭转置，其中*是一个上标。

与 QR 分解不同，所有矩阵都具有 SVD 分解。作为求解线性回归线性方程组的基础，SVD 更稳定，是首选方法。

一旦分解，就可以通过计算输入矩阵 X 的伪逆并将其乘以输出向量 y 来找到系数。

```
b = X^+ . y
```

伪逆的计算方法如下：

```
X^+ = U . D^+ . V^T
```

其中 X ^ +是 X 的伪逆，+是上标，D ^ +是对角矩阵 Sigma 的伪逆，V ^ T 是 V ^ *的转置。

> 没有为非正方形的矩阵定义矩阵求逆。 [...]当 A 的列数多于行数时，使用 pseudoinverse 求解线性方程式提供了许多可能的解决方案之一。

- 第 46 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

我们可以通过 SVD 操作获得 U 和 V.可以通过从 Sigma 创建对角矩阵并计算 Sigma 中每个非零元素的倒数来计算 D ^ +。

```
         s11,   0,   0
Sigma = (  0, s22,   0)
           0,   0, s33

     1/s11,     0,     0
D = (    0, 1/s22,     0)
         0,     0, 1/s33
```

我们可以手动计算 SVD，然后计算伪逆。相反，NumPy 提供了我们可以直接使用的函数 pinv（）。

下面列出了完整的示例。

```
# least squares via SVD with pseudoinverse
from numpy import array
from numpy.linalg import pinv
from matplotlib import pyplot
data = array([
	[0.05, 0.12],
	[0.18, 0.22],
	[0.31, 0.35],
	[0.42, 0.38],
	[0.5, 0.49],
	])
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# calculate coefficients
b = pinv(X).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
```

运行该示例将打印系数并使用红线绘制数据，以显示模型中的预测。

```
[ 1.00233226]
```

实际上，NumPy 提供了一个函数来替换你可以直接使用的 lstsq（）函数中的这两个步骤。

![Scatter Plot of SVD Solution to the Linear Regression Problem](img/90bd88eb9baf4b460d9c76f03faeb6b7.jpg)

SVD 解对线性回归问题的散点图

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   使用内置的 lstsq（）NumPy 函数实现线性回归
*   在您自己的小型人为数据集上测试每个线性回归。
*   加载表格数据集并测试每个线性回归方法并比较结果。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 7.7 节最小二乘近似解。 [线性代数无废话指南](http://amzn.to/2k76D4)，2017 年。
*   第 4.3 节最小二乘近似，[线性代数简介](http://amzn.to/2AZ7R8j)，第五版，2016 年。
*   第 11 讲，最小二乘问题，[数值线性代数](http://amzn.to/2kjEF4S)，1997。
*   第 5 章，正交化和最小二乘法，[矩阵计算](http://amzn.to/2B9xnLD)，2012。
*   第 12 章，奇异值和 Jordan 分解，[线性代数和矩阵分析统计](http://amzn.to/2A9ceNv)，2014。
*   第 2.9 节 Moore-Penrose 伪逆，[深度学习](http://amzn.to/2B3MsuU)，2016。
*   第 15.4 节一般线性最小二乘法，[数字秘籍：科学计算的艺术](http://amzn.to/2BezVEE)，第三版，2007 年。

### API

*   [numpy.linalg.inv（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.inv.html)
*   [numpy.linalg.qr（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.qr.html)
*   [numpy.linalg.svd（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html)
*   [numpy.diag（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.diag.html)
*   [numpy.linalg.pinv（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.pinv.html)
*   [numpy.linalg.lstsq（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html)

### 用品

*   [维基百科上的线性回归](https://en.wikipedia.org/wiki/Linear_regression)
*   [维基百科上的最小二乘](https://en.wikipedia.org/wiki/Least_squares)
*   [维基百科上的线性最小二乘（数学）](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics))
*   [维基百科上的超定系统](https://en.wikipedia.org/wiki/Overdetermined_system)
*   维基百科上的 [QR 分解](https://en.wikipedia.org/wiki/QR_decomposition)
*   [维基百科上的奇异值分解](https://en.wikipedia.org/wiki/Singular-value_decomposition)
*   [Moore-Penrose 逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

### 教程

*   [最小二乘回归的线性代数视图](https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b)
    [线性代数与 Python 和 NumPy](https://meshlogic.github.io/posts/jupyter/linear-algebra/linear-algebra-with-python-and-numpy-2/)

## 摘要

在本教程中，您发现了线性回归的矩阵公式以及如何使用直接和矩阵分解方法来解决它。

具体来说，你学到了：

*   线性回归与正规方程的矩阵重构。
*   如何使用 QR 矩阵分解求解线性回归。
*   如何使用 SVD 和伪逆求解线性回归。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。