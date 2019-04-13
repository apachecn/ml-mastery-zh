# 浅谈机器学习的奇异值分解

> 原文： [https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)

矩阵分解，也称为矩阵分解，涉及使用其组成元素描述给定矩阵。

也许最着名和最广泛使用的矩阵分解方法是奇异值分解或 SVD。所有矩阵都有一个 SVD，这使得它比其他方法更稳定，例如特征分解。因此，它经常用于各种应用，包括压缩，去噪和数据缩减。

在本教程中，您将发现用于将矩阵分解为其组成元素的奇异值分解方法。

完成本教程后，您将了解：

*   奇异值分解是什么以及涉及什么。
*   如何计算 SVD 并从 SVD 元素重建矩形和方形矩阵。
*   如何使用 SVD 计算伪逆并执行降维

让我们开始吧。

*   **更新 Mar / 2018** ：修复了重建中的拼写错误。为清晰起见，将代码中的 V 更改为 VT。修正了伪逆方程中的拼写错误。

![A Gentle Introduction to Singular-Value Decomposition](img/f2ce7cb34e8831e36bbc9775cd8fded9.jpg)

奇异值分解
照片由 [Chris Heald](https://www.flickr.com/photos/husker_alum/8628799410/) 拍摄，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  奇异值分解
2.  计算奇异值分解
3.  从 SVD 重构矩阵
4.  伪逆的 SVD
5.  用于降维的 SVD

## 奇异值分解

奇异值分解（简称 SVD）是一种矩阵分解方法，用于将矩阵减少到其组成部分，以使某些后续矩阵计算更简单。

为简单起见，我们将重点关注实值矩阵的 SVD，并忽略复数的情况。

```
A = U . Sigma . V^T
```

其中 A 是我们希望分解的真实 mxn 矩阵，U 是 mxm 矩阵，Sigma（通常由大写希腊字母 Sigma 表示）是 mxn 对角矩阵，V ^ T 是 nxn 矩阵的转置，其中 T 是一个上标。

> 奇异值分解是线性代数的一个亮点。

- 第 371 页，[线性代数导论](http://amzn.to/2AZ7R8j)，第五版，2016 年。

Sigma 矩阵中的对角线值称为原始矩阵 A 的奇异值.U 矩阵的列称为 A 的左奇异向量，V 列称为 A 的右奇异向量。

通过迭代数值方法计算 SVD。我们不会详细介绍这些方法。每个矩形矩阵都具有奇异值分解，尽管得到的矩阵可能包含复数，浮点运算的局限性可能会导致某些矩阵无法整齐地分解。

> 奇异值分解（SVD）提供了另一种将矩阵分解为奇异向量和奇异值的方法。 SVD 允许我们发现一些与特征分解相同的信息。但是，SVD 更普遍适用。

- 第 44-45 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

SVD 广泛用于计算其他矩阵运算，例如矩阵逆运算，但也作为机器学习中的数据简化方法。 SVD 还可用于最小二乘线性回归，图像压缩和去噪数据。

> 奇异值分解（SVD）在统计学，机器学习和计算机科学中有许多应用。将 SVD 应用于矩阵就像在 X 射线视觉中查看它...

- 第 297 页，[无线性代数废话指南](http://amzn.to/2k76D4C)，2017 年

## 计算奇异值分解

可以通过调用 svd（）函数来计算 SVD。

该函数采用矩阵并返回 U，Sigma 和 V ^ T 元素。 Sigma 对角矩阵作为奇异值的向量返回。 V 矩阵以转置的形式返回，例如， V.T.

下面的示例定义了 3×2 矩阵并计算奇异值分解。

```
# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD
U, s, VT = svd(A)
print(U)
print(s)
print(VT)
```

首先运行该示例打印定义的 3×2 矩阵，然后打印 3×3U 矩阵，2 元素 Sigma 向量和从分解计算的 2×2V ^ T 矩阵元素。

```
[[1 2]
 [3 4]
 [5 6]]

[[-0.2298477   0.88346102  0.40824829]
 [-0.52474482  0.24078249 -0.81649658]
 [-0.81964194 -0.40189603  0.40824829]]

[ 9.52551809  0.51430058]

[[-0.61962948 -0.78489445]
 [-0.78489445  0.61962948]]
```

## 从 SVD 重构矩阵

可以从 U，Sigma 和 V ^ T 元素重建原始矩阵。

从 svd（）返回的 U，s 和 V 元素不能直接相乘。

必须使用 diag（）函数将 s 向量转换为对角矩阵。默认情况下，此函数将创建一个相对于原始矩阵 m x m 的方阵。这导致问题，因为矩阵的大小不符合矩阵乘法的规则，其中矩阵中的列数必须与后续矩阵中的行数匹配。

在创建方形 Sigma 对角矩阵之后，矩阵的大小相对于我们正在分解的原始 m x n 矩阵，如下所示：

```
U (m x m) . Sigma (m x m) . V^T (n x n)
```

事实上，我们要求：

```
U (m x m) . Sigma (m x n) . V^T (n x n)
```

我们可以通过创建所有零值 m x n（例如更多行）的新 Sigma 格式来实现这一点，并用通过 diag（）计算的方形对角矩阵填充矩阵的前 n x n 部分。

```
# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)
```

首先运行该示例打印原始矩阵，然后打印从 SVD 元素重建的矩阵。

```
[[1 2]
 [3 4]
 [5 6]]

[[ 1\.  2.]
 [ 3\.  4.]
 [ 5\.  6.]]
```

上述与 Sigma 对角线的复杂性仅存在于 m 和 n 不相等的情况下。当重建方形矩阵时，可以直接使用对角矩阵，如下所述。

```
# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from scipy.linalg import svd
# define a matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create n x n Sigma matrix
Sigma = diag(s)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)
```

运行该示例打印原始 3×3 矩阵和直接从 SVD 元素重建的版本。

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]

[[ 1\.  2\.  3.]
 [ 4\.  5\.  6.]
 [ 7\.  8\.  9.]]
```

## 伪逆的 SVD

伪逆是矩形矩阵到矩形矩阵的矩阵逆的推广，其中行和列的数量不相等。

在该方法的两个独立发现者或广义逆之后，它也被称为 Moore-Penrose 逆。

> 没有为非正方形的矩阵定义矩阵求逆。 [...]当 A 的列数多于行数时，使用 pseudoinverse 求解线性方程式提供了许多可能的解决方案之一。

- 第 46 页，[深度学习](http://amzn.to/2B3MsuU)，2016 年。

伪逆表示为 A ^ +，其中 A 是被反转的矩阵，+是上标。

使用 A 的奇异值分解计算伪逆：

```
A^+ = V . D^+ . U^T
```

或者，没有点符号：

```
A^+ = VD^+U^T
```

其中 A ^ +是伪逆，D ^ +是对角矩阵 Sigma 的伪逆，U ^ T 是 U 的转置。

我们可以通过 SVD 操作获得 U 和 V.

```
A = U . Sigma . V^T
```

可以通过从 Sigma 创建对角矩阵来计算 D ^ +，计算 Sigma 中每个非零元素的倒数，并且如果原始矩阵是矩形则采用转置。

```
         s11,   0,   0
Sigma = (  0, s22,   0)
           0,   0, s33
```

```
       1/s11,     0,     0
D^+ = (    0, 1/s22,     0)
           0,     0, 1/s33
```

伪逆提供了一种求解线性回归方程的方法，特别是当行数多于列时，通常就是这种情况。

NumPy 提供函数 pinv（）来计算矩形矩阵的伪逆。

下面的示例定义了一个 4×2 矩阵并计算伪逆。

```
# Pseudoinverse
from numpy import array
from numpy.linalg import pinv
# define matrix
A = array([
	[0.1, 0.2],
	[0.3, 0.4],
	[0.5, 0.6],
	[0.7, 0.8]])
print(A)
# calculate pseudoinverse
B = pinv(A)
print(B)
```

首先运行示例打印定义的矩阵，然后打印计算的伪逆。

```
[[ 0.1  0.2]
 [ 0.3  0.4]
 [ 0.5  0.6]
 [ 0.7  0.8]]

[[ -1.00000000e+01  -5.00000000e+00   9.04289323e-15   5.00000000e+00]
 [  8.50000000e+00   4.50000000e+00   5.00000000e-01  -3.50000000e+00]]
```

我们可以通过 SVD 手动计算伪逆，并将结果与​​pinv（）函数进行比较。

首先，我们必须计算 SVD。接下来，我们必须计算 s 数组中每个值的倒数。然后可以将 s 数组转换为具有添加的零行的对角矩阵，以使其成为矩形。最后，我们可以从元素中计算出伪逆。

具体实施是：

```
A^+ = V . D^+ . U^V
```

下面列出了完整的示例。

```
# Pseudoinverse via SVD
from numpy import array
from numpy.linalg import svd
from numpy import zeros
from numpy import diag
# define matrix
A = array([
	[0.1, 0.2],
	[0.3, 0.4],
	[0.5, 0.6],
	[0.7, 0.8]])
print(A)
# calculate svd
U, s, VT = svd(A)
# reciprocals of s
d = 1.0 / s
# create m x n D matrix
D = zeros(A.shape)
# populate D with n x n diagonal matrix
D[:A.shape[1], :A.shape[1]] = diag(d)
# calculate pseudoinverse
B = VT.T.dot(D.T).dot(U.T)
print(B)
```

首先运行示例打印定义的矩形矩阵和与 pinv（）函数匹配上述结果的伪逆。

```
[[ 0.1  0.2]
 [ 0.3  0.4]
 [ 0.5  0.6]
 [ 0.7  0.8]]

[[ -1.00000000e+01  -5.00000000e+00   9.04831765e-15   5.00000000e+00]
 [  8.50000000e+00   4.50000000e+00   5.00000000e-01  -3.50000000e+00]]
```

## 用于降维的 SVD

SVD 的一种流行应用是降低尺寸。

具有大量特征的数据（例如，比观察（行）更多的特征（列））可以减少到与预测问题最相关的较小特征子集。

结果是具有较低等级的矩阵，据说接近原始矩阵。

为此，我们可以对原始数据执行 SVD 操作，并在 Sigma 中选择前 k 个最大奇异值。这些列可以从 Sigma 和从 V ^ T 中选择的行中选择。

然后可以重建原始向量 A 的近似 B.

```
B = U . Sigmak . V^Tk
```

在自然语言处理中，该方法可以用于文档中的单词出现或单词频率的矩阵，并且被称为潜在语义分析或潜在语义索引。

在实践中，我们可以保留并使用名为 T 的数据的描述子集。这是矩阵或投影的密集摘要。

```
T = U . Sigmak
```

此外，可以计算该变换并将其应用于原始矩阵 A 以及其他类似的矩阵。

```
T = V^Tk . A
```

下面的示例演示了使用 SVD 减少数据。

首先定义 3×10 矩阵，列数多于行数。计算 SVD 并仅选择前两个特征。重新组合元素以给出原始矩阵的准确再现。最后，变换以两种不同的方式计算。

```
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# reconstruct
B = U.dot(Sigma.dot(VT))
print(B)
# transform
T = U.dot(Sigma)
print(T)
T = A.dot(VT.T)
print(T)
```

首先运行该示例打印定义的矩阵然后重建近似，然后是原始矩阵的两个等效变换。

```
[[ 1  2  3  4  5  6  7  8  9 10]
 [11 12 13 14 15 16 17 18 19 20]
 [21 22 23 24 25 26 27 28 29 30]]

[[  1\.   2\.   3\.   4\.   5\.   6\.   7\.   8\.   9\.  10.]
 [ 11\.  12\.  13\.  14\.  15\.  16\.  17\.  18\.  19\.  20.]
 [ 21\.  22\.  23\.  24\.  25\.  26\.  27\.  28\.  29\.  30.]]

[[-18.52157747   6.47697214]
 [-49.81310011   1.91182038]
 [-81.10462276  -2.65333138]]

[[-18.52157747   6.47697214]
 [-49.81310011   1.91182038]
 [-81.10462276  -2.65333138]]
```

scikit-learn 提供了一个直接实现此功能的 TruncatedSVD 类。

可以创建 TruncatedSVD 类，您必须在其中指定要选择的所需要素或组件的数量，例如， 2.一旦创建，您可以通过调用 fit（）函数来拟合变换（例如，计算 V ^ Tk），然后通过调用 transform（）函数将其应用于原始矩阵。结果是上面称为 T 的 A 的变换。

下面的示例演示了 TruncatedSVD 类。

```
from numpy import array
from sklearn.decomposition import TruncatedSVD
# define array
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print(A)
# svd
svd = TruncatedSVD(n_components=2)
svd.fit(A)
result = svd.transform(A)
print(result)
```

首先运行示例打印定义的矩阵，然后打印矩阵的转换版本。

我们可以看到值与上面手动计算的值匹配，除了某些值上的符号。考虑到所涉及的计算的性质以及所使用的底层库和方法的差异，我们可以预期在符号方面存在一些不稳定性。只要对变换进行了重复训练，这种符号的不稳定性在实践中就不应成为问题。

```
[[ 1  2  3  4  5  6  7  8  9 10]
 [11 12 13 14 15 16 17 18 19 20]
 [21 22 23 24 25 26 27 28 29 30]]

[[ 18.52157747   6.47697214]
 [ 49.81310011   1.91182038]
 [ 81.10462276  -2.65333138]]
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   在您自己的数据上试验 SVD 方法。
*   研究并列出了 SVD 在机器学习中的 10 个应用。
*   将 SVD 作为数据缩减技术应用于表格数据集。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   第 12 章，奇异值和 Jordan 分解，[线性代数和矩阵分析统计](http://amzn.to/2A9ceNv)，2014。
*   第 4 章，奇异值分解和第 5 章，关于 SVD 的更多内容，[数值线性代数](http://amzn.to/2kjEF4S)，1997。
*   第 2.4 节奇异值分解，[矩阵计算](http://amzn.to/2B9xnLD)，2012。
*   第 7 章奇异值分解（SVD），[线性代数导论](http://amzn.to/2AZ7R8j)，第 5 版，2016 年。
*   第 2.8 节奇异值分解，[深度学习](http://amzn.to/2B3MsuU)，2016 年。
*   第 7.D 节极性分解和奇异值分解，[线性代数完成权](http://amzn.to/2BGuEqI)，第三版，2015 年。
*   第 3 讲奇异值分解，[数值线性代数](http://amzn.to/2BI9kRH)，1997。
*   第 2.6 节奇异值分解，[数字秘籍：科学计算的艺术](http://amzn.to/2BezVEE)，第三版，2007。
*   第 2.9 节 Moore-Penrose 伪逆，[深度学习](http://amzn.to/2B3MsuU)，2016。

### API

*   [numpy.linalg.svd（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html)
*   [numpy.matrix.H API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.H.html)
*   [numpy.diag（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.diag.html)
*   [numpy.linalg.pinv（）API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.pinv.html) 。
*   [sklearn.decomposition.TruncatedSVD API](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

### 用品

*   维基百科上的[矩阵分解](https://en.wikipedia.org/wiki/Matrix_decomposition)
*   [维基百科上的奇异值分解](https://en.wikipedia.org/wiki/Singular-value_decomposition)
*   [维基百科上的奇异值](https://en.wikipedia.org/wiki/Singular_value)
*   [维基百科上的 Moore-Penrose 逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
*   [维基百科上的潜在语义分析](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

## 摘要

在本教程中，您发现了奇异值分解方法，用于将矩阵分解为其组成元素。

具体来说，你学到了：

*   奇异值分解是什么以及涉及什么。
*   如何计算 SVD 并从 SVD 元素重建矩形和方形矩阵。
*   如何使用 SVD 计算伪逆并执行降维。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。