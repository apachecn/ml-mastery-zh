# 用于时间序列预测的 4 种通用机器学习数据变换

> 原文： [https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

时间序列数据通常需要在使用机器学习算法建模之前进行一些准备。

例如，差分运算可用于从序列中去除趋势和季节结构，以简化预测问题。一些算法（例如神经网络）更喜欢在建模之前对数据进行标准化和/或标准化。

应用于该系列的任何变换操作也需要在预测上应用类似的逆变换。这是必需的，以便得到的计算表现度量与输出变量的比例相同，并且可以与经典预测方法进行比较。

在这篇文章中，您将了解如何在机器学习中对时间序列数据执行和反转四种常见数据转换。

阅读这篇文章后，你会知道：

*   如何在 Python 中转换和反转四种方法的变换。
*   在训练和测试数据集上使用变换时的重要注意事项。
*   在数据集上需要多个操作时建议的转换顺序。

让我们开始吧。

![4 Common Machine Learning Data Transforms for Time Series Forecasting](img/d76413e9f5320b1fca77b30e22b91288.jpg)

用于时间序列预测的 4 种通用机器学习数据变换
照片由 [Wolfgang Staudt](https://www.flickr.com/photos/wolfgangstaudt/2200561848/) 拍摄，保留一些权利。

## 概观

本教程分为三个部分;他们是：

1.  时间序列数据的变换
2.  模型评估的考虑因素
3.  数据转换顺序

## 时间序列数据的变换

给定单变量时间序列数据集，在使用机器学习方法进行建模和预测时，有四种变换很流行。

他们是：

*   电力转换
*   差异变换
*   标准化
*   正常化

让我们依次快速浏览一下以及如何在 Python 中执行这些转换。

我们还将审查如何反转变换操作，因为当我们想要以原始比例评估预测时，这是必需的，以便可以直接比较表现度量。

您是否希望在时间序列数据上使用其他变换来进行机器学习方法的建模？
请在下面的评论中告诉我。

### 电力转换

[功率变换](https://en.wikipedia.org/wiki/Power_transform)从数据分布中移除偏移以使分布更正常（高斯分布）。

在时间序列数据集上，这可以消除随时间变化的方差。

流行的例子是对数变换（正值）或广义版本，例如 Box-Cox 变换（正值）或 Yeo-Johnson 变换（正值和负值）。

例如，我们可以使用 SciPy 库中的 [boxcox（）函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)在 Python 中实现 Box-Cox 变换。

默认情况下，该方法将以数字方式优化变换的 lambda 值并返回最佳值。

```
from scipy.stats import boxcox
# define data
data = ...
# box-cox transform
result, lmbda = boxcox(data)
```

变换可以反转，但需要一个名为 _invert_boxcox（）_ 的下面列出的自定义函数，它接受一个变换值和用于执行变换的 lambda 值。

```
from math import log
from math import exp
# invert a boxcox transform for one value
def invert_boxcox(value, lam):
	# log case
	if lam == 0:
		return exp(value)
	# all other cases
	return exp(log(lam * value + 1) / lam)
```

下面列出了将功率变换应用于数据集并反转变换的完整示例。

```
# example of power transform and inversion
from math import log
from math import exp
from scipy.stats import boxcox

# invert a boxcox transform for one value
def invert_boxcox(value, lam):
	# log case
	if lam == 0:
		return exp(value)
	# all other cases
	return exp(log(lam * value + 1) / lam)

# define dataset
data = [x for x in range(1, 10)]
print(data)
# power transform
transformed, lmbda = boxcox(data)
print(transformed, lmbda)
# invert transform
inverted = [invert_boxcox(x, lmbda) for x in transformed]
print(inverted)
```

运行该示例将在转换变换后打印原始数据集，幂变换的结果以及原始值（或接近它）。

```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
[0\.         0.89887536 1.67448353 2.37952145 3.03633818 3.65711928
 4.2494518  4.81847233 5.36786648] 0.7200338588580095
[1.0, 2.0, 2.9999999999999996, 3.999999999999999, 5.000000000000001, 6.000000000000001, 6.999999999999999, 7.999999999999998, 8.999999999999998]
```

### 差异变换

差分变换是从时间序列中去除系统结构的简单方法。

例如，可以通过从系列中的每个值中减去先前的值来消除趋势。这称为一阶差分。可以重复该过程（例如差异系列）以消除二阶趋势，等等。

通过从前一季节中减去观察值，可以以类似的方式去除季节性结构。 12 个步骤之前的月度数据与年度季节性结构。

可以使用下面列出的名为 _difference（）_ 的自定义函数计算系列中的单个差异值。该函数采用时间序列和差值计算的间隔，例如， 1 表示趋势差异，12 表示季节性差异。

```
# difference dataset
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]
```

同样，可以使用自定义函数反转此操作，该函数将原始值添加回名为 _invert_difference（）_ 的差值，该值采用原始序列和间隔。

```
# invert difference
def invert_difference(orig_data, diff_data, interval):
	return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]
```

我们可以在下面演示这个功能。

```
# example of a difference transform

# difference dataset
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# invert difference
def invert_difference(orig_data, diff_data, interval):
	return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]

# define dataset
data = [x for x in range(1, 10)]
print(data)
# difference transform
transformed = difference(data, 1)
print(transformed)
# invert difference
inverted = invert_difference(data, transformed, 1)
print(inverted)
```

运行该示例将打印原始数据集，差异变换的结果以及转换后的原始值。

注意，变换后序列中的第一个“间隔”值将丢失。这是因为它们在“间隔”之前的时间步长没有值，因此无法区分。

```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
[1, 1, 1, 1, 1, 1, 1, 1]
[2, 3, 4, 5, 6, 7, 8, 9]
```

### 标准化

标准化是具有高斯分布的数据的变换。

它减去均值并将结果除以数据样本的标准差。这具有将数据转换为具有零或中心的均值的效果，其标准偏差为 1.这样得到的分布称为标准高斯分布，或标准法线，因此称为变换的名称。

我们可以使用 scikit-learn 库中的 Python 中的 [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 对象执行标准化。

此类允许通过调用 _fit（）_ 将变换拟合到训练数据集上，通过调用 _transform（）_ 应用于一个或多个数据集（例如训练和测试）并且还提供通过调用 _inverse_transform（）_ 来反转变换的函数。

下面应用完整的示例。

```
# example of standardization
from sklearn.preprocessing import StandardScaler
from numpy import array
# define dataset
data = [x for x in range(1, 10)]
data = array(data).reshape(len(data), 1)
print(data)
# fit transform
transformer = StandardScaler()
transformer.fit(data)
# difference transform
transformed = transformer.transform(data)
print(transformed)
# invert difference
inverted = transformer.inverse_transform(transformed)
print(inverted)
```

运行该示例将打印原始数据集，标准化变换的结果以及转换后的原始值。

请注意，期望数据作为具有多行的列提供。

```
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]

[[-1.54919334]
 [-1.161895  
 [-0.77459667]
 [-0.38729833]
 [ 0\.        
 [ 0.38729833]
 [ 0.77459667]
 [ 1.161895  
 [ 1.54919334]]

[[1.]
 [2.]
 [3.]
 [4.]
 [5.]
 [6.]
 [7.]
 [8.]
 [9.]]
```

### 正常化

规范化是将数据从原始范围重新缩放到 0 到 1 之间的新范围。

与标准化一样，这可以使用 scikit-learn 库中的转换对象来实现，特别是 [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 类。除了规范化之外，通过在对象的构造函数中指定首选范围，此类可用于将数据重新缩放到您希望的任何范围。

它可以以相同的方式用于拟合，变换和反转变换。

下面列出了一个完整的例子。

```
# example of normalization
from sklearn.preprocessing import MinMaxScaler
from numpy import array
# define dataset
data = [x for x in range(1, 10)]
data = array(data).reshape(len(data), 1)
print(data)
# fit transform
transformer = MinMaxScaler()
transformer.fit(data)
# difference transform
transformed = transformer.transform(data)
print(transformed)
# invert difference
inverted = transformer.inverse_transform(transformed)
print(inverted)
```

运行该示例将打印原始数据集，规范化转换的结果以及转换后的原始值。

```
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]

[[0\.   
 [0.125]
 [0.25 ]
 [0.375]
 [0.5  
 [0.625]
 [0.75 ]
 [0.875]
 [1\.   ]

[[1.]
 [2.]
 [3.]
 [4.]
 [5.]
 [6.]
 [7.]
 [8.]
 [9.]]
```

## 模型评估的考虑因素

我们已经提到了能够反转模型预测变换的重要性，以便计算与其他方法直接相当的模型表现统计量。

另外，另一个问题是数据泄漏问题。

上述三个数据转换来自提供的数据集的估计系数，然后用于转换数据。特别：

*   **Power Transform** ：lambda 参数。
*   **标准化**：平均值和标准差统计量。
*   **标准化**：最小值和最大值。

必须仅在训练数据集上估计这些系数。

估计完成后，可以在评估模型之前使用系数对训练和测试数据集应用变换。

如果在分割成训练集和测试集之前使用整个数据集估计系数，则从测试集到训练数据集的信息泄漏很小。这可能导致对乐观偏见的模型技能的估计。

因此，您可能希望使用领域知识增强系数的估计值，例如将来所有时间的预期最小值/最大值。

通常，差分不会遇到相同的问题。在大多数情况下，例如一步预测，可以使用滞后观察来执行差异计算。如果不是，则可以在任何需要的地方使用滞后预测作为差异计算中真实观察的代理。

## 数据转换顺序

您可能希望尝试在建模之前将多个数据转换应用于时间序列。

这很常见，例如应用幂变换以消除增加的方差，应用季节差异来消除季节性，并应用一步差分来移除趋势。

应用转换操作的顺序很重要。

直觉上，我们可以思考变换如何相互作用。

*   应该在差分之前执行功率变换。
*   应在一步差分之前进行季节性差异。
*   标准化是线性的，应在任何非线性变换和差分后对样本进行标准化。
*   标准化是线性操作，但它应该是为保持首选标度而执行的最终变换。

因此，建议的数据转换顺序如下：

1.  电力转换。
2.  季节性差异。
3.  趋势差异。
4.  标准化。
5.  正常化。

显然，您只能使用特定数据集所需的变换。

重要的是，当变换操作被反转时，必须反转逆变换操作的顺序。具体而言，必须按以下顺序执行逆操作：

1.  正常化。
2.  标准化。
3.  趋势差异。
4.  季节性差异。
5.  电力转换。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何使用 Python 进行时间序列预测数据的电源转换](https://machinelearningmastery.com/power-transform-time-series-forecast-data-python/)
*   [如何使用 Python 中的差异变换删除趋势和季节性](https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/)
*   [如何区分时间序列数据集与 Python](https://machinelearningmastery.com/difference-time-series-dataset-python/)
*   [如何在 Python 中标准化和标准化时间序列数据](https://machinelearningmastery.com/normalize-standardize-time-series-data-python/)

### 蜜蜂

*   [scipy.stats.boxcox API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
*   [sklearn.preprocessing.MinMaxScaler API](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
*   [sklearn.preprocessing.StandardScaler API](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### 用品

*   [维基百科上的权力转换](https://en.wikipedia.org/wiki/Power_transform)

## 摘要

在这篇文章中，您了解了如何在机器学习中对时间序列数据执行和反转四种常见数据转换。

具体来说，你学到了：

*   如何在 Python 中转换和反转四种方法的变换。
*   在训练和测试数据集上使用变换时的重要注意事项。
*   在数据集上需要多个操作时建议的转换顺序。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。