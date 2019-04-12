# 如何在 Python 中扩展长短期内存网络的数据

> 原文： [https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

在训练神经网络时，可能需要缩放序列预测问题的数据，例如长期短期记忆复现神经网络。

当网络适合具有一系列值的非缩放数据（例如，数量在 10 到 100 之间）时，大输入可能会减慢网络的学习和收敛速度，并且在某些情况下会妨碍网络有效地学习问题。

在本教程中，您将了解如何规范化和标准化序列预测数据，以及如何确定将哪个用于输入和输出变量。

完成本教程后，您将了解：

*   如何在 Python 中规范化和标准化序列数据。
*   如何为输入和输出变量选择适当的缩放。
*   缩放序列数据时的实际考虑因素

让我们开始吧。

![How to Scale Data for Long Short-Term Memory Networks in Python](img/873e95dd71301bfac68e206f7783e0c0.jpg)

如何在 Python 中扩展长短期内存网络的数据
图片来自 [Mathias Appel](https://www.flickr.com/photos/mathiasappel/25527849934/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  缩放系列数据
2.  缩放输入变量
3.  缩放输出变量
4.  缩放时的实际考虑因素

## 在 Python 中扩展系列数据

您可能需要考虑两种类型的系列缩放：规范化和标准化。

这些都可以使用 scikit-learn 库来实现。

### 规范化系列数据

归一化是对原始范围内的数据进行重新缩放，以使所有值都在 0 和 1 的范围内。

标准化要求您知道或能够准确估计最小和最大可观察值。您可以从可用数据中估算这些值。如果您的时间序列趋势向上或向下，估计这些预期值可能会很困难，并且规范化可能不是用于解决问题的最佳方法。

值按如下标准化：

```
y = (x - min) / (max - min)
```

其中最小值和最大值与值 x 被归一化有关。

例如，对于数据集，我们可以将 min 和 max 可观察值猜测为 30 和-10。然后我们可以将任何值标准化，如 18.8，如下所示：

```
y = (x - min) / (max - min)
y = (18.8 - (-10)) / (30 - (-10))
y = 28.8 / 40
y = 0.72
```

您可以看到，如果提供的 x 值超出最小值和最大值的范围，则结果值将不在 0 和 1 的范围内。您可以在进行预测之前检查这些观察值并删除它们来自数据集或将它们限制为预定义的最大值或最小值。

您可以使用 scikit-learn 对象 [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 来规范化数据集。

使用 MinMaxScaler 和其他缩放技术的良好实践用法如下：

*   **使用可用的训练数据**调整定标器。对于归一化，这意味着训练数据将用于估计最小和最大可观察值。这是通过调用 fit（）函数完成的。
*   **将比例应用于训练数据**。这意味着您可以使用标准化数据来训练模型。这是通过调用 transform（）函数完成的。
*   **将比例应用于前进的数据**。这意味着您可以在将来准备要预测的新数据。

如果需要，可以反转变换。这对于将预测转换回其原始比例以进行报告或绘图非常有用。这可以通过调用 inverse_transform（）函数来完成。

下面是一个标准化 10 个量的人为序列的例子。

缩放器对象要求将数据作为行和列的矩阵提供。加载的时间序列数据作为 Pandas 系列加载。

```
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
# define contrived series
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
series = Series(data)
print(series)
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print
normalized = scaler.transform(values)
print(normalized)
# inverse transform and print
inversed = scaler.inverse_transform(normalized)
print(inversed)
```

运行该示例打印序列，打印从序列估计的最小值和最大值，打印相同的标准化序列，然后使用逆变换将值返回到其原始比例。

我们还可以看到数据集的最小值和最大值分别为 10.0 和 100.0。

```
0     10.0
1     20.0
2     30.0
3     40.0
4     50.0
5     60.0
6     70.0
7     80.0
8     90.0
9    100.0

Min: 10.000000, Max: 100.000000

[[ 0\.        ]
 [ 0.11111111]
 [ 0.22222222]
 [ 0.33333333]
 [ 0.44444444]
 [ 0.55555556]
 [ 0.66666667]
 [ 0.77777778]
 [ 0.88888889]
 [ 1\.        ]]

[[  10.]
 [  20.]
 [  30.]
 [  40.]
 [  50.]
 [  60.]
 [  70.]
 [  80.]
 [  90.]
 [ 100.]]
```

### 标准化系列数据

标准化数据集涉及重新调整值的分布，以便观察值的平均值为 0，标准差为 1。

这可以被认为是减去平均值或使数据居中。

与标准化一样，当您的数据具有不同比例的输入值时，标准化可能很有用，甚至在某些机器学习算法中也是必需的。

标准化假定您的观察结果符合高斯分布（钟形曲线），具有良好的平均值和标准偏差。如果不满足此期望，您仍然可以标准化时间序列数据，但可能无法获得可靠的结果。

标准化要求您知道或能够准确估计可观察值的均值和标准差。您可以从训练数据中估算这些值。

值标准化如下：

```
y = (x - mean) / standard_deviation
```

平均值计算如下：

```
mean = sum(x) / count(x)
```

而 standard_deviation 计算如下：

```
standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))
```

我们可以猜测平均值为 10，标准偏差约为 5.使用这些值，我们可以将第一个值 20.7 标准化如下：

```
y = (x - mean) / standard_deviation
y = (20.7 - 10) / 5
y = (10.7) / 5
y = 2.14
```

数据集的均值和标准差估计值对于新数据可能比最小值和最大值更稳健。

您可以使用 scikit-learn 对象 [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 标准化数据集。

```
from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
# define contrived series
data = [1.0, 5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]
series = Series(data)
print(series)
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
# normalize the dataset and print
standardized = scaler.transform(values)
print(standardized)
# inverse transform and print
inversed = scaler.inverse_transform(standardized)
print(inversed)
```

运行该示例打印序列，打印从序列估计的平均值和标准差，打印标准化值，然后以原始比例打印值。

我们可以看到估计的平均值和标准偏差分别约为 5.3 和 2.7。

```
0    1.0
1    5.5
2    9.0
3    2.6
4    8.8
5    3.0
6    4.1
7    7.9
8    6.3

Mean: 5.355556, StandardDeviation: 2.712568

[[-1.60569456]
 [ 0.05325007]
 [ 1.34354035]
 [-1.01584758]
 [ 1.26980948]
 [-0.86838584]
 [-0.46286604]
 [ 0.93802055]
 [ 0.34817357]]

[[ 1\. ]
 [ 5.5]
 [ 9\. ]
 [ 2.6]
 [ 8.8]
 [ 3\. ]
 [ 4.1]
 [ 7.9]
 [ 6.3]]
```

## 缩放输入变量

输入变量是网络在输入或可见层上进行的预测。

一个好的经验法则是输入变量应该是小值，可能在 0-1 范围内，或者标准化为零均值和标准差为 1。

输入变量是否需要缩放取决于问题和每个变量的具体情况。我们来看一些例子。

### 分类输入

您可能有一系列分类输入，例如字母或状态。

通常，分类输入首先是整数编码，然后是一个热编码。也就是说，为每个不同的可能输入分配唯一的整数值，然后使用 1 和 0 的二进制向量来表示每个整数值。

根据定义，一个热编码将确保每个输入是一个小的实际值，在这种情况下为 0.0 或 1.0。

### 实值输入

您可能有一系列数量作为输入，例如价格或温度。

如果数量的分布是正常的，则应该标准化，否则系列应该标准化。如果数值范围很大（10s 100s 等）或小（0.01,0.0001），则适用。

如果数量值很小（接近 0-1）并且分布有限（例如标准偏差接近 1），那么也许你可以在没有缩放系列的情况下逃脱。

### 其他输入

问题可能很复杂，如何最好地扩展输入数据可能并不清楚。

如果有疑问，请将输入序列标准化。如果您拥有这些资源，请使用原始数据，标准化数据和标准化来探索建模，并查看是否存在有益的差异。

> 如果输入变量是线性组合的，就像在 MLP [多层感知器]中一样，那么至少在理论上很少有必要对输入进行标准化。 ......但是，有很多实际的原因可以解释为什么标准化输入可以使训练更快，并减少陷入局部最优的机会。

- [我应该规范化/标准化/重新缩放数据吗？](ftp://ftp.sas.com/pub/neural/FAQ2.html#A_std) 神经网络常见问题解答

## 缩放输出变量

输出变量是网络预测的变量。

您必须确保输出变量的比例与网络输出层上的激活功能（传递函数）的比例相匹配。

> 如果输出激活函数的范围为[0,1]，那么显然您必须确保目标值位于该范围内。但通常最好选择适合目标分布的输出激活函数，而不是强制数据符合输出激活函数。

- [我应该规范化/标准化/重新缩放数据吗？](ftp://ftp.sas.com/pub/neural/FAQ2.html#A_std) 神经网络常见问题解答

以下启发式方法应涵盖大多数序列预测问题：

### 二元分类问题

如果您的问题是二元分类问题，那么输出将是类值 0 和 1.这最好使用输出层上的 sigmoid 激活函数建模。输出值将是介于 0 和 1 之间的实数值，可以捕捉到清晰的值。

### 多类分类问题

如果您的问题是多类分类问题，那么输出将是一个介于 0 和 1 之间的二进制类值的向量，每个类值一个输出。这最好使用输出层上的 softmax 激活功能建模。同样，输出值将是介于 0 和 1 之间的实数值，可以捕捉到清晰的值。

### 回归问题

如果您的问题是回归问题，那么输出将是实际值。这最好使用线性激活功能建模。如果值的分布正常，则可以标准化输出变量。否则，可以对输出变量进行标准化。

### 其他问题

可以在输出层上使用许多其他激活函数，并且您的问题的细节可能会增加混淆。

经验法则是确保网络输出与数据规模相匹配。

## 缩放时的实际考虑因素

缩放序列数据时有一些实际考虑因素。

*   **估算系数**。您可以从训练数据中估算系数（标准化的最小值和最大值或标准化的平均值和标准偏差）。检查这些首先估算并使用领域知识或领域专家来帮助改进这些估算，以便将来对所有数据进行有用的更正。
*   **保存系数**。您将需要以与用于训练模型的数据完全相同的方式对未来的新数据进行标准化。保存用于存档的系数，并在需要在进行预测时扩展新数据时加载它们。
*   **数据分析**。使用数据分析可以帮助您更好地了解数据。例如，一个简单的直方图可以帮助您快速了解数量的分布情况，看看标准化是否有意义。
*   **缩放每个系列**。如果您的问题有多个系列，请将每个系列视为单独的变量，然后分别对其进行缩放。
*   **在合适的时间缩放**。在正确的时间应用任何缩放变换非常重要。例如，如果您有一系列非静止的数量，则在首次使数据静止后进行缩放可能是合适的。在将系列转换为监督学习问题后对其进行扩展是不合适的，因为每个列的处理方式不同，这是不正确的。
*   **如果怀疑**则缩放。您可能需要重新调整输入和输出变量。如果有疑问，至少要对数据进行标准化。

## 进一步阅读

本节列出了扩展时要考虑的一些其他资源。

*   [我应该规范化/标准化/重新调整数据吗？](ftp://ftp.sas.com/pub/neural/FAQ2.html#A_std) 神经网络常见问题解答
*   [MinMaxScaler scikit-learn API 文档](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
*   [StandardScaler scikit-learn API 文档](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
*   [如何使用 Python 从头开始扩展机器学习数据](http://machinelearningmastery.com/scale-machine-learning-data-scratch-python/)
*   [如何在 Python 中标准化和标准化时间序列数据](http://machinelearningmastery.com/normalize-standardize-time-series-data-python/)
*   [如何使用 Scikit-Learn 为 Python 机器学习准备数据](http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)

## 摘要

在本教程中，您了解了在使用长短期记忆复现神经网络时如何扩展序列预测数据。

具体来说，你学到了：

*   如何在 Python 中规范化和标准化序列数据。
*   如何为输入和输出变量选择适当的缩放。
*   缩放序列数据时的实际考虑因素

您对缩放序列预测数据有任何疑问吗？
在评论中提出您的问题，我会尽力回答。