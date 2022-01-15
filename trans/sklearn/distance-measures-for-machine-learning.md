# 4 机器学习的距离度量

> 原文：<https://machinelearningmastery.com/distance-measures-for-machine-learning/>

最后更新于 2020 年 8 月 19 日

距离度量在机器学习中起着重要的作用。

它们为许多流行且有效的机器学习算法提供了基础，如用于监督学习的 k 近邻和用于非监督学习的 k 均值聚类。

必须根据数据类型选择和使用不同的距离测量。因此，重要的是要知道如何实现和计算一系列不同的流行的距离测量和最终得分的直觉。

在本教程中，您将发现机器学习中的距离度量。

完成本教程后，您将知道:

*   距离度量在机器学习算法中的作用和重要性。
*   如何实现和计算汉明、欧几里德和曼哈顿距离度量。
*   如何实现和计算推广欧几里德和曼哈顿距离度量的闵可夫斯基距离。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Distance Measures for Machine Learning](img/04784974217d9d6da3540f8f68a73f55.png)

机器学习的距离测量
图片由[罗伊王子](https://flickr.com/photos/princeroy/2092543671/)提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  距离测量的作用
2.  汉娩距
3.  欧几里得距离
4.  曼哈顿距离(出租车或城市街区)
5.  闵可夫斯基距离

## 距离测量的作用

距离度量在机器学习中起着重要的作用。

距离度量是总结问题域中两个对象之间相对差异的客观分数。

最常见的是，这两个对象是描述主题(如人、车或房子)或事件(如购买、索赔或诊断)的数据行。

也许你最有可能遇到距离度量的方式是当你使用一个特定的机器学习算法时，这个算法的核心是距离度量。这类算法中最著名的是 [k 近邻算法](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)，简称 KNN。

在 KNN 算法中，通过计算新示例(行)和训练数据集中所有示例(行)之间的距离，对新示例进行分类或回归预测。然后选择训练数据集中具有最小距离的 k 个示例，并通过对结果求平均值(类别标签的模式或回归的真实值的平均值)来进行预测。

KNN 属于一个更广泛的算法领域，称为基于案例或基于实例的学习，其中大多数以类似的方式使用距离测量。另一种流行的基于实例的使用距离度量的算法是[学习矢量量化](https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/)，或 LVQ 算法，也可以被认为是一种神经网络。

与之相关的是自组织映射算法(SOM)，它也使用距离度量，可以用于有监督或无监督的学习。另一种以距离度量为核心的无监督学习算法是 K 均值聚类算法。

> 在基于实例的学习中，训练示例被逐字存储，并且使用距离函数来确定训练集的哪个成员最接近未知的测试实例。一旦找到最近的训练实例，就为测试实例预测它的类。

—第 135 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/383LSNQ)，2016 年第 4 版。

下面是一些更流行的机器学习算法的简短列表，这些算法的核心是使用距离度量:

*   k 近邻
*   学习矢量量化(LVQ)
*   自组织映射
*   k-均值聚类

有许多基于核的方法也可以被认为是基于距离的算法。也许最广为人知的核方法是支持向量机算法，简称 SVM。

**你知道更多使用距离度量的算法吗？**
在下面的评论里告诉我。

在计算两个示例或两行数据之间的距离时，不同的示例列可能使用不同的数据类型。一个例子可能有实数值、布尔值、分类值和序数值。对于每个距离度量，可能需要不同的距离度量，这些距离度量被加在一起形成单个距离分数。

数值可能有不同的刻度。这可能会极大地影响距离度量的计算，在计算距离度量之前对数值进行标准化或规范化通常是一种很好的做法。

回归问题中的数值误差也可以视为距离。例如，预期值和预测值之间的误差是一维距离度量，可以对测试集中的所有示例进行求和或平均，以给出数据集中预期结果和预测结果之间的总距离。误差的计算，如均方误差或平均绝对误差，可能类似于标准的距离测量。

正如我们所看到的，距离度量在机器学习中起着重要的作用。也许机器学习中最常用的四种距离度量如下:

*   汉娩距
*   欧几里得距离
*   曼哈顿距离
*   闵可夫斯基距离

**你还用过或听说过哪些其他距离测量方法？**
在下面的评论里告诉我。

当从头开始实现算法时，您需要知道如何计算这些距离度量中的每一个，以及当使用利用这些距离度量的算法时，计算什么的直觉。

让我们依次仔细看看每一个。

## 汉娩距

[汉明距离](https://en.wikipedia.org/wiki/Hamming_distance)计算两个二进制向量之间的距离，也称为二进制串或简称位串。

当你对数据的分类列进行一次热编码时，你很可能会遇到位串。

例如，如果一个列有类别“*红色*”、“*绿色*”和“*蓝色*”，您可以将每个示例热编码为一个位串，每列一位。

*   红色= [1，0，0]
*   绿色= [0，1，0]
*   蓝色= [0，0，1]

红色和绿色之间的距离可以计算为两个位串之间的位差的总和或平均数。这是海明距离。

对于单热编码的字符串，总结为字符串之间的位差之和可能更有意义，它总是 0 或 1。

*   海明距离= I 至 N 绝对值之和(v1[I]–v2[I])

对于可能有许多 1 位的位串，更常见的是计算位差的平均数量，以给出 0(相同)和 1(都不同)之间的汉明距离分数。

*   海明距离=(I 至 N 绝对值之和(v1[I]–v2[I])/N

我们可以用一个计算两个位串之间汉明距离的例子来演示这一点，如下所示。

```py
# calculating hamming distance between bit strings

# calculate hamming distance
def hamming_distance(a, b):
	return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)

# define data
row1 = [0, 0, 0, 0, 0, 1]
row2 = [0, 0, 0, 0, 1, 0]
# calculate distance
dist = hamming_distance(row1, row2)
print(dist)
```

运行该示例会报告两个位串之间的汉明距离。

我们可以看到，字符串之间有两个差异，或者说 6 个比特位置中有 2 个不同，其平均值(2/6)约为 1/3 或 0.333。

```py
0.3333333333333333
```

我们也可以使用 SciPy 中的[汉明()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html)进行同样的计算。下面列出了完整的示例。

```py
# calculating hamming distance between bit strings
from scipy.spatial.distance import hamming
# define data
row1 = [0, 0, 0, 0, 0, 1]
row2 = [0, 0, 0, 0, 1, 0]
# calculate distance
dist = hamming(row1, row2)
print(dist)
```

运行这个例子，我们可以看到我们得到了相同的结果，证实了我们的手动实现。

```py
0.3333333333333333
```

## 欧几里得距离

[欧氏距离](https://en.wikipedia.org/wiki/Euclidean_distance)计算两个实值向量之间的距离。

在计算具有数值(如浮点或整数值)的两行数据之间的距离时，最有可能使用欧几里德距离。

如果列具有不同比例的值，通常在计算欧几里德距离之前对所有列的数值进行归一化或标准化。否则，具有较大值的列将主导距离度量。

> 虽然还有其他可能的选择，但大多数基于实例的学习者使用欧几里德距离。

—第 135 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/383LSNQ)，2016 年第 4 版。

欧几里得距离计算为两个向量之间的平方差之和的平方根。

*   欧几里德距离= sqrt(I 与 n 之和(v1[I]–v2[i])^2)

如果距离计算要执行数千次或数百万次，通常会移除平方根运算以加快计算速度。修改后的结果分数将具有相同的相对比例，并且仍然可以在机器学习算法中有效地用于寻找最相似的例子。

*   欧几里德距离= I 到 n 的和(v1[I]–v2[i])^2

该计算与 [L2 向量范数](https://machinelearningmastery.com/vector-norms-machine-learning/)相关，如果平方根相加，则相当于平方和误差和平方根误差。

我们可以用一个计算两个实值向量之间欧几里得距离的例子来演示这一点，如下所示。

```py
# calculating euclidean distance between vectors
from math import sqrt

# calculate euclidean distance
def euclidean_distance(a, b):
	return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))

# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = euclidean_distance(row1, row2)
print(dist)
```

运行该示例会报告两个向量之间的欧氏距离。

```py
6.082762530298219
```

我们也可以使用 SciPy 中的[欧几里德()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html)进行同样的计算。下面列出了完整的示例。

```py
# calculating euclidean distance between vectors
from scipy.spatial.distance import euclidean
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = euclidean(row1, row2)
print(dist)
```

运行这个例子，我们可以看到我们得到了相同的结果，证实了我们的手动实现。

```py
6.082762530298219
```

## 曼哈顿距离(出租车或城市街区距离)

[曼哈顿距离](https://en.wikipedia.org/wiki/Taxicab_geometry)，也称为出租车距离或城市街区距离，计算两个实值向量之间的距离。

它可能对描述统一网格上的对象的向量更有用，如棋盘或城市街区。该度量的出租车名称指的是该度量所计算的东西的直觉:出租车在城市街区之间的最短路径(网格上的坐标)。

对于整数特征空间中的两个向量，计算曼哈顿距离而不是欧几里德距离可能是有意义的。

曼哈顿距离计算为两个向量的绝对差之和。

*   曼哈坦度= I 与 N 之和| v1[I]–v2[I]|

曼哈顿距离与 [L1 向量范数](https://machinelearningmastery.com/vector-norms-machine-learning/)以及绝对误差和平均绝对误差度量有关。

我们可以用下面列出的计算两个整数向量之间曼哈顿距离的例子来证明这一点。

```py
# calculating manhattan distance between vectors
from math import sqrt

# calculate manhattan distance
def manhattan_distance(a, b):
	return sum(abs(e1-e2) for e1, e2 in zip(a,b))

# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = manhattan_distance(row1, row2)
print(dist)
```

运行该示例会报告两个向量之间的曼哈顿距离。

```py
13
```

我们也可以使用 SciPy 中的 [cityblock()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html)进行同样的计算。下面列出了完整的示例。

```py
# calculating manhattan distance between vectors
from scipy.spatial.distance import cityblock
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = cityblock(row1, row2)
print(dist)
```

运行这个例子，我们可以看到我们得到了相同的结果，证实了我们的手动实现。

```py
13
```

## 闵可夫斯基距离

[闵可夫斯基距离](https://en.wikipedia.org/wiki/Minkowski_distance)计算两个实值向量之间的距离。

它是欧几里德和曼哈顿距离度量的推广，并添加了一个参数，称为“*阶*或“ *p* ，允许计算不同的距离度量。

闵可夫斯基距离度量计算如下:

*   欧几里德距离=(I 与 n 之和(绝对值(v1[I])–v2[i]))^p)^(1/p)

其中“ *p* ”为订单参数。

当 p 设置为 1 时，计算与曼哈顿距离相同。当 p 设置为 2 时，它与欧几里得距离相同。

*   *p=1* :曼哈顿距离。
*   *p=2* :欧氏距离。

中间值提供了两种度量之间的受控平衡。

在实现使用距离度量的机器学习算法时，通常使用闵可夫斯基距离，因为它通过可调整的超参数“ *p* ”来控制用于实值向量的距离度量类型。

我们可以用一个计算两个实向量之间闵可夫斯基距离的例子来演示这个计算，如下所示。

```py
# calculating minkowski distance between vectors
from math import sqrt

# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance (p=1)
dist = minkowski_distance(row1, row2, 1)
print(dist)
# calculate distance (p=2)
dist = minkowski_distance(row1, row2, 2)
print(dist)
```

运行该示例首先计算并打印闵可夫斯基距离，将 *p* 设置为 1 以给出曼哈顿距离，然后将 *p* 设置为 2 以给出欧几里德距离，与根据前面部分的相同数据计算的值相匹配。

```py
13.0
6.082762530298219
```

我们也可以使用 SciPy 的 [minkowski_distance()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html)进行同样的计算。下面列出了完整的示例。

```py
# calculating minkowski distance between vectors
from scipy.spatial import minkowski_distance
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance (p=1)
dist = minkowski_distance(row1, row2, 1)
print(dist)
# calculate distance (p=2)
dist = minkowski_distance(row1, row2, 2)
print(dist)
```

运行这个例子，我们可以看到我们得到了相同的结果，证实了我们的手动实现。

```py
13.0
6.082762530298219
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/383LSNQ)，第 4 版，2016。

### 蜜蜂

*   [距离计算(空间距离)](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
*   [scipy . spatial . distance . hamming API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html)。
*   [scipy . spatial . distance . euclidean API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html)。
*   [scipy . spatial . distance . city block API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html)。
*   [scipy . spatial . Minkowski _ distance API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html)。

### 文章

*   [基于实例的学习，维基百科](https://en.wikipedia.org/wiki/Instance-based_learning)。
*   [汉明距离，维基百科](https://en.wikipedia.org/wiki/Hamming_distance)。
*   [欧氏距离，维基百科](https://en.wikipedia.org/wiki/Euclidean_distance)。
*   [出租车几何，维基百科](https://en.wikipedia.org/wiki/Taxicab_geometry)。
*   Minkowski 距离，维基百科。

## 摘要

在本教程中，您发现了机器学习中的距离度量。

具体来说，您了解到:

*   距离度量在机器学习算法中的作用和重要性。
*   如何实现和计算汉明、欧几里德和曼哈顿距离度量。
*   如何实现和计算推广欧几里德和曼哈顿距离度量的闵可夫斯基距离。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。