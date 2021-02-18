# 使用 Python 中的描述性统计来了解您的机器学习数据

> 原文： [https://machinelearningmastery.com/understand-machine-learning-data-descriptive-statistics-python/](https://machinelearningmastery.com/understand-machine-learning-data-descriptive-statistics-python/)

您必须了解您的数据才能获得最佳效果。

在这篇文章中，您将发现可以在 Python 中使用的 7 个秘籍，以了解有关机器学习数据的更多信息。

让我们开始吧。

*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Understand Your Machine Learning Data With Descriptive Statistics in Python](img/9beeec602e85374fd151655047d4511b.jpg)

使用 Python 中的描述性统计理解您的机器学习数据
[过路人](https://www.flickr.com/photos/passer-by/75060078/)的照片，保留一些权利。

## Python Recipes 了解您的机器学习数据

本节列出了 7 种可用于更好地了解机器学习数据的秘籍。

通过加载来自 UCI 机器学习库的 [Pima 印第安人糖尿病](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)分类数据集来证明每个秘籍（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

打开你的 python 交互式环境，依次尝试每个秘籍。

### 1.窥视您的数据

查看原始数据是无可替代的。

查看原始数据可以揭示您无法通过任何其他方式获得的见解。它还可以种植种子，这些种子可能会成为如何更好地预处理和处理机器学习任务数据的想法。

您可以使用 Pandas DataFrame 上的`head()`函数查看数据的前 20 行。

```
# View first 20 rows
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
peek = data.head(20)
print(peek)
```

您可以看到第一列列出了行号，这对于引用特定观察非常方便。

```
    preg  plas  pres  skin  test  mass   pedi  age  class
0      6   148    72    35     0  33.6  0.627   50      1
1      1    85    66    29     0  26.6  0.351   31      0
2      8   183    64     0     0  23.3  0.672   32      1
3      1    89    66    23    94  28.1  0.167   21      0
4      0   137    40    35   168  43.1  2.288   33      1
5      5   116    74     0     0  25.6  0.201   30      0
6      3    78    50    32    88  31.0  0.248   26      1
7     10   115     0     0     0  35.3  0.134   29      0
8      2   197    70    45   543  30.5  0.158   53      1
9      8   125    96     0     0   0.0  0.232   54      1
10     4   110    92     0     0  37.6  0.191   30      0
11    10   168    74     0     0  38.0  0.537   34      1
12    10   139    80     0     0  27.1  1.441   57      0
13     1   189    60    23   846  30.1  0.398   59      1
14     5   166    72    19   175  25.8  0.587   51      1
15     7   100     0     0     0  30.0  0.484   32      1
16     0   118    84    47   230  45.8  0.551   31      1
17     7   107    74     0     0  29.6  0.254   31      1
18     1   103    30    38    83  43.3  0.183   33      0
19     1   115    70    30    96  34.6  0.529   32      1
```

### 2.数据的大小

您必须非常好地处理您拥有的数据量，无论是行数还是列数。

*   太多行和算法可能需要很长时间才能进行训练。太少，也许你没有足够的数据来训练算法。
*   由于维数的诅咒，太多的功能和一些算法可能会分散注意力或表现不佳。

您可以通过在 Pandas DataFrame 上打印 shape 属性来查看数据集的形状和大小。

```
# Dimensions of your data
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
shape = data.shape
print(shape)
```

结果列在行然后列中。您可以看到数据集有 768 行和 9 列。

```
(768, 9)
```

### 3.每个属性的数据类型

每个属性的类型很重要。

字符串可能需要转换为浮点值或整数以表示分类或序数值。

如上所述，您可以通过查看原始数据来了解属性的类型。您还可以使用`dtypes`属性列出 DataFrame 用于表征每个属性的数据类型。

```
# Data Types for Each Attribute
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
types = data.dtypes
print(types)
```

您可以看到大多数属性都是整数，而 mass 和 pedi 是浮点值。

```
preg       int64
plas       int64
pres       int64
skin       int64
test       int64
mass     float64
pedi     float64
age        int64
class      int64
dtype: object
```

### 4.描述性统计

描述性统计可以让您深入了解每个属性的形状。

通常，您可以创建比您有时间审核的摘要更多的摘要。 Pandas DataFrame 上的`describe()`函数列出了每个属性的 8 个统计属性：

*   计数
*   意思
*   标准偏差
*   最低价值
*   25％百分位数
*   第 50 百分位数（中位数）
*   第 75 百分位
*   最大价值

```
# Statistical Summary
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
description = data.describe()
print(description)
```

您可以看到您确实获得了大量数据。您将注意到秘籍中对`pandas.set_option()`的一些调用，以更改数字的精度和输出的首选宽度。这是为了使这个例子更具可读性。

以这种方式描述您的数据时，值得花一些时间并从结果中查看观察结果。这可能包括缺少数据的“`NA`”值或属性的令人惊讶的分布。

```
          preg     plas     pres     skin     test     mass     pedi      age    class
count  768.000  768.000  768.000  768.000  768.000  768.000  768.000  768.000  768.000
mean     3.845  120.895   69.105   20.536   79.799   31.993    0.472   33.241    0.349
std      3.370   31.973   19.356   15.952  115.244    7.884    0.331   11.760    0.477
min      0.000    0.000    0.000    0.000    0.000    0.000    0.078   21.000    0.000
25%      1.000   99.000   62.000    0.000    0.000   27.300    0.244   24.000    0.000
50%      3.000  117.000   72.000   23.000   30.500   32.000    0.372   29.000    0.000
75%      6.000  140.250   80.000   32.000  127.250   36.600    0.626   41.000    1.000
max     17.000  199.000  122.000   99.000  846.000   67.100    2.420   81.000    1.000
```

### 5.类别分布（仅限分类）

在分类问题上，您需要知道类值的平衡程度。

高度不平衡的问题（对于一个类别的观察比另一个类别更多）是常见的，并且可能需要在项目的数据准备阶段进行特殊处理。

您可以快速了解 Pandas 中 class 属性的分布。

```
# Class Distribution
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
class_counts = data.groupby('class').size()
print(class_counts)
```

您可以看到，与 0 级（糖尿病发病）相比，0 级（无糖尿病发作）的观察数量几乎翻了一番。

```
class
0    500
1    268
```

### 6.属性之间的相关性

相关性是指两个变量之间的关系以及它们如何一起变化或不变化。

计算相关性的最常用方法是 [Pearson 相关系数](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)，它假定所涉及属性的正态分布。 -1 或 1 的相关性分别显示完全的负相关或正相关。值 0 表示根本没有相关性。

如果数据集中存在高度相关的属性，则某些机器学习算法（如线性和逻辑回归）可能会遇到表现较差的情况。因此，最好检查数据集中属性的所有成对关联。您可以使用 Pandas DataFrame 上的`corr()`函数来计算相关矩阵。

```
# Pairwise Pearson correlations
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)
```

矩阵列出了顶部和侧面的所有属性，以给出所有属性对之间的相关性（两次，因为矩阵是对称的）。您可以看到矩阵左上角到右下角的矩阵对角线显示每个属性与自身的完美关联。

```
        preg   plas   pres   skin   test   mass   pedi    age  class
preg   1.000  0.129  0.141 -0.082 -0.074  0.018 -0.034  0.544  0.222
plas   0.129  1.000  0.153  0.057  0.331  0.221  0.137  0.264  0.467
pres   0.141  0.153  1.000  0.207  0.089  0.282  0.041  0.240  0.065
skin  -0.082  0.057  0.207  1.000  0.437  0.393  0.184 -0.114  0.075
test  -0.074  0.331  0.089  0.437  1.000  0.198  0.185 -0.042  0.131
mass   0.018  0.221  0.282  0.393  0.198  1.000  0.141  0.036  0.293
pedi  -0.034  0.137  0.041  0.184  0.185  0.141  1.000  0.034  0.174
age    0.544  0.264  0.240 -0.114 -0.042  0.036  0.034  1.000  0.238
class  0.222  0.467  0.065  0.075  0.131  0.293  0.174  0.238  1.000
```

### 7.单变量分布的偏差

偏斜指的是假设高斯（正常曲线或钟形曲线）在一个方向或另一个方向上移位或压扁的分布。

许多机器学习算法假设高斯分布。知道属性有偏差可能允许您执行数据准备以纠正偏斜，然后提高模型的准确性。

您可以使用 Pandas DataFrame 上的`skew()`函数计算每个属性的偏斜。

```
# Skew for each attribute
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
skew = data.skew()
print(skew)
```

偏斜结果显示正（右）或负（左）偏斜。接近零的值显示较少的偏斜。

```
preg     0.901674
plas     0.173754
pres    -1.843608
skin     0.109372
test     2.272251
mass    -0.428982
pedi     1.919911
age      1.129597
class    0.635017
```

## 更多秘籍

这只是一些最有用的摘要和描述性统计数据的选择，您可以在机器学习数据上使用这些摘要和描述性统计数据进行分类和回归。

您还可以计算许多其他统计数据。

当您开始处理新数据集时，是否有您想要计算和审核的特定统计信息？发表评论并告诉我。

## 要记住的提示

本部分为您提供了在使用摘要统计信息查看数据时要记住的一些提示。

*   **查看数字**。生成摘要统计数据是不够的。花点时间暂停，阅读并真正考虑您所看到的数字。
*   **问为什么**。检查您的号码并提出很多问题。你是如何以及为什么看到具体的数字。考虑数字如何与一般的问题域以及观察所涉及的特定实体相关。
*   **写下想法**。写下你的观察和想法。保留一个小的文本文件或记事本，记下关于变量如何相关的所有想法，数字的含义以及稍后尝试的技巧的想法。当你想要尝试新事物时，你现在在数据新鲜时记下的东西将非常有价值。

## 摘要

在这篇文章中，您发现了在开始处理机器学习项目之前描述数据集的重要性。

您发现了使用 Python 和 Pandas 汇总数据集的 7 种不同方法：

1.  窥视你的数据
2.  您的数据的维度
3.  数据类型
4.  类分布
5.  数据摘要
6.  相关性
7.  偏态

### 行动步骤

1.  打开 Python 交互式环境。
2.  输入或复制并粘贴每个秘籍，看看它是如何工作的。
3.  让我知道你如何评论。

您对此帖子中的 Python，Pandas 或秘籍有任何疑问吗？发表评论并提出问题，我会尽力回答。