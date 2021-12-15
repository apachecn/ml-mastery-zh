# 如何使用 Python 处理缺失数据

> 原文： [https://machinelearningmastery.com/handle-missing-data-python/](https://machinelearningmastery.com/handle-missing-data-python/)

实际数据通常缺少值。

数据可能由于多种原因而丢失，例如未记录的观察和数据损坏。

处理缺失数据非常重要，因为许多机器学习算法不支持缺少值的数据。

在本教程中，您将了解如何使用 Python 处理机器学习的缺失数据。

具体来说，完成本教程后，您将了解：

*   如何在数据集中将无效或损坏的值标记为缺失。
*   如何从数据集中删除缺少数据的行。
*   如何使用数据集中的平均值来估算缺失值。

让我们开始吧。

**注意**：本文中的示例假设您安装了 Pandas，NumPy 和 Scikit-Learn 的 Python 2 或 3，特别是 scikit-learn 版本 0.18 或更高版本。

*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Handle Missing Values with Python](img/a2f5701b368353e16380aa356479861d.jpg)

如何使用 Python 处理缺失的值
照片由 [CoCreatr](https://www.flickr.com/photos/cocreatr/2265030196/) ，保留一些权利。

## 概观

本教程分为 6 个部分：

1.  **皮马印第安人糖尿病数据集**：我们查看已知缺失值的数据集。
2.  **标记缺失值**：我们学习如何在数据集中标记缺失值。
3.  **缺少值导致问题**：我们看到机器学习算法在包含缺失值时如何失败。
4.  **删除缺少值的行**：我们在哪里看到如何删除包含缺失值的行。
5.  **Impute Missing Values** ：我们用合理的值替换缺失值。
6.  **支持缺失值的算法**：我们了解支持缺失值的算法。

首先，让我们看一下缺少值的样本数据集。

## 1.皮马印第安人糖尿病数据集

[皮马印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)涉及在 Pima 印第安人中根据医疗细节预测 5 年内糖尿病的发病。

这是一个二元（2 级）分类问题。每个班级的观察数量不均衡。有 768 个观测值，有 8 个输入变量和 1 个输出变量。变量名称如下：

*   0.怀孕的次数。
*   1.口服葡萄糖耐量试验中血浆葡萄糖浓度为 2 小时。
*   2.舒张压（mm Hg）。
*   3.三头肌皮褶厚度（mm）。
*   4\. 2 小时血清胰岛素（μU/ ml）。
*   5.体重指数（体重 kg /（身高 m）^ 2）。
*   6.糖尿病谱系功能。
*   7.年龄（年）。
*   8.类变量（0 或 1）。

预测最普遍类别的基线表现是大约 65％的分类准确度。最佳结果实现了大约 77％的分类准确度。

下面列出了前 5 行的样本。

```
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
```

已知此数据集具有缺失值。

具体而言，对于标记为零值的某些列，缺少观察结果。

我们可以通过这些列的定义以及零值对这些度量无效的领域知识来证实这一点，例如：体重指数或血压为零无效。

[从这里](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)下载数据集并将其保存到当前工作目录，文件名为 _pima-indians-diabetes.csv_ （更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

## 2.标记缺失值

在本节中，我们将了解如何识别和标记缺失的值。

我们可以使用绘图和摘要统计量来帮助识别丢失或损坏的数据。

我们可以将数据集作为 Pandas DataFrame 加载，并打印每个属性的摘要统计量。

```
from pandas import read_csv
dataset = read_csv('pima-indians-diabetes.csv', header=None)
print(dataset.describe())
```

运行此示例将生成以下输出：

```
                0           1           2           3           4           5  \
count  768.000000  768.000000  768.000000  768.000000  768.000000  768.000000
mean     3.845052  120.894531   69.105469   20.536458   79.799479   31.992578
std      3.369578   31.972618   19.355807   15.952218  115.244002    7.884160
min      0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
25%      1.000000   99.000000   62.000000    0.000000    0.000000   27.300000
50%      3.000000  117.000000   72.000000   23.000000   30.500000   32.000000
75%      6.000000  140.250000   80.000000   32.000000  127.250000   36.600000
max     17.000000  199.000000  122.000000   99.000000  846.000000   67.100000

                6           7           8
count  768.000000  768.000000  768.000000
mean     0.471876   33.240885    0.348958
std      0.331329   11.760232    0.476951
min      0.078000   21.000000    0.000000
25%      0.243750   24.000000    0.000000
50%      0.372500   29.000000    0.000000
75%      0.626250   41.000000    1.000000
max      2.420000   81.000000    1.000000
```

这很有用。

我们可以看到有些列的最小值为零（0）。在某些列上，值为零没有意义，表示值无效或缺失。

具体而言，以下列具有无效的零最小值：

*   1：血浆葡萄糖浓度
*   2：舒张压
*   3：肱三头肌皮褶厚度
*   4：2 小时血清胰岛素
*   5：体重指数

让'确认这是我查看原始数据，该示例打印前 20 行数据。

```
from pandas import read_csv
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# print the first 20 rows of data
print(dataset.head(20))
```

运行该示例，我们可以清楚地看到第 2,3,4 和 5 列中的 0 值。

```
     0    1   2   3    4     5      6   7  8
0    6  148  72  35    0  33.6  0.627  50  1
1    1   85  66  29    0  26.6  0.351  31  0
2    8  183  64   0    0  23.3  0.672  32  1
3    1   89  66  23   94  28.1  0.167  21  0
4    0  137  40  35  168  43.1  2.288  33  1
5    5  116  74   0    0  25.6  0.201  30  0
6    3   78  50  32   88  31.0  0.248  26  1
7   10  115   0   0    0  35.3  0.134  29  0
8    2  197  70  45  543  30.5  0.158  53  1
9    8  125  96   0    0   0.0  0.232  54  1
10   4  110  92   0    0  37.6  0.191  30  0
11  10  168  74   0    0  38.0  0.537  34  1
12  10  139  80   0    0  27.1  1.441  57  0
13   1  189  60  23  846  30.1  0.398  59  1
14   5  166  72  19  175  25.8  0.587  51  1
15   7  100   0   0    0  30.0  0.484  32  1
16   0  118  84  47  230  45.8  0.551  31  1
17   7  107  74   0    0  29.6  0.254  31  1
18   1  103  30  38   83  43.3  0.183  33  0
19   1  115  70  30   96  34.6  0.529  32  1
```

我们可以计算每列上缺失值的数量。我们可以这样做，标记我们感兴趣的 DataFrame 子集中的所有值，其值为零。然后我们可以计算每列中的真值数。

我们可以这样做，标记我们感兴趣的 DataFrame 子集中的所有值，其值为零。然后我们可以计算每列中的真值数。

```
from pandas import read_csv
dataset = read_csv('pima-indians-diabetes.csv', header=None)
print((dataset[[1,2,3,4,5]] == 0).sum())
```

运行该示例将输出以下输出：

```
1 5
2 35
3 227
4 374
5 11
```

我们可以看到第 1,2 和 5 列只有几个零值，而第 3 列和第 4 列显示了更多，接近一半的行。

这突出了不同列可能需要不同的“缺失值”策略，例如确保仍有足够数量的记录来训练预测模型。

在 Python 中，特别是 Pandas，NumPy 和 Scikit-Learn，我们将缺失值标记为 NaN。

具有 NaN 值的值将在 sum，count 等操作中被忽略。

我们可以通过在我们感兴趣的列子集上使用 [replace（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html)，使用 Pandas DataFrame 轻松地将值标记为 NaN。

在我们标记缺失值之后，我们可以使用 [isnull（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html)将数据集中的所有 NaN 值标记为 True，并获取每列的缺失值计数。

```
from pandas import read_csv
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# count the number of NaN values in each column
print(dataset.isnull().sum())
```

运行该示例将打印每列中缺失值的数量。我们可以看到列 1：5 具有与上面标识的零值相同数量的缺失值。这表明我们已正确标记了已识别的缺失值。

我们可以看到列 1 到 5 具有与上面标识的零值相同数量的缺失值。这表明我们已正确标记了已识别的缺失值。

```
0      0
1      5
2     35
3    227
4    374
5     11
6      0
7      0
8      0
```

这是一个有用的总结。我总是喜欢看实际的数据，以确认我没有欺骗自己。

下面是相同的例子，除了我们打印前 20 行数据。

```
from pandas import read_csv
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# print the first 20 rows of data
print(dataset.head(20))
```

运行该示例，我们可以清楚地看到第 2,3,4 和 5 列中的 NaN 值。第 1 列中只有 5 个缺失值，因此我们在前 20 行中没有看到示例也就不足为奇了。

从原始数据可以清楚地看出，标记缺失值具有预期效果。

```
     0      1     2     3      4     5      6   7  8
0    6  148.0  72.0  35.0    NaN  33.6  0.627  50  1
1    1   85.0  66.0  29.0    NaN  26.6  0.351  31  0
2    8  183.0  64.0   NaN    NaN  23.3  0.672  32  1
3    1   89.0  66.0  23.0   94.0  28.1  0.167  21  0
4    0  137.0  40.0  35.0  168.0  43.1  2.288  33  1
5    5  116.0  74.0   NaN    NaN  25.6  0.201  30  0
6    3   78.0  50.0  32.0   88.0  31.0  0.248  26  1
7   10  115.0   NaN   NaN    NaN  35.3  0.134  29  0
8    2  197.0  70.0  45.0  543.0  30.5  0.158  53  1
9    8  125.0  96.0   NaN    NaN   NaN  0.232  54  1
10   4  110.0  92.0   NaN    NaN  37.6  0.191  30  0
11  10  168.0  74.0   NaN    NaN  38.0  0.537  34  1
12  10  139.0  80.0   NaN    NaN  27.1  1.441  57  0
13   1  189.0  60.0  23.0  846.0  30.1  0.398  59  1
14   5  166.0  72.0  19.0  175.0  25.8  0.587  51  1
15   7  100.0   NaN   NaN    NaN  30.0  0.484  32  1
16   0  118.0  84.0  47.0  230.0  45.8  0.551  31  1
17   7  107.0  74.0   NaN    NaN  29.6  0.254  31  1
18   1  103.0  30.0  38.0   83.0  43.3  0.183  33  0
19   1  115.0  70.0  30.0   96.0  34.6  0.529  32  1
```

在我们处理缺失值之前，让我们首先证明在数据集中缺少值可能会导致问题。

## 3.缺少值会导致问题

在数据集中缺少值可能会导致某些机器学习算法出错。

在本节中，我们将尝试在具有缺失值的数据集上评估线性判别分析（LDA）算法。

这是一种在数据集中缺少值时不起作用的算法。

下面的示例标记数据集中的缺失值，就像我们在上一节中所做的那样，然后尝试使用 3 倍交叉验证来评估 LDA 并打印平均精度。

```
from pandas import read_csv
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(result.mean())
```

运行该示例会导致错误，如下所示：

```
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
```

这是我们所期望的。

我们无法在具有缺失值的数据集上评估 LDA 算法（和其他算法）。

现在，我们可以查看处理缺失值的方法。

## 4.删除缺少值的行

处理缺失数据的最简单策略是删除包含缺失值的记录。

我们可以通过创建一个包含缺失值的行的新 Pandas DataFrame 来完成此操作。

Pandas 提供 [dropna（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)，可用于删除缺少数据的列或行。我们可以使用 dropna（）删除缺少数据的所有行，如下所示：

```
from pandas import read_csv
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# drop rows with missing values
dataset.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(dataset.shape)
```

运行此示例，我们可以看到行数已经从原始数据集中的 768 积极地减少到 392，并且所有行都包含 NaN。

```
(392, 9)
```

我们现在有一个数据集可以用来评估对缺失值敏感的算法，比如 LDA。

```
from pandas import read_csv
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# drop rows with missing values
dataset.dropna(inplace=True)
# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(result.mean())
```

该示例成功运行并打印模型的准确率。

```
0.78582892934
```

删除缺少值的行可能会对某些预测性建模问题造成太大限制，另一种方法是估算缺失值。

## 5.估算缺失值

输入是指使用模型替换缺失值。

替换缺失值时我们可以考虑许多选项，例如：

*   在域中具有意义的常量值，例如 0，与所有其他值不同。
*   来自另一个随机选择的记录的值。
*   列的平均值，中值或模式值。
*   由另一个预测模型估计的值。

当需要从最终模型做出预测时，将来必须对新数据执行对训练数据集执行的任何估算。在选择如何估算缺失值时需要考虑这一点。

例如，如果您选择使用平均列值进行估算，则需要将这些平均列值存储到文件中，以便以后用于具有缺失值的新数据。

Pandas 提供 [fillna（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)，用于替换具有特定值的缺失值。

例如，我们可以使用 fillna（）将缺失值替换为每列的平均值，如下所示：

```
from pandas import read_csv
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column
print(dataset.isnull().sum())
```

运行该示例提供了每列中缺失值数量的计数，显示零缺失值。

```
0    0
1    0
2    0
3    0
4    0
5    0
6    0
7    0
8    0
```

scikit-learn 库提供 [Imputer（）预处理类](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)，可用于替换缺失值。

它是一个灵活的类，允许您指定要替换的值（它可以是除 NaN 之外的其他东西）和用于替换它的技术（例如均值，中位数或模式）。 Imputer 类直接在 NumPy 数组而不是 DataFrame 上运行。

下面的示例使用 Imputer 类将缺失值替换为每列的平均值，然后在转换后的矩阵中打印 NaN 值的数量。

```
from pandas import read_csv
from sklearn.preprocessing import Imputer
import numpy
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
# count the number of NaN values in each column
print(numpy.isnan(transformed_values).sum())
```

运行该示例显示所有 NaN 值都已成功估算。

在任何一种情况下，我们都可以训练对变换数据集中的 NaN 值敏感的算法，例如 LDA。

下面的示例显示了在 Imputer 转换数据集中训练的 LDA 算法。

```
from pandas import read_csv
import numpy
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# fill missing values with mean column values
imputer = Imputer()
transformed_X = imputer.fit_transform(X)
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, transformed_X, y, cv=kfold, scoring='accuracy')
print(result.mean())
```

运行该示例将在已转换的数据集上打印 LDA 的准确率。

```
0.766927083333
```

尝试用其他值替换缺失的值，看看是否可以提升模型的表现。

也许缺失值在数据中有意义。

接下来，我们将研究使用在建模时将缺失值视为另一个值的算法。

## 6.支持缺失值的算法

当缺少数据时，并非所有算法都会失败。

有些算法可以对缺失数据进行鲁棒处理，例如 K 最近邻 可以在缺少值时忽略距离测量中的列。

还有一些算法可以在构建预测模型时使用缺失值作为唯一且不同的值，例如分类和回归树。

遗憾的是，决策树和 K 最近邻 的 scikit-learn 实现对于缺失值并不健壮。 [虽然正在考虑](https://github.com/scikit-learn/scikit-learn/issues/5870)。

然而，如果您考虑使用其他算法实现（例如 [xgboost](http://machinelearningmastery.com/xgboost-python-mini-course/) ）或开发自己的实现，这仍然是一个选项。

## 进一步阅读

*   [在 Pandas](http://pandas.pydata.org/pandas-docs/stable/missing_data.html) 中处理缺失数据
*   [在 scikit-learn](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values) 中对缺失值的估算

## 摘要

在本教程中，您了解了如何处理包含缺失值的机器学习数据。

具体来说，你学到了：

*   如何将数据集中的缺失值标记为 numpy.nan。
*   如何从数据集中删除包含缺失值的行。
*   如何用合理的值替换缺失值。

您对处理缺失值有任何疑问吗？
在评论中提出您的问题，我会尽力回答。