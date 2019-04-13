# 如何在 Python 中加载机器学习数据

> 原文： [https://machinelearningmastery.com/load-machine-learning-data-python/](https://machinelearningmastery.com/load-machine-learning-data-python/)

您必须能够在启动机器学习项目之前加载数据。

机器学习数据最常见的格式是 CSV 文件。有许多方法可以在 Python 中加载 CSV 文件。

在这篇文章中，您将发现可以用来在 Python 中加载机器学习数据的不同方法。

让我们开始吧。

*   **2017 年 3 月更新**：更改从二进制（'rb'）到 ASCII（'rt）的加载。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。
*   **更新 March / 2018** ：更新了来自 URL 示例的 NumPy 加载，以便与 Python 3 一起工作。

![How To Load Machine Learning Data in Python](img/49594b43959868dd764277469c14f5de.jpg)

如何在 Python 中加载机器学习数据
照片由 [Ann Larie Valentine](https://www.flickr.com/photos/sanfranannie/2905016974/) ，保留一些权利。

## 加载 CSV 数据时的注意事项

从 CSV 文件加载机器学习数据时需要考虑许多因素。

作为参考，您可以通过查看标题为[通用格式和逗号分隔值（CSV）文件的 MIME 类型](https://tools.ietf.org/html/rfc4180)的评论的 CSV 请求，了解有关 CSV 文件期望的大量信息。

### CSV 文件标题

您的数据是否有文件头？

如果是这样，这可以帮助自动为每列数据分配名称。如果没有，您可能需要手动命名属性。

无论哪种方式，您都应明确指定 CSV 文件在加载数据时是否具有文件头。

### 评论

您的数据有评论吗？

CSV 文件中的注释在行的开头用散列（“＃”）表示。

如果您的文件中有注释，则根据用于加载数据的方法，您可能需要指明是否期望注释以及期望表示注释行的字符。

### 分隔符

用于分隔字段中值的标准分隔符是逗号（“，”）字符。

您的文件可以使用不同的分隔符，如 tab（“\ t”），在这种情况下，您必须明确指定它。

### 行情

有时字段值可以包含空格。在这些 CSV 文件中，通常会引用值。

默认引号字符是双引号“\”“。可以使用其他字符，您必须指定文件中使用的引号字符。

## 机器学习数据加载秘籍

每个秘籍都是独立的。

这意味着您可以将其复制并粘贴到项目中并立即使用。

如果您对这些秘籍或建议的改进有任何疑问，请发表评论，我会尽力回答。

### 使用 Python 标准库加载 CSV

Python API 提供模块 _CSV_ 和函数 _reader（）_，可用于加载 CSV 文件。

加载后，将 CSV 数据转换为 NumPy 数组并将其用于机器学习。

例如，您可以将 [Pima Indians 数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)下载到您的本地目录（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。所有字段都是数字，没有标题行。运行下面的秘籍将加载 CSV 文件并将其转换为 NumPy 数组。

```
# Load CSV (using python)
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)
```

该示例加载一个对象，该对象可以遍历数据的每一行，并且可以轻松转换为 NumPy 数组。运行该示例将打印数组的形状。

```
(768, 9)
```

有关 _csv.reader（）_ 函数的更多信息，请参阅 Python API 文档中的 [CSV 文件读取和写入](https://docs.python.org/2/library/csv.html)。

### 使用 NumPy 加载 CSV 文件

您可以使用 NumPy 和 _numpy.loadtxt（）_ 功能加载 CSV 数据。

此函数假定没有标题行，并且所有数据都具有相同的格式。下面的示例假定文件 _pima-indians-diabetes.data.csv_ 位于您当前的工作目录中。

```
# Load CSV
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)
```

运行该示例将加载文件为 [numpy.ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.html) 并打印数据的形状：

```
(768, 9)
```

可以修改此示例以直接从 URL 加载相同的数据集，如下所示：

**注意**：此示例假设您使用的是 Python 3。

```
# Load CSV from URL using NumPy
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)
```

同样，运行该示例会产生相同的结果形状。

```
(768, 9)
```

有关 [numpy.loadtxt（）](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html)函数的更多信息，请参阅 API 文档（numpy 版本 1.10）。

### 使用 Pandas 加载 CSV 文件

您可以使用 Pandas 和 _pandas.read_csv（）_ 功能加载 CSV 数据。

此功能非常灵活，可能是我推荐的加载机器学习数据的方法。该函数返回一个 [pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) ，您可以立即开始汇总和绘图。

以下示例假定' _pima-indians-diabetes.data.csv_ '文件位于当前工作目录中。

```
# Load CSV using Pandas
import pandas
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(filename, names=names)
print(data.shape)
```

请注意，在此示例中，我们明确指定 DataFrame 的每个属性的名称。运行该示例显示数据的形状：

```
(768, 9)
```

我们还可以修改此示例以直接从 URL 加载 CSV 数据。

```
# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)
```

再次，运行该示例下载 CSV 文件，解析它并显示加载的 DataFrame 的形状。

```
(768, 9)
```

要了解有关 [pandas.read_csv（）](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)功能的更多信息，请参阅 API 文档。

## 摘要

在这篇文章中，您了解了如何使用 Python 加载机器学习数据。

您学习了三种可以使用的特定技术：

*   使用 Python 标准库加载 CSV。
*   使用 NumPy 加载 CSV 文件。
*   使用 Pandas 加载 CSV 文件。

此帖子的操作步骤是键入或复制并粘贴每个秘籍，并熟悉可以在 Python 中加载机器学习数据的不同方法。

您是否有任何关于在 Python 或此帖中加载机器学习数据的问题？在评论中提出您的问题，我会尽力回答。