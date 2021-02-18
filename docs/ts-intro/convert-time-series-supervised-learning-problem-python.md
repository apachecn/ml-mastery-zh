# 如何将时间序列转换为 Python 中的监督学习问题

> 原文： [https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

深度学习等机器学习方法可用于时间序列预测。

在可以使用机器学习之前，必须将时间序列预测问题重新定义为监督学习问题。从序列到输入和输出序列对。

在本教程中，您将了解如何将单变量和多变量时间序列预测问题转换为监督学习问题，以便与机器学习算法一起使用。

完成本教程后，您将了解：

*   如何开发将时间序列数据集转换为监督学习数据集的函数。
*   如何转换用于机器学习的单变量时间序列数据。
*   如何转换多变量时间序列数据用于机器学习。

让我们开始吧。

![How to Convert a Time Series to a Supervised Learning Problem in Python](img/cfb725b8225c2cab2204a2e6c3b0b3a6.jpg)

如何将时间序列转换为 Python 中的监督学习问题
照片由 [Quim Gil](https://www.flickr.com/photos/quimgil/8490510169/) ，保留一些权利。

## 时间序列与监督学习

在我们开始之前，让我们花一点时间来更好地理解时间序列和监督学习数据的形式。

时间序列是按时间索引排序的数字序列。这可以被认为是有序值的列表或列。

例如：

```py
0
1
2
3
4
5
6
7
8
9
```

监督学习问题由输入模式（`X`）和输出模式（`y`）组成，使得算法可以学习如何从输入模式预测输出模式。

例如：

```py
X,	y
1	2
2,	3
3,	4
4,	5
5,	6
6,	7
7,	8
8,	9
```

有关此主题的更多信息，请参阅帖子：

*   [时间序列预测作为监督学习](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

## Pandas shift（）函数

帮助将时间序列数据转换为监督学习问题的关键功能是 Pandas [shift（）](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)功能。

给定一个 DataFrame， _shift（）_ 函数可用于创建向前推送的列的副本（添加到前面的 NaN 值行）或拉回（添加到末尾的 NaN 值行） 。

这是在监督学习格式中创建滞后观察列以及时间序列数据集的预测观察列所需的行为。

让我们看一下移位函数的一些例子。

我们可以将模拟时间序列数据集定义为 10 个数字的序列，在本例中是 DataFrame 中的单个列，如下所示：

```py
from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
print(df)
```

运行该示例使用每个观察的行索引打印时间序列数据。

```py
   t
0  0
1  1
2  2
3  3
4  4
5  5
6  6
7  7
8  8
9  9
```

我们可以通过在顶部插入一个新行将所有观察结果向下移动一步。因为新行没有数据，我们可以使用 NaN 来表示“无数据”。

shift 函数可以为我们执行此操作，我们可以在原始系列旁边插入此移位列。

```py
from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
df['t-1'] = df['t'].shift(1)
print(df)
```

运行该示例为我们提供了数据集中的两列。第一个是原始观察和一个新的移动列。

我们可以看到，将系列向前移动一个步骤给了我们一个原始的监督学习问题，尽管`X`和`y`的顺序错误。忽略行标签列。由于 NaN 值，必须丢弃第一行。第二行显示第二列中的输入值 0.0（输入或`X`）和第一列中的值 1（输出或`y`）。

```py
   t  t-1
0  0  NaN
1  1  0.0
2  2  1.0
3  3  2.0
4  4  3.0
5  5  4.0
6  6  5.0
7  7  6.0
8  8  7.0
9  9  8.0
```

我们可以看到，如果我们可以用 2,3 和更多的移位重复这个过程，我们如何创建可用于预测输出值的长输入序列（`X`）（`y`）。

移位运算符也可以接受负整数值。这具有通过在末尾插入新行来提升观察结果的效果。以下是一个例子：

```py
from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
df['t+1'] = df['t'].shift(-1)
print(df)
```

运行该示例显示一个新列，其中 NaN 值为最后一个值。

我们可以看到预测列可以作为输入（`X`），第二个作为输出值（`y`）。即 0 的输入值可用于预测输出值 1。

```py
   t  t+1
0  0  1.0
1  1  2.0
2  2  3.0
3  3  4.0
4  4  5.0
5  5  6.0
6  6  7.0
7  7  8.0
8  8  9.0
9  9  NaN
```

从技术上讲，在时间序列预测术语中，当前时间（`t`）和未来时间（ _t + 1_ ， _t + n_ ）是预测时间和过去的观察结果（ _t-1_ ，`tn`）用于进行预测。

我们可以看到正向和负向移位如何用于从具有监督学习问题的输入和输出模式序列的时间序列创建新的 DataFrame。

这不仅允许经典 _X - &gt;_ 预测， _X - &gt; Y_ 其中输入和输出都可以是序列。

此外，移位功能也适用于所谓的多变量时间序列问题。在这里，我们有多个（例如温度和压力），而不是对时间序列进行一组观察。时间序列中的所有变量都可以向前或向后移动以创建多变量输入和输出序列。我们将在本教程的后面部分详细探讨。

## series_to_supervised（）函数

我们可以使用 Pandas 中的 _shift（）_ 函数在给定所需的输入和输出序列长度的情况下自动创建新的时间序列问题框架。

这将是一个有用的工具，因为它将允许我们使用机器学习算法探索时间序列问题的不同框架，以查看哪些可能导致更好的模型。

在本节中，我们将定义一个名为 _series_to_supervised（）_ 的新 Python 函数，该函数采用单变量或多变量时间序列并将其构建为监督学习数据集。

该函数有四个参数：

*   **数据**：作为列表或 2D NumPy 数组的观察序列。需要。
*   **n_in** ：作为输入的滞后观察数（`X`）。值可以在[1..len（数据）]之间可选。默认为 1。
*   **n_out** ：作为输出的观测数（`y`）。值可以在[0..len（数据）-1]之间。可选的。默认为 1。
*   **dropnan** ：布尔值是否删除具有 NaN 值的行。可选的。默认为 True。

该函数返回一个值：

*   **返回**：系列的 Pandas DataFrame 用于监督学习。

新数据集构造为 DataFrame，每列适当地由变量编号和时间步骤命名。这允许您根据给定的单变量或多变量时间序列设计各种不同的时间步长序列类型预测问题。

返回 DataFrame 后，您可以决定如何将返回的 DataFrame 的行拆分为`X`和`y`组件，以便以任何方式进行监督学习。

该函数是使用默认参数定义的，因此如果仅使用您的数据调用它，它将构建一个 DataFrame，其中 _t-1_ 为`X`和`t`为`y`。

该函数被确认与 Python 2 和 Python 3 兼容。

下面列出了完整的功能，包括功能注释。

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
```

你能看到明显的方法来使功能更强大或更具可读性吗？
请在下面的评论中告诉我。

现在我们已经拥有了整个功能，我们可以探索如何使用它。

## 一步式单变量预测

时间序列预测的标准做法是使用滞后观测值（例如 t-1）作为输入变量来预测当前时间步长（t）。

这称为一步预测。

下面的示例演示了预测当前时间步长（t）的一个滞后时间步长（t-1）。

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = [x for x in range(10)]
data = series_to_supervised(values)
print(data)
```

运行该示例将打印重新构建的时间序列的输出。

```py
   var1(t-1)  var1(t)
1        0.0        1
2        1.0        2
3        2.0        3
4        3.0        4
5        4.0        5
6        5.0        6
7        6.0        7
8        7.0        8
9        8.0        9
```

我们可以看到观察结果被命名为“`var1`”并且输入观察被恰当地命名为（t-1）并且输出时间步长被命名为（t）。

我们还可以看到具有 NaN 值的行已自动从 DataFrame 中删除。

我们可以用任意数字长度的输入序列重复这个例子，例如 3.这可以通过将输入序列的长度指定为参数来完成。例如：

```py
data = series_to_supervised(values, 3)
```

下面列出了完整的示例。

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = [x for x in range(10)]
data = series_to_supervised(values, 3)
print(data)
```

再次，运行该示例打印重构系列。我们可以看到输入序列是从正确的从左到右的顺序，输出变量在最右边预测。

```py
   var1(t-3)  var1(t-2)  var1(t-1)  var1(t)
3        0.0        1.0        2.0        3
4        1.0        2.0        3.0        4
5        2.0        3.0        4.0        5
6        3.0        4.0        5.0        6
7        4.0        5.0        6.0        7
8        5.0        6.0        7.0        8
9        6.0        7.0        8.0        9
```

## 多步或序列预测

不同类型的预测问题是使用过去的观测来预测未来观测的序列。

这可称为序列预测或多步预测。

我们可以通过指定另一个参数来构建序列预测的时间序列。例如，我们可以使用 2 个过去观测的输入序列构建预测问题，以预测 2 个未来观测如下：

```py
data = series_to_supervised(values, 2, 2)
```

完整示例如下：

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = [x for x in range(10)]
data = series_to_supervised(values, 2, 2)
print(data)
```

运行该示例显示输入（t-n）和输出（t + n）变量的区别，其中当前观察值（t）被视为输出。

```py
   var1(t-2)  var1(t-1)  var1(t)  var1(t+1)
2        0.0        1.0        2        3.0
3        1.0        2.0        3        4.0
4        2.0        3.0        4        5.0
5        3.0        4.0        5        6.0
6        4.0        5.0        6        7.0
7        5.0        6.0        7        8.0
8        6.0        7.0        8        9.0
```

## 多变量预测

另一种重要的时间序列称为多变量时间序列。

在这里，我们可能会观察到多种不同的测量方法，并且有兴趣预测其中的一种或多种。

例如，我们可能有两组时间序列观测值 obs1 和 obs2，我们希望预测其中一个或两个。

我们可以用完全相同的方式调用 _series_to_supervised（）_。

例如：

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values)
print(data)
```

运行该示例将打印数据的新框架，显示两个变量的一个时间步长的输入模式和两个变量的一个时间步的输出模式。

同样，根据问题的具体情况，可以任意选择列分为`X`和`Y`成分，例如，如果`var1`的当前观察也提供了作为输入，并且仅预测`var2`。

```py
   var1(t-1)  var2(t-1)  var1(t)  var2(t)
1        0.0       50.0        1       51
2        1.0       51.0        2       52
3        2.0       52.0        3       53
4        3.0       53.0        4       54
5        4.0       54.0        5       55
6        5.0       55.0        6       56
7        6.0       56.0        7       57
8        7.0       57.0        8       58
9        8.0       58.0        9       59
```

通过指定输入和输出序列的长度，您可以看到如何通过多变量时间序列轻松地将其用于序列预测。

例如，下面是一个重构的示例，其中 1 个时间步长作为输入，2 个时间步长作为预测序列。

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values, 1, 2)
print(data)
```

运行该示例显示了大型重构的 DataFrame。

```py
   var1(t-1)  var2(t-1)  var1(t)  var2(t)  var1(t+1)  var2(t+1)
1        0.0       50.0        1       51        2.0       52.0
2        1.0       51.0        2       52        3.0       53.0
3        2.0       52.0        3       53        4.0       54.0
4        3.0       53.0        4       54        5.0       55.0
5        4.0       54.0        5       55        6.0       56.0
6        5.0       55.0        6       56        7.0       57.0
7        6.0       56.0        7       57        8.0       58.0
8        7.0       57.0        8       58        9.0       59.0
```

试验您自己的数据集并尝试多种不同的框架，看看哪种方法效果最好。

## 摘要

在本教程中，您了解了如何将时间序列数据集重新定义为 Python 的监督学习问题。

具体来说，你学到了：

*   关于 Pandas _shift（）_ 功能以及它如何用于从时间序列数据中自动定义监督学习数据集。
*   如何将单变量时间序列重新组合成一步和多步监督学习问题。
*   如何将多变量时间序列重构为一步和多步监督学习问题。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。