# 使用 Python 进行时间序列预测的特征选择

> 原文： [https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/](https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/)

在时间序列数据上使用机器学习方法需要特征工程。

单变量时间序列数据集仅由一系列观察组成。这些必须转换为输入和输出功能，以便使用监督学习算法。

问题在于，您可以为时间序列问题设计的功能类型和数量几乎没有限制。像 correlogram 这样的经典时间序列分析工具可以帮助评估滞后变量，但在选择其他类型的特征时不会直接帮助，例如从时间戳（年，月或日）和移动统计量（如移动平均值）得出的特征。

在本教程中，您将了解在处理时间序列数据时如何使用功能重要性和功能选择的机器学习工具。

完成本教程后，您将了解：

*   如何创建和解释滞后观察的相关图。
*   如何计算和解释时间序列要素的要素重要性分数。
*   如何在时间序列输入变量上执行特征选择。

让我们开始吧。

## 教程概述

本教程分为以下 5 个步骤：

1.  **每月汽车销售数据集**：描述我们将要使用的数据集。
2.  **Make Stationary** ：描述如何使数据集静止以进行分析和预测。
3.  **自相关图**：描述如何创建时间序列数据的相关图。
4.  **滞后变量的特征重要性**：描述如何计算和查看时间序列数据的特征重要性分数。
5.  **滞后变量的特征选择**：描述如何计算和查看时间序列数据的特征选择结果。

让我们从查看标准时间序列数据集开始。

## 每月汽车销售数据集

在本教程中，我们将使用 Monthly Car Sales 数据集。

该数据集描述了 1960 年至 1968 年间加拿大魁北克省的汽车销售数量。

单位是销售数量的计数，有 108 个观察值。源数据归功于 Abraham 和 Ledolter（1983）。

[您可以从 DataMarket](https://datamarket.com/data/set/22n4/monthly-car-sales-in-quebec-1960-1968) 下载数据集。

下载数据集并将其保存到当前工作目录中，文件名为“ _car-sales.csv_ ”。请注意，您可能需要从文件中删除页脚信息。

下面的代码将数据集加载为 Pandas _ 系列 _ 对象。

```py
# line plot of time series
from pandas import Series
from matplotlib import pyplot
# load dataset
series = Series.from_csv('car-sales.csv', header=0)
# display first few rows
print(series.head(5))
# line plot of dataset
series.plot()
pyplot.show()
```

运行该示例将打印前 5 行数据。

```py
Month
1960-01-01 6550
1960-02-01 8728
1960-03-01 12026
1960-04-01 14395
1960-05-01 14587
Name: Sales, dtype: int64
```

还提供了数据的线图。

![Monthly Car Sales Dataset Line Plot](img/7c3bac9ffc6c7e1b3d33e646787ca9c6.jpg)

每月汽车销售数据集线图

## 制作文具

我们可以看到明确的季节性和数据的增长趋势。

趋势和季节性是固定的组成部分，可以添加到我们做出的任何预测中。它们很有用，但需要删除才能探索任何其他有助于预测的系统信号。

季节性和趋势被移除的时间序列称为静止。

为了消除季节性，我们可以采取季节性差异，从而产生所谓的季节性调整时间序列。

季节性的时期似乎是一年（12 个月）。下面的代码计算经季节性调整的时间序列并将其保存到文件“ _seasonally-adjusted.csv_ ”。

```py
# seasonally adjust the time series
from pandas import Series
from matplotlib import pyplot
# load dataset
series = Series.from_csv('car-sales.csv', header=0)
# seasonal difference
differenced = series.diff(12)
# trim off the first year of empty data
differenced = differenced[12:]
# save differenced dataset to file
differenced.to_csv('seasonally_adjusted.csv')
# plot differenced dataset
differenced.plot()
pyplot.show()
```

由于前 12 个月的数据没有先前的数据差异，因此必须将其丢弃。

固定数据存储在“ _seasonally-adjusted.csv_ ”中。创建差异数据的线图。

![Seasonally Differenced Monthly Car Sales Dataset Line Plot](img/3b068d6f12decff8880bd0006ebee6f1.jpg)

季节性差异月度汽车销售数据集线图

该图表明通过差分消除了季节性和趋势信息。

## 自相关图

传统上，时间序列特征是基于它们与输出变量的相关性来选择的。

这称为自相关，涉及绘制自相关图，也称为相关图。这些显示了每个滞后观察的相关性以及相关性是否具有统计学意义。

例如，下面的代码绘制了月度汽车销售数据集中所有滞后变量的相关图。

```py
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
series = Series.from_csv('seasonally_adjusted.csv', header=None)
plot_acf(series)
pyplot.show()
```

运行该示例将创建数据的相关图或自相关函数（ACF）图。

该图显示了沿 x 轴的滞后值和 y 轴上的相关性，在-1 和 1 之间分别为负相关和正相关滞后。

蓝色区域上方的点表示统计显着性。滞后值为 0 的相关性为 1 表示观察与其自身的 100％正相关。

该图显示在 1,2,12 和 17 个月时的显着滞后值。

![Correlogram of the Monthly Car Sales Dataset](img/3753299706e032a0640b0fb4b28a6892.jpg)

每月汽车销售数据集的相关图

该分析为比较提供了良好的基线。

## 监督学习的时间序列

我们可以通过将滞后观察（例如 t-1）作为输入并使用当前观察（t）作为输出变量，将单变量月度汽车销售数据集转换为[监督学习问题](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)。

我们可以在 Pandas 中使用 shift 函数创建移动观测的新列。

下面的示例创建一个新的时间序列，其中包含 12 个月的滞后值，以预测当前的观察结果。

12 个月的转变意味着前 12 行数据不可用，因为它们包含`NaN`值。

```py
from pandas import Series
from pandas import DataFrame
# load dataset
series = Series.from_csv('seasonally_adjusted.csv', header=None)
# reframe as supervised learning
dataframe = DataFrame()
for i in range(12,0,-1):
dataframe['t-'+str(i)] = series.shift(i)
dataframe['t'] = series.values
print(dataframe.head(13))
dataframe = dataframe[13:]
# save to new file
dataframe.to_csv('lags_12months_features.csv', index=False)
```

运行该示例将打印显示前 12 行和可用第 13 行的前 13 行数据。

```py
             t-12   t-11   t-10    t-9     t-8     t-7     t-6     t-5  \
1961-01-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
1961-02-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
1961-03-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
1961-04-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
1961-05-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
1961-06-01    NaN    NaN    NaN    NaN     NaN     NaN     NaN   687.0
1961-07-01    NaN    NaN    NaN    NaN     NaN     NaN   687.0   646.0
1961-08-01    NaN    NaN    NaN    NaN     NaN   687.0   646.0  -189.0
1961-09-01    NaN    NaN    NaN    NaN   687.0   646.0  -189.0  -611.0
1961-10-01    NaN    NaN    NaN  687.0   646.0  -189.0  -611.0  1339.0
1961-11-01    NaN    NaN  687.0  646.0  -189.0  -611.0  1339.0    30.0
1961-12-01    NaN  687.0  646.0 -189.0  -611.0  1339.0    30.0  1645.0
1962-01-01  687.0  646.0 -189.0 -611.0  1339.0    30.0  1645.0  -276.0

               t-4     t-3     t-2     t-1       t
1961-01-01     NaN     NaN     NaN     NaN   687.0
1961-02-01     NaN     NaN     NaN   687.0   646.0
1961-03-01     NaN     NaN   687.0   646.0  -189.0
1961-04-01     NaN   687.0   646.0  -189.0  -611.0
1961-05-01   687.0   646.0  -189.0  -611.0  1339.0
1961-06-01   646.0  -189.0  -611.0  1339.0    30.0
1961-07-01  -189.0  -611.0  1339.0    30.0  1645.0
1961-08-01  -611.0  1339.0    30.0  1645.0  -276.0
1961-09-01  1339.0    30.0  1645.0  -276.0   561.0
1961-10-01    30.0  1645.0  -276.0   561.0   470.0
1961-11-01  1645.0  -276.0   561.0   470.0  3395.0
1961-12-01  -276.0   561.0   470.0  3395.0   360.0
1962-01-01   561.0   470.0  3395.0   360.0  3440.0
```

从新数据集中删除前 12 行，结果保存在文件“`lags_12months_features.csv`”中。

这个过程可以以任意数量的时间步长重复，例如 6 个月或 24 个月，我建议进行实验。

## 滞后变量的特征重要性

决策树的集合，如袋装树，随机森林和额外树木，可用于计算特征重要性分数。

这在机器学习中很常见，用于在开发预测模型时估计输入特征的相对有用性。

我们可以使用特征重要性来帮助估计人为输入特征对时间序列预测的相对重要性。

这很重要，因为我们不仅可以设计上面的滞后观察特征，还可以设计基于观测时间戳，滚动统计等的特征。特征重要性是一种帮助理清建模时可能更有用的方法。

下面的示例加载上一节中创建的数据集的监督学习视图，拟合随机森林模型（ [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) ），并总结 12 个滞后观察中每一个的相对特征重要性分数。

使用大量树木来确保分数有些稳定。此外，初始化随机数种子以确保每次运行代码时都获得相同的结果。

```py
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# load data
dataframe = read_csv('lags_12months_features.csv', header=0)
array = dataframe.values
# split into input and output
X = array[:,0:-1]
y = array[:,-1]
# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X, y)
# show importance scores
print(model.feature_importances_)
# plot importance scores
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, model.feature_importances_)
pyplot.xticks(ticks, names)
pyplot.show()
```

首先运行该示例打印滞后观察的重要性分数。

```py
[ 0.21642244  0.06271259  0.05662302  0.05543768  0.07155573  0.08478599
  0.07699371  0.05366735  0.1033234   0.04897883  0.1066669   0.06283236]
```

然后将分数绘制为条形图。

该图显示了 t-12 观测的高度相对重要性，以及在较小程度上观察到 t-2 和 t-4 观测的重要性。

值得注意的是，与上述相关图的结果存在差异。

![Bar Graph of Feature Importance Scores on the Monthly Car Sales Dataset](img/20844559ed21c5c795e261902e397309.jpg)

月度汽车销售数据集中特征重要性得分的条形图

可以使用可以计算重要性分数的不同方法重复该过程，例如梯度增强，额外树和袋装决策树。

## 滞后变量的特征选择

我们还可以使用特征选择来自动识别和选择最具预测性的输入特征。

用于特征选择的流行方法称为递归特征选择（RFE）。

RFE 通过创建预测模型，加权特征和修剪具有最小权重的特征来工作，然后重复该过程直到剩下所需数量的特征。

下面的示例使用 RFE 和随机森林预测模型，并将所需的输入要素数设置为 4。

```py
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# load dataset
dataframe = read_csv('lags_12months_features.csv', header=0)
# separate into input and output variables
array = dataframe.values
X = array[:,0:-1]
y = array[:,-1]
# perform feature selection
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 4)
fit = rfe.fit(X, y)
# report selected features
print('Selected Features:')
names = dataframe.columns.values[0:-1]
for i in range(len(fit.support_)):
	if fit.support_[i]:
		print(names[i])
# plot feature rank
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, fit.ranking_)
pyplot.xticks(ticks, names)
pyplot.show()
```

运行该示例将打印 4 个所选要素的名称。

不出所料，结果与上一节中显示出高度重要性的特征相匹配。

```py
Selected Features:
t-12
t-6
t-4
t-2
```

还会创建一个条形图，显示每个输入要素的要素选择等级（越小越好）。

![Bar Graph of Feature Selection Rank on the Monthly Car Sales Dataset](img/e9cc19987378bd2687c0562ff4568fe5.jpg)

月度汽车销售数据集中特征选择等级的条形图

可以使用不同数量的特征重复此过程以选择 4 个以上以及除随机森林之外的不同模型。

## 摘要

在本教程中，您了解了如何使用应用机器学习工具来帮助在预测时从时间序列数据中选择要素。

具体来说，你学到了：

*   如何解释高度相关滞后观察的相关图。
*   如何计算和查看时间序列数据中的要素重要性分数。
*   如何使用特征选择来识别时间序列数据中最相关的输入变量。

对时间序列数据的特征选择有任何疑问吗？
在评论中提出您的问题，我会尽力回答。