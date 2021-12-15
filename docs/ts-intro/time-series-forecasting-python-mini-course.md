# 使用 Python 7 天迷你课程进行时间序列预测

> 原文： [https://machinelearningmastery.com/time-series-forecasting-python-mini-course/](https://machinelearningmastery.com/time-series-forecasting-python-mini-course/)

### 从开发人员到 7 天的时间序列预测。

Python 是应用机器学习增长最快的平台之一。

在这个迷你课程中，您将了解如何入门，构建准确的模型，并在 7 天内使用 Python 自信地完成预测建模时间序列预测项目。

这是一个重要且重要的帖子。您可能想要将其加入书签。

让我们开始吧。

![](img/7f4890ce13bb75974737e908494d5b9c.jpg)

使用 Python 7 天迷你课程进行时间序列预测
摄影： [Raquel M](https://www.flickr.com/photos/rmalinger/4302548106/) ，保留一些权利。

## 这个迷你课程是谁？

在我们开始之前，让我们确保您在正确的位置。

以下列表提供了有关本课程设计对象的一般指导原则。

如果你没有完全匹配这些点，请不要惊慌，你可能只需要在一个或另一个区域刷新以跟上。

*   **你是开发人员**：这是开发人员的课程。你是某种开发者。您知道如何读写代码。您知道如何开发和调试程序。
*   **你知道 Python** ：这是 Python 人的课程。你知道 Python 编程语言，或者你是一个熟练的开发人员，你可以随时随地学习它。
*   **你知道一些机器学习**：这是新手机器学习从业者的课程。你知道一些基本的实用机器学习，或者你可以快速搞清楚。

这个迷你课程既不是关于 Python 的教科书，也不是关于时间序列预测的教科书。

它将把你从一个了解一点机器学习的开发人员带到一个开发人员，他可以使用 Python 生态系统获得时间序列预测结果，这是一个不断上升的专业机器学习平台。

**注意**：这个迷你课程假设你有一个有效的 Python 2 或 3 SciPy 环境，至少安装了 NumPy，Pandas，scikit-learn 和 statsmodels。

## 迷你课程概述

这个迷你课程分为 7 课。

你可以每天完成一节课（_ 推荐 _）或者在一天内完成所有课程（_ 硬核 _）。这取决于你有空的时间和你的热情程度。

以下 7 个课程将通过 Python 中的机器学习帮助您开始并提高工作效率：

*   **第 01 课**：时间序列作为监督学习。
*   **第 02 课**：加载时间序列数据。
*   **第 03 课**：数据可视化。
*   **第 04 课**：持久性预测模型。
*   **第 05 课**：自回归预测模型。
*   **第 06 课**：ARIMA 预测模型。
*   **第 07 课**：Hello World 端到端项目。

每节课可能需要 60 秒或 30 分钟。花点时间，按照自己的进度完成课程。在下面的评论中提出问题甚至发布结果。

课程期望你去学习如何做事。我会给你提示，但每节课的部分内容是强迫你学习去寻找有关时间序列的 Python 平台的帮助（提示，我直接在这个博客上得到了所有的答案，使用搜索功能）。

我确实在早期课程中提供了更多帮助，因为我希望你建立一些自信和惯性。

#### 在评论中发布您的结果，我会为你欢呼！

挂在那里，不要放弃。

## 第 01 课：时间序列作为监督学习

时间序列问题与传统预测问题不同。

时间的增加为必须保留的观测添加了一个顺序，并且可以为学习算法提供额外的信息。

时间序列数据集可能如下所示：

```py
Time, Observation
day1, obs1
day2, obs2
day3, obs3
```

我们可以将这些数据重新设置为监督学习问题，并预测输入和输出。例如：

```py
Input,	Output
?,		obs1
obs1,	obs2
obs2,	obs3
obs3,	?
```

您可以看到重构意味着我们必须丢弃一些缺少数据的行。

一旦重新构建，我们就可以应用所有我们喜欢的学习算法，如 k-Nearest Neighbors 和 Random Forest。

如需更多帮助，请参阅帖子：

*   [时间序列预测作为监督学习](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

## 课程 02：加载时间序列数据

在开发预测模型之前，必须加载并使用时间序列数据。

Pandas 提供了以 CSV 格式加载数据的工具。

在本课程中，您将下载标准时间序列数据集，将其加载到 Pandas 中并进行探索。

从 DataMarket 以 CSV 格式下载[每日女性出生数据集](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959)，并以文件名“ _daily-births.csv_ ”保存。

您可以将时间序列数据集作为 Pandas Series 加载，并在第 0 行指定标题行，如下所示：

```py
from pandas import Series
series = Series.from_csv('daily-births.csv', header=0)
```

习惯于在 Python 中探索加载的时间序列数据：

*   使用`head()`功能打印前几行。
*   使用`size`属性打印数据集的尺寸。
*   使用日期时间字符串查询数据集。
*   打印观察的摘要统计。

如需更多帮助，请参阅帖子：

*   [如何在 Python 中加载和探索时间序列数据](http://machinelearningmastery.com/load-explore-time-series-data-python/)

## 第 03 课：数据可视化

数据可视化是时间序列预测的重要组成部分。

随着时间的推移观察的线图很受欢迎，但是您可以使用一套其他图来了解有关您的问题的更多信息。

在本课程中，您必须下载标准时间序列数据集并创建 6 种不同类型的图。

从 DataMarket 以 CSV 格式下载[月度洗发水销售数据集](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)，并使用文件名“ _shampoo-sales.csv_ ”保存。

现在创建以下 6 种类型的图：

1.  线图。
2.  直方图和密度图。
3.  盒子和晶须地块按年或季度。
4.  热图。
5.  滞后图或散点图。
6.  自相关图。

下面是一个简单的线图的示例，可以帮助您入门：

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('shampoo-sales.csv', header=0)
series.plot()
pyplot.show()
```

如需更多帮助，请参阅帖子：

*   [使用 Python 进行时间序列数据可视化](http://machinelearningmastery.com/time-series-data-visualization-with-python/)

## 课 04：持久性预测模型

建立基线预测非常重要。

您可以做的最简单的预测是使用当前观测值（t）来预测下一时间步（t + 1）的观测值。

这称为朴素预测或持久性预测，可能是某些时间序列预测问题的最佳模型。

在本课程中，您将对标准时间序列预测问题进行持久性预测。

从 DataMarket 以 CSV 格式下载[每日女性出生数据集](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959)，并以文件名“ _daily-births.csv_ ”保存。

您可以将持久性预测实现为单行函数，如下所示：

```py
# persistence model
def model_persistence(x):
	return x
```

编写代码以加载数据集并使用持久性预测对数据集中的每个时间步做出预测。请注意，您将无法对数据集中的第一个步骤做出预测，因为之前没有使用过的观察。

将所有预测存储在列表中。与实际观察结果相比，您可以计算预测的均方根误差（RMSE），如下所示：

```py
from sklearn.metrics import mean_squared_error
from math import sqrt
predictions = []
actual = series.values[1:]
rmse = sqrt(mean_squared_error(actual, predictions))
```

如需更多帮助，请参阅帖子：

*   [如何使用 Python 进行时间序列预测的基线预测](http://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)

## 第 05 课：自回归预测模型

自回归意味着开发一种线性模型，该模型使用先前时间步骤的观察来预测未来时间步骤的观察结果（“自动”意味着古希腊语中的自我）。

自回归是一种快速而有效的时间序列预测方法。

statsmodels Python 库在 [AR 类](http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.ar_model.AR.html)中提供自动回归模型。

在本课程中，您将为标准时间序列数据集开发自回归预测模型。

从 DataMarket 以 CSV 格式下载[月度洗发水销售数据集](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)，并使用文件名“ _shampoo-sales.csv_ ”保存。

您可以按如下方式安装 AR 模型：

```py
model = AR(dataset)
model_fit = model.fit()
```

您可以使用拟合 AR 模型预测下一次样本观察，如下所示：

```py
prediction = model_fit.predict(start=len(dataset), end=len(dataset))
```

您可能希望通过在半数数据集上拟合模型并预测系列的后半部分中的一个或多个来进行实验，然后将预测与实际观察结果进行比较。

如需更多帮助，请参阅帖子：

*   [使用 Python 进行时间序列预测的自回归模型](http://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)

## 第 06 课：ARIMA 预测模型

ARIMA 是时间序列预测的经典线性模型。

它结合了自回归模型（AR），差异去除趋势和季节性，称为积分（I）和移动平均模型（MA），它是一个旧名称，用于预测误差的模型，用于纠正预测。

statsmodels Python 库提供了 [ARIMA 类](http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.arima_model.ARIMA.html)。

在本课程中，您将为标准时间序列数据集开发 ARIMA 模型。

从 DataMarket 以 CSV 格式下载[月度洗发水销售数据集](https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)，并使用文件名“ _shampoo-sales.csv_ ”保存。

ARIMA 类需要一个顺序（p，d，q），其由 AR 滞后的三个参数 p，d 和 q 组成，差异数和 MA 滞后。

您可以按如下方式拟合 ARIMA 模型：

```py
model = ARIMA(dataset, order=(0,1,0))
model_fit = model.fit()
```

您可以为适合的 ARIMA 模型进行一步式样本外预测，如下所示：

```py
outcome = model_fit.forecast()[0]
```

洗发水数据集有一个趋势所以我建议 d 值为 1.尝试不同的 p 和 q 值并评估结果模型的预测。

如需更多帮助，请参阅帖子：

*   [如何使用 Python 创建用于时间序列预测的 ARIMA 模型](http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

## 第 07 课：Hello World 端到端项目

您现在可以使用工具来解决时间序列问题并开发一个简单的预测模型。

在本课程中，您将使用从所有先前课程中学到的技能来处理新的时间序列预测问题。

从 DataMarket 以 CSV 格式下载[季度 S＆amp; P 500 指数，1900-1996 数据集](https://datamarket.com/data/set/22rk/quarterly-sp-500-index-1900-1996)，并以文件名“`sp500.csv`”保存。

拆分数据，可能会将最后 4 或 8 个季度提取到单独的文件中。解决问题并为缺失的数据制定预测，包括：

1.  加载并浏览数据集。
2.  可视化数据集。
3.  开发持久性模型。
4.  开发自回归模型。
5.  开发 ARIMA 模型。
6.  可视化预测并总结预测误差。

有关完成项目的示例，请参阅帖子：

*   [Python 时间序列预测研究：法国香槟月销量[H​​TG1]](http://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/)

## 结束！
（看你有多远）

你做到了。做得好！

花点时间回顾一下你到底有多远。

你发现：

*   如何将时间序列预测问题构建为监督学习。
*   如何使用 Pandas 加载和探索时间序列数据。
*   如何以多种不同方式绘制和可视化时间序列数据。
*   如何开发一种称为持久性模型作为基线的朴素预测。
*   如何使用滞后观测开发自回归预测模型。
*   如何开发 ARIMA 模型，包括自回归，积分和移动平均元素。
*   如何将所有这些元素组合到一个端到端项目中。

不要轻视这一点，你在很短的时间内走了很长的路。

这只是您使用 Python 进行时间序列预测之旅的开始。继续练习和发展你的技能。

## 摘要

**你是如何使用迷你课程的？**
你喜欢这个迷你课吗？

你有任何问题吗？有没有任何问题？
让我知道。在下面发表评论。