# 如何用 Python 进行时间序列预测的预测

> 原文： [https://machinelearningmastery.com/make-predictions-time-series-forecasting-python/](https://machinelearningmastery.com/make-predictions-time-series-forecasting-python/)

选择时间序列预测模型只是一个开始。

在实践中使用所选模型可能带来挑战，包括数据转换和将模型参数存储在磁盘上。

在本教程中，您将了解如何最终确定时间序列预测模型并使用它在 Python 中进行预测。

完成本教程后，您将了解：

*   如何最终确定模型并将其和所需数据保存到文件中。
*   如何从文件加载最终模型并使用它来进行预测。
*   如何更新与已完成模型关联的数据以进行后续预测。

让我们开始吧。

*   **2017 年 2 月更新**：更新了布局和文件名，将 AR 案例与手册案例分开。

![How to Make Predictions for Time Series Forecasting with Python](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/01/How-to-Make-Predictions-for-Time-Series-Forecasting-with-Python.jpg)

如何使用 Python 进行时间序列预测预测
照片由 [joe christiansen](https://www.flickr.com/photos/joe_milkman/5190409637/) 保留，保留一些权利。

## 进行预测的过程

关于如何调整特定时间序列预测模型的文章很多，但对如何使用模型进行预测几乎没有帮助。

一旦您可以为数据构建和调整预测模型，进行预测的过程包括以下步骤：

1.  **型号选择**。这是您选择模型并收集证据和支持以捍卫决策的地方。
2.  **模型定稿**。所选模型将根据所有可用数据进行训练并保存到文件中以供以后使用。
3.  **预测**。加载已保存的模型并用于进行预测。
4.  **型号更新**。在有新观察的情况下更新模型的元素。

我们将在本教程中查看这些元素中的每一个，重点是将模型保存到文件和从文件加载模型，并使用加载的模型进行预测。

在我们深入研究之前，让我们首先看一下我们可以用作本教程上下文的标准单变量数据集。

## 每日女性出生数据集

该数据集描述了 1959 年加利福尼亚州每日女性出生人数。

单位是计数，有 365 个观测值。数据集的来源归功于 Newton（1988）。

[了解更多信息并从数据市场](https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959)下载数据集。

下载数据集并将其放在当前工作目录中，文件名为“ _daily-total-female-births.csv_ ”。

我们可以将数据集加载为 Pandas 系列。下面的代码片段会加载并绘制数据集。

```py
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-total-female-births.csv', header=0)
print(series.head())
series.plot()
pyplot.show()
```

运行该示例将打印数据集的前 5 行。

```py
Date
1959-01-01    35
1959-01-02    32
1959-01-03    30
1959-01-04    31
1959-01-05    44
```

然后将该系列图作为线图。

![Daily Female Birth Dataset Line Plot](img/0a85aaf450bbe4811635841b3f69508b.jpg)

每日女性出生数据集线图

## 1.选择时间序列预测模型

您必须选择一个模型。

这是大部分工作将用于准备数据，执行分析以及最终选择最能捕获数据关系的模型和模型超参数。

在这种情况下，我们可以在差异数据集上任意选择滞后为 6 的自回归模型（AR）。

我们可以在下面演示这个模型。

首先，通过差分对数据进行变换，每次观察变换为：

```py
value(t) = obs(t) - obs(t - 1)
```

接下来，AR（6）模型在 66％的历史数据上进行训练。提取模型学习的回归系数，并用于跨测试数据集以滚动方式进行预测。

当执行测试数据集中的每个时间步骤时，使用系数进行预测并存储。然后，可以获得并存储对时间步骤的实际观察，以用作将来预测的滞后变量。

```py
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

series = Series.from_csv('daily-total-female-births.csv', header=0)
# split dataset
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
# train autoregression
model = AR(train)
model_fit = model.fit(maxlag=6, disp=False)
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

运行该示例首先打印预测的均方误差（MSE），即 52（如果我们采用平方根将错误分数返回到原始单位，则平均约为 7 次出生）。

这是我们期望模型在对新数据进行预测时平均执行的程度。

```py
Test MSE: 52.696
```

最后，创建一个图表，显示测试数据集中的实际观察结果（蓝色）与预测（红色）相比较。

![Predictions vs Actual Daily Female Birth Dataset Line Plot](img/ed9d42a58e54333389c69127f0f36f85.jpg)

预测与实际每日女性出生数据集线图

这可能不是我们在这个问题上可以开发的最好的模型，但它是合理和熟练的。

## 2.完成并保存时间序列预测模型

选择模型后，我们必须完成它。

这意味着保存模型学习的显着信息，这样我们就不必在每次需要预测时重新创建它。

这包括首先在所有可用数据上训练模型，然后将模型保存到文件。

时间序列模型的`statsmodels`实现通过在拟合[上调用`save()`和`load()`来提供内置的保存和加载模型的功能。 ] ARResults](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.ar_model.ARResults.html) 对象。

例如，下面的代码将在整个女性出生数据集上训练 AR（6）模型，并使用内置的`save()`函数保存它，这将基本上腌制`ARResults`对象。

还必须保存差异训练数据，既可以用于进行预测所需的滞后变量，也可以用于 _ 预测（）_ ARResults 的函数所需观察数量的知识。 对象。

最后，我们需要能够将差异数据集转换回原始形式。要做到这一点，我们必须跟踪最后的实际观察。这样可以将预测的差值添加到其中。

```py
# fit an AR model and save the whole model to file
from pandas import Series
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import ARResults
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = Series.from_csv('daily-total-female-births.csv', header=0)
X = difference(series.values)
# fit model
model = AR(X)
model_fit = model.fit(maxlag=6, disp=False)
# save model to file
model_fit.save('ar_model.pkl')
# save the differenced dataset
numpy.save('ar_data.npy', X)
# save the last ob
numpy.save('ar_obs.npy', [series.values[-1]])
```

此代码将创建一个文件`ar_model.pkl`，您可以稍后加载该文件并用于进行预测。

整个训练数据集保存为`ar_data.npy`，最后一个观察结果保存在文件`ar_obs.npy`中，作为带有一个项目的数组。

NumPy [save（）](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html)功能用于保存差异训练数据和观察。然后可以使用 [load（）](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)函数来加载这些数组。

下面的代码段将加载模型，差异数据和最后一次观察。

```py
# load the AR model from file
from statsmodels.tsa.ar_model import ARResults
import numpy
loaded = ARResults.load('ar_model.pkl')
print(loaded.params)
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
print(last_ob)
```

运行该示例打印系数和最后一个观察。

```py
[ 0.12129822 -0.75275857 -0.612367   -0.51097172 -0.4176669  -0.32116469
 -0.23412997]
[50]
```

我认为这对大多数情况都有好处，但也很重要。您可能会对 statsmodels API 进行更改。

我倾向于直接使用模型的系数，如上面的情况，使用滚动预测来评估模型。

在这种情况下，您可以简单地存储模型系数，然后加载它们并进行预测。

下面的示例仅保存模型中的系数，以及进行下一次预测所需的最小差异滞后值以及转换下一次预测所需的最后一次观察。

```py
# fit an AR model and manually save coefficients to file
from pandas import Series
from statsmodels.tsa.ar_model import AR
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = Series.from_csv('daily-total-female-births.csv', header=0)
X = difference(series.values)
# fit model
window_size = 6
model = AR(X)
model_fit = model.fit(maxlag=window_size, disp=False)
# save coefficients
coef = model_fit.params
numpy.save('man_model.npy', coef)
# save lag
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
# save the last ob
numpy.save('man_obs.npy', [series.values[-1]])
```

系数保存在本地文件`man_model.npy`中，滞后历史记录保存在文件`man_data.npy`中，最后一次观察保存在文件 _man_obs 中。 npy_ 。

然后可以按如下方式再次加载这些值：

```py
# load the manually saved model from file
import numpy
coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)
```

运行此示例将打印加载的数据以供查看。我们可以看到系数和最后一个观察与前一个例子的输出相匹配。

```py
[ 0.12129822 -0.75275857 -0.612367   -0.51097172 -0.4176669  -0.32116469
 -0.23412997]
[-10   3  15  -4   7  -5]
[50]
```

现在我们知道如何保存最终模型，我们可以使用它来进行预测。

## 3.制作时间序列预测

进行预测涉及加载已保存的模型并在下一个时间步骤估计观测值。

如果 ARResults 对象被序列化，我们可以使用`predict()`函数来预测下一个时间段。

下面的示例显示了如何预测下一个时间段。

从文件加载模型，训练数据和最后观察。

该周期被指定为`predict()`函数，作为训练数据集结束后的下一个时间索引。该索引可以直接存储在文件中，而不是存储整个训练数据，这可以是效率。

进行预测，该预测是在差异数据集的背景下进行的。要将预测转回原始单位，必须将其添加到最后已知的观察中。

```py
# load AR model from file and make a one-step prediction
from statsmodels.tsa.ar_model import ARResults
import numpy
# load model
model = ARResults.load('ar_model.pkl')
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
# make prediction
predictions = model.predict(start=len(data), end=len(data))
# transform prediction
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)
```

运行该示例将打印预测。

```py
Prediction: 46.755211
```

我们也可以使用类似的技巧来加载原始系数并进行手动预测。
下面列出了完整的示例。

```py
# load a coefficients and from file and make a manual prediction
import numpy

def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

# load model
coef = numpy.load('man_model.npy')
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
# make prediction
prediction = predict(coef, lag)
# transform prediction
yhat = prediction + last_ob[0]
print('Prediction: %f' % yhat)
```

运行该示例，我们实现了与我们预期相同的预测，因为用于进行预测的基础模型和方法是相同的。

```py
Prediction: 46.755211
```

## 4.更新预测模型

我们的工作没有完成。

一旦下一次真实观察可用，我们必须更新与模型相关的数据。

具体来说，我们必须更新：

1.  差异训练数据集用作进行后续预测的输入。
2.  最后一次观察，为预测的差异值提供背景。

让我们假设该系列中的下一个实际观察结果为 48\.
新观察必须首先与最后一次观察不同。然后可以将其存储在差异观察列表中。最后，该值可以存储为最后一个观察值。

对于存储的 AR 模型，我们可以更新`ar_data.npy`和`ar_obs.npy`文件。完整示例如下：

```py
# update the data for the AR model with a new obs
import numpy
# get real observation
observation = 48
# load the saved data
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
# update and save differenced observation
diffed = observation - last_ob[0]
data = numpy.append(data, [diffed], axis=0)
numpy.save('ar_data.npy', data)
# update and save real observation
last_ob[0] = observation
numpy.save('ar_obs.npy', last_ob)
```

我们可以对手动案例的数据文件进行相同的更改。具体来说，我们可以分别更新`man_data.npy`和`man_obs.npy`。

下面列出了完整的示例。

```py
# update the data for the manual model with a new obs
import numpy
# get real observation
observation = 48
# update and save differenced observation
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
diffed = observation - last_ob[0]
lag = numpy.append(lag[1:], [diffed], axis=0)
numpy.save('man_data.npy', lag)
# update and save real observation
last_ob[0] = observation
numpy.save('man_obs.npy', last_ob)
```

我们专注于一步预测。

通过重复使用模型并使用先前时间步长的预测作为输入滞后值来预测后续时间步骤的观测，这些方法对于多步预测同样容易。

## 考虑存储所有观察结果

通常，跟踪所有观察结果是个好主意。

这将允许您：

*   为进一步的时间序列分析提供上下文，以了解数据中的新变化。
*   根据最新数据在未来训练新模型。
*   回溯测试新的和不同的模型，看看表现是否可以提高。

对于小型应用程序，您可以将原始观察结果存储在模型旁边的文件中。

还可能希望以简单文本存储模型系数和所需滞后数据以及最后观察以便于查看。

对于较大的应用程序，也许可以使用数据库系统来存储观察结果。

## 摘要

在本教程中，您了解了如何最终确定时间序列模型并使用它来使用 Python 进行预测。

具体来说，你学到了：

*   如何将时间序列预测模型保存到文件。
*   如何从文件加载保存的时间序列预测并进行预测。
*   如何使用新观察更新时间序列预测模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。