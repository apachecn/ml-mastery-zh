# 使用 Python 进行时间序列预测表现测量

> 原文： [https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/](https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/)

时间序列预测表现度量提供了做出预测的预测模型的技能和能力的总结。

有许多不同的绩效指标可供选择。知道使用哪种措施以及如何解释结果可能会令人困惑。

在本教程中，您将发现使用 Python 评估时间序列预测的表现度量。

时间序列通常关注实际值的预测，称为回归问题。因此，本教程中的表现度量将侧重于评估实值预测的方法。

完成本教程后，您将了解：

*   预测绩效的基本衡量指标，包括残差预测误差和预测偏差。
*   时间序列预测误差计算与预期结果具有相同的单位，例如平均绝对误差。
*   广泛使用的错误计算可以惩罚大的错误，例如均方误差和均方根误差。

让我们开始吧。

![Time Series Forecasting Performance Measures With Python](img/cbffbfe7ae4e747e3ecd6d21d63ab45d.jpg)

时间序列预测表现测量与 Python
照片由[汤姆霍尔](https://www.flickr.com/photos/tom_hall_nz/14917023204/)，保留一些权利。

## 预测误差（或残差预测误差）

[预测误差](https://en.wikipedia.org/wiki/Forecast_error)被计算为期望值减去预测值。

这称为预测的残差。

```py
forecast_error = expected_value - predicted_value
```

可以为每个预测计算预测误差，提供预测误差的时间序列。

下面的示例演示了如何计算一系列 5 个预测与 5 个预期值相比的预测误差。这个例子是出于演示目的而设计的。

```py
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
print('Forecast Errors: %s' % forecast_errors)
```

运行该示例计算 5 个预测中每个预测的预测误差。然后打印预测错误列表。

```py
Forecast Errors: [-0.2, 0.09999999999999998, -0.1, -0.09999999999999998, -0.2]
```

预测误差的单位与预测的单位相同。预测误差为零表示没有错误或该预测的完美技能。

## 平均预测误差（或预测偏差）

平均预测误差计算为预测误差值的平均值。

```py
mean_forecast_error = mean(forecast_error)
```

预测错误可能是积极的和消极的。这意味着当计算这些值的平均值时，理想的平均预测误差将为零。

除零以外的平均预测误差值表明模型倾向于过度预测（正误差）或低于预测（负误差）。因此，平均预测误差也称为[预测偏差](https://en.wikipedia.org/wiki/Forecast_bias)。

预测误差可以直接计算为预测值的平均值。下面的示例演示了如何手动计算预测误差的平均值。

```py
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
bias = sum(forecast_errors) * 1.0/len(expected)
print('Bias: %f' % bias)
```

运行该示例将打印平均预测误差，也称为预测偏差。

```py
Bias: -0.100000
```

预测偏差的单位与预测的单位相同。零预测偏差或接近零的非常小的数字表示无偏模型。

## 平均绝对误差

[平均绝对误差](https://en.wikipedia.org/wiki/Mean_absolute_error)或 MAE 计算为预测误差值的平均值，其中所有预测值都被强制为正值。

强制值为正值称为绝对值。这由绝对函数`abs()`表示，或者在数学上显示为值周围的两个管道字符： _| value |_ 。

```py
mean_absolute_error = mean( abs(forecast_error) )
```

其中`abs()`使值为正，`forecast_error`是一个或一系列预测误差，`mean()`计算平均值。

我们可以使用 scikit-learn 库中的 [mean_absolute_error（）](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)函数来计算预测列表的平均绝对误差。以下示例演示了此功能。

```py
from sklearn.metrics import mean_absolute_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)
```

运行该示例计算并打印 5 个预期值和预测值的列表的平均绝对误差。

```py
MAE: 0.140000
```

这些误差值以预测值的原始单位表示。平均绝对误差为零表示没有错误。

## 均方误差

[均方误差](https://en.wikipedia.org/wiki/Mean_squared_error)或 MSE 计算为平方预测误差值的平均值。平方预测误差值迫使它们为正;它还具有加重大错误的效果。

非常大或异常的预测误差是平方的，这反过来又具有拖动平方预测误差的平均值的效果，导致更大的均方误差分数。实际上，得分会给那些做出大错误预测的模型带来更差的表现。

```py
mean_squared_error = mean(forecast_error^2)
```

我们可以使用 scikit-learn 中的 [mean_squared_error（）](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)函数来计算预测列表的均方误差。以下示例演示了此功能。

```py
from sklearn.metrics import mean_squared_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mse = mean_squared_error(expected, predictions)
print('MSE: %f' % mse)
```

运行该示例计算并打印预期值和预测值列表的均方误差。

```py
MSE: 0.022000
```

误差值以预测值的平方单位表示。均方误差为零表示技能完美，或没有错误。

## 均方根误差

上述均方误差是预测的平方单位。

通过取平均误差分数的平方根，可以将其转换回预测的原始单位。这称为[均方根误差](https://en.wikipedia.org/wiki/Root-mean-square_deviation)或 RMSE。

```py
rmse = sqrt(mean_squared_error)
```

这可以通过使用`sqrt()`数学函数计算平均误差，使用`mean_squared_error()`scikit-learn 函数计算得出。

```py
from sklearn.metrics import mean_squared_error
from math import sqrt
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mse = mean_squared_error(expected, predictions)
rmse = sqrt(mse)
print('RMSE: %f' % rmse)
```

运行该示例计算均方根误差。

```py
RMSE: 0.148324
```

RMES 错误值与预测的单位相同。与均方误差一样，RMSE 为零表示没有错误。

## 进一步阅读

以下是有关进一步阅读时间序列预测误差测量的一些参考资料。

*   第 3.3 节测量预测准确度，[实际时间序列预测与 R：动手指南](http://www.amazon.com/dp/0997847913?tag=inspiredalgor-20)。
*   第 2.5 节评估预测准确度，[预测：原则和实践](http://www.amazon.com/dp/0987507109?tag=inspiredalgor-20)
*   [scikit-learn Metrics API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
*   第 3.3.4 节。 [回归指标](http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)，scikit-learn API 指南

## 摘要

在本教程中，您在 Python 中发现了一套 5 个标准时间序列表现度量。

具体来说，你学到了：

*   如何计算预测残差以及如何估计预测列表中的偏差。
*   如何计算平均绝对预测误差，以与预测相同的单位描述误差。
*   如何计算预测的广泛使用的均方误差和均方根误差。

您对时间序列预测表现指标或本教程有任何疑问
请在下面的评论中提出您的问题，我会尽力回答。