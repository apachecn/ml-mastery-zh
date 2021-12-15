# 通过提前停止(early stopping)避免Python中应用XGBoost时发生得过拟合（overfitting）现象

> 原文： [https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/)

过拟合是非线性学习算法（例如梯度提升）中常见的一个复杂问题。

在这篇文章中，您将了解如何通过提前停止(early stopping)来抑制在Python中应用XGBoost时的过拟合现象。

阅读这篇文章，您会学习到：

*   提前停止(early stopping)是减少训练数据过拟合的一种方法。
*   如何在训练期间监测XGBoost模型的表现并绘制学习曲线。
*   如何使用提前停止(early stopping)来适时及早终止训练处于最佳epoch中的XGBoost模型。

请在我的[新书](https://machinelearningmastery.com/xgboost-with-python/)中找到如何通过XGBoost配置、训练、调试和评估梯度提升模型，其中包括了15个手把手（Step-by-Step）的示例课程以及完整的Python代码。

让我们开始吧。

*   **2017年1月更新**：此次更新为对应scikit-learn API版本0.18.1中的更改。
*   **2018年3月更新**：为下载数据集添加了备用链接，旧链接已被移除。

![Avoid Overfitting By Early Stopping With XGBoost In Python](img/3b5a137b9d5bd85033c44aac0f3068ff.jpg)

通过提前停止(early stopping)避免Python中应用XGBoost时发生得过拟合（overfitting）现象
照片由[Michael Hamann](https://www.flickr.com/photos/michitux/7218180540/)拍摄，保留部分版权。

## 通过提前停止(early stopping)避免过拟合

[提前停止(early stopping)](https://en.wikipedia.org/wiki/Early_stopping)是一种训练复杂机器学习模型时避免过拟合的方法。

它通过监测在单独的测试数据集上训练模型的表现，并且观察到一旦在固定数量的训练迭代之后测试数据集上的表现没有得到改善，就会停止训练过程。

通过尝试自动选出测试数据集上的表现开始降低而训练数据集上的表现继续提高这样的过拟合发生迹象拐点来避免过拟合。

表现度量可以是通过训练模型而进行优化的损失函数（例如对数损失函数（logarithmic loss）），或者通常情况下问题所关注的外部指标（例如分类精度）。

## 在XGBoost中监测训练表现

XGBoost模型可以在训练期间评估和报告模型在测试集上的表现。

它通过在训练模型和获取verbose output中调用 **model.fit（）**的同时指定测试数据集以及评估度量（evaluation metric）来支持此功能。

例如，我们可以在训练XGBoost模型时，在独立测试集（ **eval_set** ）上报告二值分类误差（"error"），如下所示：

```py
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
```

XGBoost所支持的评估度量（evaluation metric）集合包括但不仅限于：

*   “rmse”表示均方根误差。
*   “mae”表示平均绝对误差。
*   “logloss”表示二值对数损失，“mlogloss”表示多类对数损失（交叉熵）。
*   “error”表示分类误差。
*   “auc”表示ROC 曲线下的面积。

完整列表请参照XGBoost参数网页“[学习任务参数(Learning Task Parameters)](http://xgboost.readthedocs.io/en/latest//parameter.html)”。

例如，我们可以展示如何追踪XGBoost模型训练的表现，应用对象是[Pima印第安人糖尿病数据集(Pima Indians onset of diabetes dataset)](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)，可以从UCI机器学习库(UCI Machine Learning Repository)获取下载（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

完整示例代码如下：

```py
# monitor training performance
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model no training data
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行这个例子将会在67％的数据上训练模型，并在剩余33％的测试数据集上，于每一个训练epoch评估一次模型。

每一次迭代的结果都将会报告分类误差，而分类精度将会在最后给出。

下面展示了结果输出，为简洁明了，只截取末尾部分。我们可以看到每次训练迭代都会报告分类误差（在每个boosted tree被添加到模型之后）。

```py
...
[89]	validation_0-error:0.204724
[90]	validation_0-error:0.208661
[91]	validation_0-error:0.208661
[92]	validation_0-error:0.208661
[93]	validation_0-error:0.208661
[94]	validation_0-error:0.208661
[95]	validation_0-error:0.212598
[96]	validation_0-error:0.204724
[97]	validation_0-error:0.212598
[98]	validation_0-error:0.216535
[99]	validation_0-error:0.220472
Accuracy: 77.95%
```

回顾所有输出，我们可以看到在测试集上模型表现平稳，不过在训练即将结束时表现有些下降。

## 通过学习曲线评估XGBoost模型

我们可以在评估数据集上检视模型的表现，并通过绘制图像以更深入地展开了解训练中是如何学习的。

在训练XGBoost模型时，我们为**eval_metric**参数提供了一对X和y数组。除了测试集，我们也可以一并提供训练数据集。它将说明模型在训练期间在训练集和测试集上分别表现的情况。

例如：

```py
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
```

此外，通过调用 **model.evals_result（）**函数，在每个评估集上训练的模型都可以存储并再度可用。这将返回评估数据集和评分的dictionary，例如：

```py
results = model.evals_result()
print(results)
```

这将print如下结果（为简洁明了，只截取部分作说明）：

```py
{
	'validation_0': {'error': [0.259843, 0.26378, 0.26378, ...]},
	'validation_1': {'error': [0.22179, 0.202335, 0.196498, ...]}
}
```

'validation_0'和'validation_1'对应于在**fit（）**调用中向 **eval_set** 参数提供数据集的顺序。

若需要访问特定的结果数组，例如针对第一个数据集和其误差指标，可以操作如下：

```py
results['validation_0']['error']
```

此外，我们可以通过向**fit（）**函数的eval_metric参数提供度量数组，来指定更多的评估度量（evaluation metric）用于评价和汇总。

我们可以使用这些汇总的表现度量来创建曲线图，并进一步解读模型在训练epochs过程中分别在训练数据集和测试数据集上的表现。

下面是完整的代码示例，显示了如何在曲线图上可视化汇总的结果。

```py
# plot learning curve
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model no training data
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
```

运行这段代码会显示每个epoch中训练数据集和测试数据集的分类误差。我们可以通过在 **fit（）**函数的调用中设置 **verbose = False** （默认值）来关闭这个功能。

我们看到结果创建了两张图。第一张图显示了训练数据集和测试数据集每个epoch中XGBoost模型的对数损失（logarithmic loss）。

![XGBoost Learning Curve Log Loss](img/3dd164f486ba1862fa97f82eb6693360.jpg)

XGBoost学习曲线（对数损失（log loss））

第二张图显示了训练数据集和测试数据集每个epoch中XGBoost模型的分类误差（classification error）。

![XGBoost Learning Curve Classification Error](img/cdfec3000bac01a37daacb6f874ff978.jpg)

XGBoost学习曲线（分类误差（classification error））

通过回顾logloss的图像，我们看起来是有提前停止学习过程的机会，也许在epoch 20到epoch 40之间的某个阶段。

在分类误差的图像中，我们也观察到了类似的情况，误差似乎在epoch 40左右出现上升。

## 在XGBoost中使用提前停止(early stopping)

XGBoost可以支持在固定次数的迭代后提前停止(early stopping)。

除了为每个epoch指定用于评估的度量和测试数据集之外，还必须指定一个epoch窗长，它代表没有观察到任何改善的epoch数目。它可以在**early_stopping_rounds**参数中实现。

例如，我们可以在10个epoch中检查对数损失没有得到改善，代码如下：

```py
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
```

如果提供了多个评估数据集或多个评估度量（evaluation metric），则提前停止(early stopping)将使用列表中的最后一个。

下面展示提前停止(early stopping)的一个完整示例。

```py
# early stopping
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行示例代码将给出如下输出（为简洁明了，只截取部分作说明）：

```py
...
[35]	validation_0-logloss:0.487962
[36]	validation_0-logloss:0.488218
[37]	validation_0-logloss:0.489582
[38]	validation_0-logloss:0.489334
[39]	validation_0-logloss:0.490969
[40]	validation_0-logloss:0.48978
[41]	validation_0-logloss:0.490704
[42]	validation_0-logloss:0.492369
Stopping. Best iteration:
[32]	validation_0-logloss:0.487297
```

我们可以看到模型在epoch 42停止训练（接近我们对学习曲线人为判断的预期），并且在epoch 32观察到具有最佳损失结果的模型。

通常情况下，选择 **early_stopping_rounds** 作为训练epoch总数（在这种情况下为 10％）是个不错的主意，或者尝试找到可能观察到的学习曲线拐点时期。

## 总结

在这篇文章中，您了解到了如何监测表现表现和提前停止(early stopping)。

所学到的要点是：

*   提前停止(early stopping)技术能够在训练数据集发生过拟合之前就停止模型训练。
*   如何在训练期间监测XGBoost模型的表现并绘制学习曲线。
*   如何在训练XGBoost模型中配置提前停止(early stopping)。

您对过拟合或这篇文章有任何疑问吗？请在评论中提出您的问题，我将会尽力回答。
