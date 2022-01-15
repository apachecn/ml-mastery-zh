# 用于多输出回归的深度学习模型

> 原文：<https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/>

最后更新于 2020 年 8 月 28 日

多输出回归包括预测两个或多个数值变量。

与为每个样本预测单个值的正常回归不同，多输出回归需要支持为每个预测输出多个变量的专门机器学习算法。

深度学习神经网络是原生支持多输出回归问题的算法的一个例子。使用 Keras 深度学习库可以轻松定义和评估多输出回归任务的神经网络模型。

在本教程中，您将发现如何为多输出回归开发深度学习模型。

完成本教程后，您将知道:

*   多输出回归是一项预测建模任务，涉及两个或多个数值输出变量。
*   可以为多输出回归任务配置神经网络模型。
*   如何评价多输出回归的神经网络并对新数据进行预测？

我们开始吧。

![Deep Learning Models for Multi-Output Regression](img/b8a485b210d5c99cce2915aca56c7838.png)

多输出回归的深度学习模型
图片由[克里斯蒂安·科林斯](https://flickr.com/photos/collins_family/25354279379/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  多输出回归
2.  多输出神经网络
3.  多输出回归的神经网络

## 多输出回归

回归是一项预测建模任务，包括在给定输入的情况下预测数值输出。

它不同于涉及预测类别标签的分类任务。

通常，回归任务包括预测单个数值。但是，有些任务需要预测多个数值。这些任务被称为**多输出回归**，简称多输出回归。

在多输出回归中，每个输入样本需要两个或多个输出，并且同时需要输出。假设输出是输入的函数。

我们可以使用 scikit-learn 库中的[make _ revolution()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个合成的多输出回归数据集。

我们的数据集将有 1000 个样本，包含 10 个输入要素，其中 5 个与输出相关，5 个是冗余的。对于每个样本，数据集将有三个数字输出。

下面列出了创建和总结合成多输出回归数据集的完整示例。

```py
# example of a multi-output regression problem
from sklearn.datasets import make_regression
# create dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
# summarize shape
print(X.shape, y.shape)
```

运行该示例将创建数据集并总结输入和输出元素的形状。

我们可以看到，正如预期的那样，有 1000 个样本，每个样本有 10 个输入特征和 3 个输出特征。

```py
(1000, 10) (1000, 3)
```

接下来，让我们看看如何为多输出回归任务开发神经网络模型。

## 多输出神经网络

许多机器学习算法本身支持多输出回归。

常见的例子是决策树和决策树的集合。用于多输出回归的决策树的一个限制是输入和输出之间的关系可以是块状的或者基于训练数据的高度结构化的。

神经网络模型还支持多输出回归，并且具有学习连续函数的好处，该函数可以在输入和输出的变化之间建立更优美的关系。

只要将问题中存在的目标变量的数量指定为输出层中的节点数量，神经网络就可以直接支持多输出回归。例如，具有三个输出变量的任务将需要一个神经网络输出层，在输出层中有三个节点，每个节点都具有线性(默认)激活函数。

我们可以使用 Keras 深度学习库来演示这一点。

我们将为上一节中定义的多输出回归任务定义一个多层感知器(MLP)模型。

每个样本有 10 个输入和 3 个输出，因此，网络需要一个输入层，它期望通过第一个隐藏层中的“ *input_dim* ”参数和输出层中的 3 个节点指定 10 个输入。

我们将使用隐藏层中流行的 [ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) 激活函数。隐藏层有 20 个节点，这些节点是经过反复试验选择的。我们将使用平均绝对误差损失和随机梯度下降的亚当版本来拟合模型。

下面列出了多输出回归任务的网络定义。

```py
...
# define the model
model = Sequential()
model.add(Dense(20, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(3))
model.compile(loss='mae', optimizer='adam')
```

您可能希望将此模型用于自己的多输出回归任务，因此，我们可以创建一个函数来定义和返回模型，其中输入变量的数量和输出变量的数量作为参数提供。

```py
# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model
```

现在我们已经熟悉了如何定义多输出回归的 MLP，让我们来探索如何评估这个模型。

## 多输出回归的神经网络

如果数据集很小，最好在同一数据集上重复评估神经网络模型，并报告重复的平均表现。

这是因为学习算法的随机性。

此外，在对新数据进行预测时，最好使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)代替数据集的训练/测试分割，以获得模型表现的无偏估计。同样，只有当没有太多的数据并且该过程可以在合理的时间内完成时。

考虑到这一点，我们将使用 10 倍和 3 倍重复的重复 k 倍交叉验证来评估多输出回归任务的 MLP 模型。

模型的每个折叠都被定义、拟合和评估。收集分数，并通过报告平均值和标准偏差进行汇总。

下面的 *evaluate_model()* 函数获取数据集，对模型进行评估，并返回评估分数列表，在本例中为 MAE 分数。

```py
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results
```

然后，我们可以加载数据集，评估模型，并报告平均表现。

将这些联系在一起，完整的示例如下所示。

```py
# mlp for multi-output regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results

# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))
```

运行该示例报告每个折叠和每个重复的 MAE，以给出评估进度的想法。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

最后，报告了平均和标准差。在这种情况下，该模型显示的 MAE 约为 8.184。

您可以使用此代码作为模板，在自己的多输出回归任务中评估 MLP 模型。模型中节点和图层的数量可以根据数据集的复杂性轻松调整和定制。

```py
...
>8.054
>7.562
>9.026
>8.541
>6.744
MAE: 8.184 (1.032)
```

一旦选择了模型配置，我们就可以使用它来拟合所有可用数据的最终模型，并对新数据进行预测。

下面的示例演示了这一点，首先在整个多输出回归数据集上拟合 MLP 模型，然后在保存的模型上调用 *predict()* 函数，以便对新的数据行进行预测。

```py
# use mlp for prediction on multi-output regression
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model

# load dataset
X, y = get_dataset()
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=100)
# make a prediction for new data
row = [-0.99859353,2.19284309,-0.42632569,-0.21043258,-1.13655612,-0.55671602,-0.63169045,-0.87625098,-0.99445578,-0.3677487]
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])
```

运行该示例符合模型，并对新行进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

正如预期的那样，预测包含多输出回归任务所需的三个输出变量。

```py
Predicted: [-152.22713 -78.04891 -91.97194]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [sklearn . datasets . make _ revolution API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)。
*   [硬主页](https://keras.io/) 。
*   [sklearn.model_selection。重复应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)。

## 摘要

在本教程中，您发现了如何为多输出回归开发深度学习模型。

具体来说，您了解到:

*   多输出回归是一项预测建模任务，涉及两个或多个数值输出变量。
*   可以为多输出回归任务配置神经网络模型。
*   如何评价多输出回归的神经网络并对新数据进行预测？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。