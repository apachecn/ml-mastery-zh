# 深度学习多标签分类

> 原文:[https://machinelearning master . com/多标签-分类-带深度学习/](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)

最后更新于 2020 年 8 月 31 日

多标签分类包括预测零个或多个类别标签。

与类别标签互斥的正常分类任务不同，多标签分类需要支持预测多个互斥类别或“标签”的专门机器学习算法。

深度学习神经网络是原生支持多标签分类问题的算法的一个例子。使用 Keras 深度学习库可以轻松定义和评估用于多标签分类任务的神经网络模型。

在本教程中，您将发现如何为多标签分类开发深度学习模型。

完成本教程后，您将知道:

*   多标签分类是一项预测建模任务，涉及预测零个或多个互斥的类别标签。
*   可以为多标签分类任务配置神经网络模型。
*   如何评价多标签分类的神经网络并对新数据进行预测？

我们开始吧。

![Multi-Label Classification with Deep Learning](img/c51fb36d44b8695354382f59d8d7d56b.png)

深度学习多标签分类
图片由[特雷弗·马龙](https://flickr.com/photos/141333312@N03/26888851517/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

*   多标签分类
*   多标签神经网络
*   用于多标签分类的神经网络

## 多标签分类

分类是一个预测建模问题，它涉及输出给定输入的类标签

它不同于涉及预测数值的回归任务。

通常，分类任务包括预测单个标签。或者，它可能涉及预测跨越两个或更多类别标签的可能性。在这些情况下，类是互斥的，这意味着分类任务假设输入只属于一个类。

一些分类任务需要预测多个类别标签。这意味着类标签或类成员并不相互排斥。这些任务被称为**多标签分类**，简称多标签分类。

在多标签分类中，每个输入样本需要零个或多个标签作为输出，并且同时需要输出。假设输出标签是输入的函数。

我们可以使用 scikit-learn 库中的[make _ multi label _ classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html)创建一个合成的多标签分类数据集。

我们的数据集将有 1000 个样本和 10 个输入要素。对于每个样本，数据集将有三个类别标签输出，每个类别将有一个或两个值(0 或 1，例如存在或不存在)。

下面列出了创建和总结合成多标签分类数据集的完整示例。

```
# example of a multi-label classification task
from sklearn.datasets import make_multilabel_classification
# define dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
# summarize dataset shape
print(X.shape, y.shape)
# summarize first few examples
for i in range(10):
	print(X[i], y[i])
```

运行该示例将创建数据集并总结输入和输出元素的形状。

我们可以看到，正如预期的那样，有 1000 个样本，每个样本有 10 个输入特征和 3 个输出特征。

对前 10 行输入和输出进行了汇总，我们可以看到该数据集的所有输入都是数字，并且输出类标签在三个类标签中各有 0 或 1 个值。

```
(1000, 10) (1000, 3)

[ 3\.  3\.  6\.  7\.  8\.  2\. 11\. 11\.  1\.  3.] [1 1 0]
[7\. 6\. 4\. 4\. 6\. 8\. 3\. 4\. 6\. 4.] [0 0 0]
[ 5\.  5\. 13\.  7\.  6\.  3\.  6\. 11\.  4\.  2.] [1 1 0]
[1\. 1\. 5\. 5\. 7\. 3\. 4\. 6\. 4\. 4.] [1 1 1]
[ 4\.  2\.  3\. 13\.  7\.  2\.  4\. 12\.  1\.  7.] [0 1 0]
[ 4\.  3\.  3\.  2\.  5\.  2\.  3\.  7\.  2\. 10.] [0 0 0]
[ 3\.  3\.  3\. 11\.  6\.  3\.  4\. 14\.  1\.  3.] [0 1 0]
[ 2\.  1\.  7\.  8\.  4\.  5\. 10\.  4\.  6\.  6.] [1 1 1]
[ 5\.  1\.  9\.  5\.  3\.  4\. 11\.  8\.  1\.  8.] [1 1 1]
[ 2\. 11\.  7\.  6\.  2\.  2\.  9\. 11\.  9\.  3.] [1 1 1]
```

接下来，让我们看看如何为多标签分类任务开发神经网络模型。

## 多标签神经网络

一些机器学习算法本身支持多标签分类。

神经网络模型可以被配置为支持多标签分类，并且可以根据分类任务的具体情况表现良好。

只要将问题中的目标标签数量指定为输出层中的节点数量，神经网络就可以直接支持多标签分类。例如，具有三个输出标签(类)的任务将需要在输出层中具有三个节点的神经网络输出层。

输出层的每个节点都必须使用 sigmoid 激活。这将预测标签的类成员概率，一个介于 0 和 1 之间的值。最后，模型必须符合[二元交叉熵损失函数](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)。

总之，要配置用于多标签分类的神经网络模型，具体如下:

*   输出图层中的节点数与标签数相匹配。
*   输出层每个节点的 Sigmoid 激活。
*   二元交叉熵损失函数

我们可以使用 Keras 深度学习库来演示这一点。

我们将为上一节中定义的多标签分类任务定义一个多层感知器(MLP)模型。

每个样本有 10 个输入和 3 个输出；因此，网络需要一个输入层，该输入层期望通过第一个隐藏层中的“ *input_dim* ”参数指定的 10 个输入和输出层中的三个节点。

我们将使用隐藏层中流行的 [ReLU 激活功能](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)。隐藏层有 20 个节点，是经过反复试验选择的。我们将使用二元交叉熵损失和随机梯度下降的亚当版本来拟合模型。

下面列出了多标签分类任务的网络定义。

```
# define the model
model = Sequential()
model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
```

您可能希望将此模型用于您自己的多标签分类任务；因此，我们可以创建一个函数来定义和返回模型，其中输入和输出变量的数量作为参数提供。

```
# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
```

现在我们已经熟悉了如何定义多标签分类的 MLP，让我们来探索如何评估这个模型。

## 用于多标签分类的神经网络

如果数据集很小，最好在同一数据集上重复评估神经网络模型，并报告重复的平均性能。

这是因为学习算法的随机性。

此外，在对新数据进行预测时，最好使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)代替数据集的训练/测试分割，以获得模型性能的无偏估计。同样，只有在没有太多数据的情况下，这个过程才能在合理的时间内完成。

考虑到这一点，我们将使用 10 倍和 3 倍重复的重复 k 倍交叉验证来评估多输出回归任务的 MLP 模型。

默认情况下，MLP 模型将预测每个类别标签的概率。这意味着它将为每个样本预测三种概率。通过将这些值舍入为 0 或 1，可以将其转换为清晰的类标签。然后，我们可以计算清晰类别标签的分类精度。

```
...
# make a prediction on the test set
yhat = model.predict(X_test)
# round probabilities to class labels
yhat = yhat.round()
# calculate accuracy
acc = accuracy_score(y_test, yhat)
```

收集分数，并通过报告所有重复和交叉验证折叠的平均值和标准偏差进行汇总。

下面的 *evaluate_model()* 函数获取数据集，对模型进行评估，并返回评估分数列表，在本例中是准确性分数。

```
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
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results
```

然后，我们可以加载数据集，评估模型，并报告平均性能。

将这些联系在一起，完整的示例如下所示。

```
# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
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
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results

# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
```

运行该示例会报告每个折叠和每个重复的分类准确性，从而给出评估进度的想法。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

最后，报告了平均值和标准偏差的准确性。在这种情况下，模型显示达到了大约 81.2%的精度。

您可以使用此代码作为模板，在自己的多标签分类任务中评估 MLP 模型。模型中节点和图层的数量可以根据数据集的复杂性轻松调整和定制。

```
...
>0.780
>0.820
>0.790
>0.810
>0.840
Accuracy: 0.812 (0.032)
```

一旦选择了模型配置，我们就可以使用它来拟合所有可用数据的最终模型，并对新数据进行预测。

下面的示例演示了这一点，首先在整个多标签分类数据集上拟合 MLP 模型，然后在保存的模型上调用 *predict()* 函数，以便对新的数据行进行预测。

```
# use mlp for prediction on multi-label classification
from numpy import asarray
from sklearn.datasets import make_multilabel_classification
from keras.models import Sequential
from keras.layers import Dense

# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# load dataset
X, y = get_dataset()
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=100)
# make a prediction for new data
row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])
```

运行该示例符合模型，并对新行进行预测。正如预期的那样，预测包含多标签分类任务所需的三个输出变量:每个类别标签的概率。

```
Predicted: [0.9998627 0.9849341 0.00208042]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [多标签分类，维基百科](https://en.wikipedia.org/wiki/Multi-label_classification)。
*   [sklearn . datasets . make _ multi label _ classification API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html)。
*   [硬主页](https://keras.io/) 。
*   [sklearn.model_selection。重复的策略应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)。

## 摘要

在本教程中，您发现了如何为多标签分类开发深度学习模型。

具体来说，您了解到:

*   多标签分类是一项预测建模任务，涉及预测零个或多个互斥的类别标签。
*   可以为多标签分类任务配置神经网络模型。
*   如何评价多标签分类的神经网络并对新数据进行预测？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。