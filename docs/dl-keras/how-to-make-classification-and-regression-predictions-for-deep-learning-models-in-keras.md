# 如何用 Keras 进行预测

> 原文： [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)

一旦您在 Keras 中选择并适合最终的深度学习模型，您就可以使用它来对新数据实例进行预测。

初学者对如何做到这一点有一些困惑。我经常看到以下问题：

> 如何在 Keras 中使用我的模型进行预测？

在本教程中，您将了解如何使用 Keras Python 库通过最终的深度学习模型进行分类和回归预测。

完成本教程后，您将了解：

*   如何最终确定模型以便为预测做好准备。
*   如何对 Keras 中的分类问题进行类和概率预测。
*   如何在 Keras 中进行回归预测。

让我们开始吧。

![How to Make Classification and Regression Predictions for Deep Learning Models in Keras](img/f0f15aa8316dd7ee6c7b1361b8727f8a.png)

如何对 Keras 中的深度学习模型进行分类和回归预测
[mstk east](https://www.flickr.com/photos/120248737@N03/16306796118/) 的照片，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  完成模型
2.  分类预测
3.  回归预测

## 1.完成模型

在进行预测之前，必须训练最终模型。

您可能使用 k 折交叉验证或训练/测试分割数据来训练模型。这样做是为了让您估算模型中样本数据的技能，例如：新数据。

这些模型已达到目的，现在可以丢弃。

您现在必须在所有可用数据上训练最终模型。您可以在此处了解有关如何训练最终模型的更多信息：

*   [如何训练最终机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)

## 2.分类预测

分类问题是模型学习输入要素和作为标签的输出要素之间的映射的问题，例如“_ 垃圾邮件 _”和“_ 不垃圾邮件 _”。

下面是 Keras 中针对简单的两类（二元）分类问题开发的最终神经网络模型的示例。

如果在 Keras 中开发神经网络模型对您来说是新的，请参阅帖子：

*   [用 Keras 逐步开发 Python 中的第一个神经网络](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

```py
# example of training a final classification model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, verbose=0)
```

完成后，您可能希望将模型保存到文件，例如通过 Keras API。保存后，您可以随时加载模型并使用它进行预测。有关此示例，请参阅帖子：

*   [保存并加载您的 Keras 深度学习模型](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)

为简单起见，我们将跳过本教程中的示例。

我们可能希望使用最终模型进行两种类型的分类预测;它们是阶级预测和概率预测。

### 阶级预测

给出最终模型和一个或多个数据实例的类预测，预测数据实例的类。

我们不知道新数据的结果类。这就是我们首先需要模型的原因。

我们可以使用 _predict_classes（）_ 函数在 Keras 中使用我们最终的分类模型来预测新数据实例的类。请注意，此功能仅适用于 _Sequential_ 型号，而不适用于使用功能 API 开发的型号。

例如，我们在名为 _Xnew_ 的数组中有一个或多个数据实例。这可以传递给我们模型上的 _predict_classes（）_ 函数，以便预测数组中每个实例的类值。

```py
Xnew = [[...], [...]]
ynew = model.predict_classes(Xnew)
```

让我们通过一个例子来具体化：

```py
# example making new class predictions for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=500, verbose=0)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
# make a prediction
ynew = model.predict_classes(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行该示例预测三个新数据实例的类，然后将数据和预测一起打印。

```py
X=[0.89337759 0.65864154], Predicted=[0]
X=[0.29097707 0.12978982], Predicted=[1]
X=[0.78082614 0.75391697], Predicted=[0]
```

如果您只有一个新的数据实例，则可以将其作为包含在数组中的实例提供给 _predict_classes（）_ 函数;例如：

```py
# example making new class prediction for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from numpy import array
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=500, verbose=0)
# new instance where we do not know the answer
Xnew = array([[0.89337759, 0.65864154]])
# make a prediction
ynew = model.predict_classes(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

运行该示例将打印单个实例和预测类。

```py
X=[0.89337759 0.65864154], Predicted=[0]
```

### 关于类标签的注释

请注意，在准备数据时，您将把域中的类值（例如字符串）映射到整数值。您可能使用过 _[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder)_ 。

此 _LabelEncoder_ 可用于通过 _inverse_transform（）_ 函数将整数转换回字符串值。

因此，您可能希望在拟合最终模型时保存（pickle）用于编码 _y_ 值的 _LabelEncoder_ 。

### 概率预测

您可能希望进行的另一种类型的预测是数据实例属于每个类的概率。

这被称为概率预测，其中，给定新实例，模型将每个结果类的概率返回为 0 和 1 之间的值。

您可以通过调用 _predict_proba（）_ 函数在 Keras 中进行这些类型的预测;例如：

```py
Xnew = [[...], [...]]
ynew = model.predict_proba(Xnew)
```

在两类（二进制）分类问题的情况下，通常在输出层中使用 S 形激活函数。预测概率被视为观察属于类 1 的可能性，或被反转（1-概率）以给出类 0 的概率。

在多类分类问题的情况下，通常在输出层上使用 softmax 激活函数，并且将每个类的观察的可能性作为向量返回。

以下示例对数据实例的 _Xnew_ 数组中的每个示例进行概率预测。

```py
# example making new probability predictions for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=500, verbose=0)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行实例会进行概率预测，然后打印输入数据实例以及属于类 1 的每个实例的概率。

```py
X=[0.89337759 0.65864154], Predicted=[0.0087348]
X=[0.29097707 0.12978982], Predicted=[0.82020265]
X=[0.78082614 0.75391697], Predicted=[0.00693122]
```

如果您想向用户提供专家解释的概率，这在您的应用程序中会有所帮助。

## 3.回归预测

回归是一种监督学习问题，在给定输入示例的情况下，模型学习到适当输出量的映射，例如“0.1”和“0.2”等。

下面是用于回归的最终 Keras 模型的示例。

```py
# example of training a final regression model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(100,1))
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)
```

我们可以通过在最终模型上调用 _predict（）_ 函数来使用最终的回归模型预测数量。

_predict（）_ 函数采用一个或多个数据实例的数组。

下面的示例演示了如何对具有未知预期结果的多个数据实例进行回归预测。

```py
# example of making predictions for a regression problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(100,1))
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)
# new instances where we do not know the answer
Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
Xnew = scalarX.transform(Xnew)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行该示例会进行多次预测，然后并排打印输入和预测以供审阅。

```py
X=[0.29466096 0.30317302], Predicted=[0.17097184]
X=[0.39445118 0.79390858], Predicted=[0.7475489]
X=[0.02884127 0.6208843 ], Predicted=[0.43370453]
```

可以使用相同的函数来对单个数据实例进行预测，只要它适当地包装在周围的列表或数组中即可。

例如：

```py
# example of making predictions for a regression problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from numpy import array
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(100,1))
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)
# new instance where we do not know the answer
Xnew = array([[0.29466096, 0.30317302]])
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

运行该示例进行单个预测并打印数据实例和预测以供审阅。

```py
X=[0.29466096 0.30317302], Predicted=[0.17333156]
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [如何训练最终机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)
*   [保存并加载您的 Keras 深度学习模型](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
*   [用 Keras 逐步开发 Python 中的第一个神经网络](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
*   [Keras 中长期短期记忆模型的 5 步生命周期](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)
*   [如何用 Keras 中的长短期记忆模型进行预测](https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/)

## 摘要

在本教程中，您了解了如何使用 Keras Python 库通过最终的深度学习模型进行分类和回归预测。

具体来说，你学到了：

*   如何最终确定模型以便为预测做好准备。
*   如何对 Keras 中的分类问题进行类和概率预测。
*   如何在 Keras 中进行回归预测。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。