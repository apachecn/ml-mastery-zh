# 如何使用 scikit-learn 做出预测

> 原文： [https://machinelearningmastery.com/make-predictions-scikit-learn/](https://machinelearningmastery.com/make-predictions-scikit-learn/)

#### 如何使用 Python 中的 scikit-learn 模型预测分类或回归结果
。

一旦您在 scikit-learn 中选择并适合最终的机器学习模型，您就可以使用它来对新数据实例做出预测。

初学者对如何做到这一点有一些困惑。我经常看到以下问题：

如何在 scikit-learn 中使用我的模型做出预测？

在本教程中，您将了解如何使用 scikit-learn Python 库中的最终机器学习模型进行分类和回归预测。

完成本教程后，您将了解：

*   如何最终确定模型以便为预测做好准备。
*   如何在 scikit-learn 中进行类和概率预测。
*   如何在 scikit-learn 中进行回归预测。

让我们开始吧。

![Gentle Introduction to Vector Norms in Machine Learning](img/e4a4a58243578310240b1d079ff99795.jpg)

机器学习中向量规范的温和介绍
Cosimo 的照片，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  首先完成您的模型
2.  如何用分类模型预测
3.  如何用回归模型预测

## 1.首先完成您的模型

在做出预测之前，必须训练最终模型。

您可能使用 k 折交叉验证或训练/测试分割数据来训练模型。这样做是为了让您估计模型对样本外数据的技能，例如：新数据。

这些模型已达到目的，现在可以丢弃。

您现在必须在所有可用数据上训练最终模型。

您可以在此处了解有关如何训练最终模型的更多信息：

*   [如何训练最终机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)

## 2.如何用分类模型预测

分类问题是模型学习输入要素和作为标签的输出要素之间的映射的问题，例如“_ 垃圾邮件 _”和“_ 不是垃圾邮件 _”。

下面是针对简单二分类问题的最终 [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 模型的示例代码。

虽然我们在本教程中使用`LogisticRegression`，但在 scikit-learn 中几乎所有的分类算法都可以使用相同的函数。

```
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
```

完成模型后，您可能希望将模型保存到文件，例如通过泡菜。保存后，您可以随时加载模型并使用它做出预测。有关此示例，请参阅帖子：

*   [用 scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/) 在 Python 中保存和加载机器学习模型

为简单起见，我们将跳过本教程中的示例。

我们可能希望使用最终模型进行两种类型的分类预测;它们是阶级预测和概率预测。

### 阶级预测

类预测是：给定最终模型和一个或多个数据实例，预测数据实例的类。

我们不知道新数据的结果类。这就是我们首先需要模型的原因。

我们可以使用`predict()`函数在 scikit-learn 中使用我们最终的分类模型来预测新数据实例的类。

例如，我们在名为`Xnew`的数组中有一个或多个数据实例。这可以传递给我们模型上的`predict()`函数，以预测数组中每个实例的类值。

```
Xnew = [[...], [...]]
ynew = model.predict(Xnew)
```

### 多类预测

让我们通过一次预测多个数据实例的示例来具体化。

```
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行该示例预测三个新数据实例的类，然后将数据和预测一起打印。

```
X=[-0.79415228  2.10495117], Predicted=0
X=[-8.25290074 -4.71455545], Predicted=1
X=[-2.18773166  3.33352125], Predicted=0
```

### 单一类预测

如果您只有一个新数据实例，则可以将此数据包装为`predict()`函数;例如：

```
# example of making a single class prediction
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# define one new instance
Xnew = [[-0.79415228, 2.10495117]]
# make a prediction
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

运行该示例将打印单个实例和预测类。

```
X=[-0.79415228, 2.10495117], Predicted=0
```

### 关于类标签的注释

准备好数据后，您将把域中的类值（例如字符串）映射到整数值。您可能使用过 [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) 。

此`LabelEncoder`可用于通过`inverse_transform()`函数将整数转换回字符串值。

因此，您可能希望在拟合最终模型时保存（pickle）用于编码 y 值的`LabelEncoder`。

### 概率预测

您可能希望进行的另一种类型的预测是数据实例属于每个类的概率。

这被称为概率预测，其中给定新实例，模型将每个结果类的概率返回为 0 和 1 之间的值。

您可以通过调用`predict_proba()`函数在 scikit-learn 中进行这些类型的预测，例如：

```
Xnew = [[...], [...]]
ynew = model.predict_proba(Xnew)
```

此功能仅适用于能够进行概率预测的分类模型，这是大多数但不是所有模型。

以下示例对数据实例的`Xnew`数组中的每个示例进行概率预测。

```
# example of making multiple probability predictions
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行实例进行概率预测，然后打印输入数据实例以及每个实例属于第一和第二类（0 和 1）的概率。

```
X=[-0.79415228 2.10495117], Predicted=[0.94556472 0.05443528]
X=[-8.25290074 -4.71455545], Predicted=[3.60980873e-04 9.99639019e-01]
X=[-2.18773166 3.33352125], Predicted=[0.98437415 0.01562585]
```

如果您想向用户提供专家解释的概率，这在您的应用程序中会有所帮助。

## 3.如何用回归模型预测

回归是一种监督学习问题，在给定输入示例的情况下，模型学习映射到合适的输出量，例如“0.1”和“0.2”等。

下面是最终的`LinearRegression`模型的示例。同样，用于进行回归预测的函数适用于 scikit-learn 中可用的所有回归模型。

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
# fit final model
model = LinearRegression()
model.fit(X, y)
```

我们可以通过在最终模型上调用`predict()`函数来使用最终的回归模型预测数量。

与分类一样，predict（）函数采用一个或多个数据实例的列表或数组。

### 多元回归预测

下面的示例演示了如何对具有未知预期结果的多个数据实例进行回归预测。

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

运行该示例会进行多次预测，然后并排打印输入和预测以供审阅。

```
X=[-1.07296862 -0.52817175], Predicted=-61.32459258381131
X=[-0.61175641 1.62434536], Predicted=-30.922508147981667
X=[-2.3015387 0.86540763], Predicted=-127.34448527071137
```

### 单回归预测

可以使用相同的函数来对单个数据实例做出预测，只要它适当地包装在周围的列表或数组中即可。

例如：

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# define one new data instance
Xnew = [[-1.07296862, -0.52817175]]
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

运行该示例进行单个预测并打印数据实例和预测以供审阅。

```
X=[-1.07296862, -0.52817175], Predicted=-77.17947088762787
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [如何训练最终机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)
*   [用 scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/) 在 Python 中保存和加载机器学习模型
*   [scikit-learn API 参考](http://scikit-learn.org/stable/modules/classes.html)

### 摘要

在本教程中，您了解了如何使用 scikit-learn Python 库中的最终机器学习模型进行分类和回归预测。

具体来说，你学到了：

*   如何最终确定模型以便为预测做好准备。
*   如何在 scikit-learn 中进行类和概率预测。
*   如何在 scikit-learn 中进行回归预测。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。