# 如何将模型输入数据与机器学习的预测联系起来

> 原文：<https://machinelearningmastery.com/how-to-connect-model-input-data-with-predictions-for-machine-learning/>

最后更新于 2020 年 8 月 19 日

如今，使用 scikit-learn 这样的库，将模型拟合到训练数据集非常容易。

只需几行代码，就可以在数据集上拟合和评估模型。太容易了，都成问题了。

同样的几行代码一遍又一遍地重复，实际上如何使用模型进行预测可能并不明显。或者，如果进行了预测，如何将预测值与实际输入值联系起来。

我知道情况是这样的，因为我收到了许多邮件，其中有这样一个问题:

> *如何将预测值与输入数据联系起来？*

这是一个常见的问题。

在本教程中，您将发现如何将预测值与机器学习模型的输入相关联。

完成本教程后，您将知道:

*   如何在训练数据集上拟合和评估模型？
*   如何使用拟合模型一次一个批量地进行预测？
*   如何将预测值与模型的输入联系起来？

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 scikit-learn v0.22 API 的变化进行了更新。

![How to Connect Model Input Data With Predictions for Machine Learning](img/e8acc6b3f758c6c5c194e7b0bf251b3a.png)

如何将模型输入数据与机器学习预测联系起来
图片由[伊恩·基廷](https://www.flickr.com/photos/ian-arlett/30798942798/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  准备训练数据集
2.  如何在训练数据集中拟合模型
3.  如何将预测与模型输入联系起来

## 准备训练数据集

让我们从定义一个可以在模型中使用的数据集开始。

您可能在 CSV 文件或内存中的 NumPy 数组中有自己的数据集。

在这种情况下，我们将使用带有两个数字输入变量的简单两类或二进制分类问题。

*   **输入**:两个数值输入变量:
*   **输出**:类别标签为 0 或 1。

我们可以使用 [make_blobs() scikit-learn 函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)创建这个包含 1000 个示例的数据集。

下面的示例为输入( *X* )和输出( *y* )创建了具有独立数组的数据集。

```py
# example of creating a test dataset
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# summarize the shape of the arrays
print(X.shape, y.shape)
```

运行该示例将创建数据集并打印每个数组的形状。

我们可以看到数据集中的 1000 个样本有 1000 行。我们还可以看到，对于两个输入变量，输入数据有两列，输出数组是输入数据中每一行的一个长的类标签数组。

```py
(1000, 2) (1000,)
```

接下来，我们将在这个训练数据集上拟合一个模型。

## 如何在训练数据集中拟合模型

现在我们有了一个训练数据集，我们可以在数据上拟合一个模型。

这意味着我们将向学习算法提供所有的训练数据，并让学习算法发现输入和输出类标签之间的映射，从而最小化预测误差。

在这种情况下，由于是两类问题，我们将尝试逻辑回归分类算法。

这可以通过 scikit-learn 中的[物流配送类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)来实现。

首先，必须用我们需要的任何特定配置来定义模型。在这种情况下，我们将使用高效的“ *lbfgs* ”求解器。

接下来，通过调用 *fit()* 函数并传入训练数据集，在训练数据集上拟合模型。

最后，我们可以对模型进行评估，首先使用它通过调用 *predict()* 对训练数据集进行预测，然后将预测与期望的类标签进行比较并计算精确率。

下面列出了完整的示例。

```py
# fit a logistic regression on the training dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# evaluate predictions
acc = accuracy_score(y, yhat)
print(acc)
```

运行该示例使模型适合训练数据集，然后打印分类精确率。

在这种情况下，我们可以看到该模型在训练数据集上具有 100%的分类精确率。

```py
1.0
```

既然我们知道了如何在训练数据集上拟合和评估模型，那么让我们来看看问题的根源。

*如何将模型的输入连接到输出？*

## 如何将预测与模型输入联系起来

合适的机器学习模型接受输入并进行预测。

这可以是一次一行数据；例如:

*   **输入** : 2.12309797 -1.41131072
*   **输出** : 1

这在我们的模型中很简单。

例如，我们可以用一个数组输入进行预测，得到一个输出，我们知道这两个是直接相连的。

输入必须定义为一个数字数组，特别是 1 行 2 列。我们可以通过将示例定义为一个行列表，每行有一个列列表来实现这一点；例如:

```py
...
# define input
new_input = [[2.12309797, -1.41131072]]
```

然后，我们可以将此作为输入提供给模型，并进行预测。

```py
...
# get prediction for new input
new_output = model.predict(new_input)
```

将这一点与前一节中的模型相结合，下面列出了完整的示例。

```py
# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# define input
new_input = [[2.12309797, -1.41131072]]
# get prediction for new input
new_output = model.predict(new_input)
# summarize input and output
print(new_input, new_output)
```

运行该示例定义新输入并进行预测，然后打印输入和输出。

我们可以看到，在这种情况下，模型为输入预测类标签 1。

```py
[[2.12309797, -1.41131072]] [1]
```

如果我们在自己的应用程序中使用模型，模型的这种用法将允许我们直接关联每个预测的输入和输出。

如果我们需要将标签 0 和 1 替换为像“*垃圾邮件*”和“*非垃圾邮件*这样有意义的东西，我们可以用一个简单的 If 语句来完成。

目前为止一切顺利。

***用模型一次做多个预测会发生什么？**T3】*

也就是说，当同时向模型提供多行或多样本时，我们如何将预测与输入联系起来？

例如，我们可以对训练数据集中的 1，000 个示例中的每一个进行预测，就像我们在评估模型时上一节所做的那样。在这种情况下，模型将进行 1，000 次不同的预测，并返回 1，000 个整数值的数组。对 1，000 个输入数据行中的每一行进行一次预测。

重要的是，输出数组中预测的顺序与预测时作为模型输入提供的行的顺序相匹配。这意味着索引 0 处的输入行与索引 0 处的预测相匹配；指数 1、指数 2 也是如此，一直到指数 999。

因此，我们可以根据输入和输出的索引直接将它们联系起来，知道在对多行输入进行预测时会保留顺序。

让我们用一个例子来具体说明。

首先，我们可以对训练数据集中的每一行输入进行预测:

```py
...
# make predictions on the entire training dataset
yhat = model.predict(X)
```

然后，我们可以遍历索引，访问每个索引的输入和预测输出。

这精确地显示了如何将预测与输入行联系起来。例如，第 0 行的输入和索引 0 的预测:

```py
...
print(X[0], yhat[0])
```

在这种情况下，我们将只查看前 10 行及其预测。

```py
...
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
```

将这些联系在一起，下面列出了为训练数据中的每一行进行预测并将预测与输入连接起来的完整示例。

```py
# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions on the entire training dataset
yhat = model.predict(X)
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
```

运行该示例时，模型对训练数据集中的 1，000 行进行 1，000 次预测，然后将输入与前 10 个示例的预测值联系起来。

这提供了一个模板，您可以使用该模板并对自己的预测建模项目进行调整，以便通过行索引将预测连接到输入行。

```py
[ 1.23839154 -2.8475005 ] 1
[-1.25884111 -8.57055785] 0
[ -0.86599821 -10.50446358] 0
[ 0.59831673 -1.06451727] 1
[ 2.12309797 -1.41131072] 1
[-1.53722693 -9.61845366] 0
[ 0.92194131 -0.68709327] 1
[-1.31478732 -8.78528161] 0
[ 1.57989896 -1.462412  ] 1
[ 1.36989667 -1.3964704 ] 1
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 邮件

*   [你的第一个 Python 机器学习项目循序渐进](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
*   [如何使用 scikit 进行预测-学习](https://machinelearningmastery.com/make-predictions-scikit-learn/)

### 蜜蜂

*   [sklearn . dataset . make _ blobs API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
*   [sklearn . metrics . accuracy _ score API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
*   [sklearn.linear_model。物流配送应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## 摘要

在本教程中，您发现了如何将预测值与机器学习模型的输入相关联。

具体来说，您了解到:

*   如何在训练数据集上拟合和评估模型？
*   如何使用拟合模型一次一个批量地进行预测？
*   如何将预测值与模型的输入联系起来？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。