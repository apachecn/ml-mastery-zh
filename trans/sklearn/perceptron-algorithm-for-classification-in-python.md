# Python 中分类的感知器算法

> 原文：<https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/>

**感知器**是一种用于二进制分类任务的线性机器学习算法。

它可以被认为是第一种也是最简单的人工神经网络。这绝对不是“深度”学习，而是一个重要的组成部分。

像逻辑回归一样，它可以在特征空间中快速学习两类分类任务的线性分离，尽管与逻辑回归不同，它使用随机梯度下降优化算法学习，并且不预测校准概率。

在本教程中，您将发现感知器分类机器学习算法。

完成本教程后，您将知道:

*   感知器分类器是一种线性算法，可以应用于二进制分类任务。
*   如何使用带有 Scikit-Learn 的感知器模型进行拟合、评估和预测。
*   如何在给定的数据集上调整感知器算法的超参数。

我们开始吧。

![Perceptron Algorithm for Classification in Python](img/64910bd2c828ab5329b064d3805c7f5a.png)

Python 中分类的感知器算法
图片由[贝琳达·诺维卡](https://flickr.com/photos/bnovika/34068980060/)提供，保留部分权利。

## 教程概述

本教程分为 3 =三个部分；它们是:

1.  感知器算法
2.  带有 Scikit 的感知器-学习
3.  调整感知器超参数

## 感知器算法

[感知器算法](https://en.wikipedia.org/wiki/Perceptron)是一种两类(二进制)分类机器学习算法。

这是一种神经网络模型，也许是最简单的神经网络模型。

它由单个节点或神经元组成，以一行数据作为输入，并预测一个类标签。这是通过计算输入和偏差(设置为 1)的加权和来实现的。模型输入的加权和称为激活。

*   **激活** =权重*输入+偏置

如果激活高于 0.0，模型将输出 1.0；否则，它将输出 0.0。

*   **预测 1** :如果激活> 0.0
*   **预测 0** :如果激活< = 0.0

假设输入乘以模型系数，如线性回归和逻辑回归，在使用模型之前对数据进行规范化或标准化是一个很好的做法。

感知器是一种线性分类算法。这意味着它学习一个决策边界，该边界使用特征空间中的一条线(称为超平面)分隔两个类。因此，它适用于那些类可以被线或线性模型很好地分开的问题，称为线性可分的。

模型的系数被称为输入权重，并使用随机梯度下降优化算法进行训练。

训练数据集中的示例一次一个地显示给模型，模型进行预测，并计算误差。然后更新模型的权重以减少示例的误差。这称为感知器更新规则。对训练数据集中的所有示例重复该过程，称为[时期](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)。然后，这个使用例子更新模型的过程会重复很多个时期。

每批使用一小部分误差更新模型权重，该比例由称为学习率的超参数控制，通常设置为小值。这是为了确保学习不会发生得太快，导致可能较低的技能模型，称为模型权重优化(搜索)过程的过早收敛。

*   权重(t + 1) =权重(t) +学习率*(预期 _i–预测 _) *输入 _ I

当模型产生的误差降至较低水平或不再改善，或者执行了最大数量的时期时，停止训练。

模型权重的初始值被设置为小的随机值。此外，训练数据集在每个训练时期之前被打乱。这是通过设计来加速和改进模型训练过程。因此，学习算法是随机的，每次运行时可能会获得不同的结果。因此，使用重复评估和报告平均分类精度来总结算法在数据集上的性能是一种很好的做法。

学习速率和训练时期的数量是算法的超参数，可以使用试探法或超参数调整来设置。

有关感知器算法的更多信息，请参见教程:

*   [如何在 Python 中从头实现感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)

现在我们已经熟悉了感知器算法，让我们探索如何在 Python 中使用该算法。

## 带有 Scikit 的感知器-学习

感知机算法可通过[感知机类](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)在 scikit-learn Python 机器学习库中获得。

该课程允许您配置学习率( *eta0* )，默认为 1.0。

```py
...
# define model
model = Perceptron(eta0=1.0)
```

该实现还允许您配置训练时期的总数( *max_iter* ，默认为 1，000。

```py
...
# define model
model = Perceptron(max_iter=1000)
```

感知器算法的 scikit-learn 实现还提供了您可能想要探索的其他配置选项，例如提前停止和使用惩罚损失。

我们可以用一个工作示例来演示感知器分类器。

首先，让我们定义一个综合分类数据集。

我们将使用 [make_classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 1000 个示例的数据集，每个示例有 20 个输入变量。

该示例创建并汇总数据集。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认数据集的行数和列数。

```py
(1000, 10) (1000,)
```

我们可以通过[repeated stratifiedfold 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)使用重复的分层 k 重交叉验证来拟合和评估感知器模型。我们将在测试装具中使用 10 次折叠和三次重复。

我们将使用默认配置。

```py
...
# create the model
model = Perceptron()
```

下面列出了评估用于合成二进制分类任务的感知器模型的完整示例。

```py
# evaluate a perceptron model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例在合成数据集上评估感知器算法，并报告 10 倍交叉验证的三次重复的平均准确性。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型达到了大约 84.7%的平均准确率。

```py
Mean Accuracy: 0.847 (0.052)
```

我们可能会决定使用感知器分类器作为最终模型，并对新数据进行预测。

这可以通过在所有可用数据上拟合模型管道并调用 predict()函数传入新的数据行来实现。

我们可以用下面列出的完整示例来演示这一点。

```py
# make a prediction with a perceptron model on the dataset
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron()
# fit model
model.fit(X, y)
# define new data
row = [0.12777556,-3.64400522,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,1.01001752,0.56768485]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat)
```

运行该示例符合模型，并对新的数据行进行类别标签预测。

```py
Predicted Class: 1
```

接下来，我们可以看看如何配置模型超参数。

## 调整感知器超参数

感知器算法的超参数必须针对您的特定数据集进行配置。

也许最重要的超参数是学习率。

较高的学习率可以使模型学习得更快，但代价可能是较低的技能。较小的学习速率可以产生性能更好的模型，但可能需要很长时间来训练模型。

您可以在教程中了解关于探索学习率的更多信息:

*   [训练深度学习神经网络时如何配置学习速率](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)

通常在一个小值(如 1e-4(或更小)和 1.0 之间的对数标度上测试学习速率。在这种情况下，我们将测试以下值:

```py
...
# define grid
grid = dict()
grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

下面的例子使用 [GridSearchCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)和我们定义的值网格来演示这一点。

```py
# grid search learning rate for the perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到，比默认值更小的学习速率导致更好的性能，学习速率 0.0001 和 0.001 都实现了大约 85.7%的分类准确率，而默认值 1.0 实现了大约 84.7%的准确率。

```py
Mean Accuracy: 0.857
Config: {'eta0': 0.0001}
>0.857 with: {'eta0': 0.0001}
>0.857 with: {'eta0': 0.001}
>0.853 with: {'eta0': 0.01}
>0.847 with: {'eta0': 0.1}
>0.847 with: {'eta0': 1.0}
```

另一个重要的超参数是使用多少个时期来训练模型。

这可能取决于训练数据集，并且可能有很大差异。同样，我们将在 1 到 1e+4 之间的对数标度上探索配置值。

```py
...
# define grid
grid = dict()
grid['max_iter'] = [1, 10, 100, 1000, 10000]
```

我们将使用上一次搜索中发现的 0.0001 的良好学习率。

```py
...
# define model
model = Perceptron(eta0=0.0001)
```

下面列出了网格搜索训练时期数量的完整示例。

```py
# grid search total epochs for the perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron(eta0=0.0001)
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['max_iter'] = [1, 10, 100, 1000, 10000]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到 10 到 10，000 个纪元导致了大约相同的分类精度。一个有趣的例外是探索同时配置学习速率和训练时期的数量，看看是否能取得更好的结果。

```py
Mean Accuracy: 0.857
Config: {'max_iter': 10}
>0.850 with: {'max_iter': 1}
>0.857 with: {'max_iter': 10}
>0.857 with: {'max_iter': 100}
>0.857 with: {'max_iter': 1000}
>0.857 with: {'max_iter': 10000}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何在 Python 中从头开始实现感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)
*   [了解学习率对神经网络性能的影响](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)
*   [训练深度学习神经网络时如何配置学习速率](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)

### 书

*   [用于模式识别的神经网络](https://amzn.to/2VoOp01)，1995。
*   [模式识别与机器学习](https://amzn.to/3a76mWm)，2006。
*   [人工智能:现代方法](https://amzn.to/3b6uwl9)，第 3 版，2015 年。

### 蜜蜂

*   [sklearn.linear_model。感知器 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) 。

### 文章

*   [感知，维基百科](https://en.wikipedia.org/wiki/Perceptron)。
*   [感知器(书)，维基百科](https://en.wikipedia.org/wiki/Perceptrons_(book))。

## 摘要

在本教程中，您发现了感知器分类机器学习算法。

具体来说，您了解到:

*   感知器分类器是一种线性算法，可以应用于二进制分类任务。
*   如何使用带有 Scikit-Learn 的感知器模型进行拟合、评估和预测。
*   如何在给定的数据集上调整感知器算法的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。