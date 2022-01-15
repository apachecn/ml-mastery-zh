# 用 Python 进行分类的高斯过程

> 原文：<https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/>

**高斯过程分类器**是一种分类机器学习算法。

高斯过程是高斯概率分布的推广，可以用作复杂的非参数机器学习算法的基础，用于分类和回归。

它们是一种内核模型，就像支持向量机一样，与支持向量机不同，它们能够预测高度校准的类成员概率，尽管作为方法核心的内核的选择和配置可能具有挑战性。

在本教程中，您将发现高斯过程分类器分类机器学习算法。

完成本教程后，您将知道:

*   高斯过程分类器是一种非参数算法，可以应用于二进制分类任务。
*   如何使用带有 Scikit-Learn 的高斯过程分类器模型进行拟合、评估和预测。
*   如何在给定数据集上调整高斯过程分类器算法的超参数。

我们开始吧。

![Gaussian Processes for Classification With Python](img/d317c457b25ddcec7140199830554cd1.png)

用 Python 进行分类的高斯过程
图片由[马克考](https://flickr.com/photos/67415843@N05/37583657721/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  高斯分类过程
2.  用 Scikit 学习高斯过程
3.  调整高斯过程超参数

## 高斯分类过程

高斯过程，简称 GP，是[高斯概率分布](https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/)的推广(例如钟形函数)。

高斯概率分布函数总结了随机变量的分布，而高斯过程总结了函数的性质，例如函数的参数。因此，您可以将高斯过程视为高斯函数之上的一个抽象层次或间接层次。

> 高斯过程是高斯概率分布的推广。概率分布描述的是标量或向量的随机变量(对于多元分布)，而随机过程控制函数的性质。

—第 2 页，[机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。

高斯过程可以用作分类预测建模的机器学习算法。

高斯过程是一种核方法，像支持向量机一样，虽然它们能够预测高度校准的概率，不像支持向量机。

高斯过程需要指定一个内核来控制示例如何相互关联；具体来说，它定义了数据的协方差函数。这被称为潜在功能或“T0”骚扰功能。

> 潜在函数 f 扮演了一个讨厌函数的角色:我们不观察 f 本身的值(我们只观察输入 X 和类标签 y)，我们对 f 的值也不特别感兴趣…

—第 40 页，[机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。

使用内核对示例进行分组的方式控制模型“*如何感知*”示例，假设它假设“*彼此接近的*示例具有相同的类标签。

因此，测试模型的不同内核函数和复杂内核函数的不同配置都很重要。

> …协方差函数是高斯过程预测器的关键要素，因为它编码了我们对希望学习的函数的假设。

—第 79 页，[机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。

它还需要一个链接函数来解释内部表示，并预测类成员的概率。可以使用逻辑函数，允许对用于二进制分类的[二项式概率分布](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)进行建模。

> 对于二元判别情况，一个简单的想法是使用响应函数(链接函数的逆函数)将回归模型的输出转化为类概率，该函数将位于域(inf，inf)中的自变量“挤压”到范围[0，1]内，从而保证有效的概率解释。

—第 35 页，[机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。

高斯过程和高斯过程的分类是一个复杂的话题。

要了解更多信息，请参阅文本:

*   [机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。

## 用 Scikit 学习高斯过程

高斯过程分类器可通过[高斯过程分类器类](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)在 scikit-learn Python 机器学习库中获得。

该类允许您通过“*内核*”参数指定要使用的内核，并默认为 1 *径向基函数(1.0)，例如径向基函数内核。

```py
...
# define model
model = GaussianProcessClassifier(kernel=1*RBF(1.0))
```

假设指定了内核，模型将尝试为训练数据集最佳地配置内核。

这是通过设置“*优化器*”来控制的，通过“ *max_iter_predict* 来控制优化器的迭代次数，以及为了克服局部最优而执行的优化过程的重复次数”*n _ restaults _ optimizer*。

默认情况下，执行单次优化运行，这可以通过将“*优化*”设置为*无*来关闭。

```py
...
# define model
model = GaussianProcessClassifier(optimizer=None)
```

我们可以用一个工作示例来演示高斯过程分类器。

首先，让我们定义一个综合分类数据集。

我们将使用 [make_classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 100 个示例的数据集，每个示例有 20 个输入变量。

下面的示例创建并汇总了数据集。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认数据集的行数和列数。

```py
(100, 20) (100,)
```

我们可以通过[repeated stratifiedfold 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)使用重复的分层 k 重交叉验证来拟合和评估高斯过程分类器模型。我们将在测试装具中使用 10 次折叠和三次重复。

我们将使用默认配置。

```py
...
# create the model
model = GaussianProcessClassifier()
```

下面列出了评估合成二进制分类任务的高斯过程分类器模型的完整示例。

```py
# evaluate a gaussian process classifier model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = GaussianProcessClassifier()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估合成数据集上的高斯过程分类器算法，并报告 10 倍交叉验证的三次重复的平均准确性。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型实现了大约 79.0%的平均精确率。

```py
Mean Accuracy: 0.790 (0.101)
```

我们可能会决定使用高斯过程分类器作为最终模型，并对新数据进行预测。

这可以通过在所有可用数据上拟合模型管道并调用传递新数据行的 *predict()* 函数来实现。

我们可以用下面列出的完整示例来演示这一点。

```py
# make a prediction with a gaussian process classifier model on the dataset
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = GaussianProcessClassifier()
# fit model
model.fit(X, y)
# define new data
row = [2.47475454,0.40165523,1.68081787,2.88940715,0.91704519,-3.07950644,4.39961206,0.72464273,-4.86563631,-6.06338084,-1.22209949,-0.4699618,1.01222748,-0.6899355,-0.53000581,6.86966784,-3.27211075,-6.59044146,-2.21290585,-3.139579]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat)
```

运行该示例符合模型，并对新的数据行进行类别标签预测。

```py
Predicted Class: 0
```

接下来，我们可以看看配置模型超参数。

## 调整高斯过程超参数

必须为您的特定数据集配置高斯过程分类器方法的超参数。

也许最重要的超参数是通过“*内核*”参数控制的内核。scikit-learn 库提供了许多可以使用的内置内核。

也许一些更常见的例子包括:

*   肾血流量（renal blood flow 的缩写）
*   点产品
*   成熟的
*   有理二次型
*   WhiteKernel

您可以在这里了解更多关于该库提供的内核的信息:

*   [高斯过程内核，Scikit-Learn 用户指南](https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes)。

我们将使用默认参数来评估高斯过程分类器的性能。

```py
...
# define grid
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(), 1*RationalQuadratic(), 1*WhiteKernel()]
```

下面的例子使用 [GridSearchCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)和我们定义的值网格来演示这一点。

```py
# grid search kernel for gaussian process classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = GaussianProcessClassifier()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到*比率二次*内核以大约 91.3%的精确率提升了性能，而上一节中的径向基函数内核达到了 79.0%。

```py
Best Mean Accuracy: 0.913
Best Config: {'kernel': 1**2 * RationalQuadratic(alpha=1, length_scale=1)}
>0.790 with: {'kernel': 1**2 * RBF(length_scale=1)}
>0.800 with: {'kernel': 1**2 * DotProduct(sigma_0=1)}
>0.830 with: {'kernel': 1**2 * Matern(length_scale=1, nu=1.5)}
>0.913 with: {'kernel': 1**2 * RationalQuadratic(alpha=1, length_scale=1)}
>0.510 with: {'kernel': 1**2 * WhiteKernel(noise_level=1)}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [机器学习的高斯过程](https://amzn.to/3aY1nsu)，2006。
*   [机器学习的高斯过程，主页](http://www.gaussianprocess.org/gpml/)。
*   [机器学习:概率视角](https://amzn.to/2V8wc6Y)，2012。
*   [模式识别与机器学习](https://amzn.to/34qHQOW)，2006。

### 蜜蜂

*   [sklearn.gaussian_process。高斯过程分类器应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)。
*   [sklearn.gaussian_process。高斯处理器回归器应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)。
*   [高斯过程，Scikit-Learn 用户指南](https://scikit-learn.org/stable/modules/gaussian_process.html)。
*   [高斯过程核应用编程接口](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)。

### 文章

*   [高斯过程，维基百科](https://en.wikipedia.org/wiki/Gaussian_process)。

## 摘要

在本教程中，您发现了高斯过程分类器分类机器学习算法。

具体来说，您了解到:

*   高斯过程分类器是一种非参数算法，可以应用于二进制分类任务。
*   如何使用带有 Scikit-Learn 的高斯过程分类器模型进行拟合、评估和预测。
*   如何在给定数据集上调整高斯过程分类器算法的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。