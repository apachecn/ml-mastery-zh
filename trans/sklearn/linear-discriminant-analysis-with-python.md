# Python 线性判别分析

> 原文:[https://machinelearning master . com/linear-判别分析-with-python/](https://machinelearningmastery.com/linear-discriminant-analysis-with-python/)

**线性判别分析**是一种线性分类机器学习算法。

该算法包括基于每个输入变量的观测值的特定分布，为每个类开发一个概率模型。然后，通过计算一个新例子属于每个类别的条件概率并选择具有最高概率的类别来对其进行分类。

因此，这是一个相对简单的概率分类模型，它对每个输入变量的分布做出了强有力的假设，尽管它可以做出有效的预测，即使这些预期被违反(例如，它优雅地失败了)。

在本教程中，您将发现 Python 中的线性判别分析分类机器学习算法。

完成本教程后，您将知道:

*   线性判别分析是一种简单的线性机器学习分类算法。
*   如何使用 Scikit-Learn 的线性判别分析模型进行拟合、评估和预测。
*   如何在给定数据集上调整线性判别分析算法的超参数。

我们开始吧。

![Linear Discriminant Analysis With Python](img/d7bd67842587e2a47a3a4405df42191c.png)

用 Python 进行线性判别分析
图片由[密海 Lucîț](https://flickr.com/photos/revoltatul/27752939787/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  线性判别分析
2.  基于 scikit 学习的线性判别分析
3.  调整线性判别分析超参数

## 线性判别分析

线性判别分析，简称 LDA，是一种分类机器学习算法。

它的工作原理是按类别标签计算输入要素的汇总统计数据，如平均值和标准偏差。这些统计数据表示从训练数据中学习到的模型。在实践中，线性代数运算被用来通过矩阵分解有效地计算所需的量。

根据每个输入要素的值，通过估计新示例属于每个类别标签的概率来进行预测。然后将产生最大概率的类分配给该示例。因此，LDA 可以被认为是[贝叶斯定理](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)在分类中的简单应用。

线性判别分析假设输入变量是数值型的正态分布，并且它们具有相同的方差(分布)。如果不是这样，可能需要将数据转换为高斯分布，并在建模之前对数据进行标准化或规范化。

> LDA 分类器是假设每个类别内的观测值来自具有特定类别均值向量和共同方差的正态分布而产生的

—第 142 页，[R](https://amzn.to/2xW4hPy)中应用的统计学习介绍，2014。

它还假设输入变量不相关；如果是，PCA 变换可能有助于消除线性相关性。

> ……从业者在使用 LDA 之前，应对数据进行特别严格的预处理。我们建议对预测值进行居中和缩放，并移除接近零的方差预测值。

—第 293 页，[应用预测建模](https://amzn.to/2wfqnw0)，2013 年。

然而，该模型可以表现良好，即使违反了这些期望。

LDA 模型自然是多类的。这意味着它支持两类分类问题，并且可以扩展到两类以上(多类分类)，而无需修改或增加。

它是一种线性分类算法，就像逻辑回归一样。这意味着类在特征空间中由线或超平面分隔。该方法的扩展可以用于允许其他形状，如二次判别分析(QDA)，它允许在决策边界弯曲的形状。

> ……与 LDA 不同，QDA 假设每个类都有自己的协方差矩阵。

—第 149 页，[R](https://amzn.to/2xW4hPy)中应用的统计学习介绍，2014。

现在我们已经熟悉了 LDA，让我们看看如何使用 scikit-learn 库来拟合和评估模型。

## 基于 scikit 学习的线性判别分析

线性判别分析可通过[线性判别分析类](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)在 scikit-learn Python 机器学习库中获得。

该方法可以在没有配置的情况下直接使用，尽管该实现确实提供了自定义参数，例如选择求解器和使用惩罚。

```py
...
# create the lda model
model = LinearDiscriminantAnalysis()
```

我们可以用一个实例来说明线性判别分析方法。

首先，让我们定义一个综合分类数据集。

我们将使用 [make_classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 1000 个示例的数据集，每个示例有 10 个输入变量。

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

我们可以通过[重复分层 k 重交叉验证类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)来拟合和评估线性判别分析模型。我们将在测试装具中使用 10 次折叠和三次重复。

下面列出了评估综合二元分类任务的线性判别分析模型的完整示例。

```py
# evaluate a lda model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估合成数据集上的线性判别分析算法，并报告 10 倍交叉验证的三次重复的平均准确性。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型实现了大约 89.3%的平均精度。

```py
Mean Accuracy: 0.893 (0.033)
```

我们可能会决定使用线性判别分析作为我们的最终模型，并对新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用 predict()函数传入新的数据行来实现。

我们可以用下面列出的完整示例来演示这一点。

```py
# make a prediction with a lda model on the dataset
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
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

## 调整线性判别分析超参数

必须为特定数据集配置线性判别分析方法的超参数。

一个重要的超参数是解算器，它默认为“ *svd* ”，但也可以设置为支持收缩能力的解算器的其他值。

下面的示例使用带有不同解算器值的网格的[网格分类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)来演示这一点。

```py
# grid search solver for lda
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到，与其他内置解算器相比，默认的 SVD 解算器的性能最好。

```py
Mean Accuracy: 0.893
Config: {'solver': 'svd'}
```

接下来，我们可以探索在模型中使用收缩是否会提高性能。

收缩给模型增加了一种惩罚，作为一种正则化，降低了模型的复杂性。

> 正则化以潜在增加的偏差为代价，减少了与基于样本的估计相关的方差。这种偏差方差权衡通常由一个或多个(置信程度)参数调节，这些参数控制朝向“似是而非”的一组(总体)参数值的偏差强度。

——[正则化判别分析](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752)，1989。

这可以通过“*收缩*”参数设置，并且可以设置为 0 到 1 之间的值。我们将在间距为 0.01 的网格上测试值。

为了使用惩罚，必须选择支持此功能的解算器，例如“*特征值*或“ *lsqr* ”。在这种情况下，我们将使用后者。

下面列出了调整收缩超参数的完整示例。

```py
# grid search shrinkage for lda
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis(solver='lsqr')
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['shrinkage'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到使用收缩将性能从大约 89.3%略微提升到大约 89.4%，值为 0.02。

```py
Mean Accuracy: 0.894
Config: {'shrinkage': 0.02}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习的线性判别分析](https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/)

### 报纸

*   [正则化判别分析](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752)，1989。

### 书

*   [应用预测建模](https://amzn.to/2wfqnw0)，2013。
*   [R](https://amzn.to/2xW4hPy)中应用的统计学习导论，2014。

### 蜜蜂

*   [sklearn . discriminal _ analysis。线性判别分析应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)。
*   [线性和二次判别分析，scikit-learn](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda) 。

### 文章

*   [线性判别分析，维基百科](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)。

## 摘要

在本教程中，您发现了 Python 中的线性判别分析分类机器学习算法。

具体来说，您了解到:

*   线性判别分析是一种简单的线性机器学习分类算法。
*   如何使用 Scikit-Learn 的线性判别分析模型进行拟合、评估和预测。
*   如何在给定数据集上调整线性判别分析算法的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。