# 蟒蛇最近的缩小形心

> 原文:[https://machinelearning master . com/最近-缩小-质心-带-python/](https://machinelearningmastery.com/nearest-shrunken-centroids-with-python/)

最近质心是一种线性分类机器学习算法。

它包括根据训练数据集中的基于类的质心预测新示例的类标签。

**最近收缩质心**算法是一种扩展，它涉及将基于类的质心向整个训练数据集的质心移动，并移除那些在区分类时不太有用的输入变量。

因此，最近收缩质心算法执行自动形式的特征选择，使其适用于具有大量输入变量的数据集。

在本教程中，您将发现最近收缩质心分类机器学习算法。

完成本教程后，您将知道:

*   最近收缩质心是一种简单的线性机器学习分类算法。
*   如何使用 Scikit-Learn 的最近收缩质心模型进行拟合、评估和预测。
*   如何在给定数据集上调整最近收缩质心算法的超参数。

我们开始吧。

![Nearest Shrunken Centroids With Python](img/a0b81176fcb54f89f21a5b771d530f04.png)

Giuseppe Milo 拍摄的最近的蟒蛇缩小质心
照片，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  最近质心算法
2.  用 Scikit 学习最近的质心
3.  调谐最近质心超参数

## 最近质心算法

最近质心是一种分类机器学习算法。

该算法首先将训练数据集总结成一组质心(中心)，然后使用质心对新的例子进行预测。

> 对于每个类别，通过取训练集中每个预测器(每个类别)的平均值来找到数据的质心。使用来自所有类别的数据计算总质心。

—第 307 页，[应用预测建模](https://amzn.to/2wfqnw0)，2013 年。

A [质心](https://en.wikipedia.org/wiki/Centroid)是数据分布的几何中心，如平均值。在多个维度中，这将是每个维度的平均值，形成每个变量的分布中心点。

最近质心算法假设输入特征空间中的质心对于每个目标标签是不同的。训练数据通过类标签分成组，然后计算每组数据的质心。每个质心只是每个输入变量的平均值。如果有两类，则计算两个质心或点；三个类给出三个质心，以此类推。

质心代表“*模型*”给定新的示例，例如测试集中的示例或新数据，计算给定数据行与每个质心之间的距离，并使用最近的质心为示例分配类别标签。

[距离度量](https://machinelearningmastery.com/distance-measures-for-machine-learning/)，如欧几里德距离，用于数值数据或汉明距离用于分类数据，在这种情况下，最佳实践是在训练模型之前通过标准化或规范化来缩放输入变量。这是为了确保具有大值的输入变量不会主导距离计算。

分类的最近质心方法的扩展是将每个输入变量的质心向整个训练数据集的质心收缩。那些缩小到数据质心值的变量可以被移除，因为它们无助于区分类别标签。

因此，应用于质心的收缩量是一个超参数，可以针对数据集进行调整，并用于执行自动形式的特征选择。因此，它适用于具有大量输入变量的数据集，其中一些变量可能是不相关的或有噪声的。

> 因此，最近的收缩质心模型也在模型训练过程中进行特征选择。

—第 307 页，[应用预测建模](https://amzn.to/2wfqnw0)，2013 年。

这种方法被称为“*最近的收缩着丝粒*”，最早由[罗伯特·蒂比什拉尼](https://statweb.stanford.edu/~tibs/)等人在他们 2002 年发表的题为“[通过基因表达的收缩着丝粒](https://www.pnas.org/content/99/10/6567.short)诊断多种癌症类型”的论文中描述

## 用 Scikit 学习最近的质心

最近收缩质心可通过[最近质心类](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)在 scikit-learn Python 机器学习库中获得。

该类允许通过“*度量*”参数配置算法中使用的距离度量，对于欧几里德距离度量，该参数默认为“*欧几里德*”。

这可以更改为其他内置指标，如“*曼哈顿*”

```
...
# create the nearest centroid model
model = NearestCentroid(metric='euclidean')
```

默认情况下，不使用收缩，但是可以通过“*收缩 _ 阈值*参数指定收缩，该参数采用 0 到 1 之间的浮点值。

```
...
# create the nearest centroid model
model = NearestCentroid(metric='euclidean', shrink_threshold=0.5)
```

我们可以用一个工作示例来演示最近的收缩形心。

首先，让我们定义一个综合分类数据集。

我们将使用 [make_classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 1000 个示例的数据集，每个示例有 20 个输入变量。

该示例创建并汇总数据集。

```
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认数据集的行数和列数。

```
(1000, 20) (1000,)
```

我们可以通过[重复分层 k 重交叉验证类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)来拟合和评估最近收缩质心模型。我们将在测试装具中使用 10 次折叠和三次重复。

我们将使用欧几里德距离和无收缩的默认配置。

```
...
# create the nearest centroid model
model = NearestCentroid()
```

下面列出了评估合成二进制分类任务的最近收缩质心模型的完整示例。

```
# evaluate an nearest centroid model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestCentroid
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = NearestCentroid()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估合成数据集上的最近收缩质心算法，并报告 10 倍交叉验证的三次重复的平均精度。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型达到了大约 71%的平均精度。

```
Mean Accuracy: 0.711 (0.055)
```

我们可能会决定使用最近的收缩质心作为最终模型，并根据新数据进行预测。

这可以通过在所有可用数据上拟合模型并调用传递新数据行的 *predict()* 函数来实现。

我们可以用下面列出的完整示例来演示这一点。

```
# make a prediction with a nearest centroid model on the dataset
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = NearestCentroid()
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

```
Predicted Class: 0
```

接下来，我们可以看看配置模型超参数。

## 调谐最近质心超参数

必须为特定数据集配置最近收缩形心方法的超参数。

也许最重要的超参数是通过“*收缩阈值*参数控制的收缩。在值网格(如 0.1 或 0.01)上测试 0 到 1 之间的值是一个好主意。

下面的例子使用 [GridSearchCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)和我们定义的值网格来演示这一点。

```
# grid search shrinkage for nearest centroid
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestCentroid
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = NearestCentroid()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['shrink_threshold'] = arange(0, 1.01, 0.01)
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

在这种情况下，我们可以看到我们获得了比默认情况下稍好的结果，71.4%对 71.1%。我们可以看到模型分配了一个 0.53 的*收缩阈值*值。

```
Mean Accuracy: 0.714
Config: {'shrink_threshold': 0.53}
```

另一个关键配置是使用的距离度量，可以根据输入变量的分布来选择。

可以使用任何内置的距离测量，如下所列:

*   [metrics . pair . pair _ distance API](https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html)。

常见的距离测量包括:

*   cityblock '，'余弦'，'欧几里德'，' l1 '，' l2 '，'曼哈顿'

有关如何计算这些距离度量的更多信息，请参见教程:

*   [4 机器学习的距离度量](https://machinelearningmastery.com/distance-measures-for-machine-learning/)

假设我们的输入变量是数字，我们的数据集只支持“*欧几里德*”和“*曼哈顿*”

我们可以在网格搜索中包含这些指标；下面列出了完整的示例。

```
# grid search shrinkage and distance metric for nearest centroid
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestCentroid
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = NearestCentroid()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['shrink_threshold'] = arange(0, 1.01, 0.01)
grid['metric'] = ['euclidean', 'manhattan']
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例符合模型，并使用交叉验证发现给出最佳结果的超参数。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到，使用无收缩和曼哈顿代替欧几里德距离测量，我们获得了略好的 75%的精度。

```
Mean Accuracy: 0.750
Config: {'metric': 'manhattan', 'shrink_threshold': 0.0}
```

这些实验的一个很好的扩展是将数据规范化或标准化作为建模[管道](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)的一部分添加到数据中。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [4 机器学习的距离度量](https://machinelearningmastery.com/distance-measures-for-machine-learning/)

### 报纸

*   [通过基因表达的收缩质心诊断多种癌症类型](https://www.pnas.org/content/99/10/6567.short)，2002。

### 书

*   [第 12.6 节最近的收缩质心，应用预测建模](https://amzn.to/2wfqnw0)，2013。

### 蜜蜂

*   [sklearn . neights . nearest 质心 API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) 。
*   [metrics . pair . pair _ distance API](https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html)。

### 文章

*   [最近质心分类器，维基百科](https://en.wikipedia.org/wiki/Nearest_centroid_classifier)。
*   [Centroid，维基百科](https://en.wikipedia.org/wiki/Centroid)。

## 摘要

在本教程中，您发现了最近收缩质心分类机器学习算法。

具体来说，您了解到:

*   最近收缩质心是一种简单的线性机器学习分类算法。
*   如何使用 Scikit-Learn 的最近收缩质心模型进行拟合、评估和预测。
*   如何在给定数据集上调整最近收缩质心算法的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。