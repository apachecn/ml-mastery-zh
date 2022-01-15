# 基于 Python 的半径邻居分类器算法

> 原文:[https://machinelearning master . com/radius-neighbors-classifier-algorithm-with-python/](https://machinelearningmastery.com/radius-neighbors-classifier-algorithm-with-python/)

半径邻居分类器是一种分类机器学习算法。

它是 k 近邻算法的扩展，使用新示例半径内的所有示例而不是 k 近邻进行预测。

因此，基于半径的选择邻居的方法更适合于[稀疏数据](https://machinelearningmastery.com/sparse-matrices-for-machine-learning/)，防止在特征空间中较远的例子对预测做出贡献。

在本教程中，您将发现**半径邻居分类器**分类机器学习算法。

完成本教程后，您将知道:

*   最近半径近邻分类器是 k 近邻分类算法的简单扩展。
*   如何使用带有 Scikit-Learn 的半径邻居分类器模型进行拟合、评估和预测。
*   如何在给定数据集上调整半径邻居分类器算法的超参数。

我们开始吧。

![Radius Neighbors Classifier Algorithm With Python](img/c4528e6d194edcc53a463b4a727d256b.png)

带 Python 的半径邻居分类器算法
图片由 [J .特里普克](https://flickr.com/photos/piro007/16672314669/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  半径邻居分类器
2.  基于 Scikit-Learn 的半径邻居分类器
3.  调整半径邻居分类器超参数

## 半径邻居分类器

半径邻居是一种分类机器学习算法。

它基于 k 近邻算法，或称 kNN。kNN 包括获取整个训练数据集并存储它。然后，在预测时，为我们要预测的每个新示例定位训练数据集中 k 个最接近的示例。然后，来自 k 个邻居的模式(最常见的值)类别标签被分配给新示例。

有关 k 近邻算法的更多信息，请参见教程:

*   [从头开始开发 Python 中的 k 近邻](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

半径邻居分类器的相似之处在于，训练包括存储整个训练数据集。预测期间使用训练数据集的方式不同。

半径邻居分类器不是定位 k 邻居，而是定位训练数据集中新示例给定半径内的所有示例。半径邻居随后被用于对新示例进行预测。

半径是在特征空间中定义的，并且通常假设输入变量是数字，并且被缩放到 0-1 的范围，例如归一化。

基于半径的邻居定位方法适用于那些希望邻居的贡献与特征空间中的示例密度成比例的数据集。

给定固定的半径，特征空间的密集区域将贡献更多的信息，而稀疏区域将贡献更少的信息。后一种情况是最理想的，并且它防止在特征空间中距离新示例很远的示例对预测做出贡献。

因此，半径邻居分类器可能更适合于存在特征空间稀疏区域的预测问题。

假设半径在要素空间的所有维度上都是固定的，随着输入要素数量的增加，半径的有效性会降低，这将导致要素空间中的示例越来越分散。这个属性被称为维度的[诅咒](https://en.wikipedia.org/wiki/Curse_of_dimensionality)。

## 基于 Scikit-Learn 的半径邻居分类器

半径邻居分类器可通过[半径邻居分类器类](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)在 scikit-learn Python 机器学习库中获得。

该类允许您通过“*半径*”参数指定进行预测时使用的半径大小，该参数默认为 1.0。

```py
...
# create the model
model = RadiusNeighborsClassifier(radius=1.0)
```

另一个重要的超参数是“*权重*”参数，该参数控制邻居是以一致的*方式对预测做出贡献，还是与该示例的距离(*距离*)成反比。默认情况下使用统一重量。*

```py
...
# create the model
model = RadiusNeighborsClassifier(weights='uniform')
```

我们可以用一个工作示例来演示半径邻居分类器。

首先，让我们定义一个综合分类数据集。

我们将使用 [make_classification()函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 1000 个示例的数据集，每个示例有 20 个输入变量。

下面的示例创建并汇总了数据集。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认数据集的行数和列数。

```py
(1000, 20) (1000,)
```

我们可以通过[repeated stratifiedfold 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)使用重复的分层 k 重交叉验证来拟合和评估半径邻居分类器模型。我们将在测试装具中使用 10 次折叠和三次重复。

我们将使用默认配置。

```py
...
# create the model
model = RadiusNeighborsClassifier()
```

在准备和使用模型之前，对特征空间进行缩放是很重要的。

我们可以通过使用[最小最大缩放器](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来归一化输入特征，并使用[管道](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)来首先应用缩放，然后使用模型。

```py
...
# define model
model = RadiusNeighborsClassifier()
# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])
```

下面列出了为合成二进制分类任务评估半径邻居分类器模型的完整示例。

```py
# evaluate an radius neighbors classifier model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import RadiusNeighborsClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = RadiusNeighborsClassifier()
# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例在合成数据集上评估半径邻居分类器算法，并报告 10 倍交叉验证的三次重复的平均准确性。

鉴于学习算法的随机性，您的具体结果可能会有所不同。考虑运行这个例子几次。

在这种情况下，我们可以看到模型达到了大约 75.4%的平均精度。

```py
Mean Accuracy: 0.754 (0.042)
```

我们可能会决定使用半径邻居分类器作为最终模型，并对新数据进行预测。

这可以通过在所有可用数据上拟合模型管道并调用传递新数据行的 *predict()* 函数来实现。

我们可以用下面列出的完整示例来演示这一点。

```py
# make a prediction with a radius neighbors classifier model on the dataset
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import RadiusNeighborsClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = RadiusNeighborsClassifier()
# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])
# fit model
pipeline.fit(X, y)
# define new data
row = [2.47475454,0.40165523,1.68081787,2.88940715,0.91704519,-3.07950644,4.39961206,0.72464273,-4.86563631,-6.06338084,-1.22209949,-0.4699618,1.01222748,-0.6899355,-0.53000581,6.86966784,-3.27211075,-6.59044146,-2.21290585,-3.139579]
# make a prediction
yhat = pipeline.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat)
```

运行该示例符合模型，并对新的数据行进行类别标签预测。

```py
Predicted Class: 0
```

接下来，我们可以看看配置模型超参数。

## 调整半径邻居分类器超参数

必须为您的特定数据集配置半径邻居分类器方法的超参数。

也许最重要的超参数是通过“*半径*”参数控制的半径。测试一系列值是个好主意，可能在 1.0 左右。

我们将在合成数据集上探索 0.8 到 1.5 之间的值，网格为 0.01。

```py
...
# define grid
grid = dict()
grid['model__radius'] = arange(0.8, 1.5, 0.01)
```

请注意，我们正在*管线*内对*半径邻居分类器*的“*半径*超参数进行网格搜索，其中模型名为“*模型*”，因此半径参数通过带有双下划线( *__* )分隔符的*模型- >半径*访问，例如“*模型 __ 半径*”。

下面的例子使用 [GridSearchCV 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)和我们定义的值网格来演示这一点。

```py
# grid search radius for radius neighbors classifier
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import RadiusNeighborsClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = RadiusNeighborsClassifier()
# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['model__radius'] = arange(0.8, 1.5, 0.01)
# define search
search = GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例将使用重复的交叉验证来评估配置的每个组合。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到，使用 0.8 的半径获得了更好的结果，精度约为 87.2%，而前面示例中的半径为 1.0，精度约为 75.4%。

```py
Mean Accuracy: 0.872
Config: {'model__radius': 0.8}
```

另一个关键超参数是半径中的示例通过“*权重*”参数对预测做出贡献的方式。这可以设置为“*制服*”(默认)、“*距离*”为逆距离，或自定义功能。

我们可以测试这两个内置权重，看看半径为 0.8 时哪个表现更好。

```py
...
# define grid
grid = dict()
grid['model__weights'] = ['uniform', 'distance']
```

下面列出了完整的示例。

```py
# grid search weights for radius neighbors classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import RadiusNeighborsClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define model
model = RadiusNeighborsClassifier(radius=0.8)
# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['model__weights'] = ['uniform', 'distance']
# define search
search = GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

运行该示例符合模型，并使用交叉验证发现给出最佳结果的超参数。

鉴于学习算法的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到平均分类准确率从上一个示例中的 87.2%左右的“T0”统一权重提升到本例中的 89.3%左右的“T2”距离权重。

```py
Mean Accuracy: 0.893
Config: {'model__weights': 'distance'}
```

您可能希望探索的另一个度量是通过默认为“*闵可夫斯基*的“*度量*参数使用的距离度量。

将结果与‘欧几里德距离’和‘T2 街区’进行比较可能会很有趣。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k-机器学习的最近邻](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)
*   [从头开始开发 Python 中的 k 近邻](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

### 书

*   [应用预测建模](https://amzn.to/2wfqnw0)，2013。
*   [R](https://amzn.to/2xW4hPy)中应用的统计学习导论，2014。

### 蜜蜂

*   [最近邻居，sci kit-学习用户指南](https://scikit-learn.org/stable/modules/neighbors.html)。
*   [sklearn . neights . radiusNeightosclassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)。
*   [sklearn . pipeline . pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)。
*   [硬化。预处理。MinMaxScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 。

### 文章

*   [k 近邻算法，维基百科](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)。
*   [维度的诅咒，维基百科](https://en.wikipedia.org/wiki/Curse_of_dimensionality)。

## 摘要

在本教程中，您发现了半径邻居分类器分类机器学习算法。

具体来说，您了解到:

*   最近半径近邻分类器是 k 近邻分类算法的简单扩展。
*   如何使用带有 Scikit-Learn 的半径邻居分类器模型进行拟合、评估和预测。
*   如何在给定数据集上调整半径邻居分类器算法的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。*