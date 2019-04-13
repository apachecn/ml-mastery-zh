# 如何用 Scikit-Learn 调整算法参数

> 原文： [https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/)

机器学习模型被参数化，以便可以针对给定问题调整它们的行为。

模型可以有许多参数，找到最佳参数组合可以视为搜索问题。

在这篇文章中，您将了解如何使用 [scikit-learn 库](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library")在 Python 中调整机器学习算法的参数。

*   **2017 年 1 月更新**：已更新，以反映版本 0.18 中 scikit-learn API 的更改。

[![fine tuning](img/7370d2ed53d2b7c4618d0e9ac63bad88.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/fine-tuning.jpg)

调整像 [Katie Fricker](https://www.flickr.com/photos/frickfrack/6261857231) 调整钢琴
照片的算法，保留一些权利

## 机器学习算法参数

[算法调整](http://machinelearningmastery.com/how-to-improve-machine-learning-results/ "How to Improve Machine Learning Results")是在呈现结果之前应用机器学习过程的最后一步。

它有时被称为[超参数优化](http://en.wikipedia.org/wiki/Hyperparameter_optimization)，其中算法参数被称为超参数，而机器学习算法本身找到的系数被称为参数。优化表明了问题的搜索性质。

作为搜索问题，您可以使用不同的搜索策略来查找针对给定问题的算法的良好且稳健的参数或参数集。

两种简单易用的搜索策略是网格搜索和随机搜索。 Scikit-learn 为算法参数调整提供了这两种方法，下面提供了每种方法的示例。

## 网格搜索参数调整

网格搜索是一种参数调整方法，它将为网格中指定的每个算法参数组合有条不紊地构建和评估模型。

下面的秘籍评估标准糖尿病数据集上的岭回归算法的不同 alpha 值。这是一维网格搜索。

Grid Search for Algorithm Tuning Python

```
# Grid Search for Algorithm Tuning
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
# load the diabetes datasets
dataset = datasets.load_diabetes()
# prepare a range of alpha values to test
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(dataset.data, dataset.target)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)
```

有关更多信息，请参阅用户指南中 GridSearchCV 和[穷举网格搜索](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)部分的 [API。](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## 随机搜索参数调整

随机搜索是一种参数调整方法，它将从随机分布（即均匀）中对算法参数进行采样，进行固定次数的迭代。为所选择的每个参数组合构建和评估模型。

下面的秘籍评估标准糖尿病数据集上的岭回归算法的 0 到 1 之间的不同 alpha 随机值。

Randomized Search for Algorithm Tuning Python

```
# Randomized Search for Algorithm Tuning
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
# load the diabetes datasets
dataset = datasets.load_diabetes()
# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(dataset.data, dataset.target)
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
```

有关更多信息，请参阅用户指南中的 [API for RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV) 和[随机参数优化](http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)部分。

## 摘要

算法参数调整是在呈现结果或准备生产系统之前提高算法表现的重要步骤。

在这篇文章中，您发现了算法参数调优和两种方法，您现在可以在 Python 和 scikit-learn 库中使用它们来改进算法结果。特别是网格搜索和随机搜索。