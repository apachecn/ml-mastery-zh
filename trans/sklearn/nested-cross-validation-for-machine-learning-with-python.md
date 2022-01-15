# Python 机器学习的嵌套交叉验证

> 原文：<https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/>

最后更新于 2021 年 11 月 20 日

k-fold 交叉验证程序用于在对训练期间未使用的数据进行预测时估计机器学习模型的性能。

此过程既可用于优化数据集模型的超参数，也可用于比较和选择数据集模型。当使用相同的交叉验证过程和数据集来调整和选择模型时，很可能导致对模型性能的乐观偏见评估。

克服这种偏差的一种方法是将超参数优化过程嵌套在模型选择过程之下。这被称为**双重交叉验证**或**嵌套交叉验证**，是评估和比较调整后的机器学习模型的首选方式。

在本教程中，您将发现用于评估调整后的机器学习模型的嵌套交叉验证。

完成本教程后，您将知道:

*   超参数优化可以对数据集进行过度优化，并提供不应用于模型选择的模型的乐观评估。
*   嵌套交叉验证提供了一种减少组合超参数调整和模型选择偏差的方法。
*   如何在 scikit-learn 中实现嵌套交叉验证来评估调整后的机器学习算法？

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2021 年 1 月更新**:增加了管道思维部分和相关教程的链接。

![Nested Cross-Validation for Machine Learning with Python](img/538041b2c9f023f0bf9529b42f6d7bf9.png)

Python 机器学习的嵌套交叉验证
图片由[安德鲁·伯恩](https://flickr.com/photos/andreboeni/37086944973/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  组合超参数调整和模型选择
2.  什么是嵌套交叉验证
    1.  嵌套交叉验证的成本是多少？
    2.  你如何设置 k？
    3.  如何配置最终模型？
    4.  内环选择了什么配置？
3.  使用 Scikit-Learn 进行嵌套交叉验证

## 组合超参数调整和模型选择

使用 k 重交叉验证在数据集上评估机器学习模型是很常见的。

k 折叠交叉验证过程将有限的数据集分成 k 个不重叠的折叠。k 个折叠中的每一个都有机会用作保留测试集，而所有其他折叠一起用作训练数据集。在 k 个保持测试集上对总共 k 个模型进行拟合和评估，并报告平均性能。

有关 k-fold 交叉验证过程的更多信息，请参见教程:

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

当对训练期间未使用的数据进行预测时，该过程提供对数据集的模型性能的估计。它比其他一些技术更少偏向，比如针对小到中等大小数据集的单一训练测试分割。k 的常见值有 k=3、k=5 和 k=10。

每个机器学习算法包括一个或多个超参数，这些超参数允许算法行为适合特定的数据集。问题是，关于如何为数据集配置模型超参数，很少有好的启发式方法。相反，优化过程用于发现数据集上表现良好或最好的一组超参数。优化算法的常见示例包括网格搜索和随机搜索，每组不同的模型超参数通常使用 k 倍交叉验证进行评估。

这突出表明，k-fold 交叉验证过程既用于选择模型超参数以配置每个模型，也用于选择已配置的模型。

k-fold 交叉验证程序是评估模型性能的有效方法。然而，该过程的一个限制是，如果它与同一算法一起使用多次，它会导致过拟合。

每次在数据集上评估具有不同模型超参数的模型时，它都会提供关于数据集的信息。具体来说，有噪声的数据集通常得分更低。可以在模型配置过程中利用数据集模型的这些知识来为数据集找到性能最佳的配置。k-fold 交叉验证过程试图减少这种影响，但它不能完全消除，将执行某种形式的爬山或模型超参数到数据集的过度拟合。这是超参数优化的正常情况。

问题是，如果仅使用这个分数来选择模型，或者使用相同的数据集来评估调整后的模型，那么选择过程将会由于这种无意的过度拟合而有所偏差。结果是对模型性能过于乐观的估计，不能推广到新数据。

需要一个过程，允许两个模型为数据集选择性能良好的超参数，并在数据集上配置良好的模型集合中进行选择。

解决这个问题的一种方法叫做**嵌套交叉验证**。

## 什么是嵌套交叉验证

嵌套交叉验证是一种模型超参数优化和模型选择的方法，试图克服训练数据集过拟合的问题。

> 为了克服性能评估中的偏差，模型选择应被视为模型拟合程序的一个组成部分，并应在每个试验中独立进行，以防止选择偏差，因为它反映了操作使用中的最佳实践。

——[关于模型选择中的过度拟合和绩效评估中的后续选择偏差](http://www.jmlr.org/papers/v11/cawley10a.html)，2010。

该过程包括将模型超参数优化视为模型本身的一部分，并在更广泛的 k 倍交叉验证过程中对其进行评估，以评估模型进行比较和选择。

因此，模型超参数优化的 k 倍交叉验证程序嵌套在模型选择的 *k* 倍交叉验证程序中。两个交叉验证循环的使用也导致该程序被称为“T2”双交叉验证

通常，k 折叠交叉验证过程包括在除一个折叠之外的所有折叠上拟合模型，并在保持折叠上评估拟合模型。让我们将用于训练模型的折叠集合称为“*训练数据集*”，将伸出的折叠称为“*测试数据集*”

然后，每个训练数据集被提供给超参数优化过程，例如网格搜索或随机搜索，其为模型找到最优的超参数集。每组超参数的评估使用 k-fold 交叉验证来执行，该交叉验证将所提供的训练数据集分成 *k* 个折叠，而不是原始数据集。

> 这被称为“内部”协议，因为模型选择过程是在重采样过程的每个折叠中独立执行的。

——[关于模型选择中的过度拟合和绩效评估中的后续选择偏差](http://www.jmlr.org/papers/v11/cawley10a.html)，2010。

在此过程中，超参数搜索没有机会对数据集进行过度填充，因为它只暴露给外部交叉验证过程提供的数据集子集。这降低了(如果不是消除的话)搜索过程过度拟合原始数据集的风险，并且应该提供对数据集上的优化模型性能的偏差较小的估计。

> 以这种方式，性能估计包括适当考虑由模型选择标准的过度拟合引入的误差的组件。

——[关于模型选择中的过度拟合和绩效评估中的后续选择偏差](http://www.jmlr.org/papers/v11/cawley10a.html)，2010。

### 嵌套交叉验证的成本是多少？

嵌套交叉验证的缺点是执行的模型评估数量急剧增加。

如果 *n * k* 模型作为给定模型的传统交叉验证超参数搜索的一部分被拟合和评估，那么这将增加到 *k * n * k* ，因为在嵌套交叉验证的外部循环中，该过程随后对每个折叠执行 *k* 更多次。

为了使其具体化，您可以使用 *k=5* 进行超参数搜索，并测试 100 个模型超参数组合。因此，传统的超参数搜索将适合并评估 *5 * 100* 或 500 型号。外环中带有 *k=10* 折叠的嵌套交叉验证将适合并评估 5000 个模型。在这种情况下增加了 10 倍。

### 你如何设置 k？

内环和外环的 k 值应该像您为单个 *k* 折叠交叉验证程序设置 *k* 值一样进行设置。

您必须为数据集选择一个 *k* 值，该值平衡评估过程的计算成本(不要有太多的模型评估)和模型性能的无偏估计。

外环通常使用 *k=10* ，内环使用较小的 k 值，如 *k=3* 或 *k=5* 。

有关设置 k 的更多常规帮助，请参见本教程:

*   [如何配置 k 重交叉验证](https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/)

### 如何配置最终模型？

最终的模型使用在外环的一个过程中应用的过程进行配置和拟合，例如外环应用于整个数据集。

如下所示:

1.  基于算法在嵌套交叉验证外环上的性能来选择算法。
2.  然后将内部过程应用于整个数据集。
3.  在最终搜索过程中找到的超参数随后被用于配置最终模型。
4.  最终模型适合整个数据集。

这个模型可以用来对新数据进行预测。根据最终模型调优过程中提供的分数，我们知道它的平均性能。

### 内环选择了什么配置？

没关系，这就是全部想法。

使用了自动配置过程，而不是特定配置。只有一个最终模型，但是在最终运行时，通过选择的搜索过程可以找到该最终模型的最佳配置。

您不再需要深入所选的特定模型配置，就像在下一个级别，您不再需要每个交叉验证文件夹中的特定模型系数一样。

这需要思维的转变，并且可能具有挑战性，例如从“*我这样配置我的模型…* ”转变为“*我使用了带有这些约束的自动模型配置过程…* ”。

本教程有更多关于*管道思维*的话题，可能会有所帮助:

*   [机器学习建模管道的温和介绍](https://machinelearningmastery.com/machine-learning-modeling-pipelines/)

现在我们已经熟悉了嵌套交叉验证，让我们回顾一下如何在实践中实现它。

## 使用 Scikit-Learn 进行嵌套交叉验证

k-fold 交叉验证程序可通过 [KFold 类](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)在 scikit-learn Python 机器学习库中获得。

该类配置了折叠(拆分)的数量，然后调用 *split()* 函数，在数据集中传递。枚举 *split()* 函数的结果，以给出列车的行索引和每个折叠的测试集。

例如:

```py
...
# configure the cross-validation procedure
cv = KFold(n_splits=10, random_state=1)
# perform cross-validation procedure
for train_ix, test_ix in cv_outer.split(X):
	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]
	# fit and evaluate a model
	...
```

此类可用于执行嵌套交叉验证过程的外部循环。

scikit-learn 库分别通过[随机化搜索 CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 和 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 类提供交叉验证随机搜索和网格搜索超参数优化。通过创建类并指定模型、数据集、要搜索的超参数和交叉验证过程来配置过程。

例如:

```py
...
# configure the cross-validation procedure
cv = KFold(n_splits=3, shuffle=True, random_state=1)
# define search space
space = dict()
...
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
```

这些类可用于嵌套交叉验证的内部循环，其中由外部循环定义的训练数据集用作内部循环的数据集。

我们可以将这些元素联系在一起，并实现嵌套的交叉验证过程。

重要的是，我们可以配置超参数搜索，使用搜索过程中找到的最佳超参数，用整个训练数据集重新调整最终模型。这可以通过将“ *refit* ”参数设置为 True，然后通过搜索结果上的“ *best_estimator_* ”属性检索模型来实现。

```py
...
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv_inner, refit=True)
# execute search
result = search.fit(X_train, y_train)
# get the best performing model fit on the whole training set
best_model = result.best_estimator_
```

然后，该模型可用于对来自外环的保持数据进行预测，并估计模型的性能。

```py
...
# evaluate model on the hold out dataset
yhat = best_model.predict(X_test)
```

将所有这些联系在一起，我们可以在一个综合分类数据集上演示[随机森林分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)的嵌套交叉验证。

我们将保持简单，只调整两个各有三个值的超参数，例如( *3 * 3* ) 9 的组合。我们将在外部交叉验证中使用 10 个折叠，在内部交叉验证中使用 3 个折叠，从而得到( *10 * 9 * 3* )或 270 个模型评估。

下面列出了完整的示例。

```py
# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]
	# configure the cross-validation procedure
	cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
	# define the model
	model = RandomForestClassifier(random_state=1)
	# define search space
	space = dict()
	space['n_estimators'] = [10, 100, 500]
	space['max_features'] = [2, 4, 6]
	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
	# execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)
	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	# store the result
	outer_results.append(acc)
	# report progress
	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
```

运行该示例使用合成分类数据集上的嵌套交叉验证来评估[随机森林](https://machinelearningmastery.com/random-forest-ensemble-in-python/)。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

您可以使用该示例作为起点，并对其进行调整以评估不同的算法超参数、不同的算法或不同的数据集。

外部交叉验证过程的每一次迭代都报告最佳性能模型的估计性能(使用 3 倍交叉验证)和被发现表现最佳的超参数，以及保持数据集的精度。

这是有见地的，因为我们可以看到，实际和估计的准确性是不同的，但在这种情况下，类似的。我们还可以看到，每次迭代都会发现不同的超参数，这表明这个数据集上的好的超参数取决于数据集的细节。

然后报告最终的平均分类精度。

```py
>acc=0.900, est=0.932, cfg={'max_features': 4, 'n_estimators': 100}
>acc=0.940, est=0.924, cfg={'max_features': 4, 'n_estimators': 500}
>acc=0.930, est=0.929, cfg={'max_features': 4, 'n_estimators': 500}
>acc=0.930, est=0.927, cfg={'max_features': 6, 'n_estimators': 100}
>acc=0.920, est=0.927, cfg={'max_features': 4, 'n_estimators': 100}
>acc=0.950, est=0.927, cfg={'max_features': 4, 'n_estimators': 500}
>acc=0.910, est=0.918, cfg={'max_features': 2, 'n_estimators': 100}
>acc=0.930, est=0.924, cfg={'max_features': 6, 'n_estimators': 500}
>acc=0.960, est=0.926, cfg={'max_features': 2, 'n_estimators': 500}
>acc=0.900, est=0.937, cfg={'max_features': 4, 'n_estimators': 500}
Accuracy: 0.927 (0.019)
```

我们可以执行相同程序的一个更简单的方法是使用 [cross_val_score()函数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)，该函数将执行外部交叉验证程序。这可以在已配置的 *GridSearchCV* 上直接执行，它将从外环自动使用测试集上的改装最佳性能模型。

这大大减少了执行嵌套交叉验证所需的代码量。

下面列出了完整的示例。

```py
# automatic nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
# configure the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# define the model
model = RandomForestClassifier(random_state=1)
# define search space
space = dict()
space['n_estimators'] = [10, 100, 500]
space['max_features'] = [2, 4, 6]
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# execute the nested cross-validation
scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行示例在随机森林算法上执行嵌套交叉验证，达到了与我们的手动过程相匹配的平均精度。

```py
Accuracy: 0.927 (0.019)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)
*   [如何配置 k 重交叉验证](https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/)
*   [机器学习建模管道的温和介绍](https://machinelearningmastery.com/machine-learning-modeling-pipelines/)

### 报纸

*   [统计预测的交叉验证选择和评估](https://www.jstor.org/stable/2984809?seq=1)，1974 年。
*   [关于模型选择中的过度拟合和绩效评估中的后续选择偏差](http://www.jmlr.org/papers/v11/cawley10a.html)，2010。
*   [选择和评估回归和分类模型时的交叉验证陷阱](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10)，2014。
*   [在大多数实际应用中，选择分类器时的嵌套交叉验证过于热心](https://arxiv.org/abs/1809.09446)，2018。

### 蜜蜂

*   [交叉验证:评估评估者绩效，scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) 。
*   [嵌套与非嵌套交叉验证，scikit-learn 示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。
*   [sklearn.model_selection。KFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 。
*   [sklearn.model_selection。GridSearchCV API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 。
*   [硬化。一起。随机应变分类 API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 。
*   [sklearn . model _ selection . cross _ val _ score API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

## 摘要

在本教程中，您发现了用于评估调整后的机器学习模型的嵌套交叉验证。

具体来说，您了解到:

*   超参数优化可以对数据集进行过度优化，并提供不应用于模型选择的模型的乐观评估。
*   嵌套交叉验证提供了一种减少组合超参数调整和模型选择偏差的方法。
*   如何在 scikit-learn 中实现嵌套交叉验证来评估调整后的机器学习算法？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。