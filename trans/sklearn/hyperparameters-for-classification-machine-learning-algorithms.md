# 调整分类机器学习算法的超参数

> 原文：<https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/>

最后更新于 2020 年 8 月 28 日

机器学习算法具有超参数，允许您根据特定数据集定制算法的行为。

[超参数](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)不同于参数，参数是通过学习算法找到的模型的内部系数或权重。与参数不同，超参数由从业者在配置模型时指定。

通常，很难知道给定数据集上给定算法的超参数使用什么值，因此通常对不同的超参数值使用随机或网格搜索策略。

需要调整的算法超参数越多，调整过程就越慢。因此，希望选择模型超参数的最小子集来搜索或调整。

并非所有的模型超参数都同等重要。一些超参数对行为有着巨大的影响，进而影响机器学习算法的表现。

作为机器学习的实践者，你必须知道要关注哪些超参数才能快速得到好的结果。

在本教程中，您将发现那些对一些顶级机器学习算法最重要的超参数。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 scikit-learn v0.22 API 的变化进行了更新。

![Hyperparameters for Classification Machine Learning Algorithms](img/d2e99c384a8c9a76f647bb3994df8a36.png)

分类机器学习算法的超参数
图片由 [shuttermonkey](https://flickr.com/photos/shuttermonkey/4934194353/) 提供，保留部分权利。

## 分类算法综述

我们将仔细研究您可能用于分类的顶级机器学习算法的重要超参数。

我们将查看您需要关注的超参数以及在数据集上调整模型时尝试的建议值。

这些建议是基于教科书中关于算法的建议和从业者的实际建议，以及我自己的一点经验。

我们将研究的七种分类算法如下:

1.  逻辑回归
2.  脊分类器
3.  k 近邻(KNN)
4.  支持向量机(SVM)
5.  袋装决策树(袋装)
6.  随机森林
7.  随机梯度升压

我们将在 scikit-learn 实现(Python)的背景下考虑这些算法；尽管如此，您可以将相同的超参数建议用于其他平台，例如 Weka 和 r。

还为每个算法提供了一个小的网格搜索示例，您可以将其用作自己的分类预测建模项目的起点。

**注**:如果你用不同的超参数值，甚至不同于本教程建议的超参数成功过，请在下面的评论中告诉我。我很想听听。

让我们开始吧。

## 逻辑回归

逻辑回归实际上没有任何关键的超参数需要调整。

有时，您可以看到不同解算器(*解算器*)在表现或收敛性方面的有用差异。

*   **求解器**在['newton-cg '，' lbfgs '，' liblinear '，' sag '，' saga']中

正规化(*惩罚*)有时会有帮助。

*   **处罚**在['无'，' l1 '，' l2 '，'弹性']

**注**:并非所有解算器都支持所有正则化项。

C 参数控制惩罚力度，这也是有效的。

*   **C** 在【100，10，1.0，0.1，0.01】

有关超参数的完整列表，请参见:

*   [sklearn.linear_model。物流配送应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。

下面的示例演示了在合成二进制分类数据集上网格搜索物流分类的关键超参数。

为了减少警告/错误，省略了一些组合。

```py
# example of grid searching key hyperparametres for logistic regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.945333 using {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
0.936333 (0.016829) with: {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
0.937667 (0.017259) with: {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
0.938667 (0.015861) with: {'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}
0.936333 (0.017413) with: {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
0.938333 (0.017904) with: {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
0.939000 (0.016401) with: {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
0.937333 (0.017114) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'}
0.939000 (0.017195) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
0.939000 (0.015780) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}
0.940000 (0.015706) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
0.940333 (0.014941) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
0.941000 (0.017000) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
0.943000 (0.016763) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
0.943000 (0.016763) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
0.945333 (0.017651) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
```

## 脊分类器

岭回归是一种用于预测数值的惩罚线性回归模型。

然而，当应用于分类时，它可能非常有效。

也许最重要的调整参数是正则化强度(*α*)。一个好的起点可能是[0.1 到 1.0]范围内的值

*   **α**在【0.1，0.2，0.3，0.4，0.5，0.6，0.7，0.8，0.9，1.0】

有关超参数的完整列表，请参见:

*   [sklearn.linear_model。脊分类器 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) 。

下面的示例演示了在合成二进制分类数据集上网格搜索 RidgeClassifier 的关键超参数。

```py
# example of grid searching key hyperparametres for ridge classifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = RidgeClassifier()
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# define grid search
grid = dict(alpha=alpha)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.974667 using {'alpha': 0.1}
0.974667 (0.014545) with: {'alpha': 0.1}
0.974667 (0.014545) with: {'alpha': 0.2}
0.974667 (0.014545) with: {'alpha': 0.3}
0.974667 (0.014545) with: {'alpha': 0.4}
0.974667 (0.014545) with: {'alpha': 0.5}
0.974667 (0.014545) with: {'alpha': 0.6}
0.974667 (0.014545) with: {'alpha': 0.7}
0.974667 (0.014545) with: {'alpha': 0.8}
0.974667 (0.014545) with: {'alpha': 0.9}
0.974667 (0.014545) with: {'alpha': 1.0}
```

## k 近邻(KNN)

对于 KNN 来说，最重要的超参数是邻居的数量( *n_neighbors* )。

测试值至少在 1 到 21 之间，可能只是奇数。

*   【1 至 21】中的 **n_neighbors**

测试不同的距离度量(*度量*)来选择邻域的组成可能也很有趣。

*   **公制**用['欧几里得'，'曼哈顿'，'闵可夫斯基']表示

有关更完整的列表，请参见:

*   [sklearn . neights . distance metric API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)

通过不同的权重(权重*权重*)来测试邻域成员的贡献可能也很有趣。

*   **重量**单位为['统一'，'距离']

有关超参数的完整列表，请参见:

*   [sklearn . neighborsclassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)。

下面的示例演示了在合成二进制分类数据集上网格搜索 KNeighborsClassifier 的关键超参数。

```py
# example of grid searching key hyperparametres for KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.937667 using {'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'uniform'}
0.833667 (0.031674) with: {'metric': 'euclidean', 'n_neighbors': 1, 'weights': 'uniform'}
0.833667 (0.031674) with: {'metric': 'euclidean', 'n_neighbors': 1, 'weights': 'distance'}
0.895333 (0.030081) with: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}
0.895333 (0.030081) with: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
0.909000 (0.021810) with: {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}
0.909000 (0.021810) with: {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}
0.925333 (0.020774) with: {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'uniform'}
0.925333 (0.020774) with: {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}
0.929000 (0.027368) with: {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'}
0.929000 (0.027368) with: {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'distance'}
...
```

## 支持向量机(SVM)

SVM 算法，像梯度增强一样，非常流行，非常有效，并且提供了大量的超参数来调整。

也许第一个重要的参数是内核的选择，它将控制输入变量的投影方式。可供选择的有很多，但最常见的是线性、多项式和径向基函数，实际上可能只有线性和径向基函数。

*   **核**在['线性'，'多边形'，'径向基函数'，' sigmoid']

如果多项式核成立，那么深入到度超参数是个好主意。

另一个关键参数是惩罚( *C* )，它可以采用一系列值，并对每个类别的结果区域的形状产生显著影响。对数标度可能是一个很好的起点。

*   **C** 在【100，10，1.0，0.1，0.001】

有关超参数的完整列表，请参见:

*   [硬化. svm.SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) 。

下面的示例演示了在合成二进制分类数据集上网格搜索支持向量机的关键超参数。

```py
# example of grid searching key hyperparametres for SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.974333 using {'C': 1.0, 'gamma': 'scale', 'kernel': 'poly'}
0.973667 (0.012512) with: {'C': 50, 'gamma': 'scale', 'kernel': 'poly'}
0.970667 (0.018062) with: {'C': 50, 'gamma': 'scale', 'kernel': 'rbf'}
0.945333 (0.024594) with: {'C': 50, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.973667 (0.012512) with: {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
0.970667 (0.018062) with: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
0.957000 (0.016763) with: {'C': 10, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.974333 (0.012565) with: {'C': 1.0, 'gamma': 'scale', 'kernel': 'poly'}
0.971667 (0.016948) with: {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
0.966333 (0.016224) with: {'C': 1.0, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.972333 (0.013585) with: {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}
0.974000 (0.013317) with: {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
0.971667 (0.015934) with: {'C': 0.1, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.972333 (0.013585) with: {'C': 0.01, 'gamma': 'scale', 'kernel': 'poly'}
0.973667 (0.014716) with: {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}
0.974333 (0.013828) with: {'C': 0.01, 'gamma': 'scale', 'kernel': 'sigmoid'}
```

## 袋装决策树(袋装)

袋装决策树最重要的参数是树的数量(*n _ evaluator*)。

理想情况下，这应该增加，直到在模型中没有看到进一步的改进。

好的值可能是从 10 到 1，000 的对数标度。

*   【10，100，1000】中的**n _ 估算器**

有关超参数的完整列表，请参见:

*   [硬化。一起。bagginclassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)

下面的示例演示了在合成二进制分类数据集上网格搜索巴金分类器的关键超参数。

```py
# example of grid searching key hyperparameters for BaggingClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = BaggingClassifier()
n_estimators = [10, 100, 1000]
# define grid search
grid = dict(n_estimators=n_estimators)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.873667 using {'n_estimators': 1000}
0.839000 (0.038588) with: {'n_estimators': 10}
0.869333 (0.030434) with: {'n_estimators': 100}
0.873667 (0.035070) with: {'n_estimators': 1000}
```

## 随机森林

最重要的参数是每个分割点要采样的随机特征的数量( *max_features* )。

您可以尝试一个整数值范围，例如 1 到 20，或者输入要素数量的 1 到一半。

*   **最大功能**【1 至 20】

或者，您可以尝试一套不同的默认值计算器。

*   **max_features** in ['sqrt '，' log2']

随机森林的另一个重要参数是树的数量(*n _ estimator*)。

理想情况下，这应该增加，直到在模型中没有看到进一步的改进。

好的值可能是从 10 到 1，000 的对数标度。

*   【10，100，1000】中的**n _ 估算器**

有关超参数的完整列表，请参见:

*   [硬化。一起。随机应变分类 API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 。

下面的示例演示了在合成二进制分类数据集上网格搜索巴金分类器的关键超参数。

```py
# example of grid searching key hyperparameters for RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.952000 using {'max_features': 'log2', 'n_estimators': 1000}
0.841000 (0.032078) with: {'max_features': 'sqrt', 'n_estimators': 10}
0.938333 (0.020830) with: {'max_features': 'sqrt', 'n_estimators': 100}
0.944667 (0.024998) with: {'max_features': 'sqrt', 'n_estimators': 1000}
0.817667 (0.033235) with: {'max_features': 'log2', 'n_estimators': 10}
0.940667 (0.021592) with: {'max_features': 'log2', 'n_estimators': 100}
0.952000 (0.019562) with: {'max_features': 'log2', 'n_estimators': 1000}
```

## 随机梯度升压

也称为梯度增强机(GBM)或以特定实现命名，如 XGBoost。

梯度增强算法有许多参数需要调整。

有一些参数配对需要考虑。第一个是学习率，也称为收缩率或 eta ( *学习率*)和模型中的树的数量(*n _ estimates*)。两者都可以用对数尺度来考虑，尽管方向不同。

*   **学习 _ 率**在【0.001，0.01，0.1】
*   **n _ 估算器**【10，100，1000】

另一个配对是每棵树要考虑的行数或数据子集(*子样本*)和每棵树的深度( *max_depth* )。这些可以分别以 0.1 和 1 的间隔进行网格搜索，尽管可以直接测试公共值。

*   【0.5，0.7，1.0】中的**子样本**
*   【3，7，9】中的**最大深度**

有关调优 XGBoost 实现的更详细建议，请参见:

*   [如何配置梯度增强算法](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)

有关超参数的完整列表，请参见:

*   [硬化。集合。梯度助推器 API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) 。

下面的示例演示了在合成二进制分类数据集上网格搜索 gradientboosting 分类器的关键超参数。

```py
# example of grid searching key hyperparameters for GradientBoostingClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例将打印最佳结果以及所有评估组合的结果。

```py
Best: 0.936667 using {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
0.803333 (0.042058) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 10, 'subsample': 0.5}
0.783667 (0.042386) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 10, 'subsample': 0.7}
0.711667 (0.041157) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 10, 'subsample': 1.0}
0.832667 (0.040244) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.5}
0.809667 (0.040040) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.7}
0.741333 (0.043261) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}
0.881333 (0.034130) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
0.866667 (0.035150) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.7}
0.838333 (0.037424) with: {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 1.0}
0.838333 (0.036614) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 10, 'subsample': 0.5}
0.821667 (0.040586) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 10, 'subsample': 0.7}
0.729000 (0.035903) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 10, 'subsample': 1.0}
0.884667 (0.036854) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.5}
0.871333 (0.035094) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.7}
0.729000 (0.037625) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 100, 'subsample': 1.0}
0.905667 (0.033134) with: {'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.5}
...
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [scikit-learn API](https://scikit-learn.org/stable/modules/classes.html)
*   [插入符号算法和调谐参数列表](https://topepo.github.io/caret/available-models.html)

## 摘要

在本教程中，您发现了顶级超参数以及如何为顶级机器学习算法配置它们。

你有其他的超参数建议吗？请在下面的评论中告诉我。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。