# 随机搜索和网格搜索的超参数优化

> 原文：<https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/>

最后更新于 2020 年 9 月 19 日

机器学习模型有您必须设置的超参数，以便根据数据集定制模型。

通常，超参数对模型的一般影响是已知的，但是如何为给定的数据集最佳地设置超参数和交互超参数的组合是具有挑战性的。通常有配置超参数的通用试探法或经验法则。

一种更好的方法是客观地搜索模型超参数的不同值，并选择一个子集，该子集导致在给定数据集上获得最佳性能的模型。这被称为**超参数优化**或超参数调整，可在 scikit-learn Python 机器学习库中获得。超参数优化的结果是一组性能良好的超参数，可用于配置模型。

在本教程中，您将在 Python 中发现机器学习的超参数优化。

完成本教程后，您将知道:

*   需要超参数优化来充分利用机器学习模型。
*   如何为分类任务配置随机和网格搜索超参数优化？
*   如何为回归任务配置随机和网格搜索超参数优化？

我们开始吧。

![Hyperparameter Optimization With Random Search and Grid Search](img/5dbbfc3546d44b0de5a673c46e1d0cb8.png)

带有随机搜索和网格搜索的超参数优化
詹姆斯·圣约翰摄，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  模型超参数优化
2.  超参数优化套件-学习应用编程接口
3.  分类的超参数优化
    1.  分类的随机搜索
    2.  用于分类的网格搜索
4.  回归的超参数优化
    1.  回归的随机搜索
    2.  回归的网格搜索
5.  超参数优化的常见问题

## 模型超参数优化

机器学习模型有超参数。

超参数是允许为特定任务或数据集定制机器学习模型的选择点或配置点。

*   **超参数**:开发人员指定的模型配置参数，用于指导特定数据集的学习过程。

机器学习模型也有参数，这些参数是通过在训练数据集上训练或优化模型而设置的内部系数。

参数不同于超参数。参数自动学习；超参数是手动设置的，有助于指导学习过程。

有关参数和超参数之间区别的更多信息，请参见教程:

*   [参数和超参数有什么区别？](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)

通常，超参数在一般意义上对模型有已知的影响，但是不清楚如何为给定的数据集设置最佳的超参数。此外，许多机器学习模型具有一系列超参数，并且它们可能以非线性方式相互作用。

因此，通常需要搜索一组超参数，这些超参数会使模型在数据集上表现最佳。这称为超参数优化、超参数调整或超参数搜索。

优化过程包括定义搜索空间。这可以在几何学上被认为是 n 维体积，其中每个超参数代表不同的维度，维度的尺度是超参数可以采用的值，例如实值、整数值或分类。

*   **搜索空间**:每个维度代表一个超参数，每个点代表一个模型配置的待搜索体积。

搜索空间中的一个点是一个向量，每个超参数值都有一个特定的值。优化过程的目标是找到一个学习后模型性能最好的向量，如最大精度或最小误差。

可以使用一系列不同的优化算法，尽管最简单和最常见的两种方法是随机搜索和网格搜索。

*   **随机搜索**。将搜索空间定义为超参数值的有界域，并在该域中随机采样点。
*   **网格搜索**。将搜索空间定义为超参数值网格，并计算网格中的每个位置。

网格搜索非常适合抽查那些通常表现良好的组合。随机搜索非常适合于发现和获得你凭直觉无法猜到的超参数组合，尽管它通常需要更多的时间来执行。

有时使用更高级的方法，如[贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)和进化优化。

现在我们已经熟悉了超参数优化，让我们看看如何在 Python 中使用这个方法。

## 超参数优化套件-学习应用编程接口

scikit-learn Python 开源机器学习库提供了调整模型超参数的技术。

具体来说，它提供了用于随机搜索的[随机化搜索 CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 和用于网格搜索的[网格搜索 CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 。这两种技术都使用交叉验证来评估给定超参数向量的模型，因此每个类名都有“ *CV* ”后缀。

两个类都需要两个参数。首先是你正在优化的模型。这是模型的一个实例，其中包含要优化的超参数集的值。第二是搜索空间。这被定义为[字典](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)，其中名称是模型的超参数参数，值是离散值或随机搜索情况下要采样的值的分布。

```py
...
# define model
model = LogisticRegression()
# define search space
space = dict()
...
# define search
search = GridSearchCV(model, space)
```

这两个类都提供了“ *cv* ”参数，该参数允许指定整数个折叠，例如 5，或者配置交叉验证对象。我建议定义和指定一个交叉验证对象，以获得对模型评估的更多控制，并使评估过程清晰明了。

在分类任务的情况下，我建议使用[repeated stratifiedfold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)类，对于回归任务，我建议使用 [RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) 进行适当次数的折叠和重复，比如 10 次折叠和 3 次重复。

```py
...
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
search = GridSearchCV(..., cv=cv)
```

两个超参数优化类还提供了一个“*评分*”参数，该参数采用一个指示要优化的度量的字符串。

度量必须是最大化的，这意味着更好的模型导致更高的分数。对于分类，这可能是“*精度*”。对于回归，这是负误差度量，例如负版本的平均绝对误差的“ *neg_mean_absolute_error* ”，其中更接近零的值表示模型的预测误差更小。

```py
...
# define search
search = GridSearchCV(..., scoring='neg_mean_absolute_error')
```

您可以在这里看到一个内置评分指标列表:

*   [评分参数:定义模型评价规则](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

最后，搜索可以并行进行，例如，通过将“ *n_jobs* ”参数指定为系统内核数量的整数，例如 8，来使用所有的 CPU 内核。或者，您可以将其设置为-1，以自动使用系统中的所有内核。

```py
...
# define search
search = GridSearchCV(..., n_jobs=-1)
```

定义后，通过调用 *fit()* 函数并提供一个数据集来执行搜索，该数据集用于使用交叉验证来训练和评估模型超参数组合。

```py
...
# execute search
result = search.fit(X, y)
```

运行搜索可能需要几分钟或几小时，具体取决于搜索空间的大小和硬件的速度。你会经常想要根据你有多少时间来定制搜索，而不是搜索什么的可能性。

在搜索结束时，您可以通过类的属性访问所有结果。也许最重要的属性是观察到的**最佳得分**和获得最佳得分的**超参数**。

```py
...
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

一旦知道了获得最佳结果的超参数集，就可以定义一个新模型，设置每个超参数的值，然后在所有可用数据上拟合该模型。这个模型可以用来对新数据进行预测。

现在我们已经熟悉了 scikit-learn 中的超参数优化 API，让我们来看看一些工作示例。

## 分类的超参数优化

在本节中，我们将使用超参数优化来发现声纳数据集的性能良好的模型配置。

声纳数据集是一个标准的机器学习数据集，包括 208 行具有 60 个数字输入变量的数据和一个具有两个类值的目标变量，例如二进制分类。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以达到大约 53%的准确率。一个性能最好的模型在同样的测试设备上可以达到 88%的准确率。这提供了此数据集的预期性能范围。

该数据集包括预测声纳回波显示的是岩石还是模拟地雷。

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集描述(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```py
# summarize the sonar dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 208 行数据，60 个输入变量。

```py
(208, 60) (208,)
```

接下来，让我们使用随机搜索为声纳数据集找到一个好的模型配置。

为了简单起见，我们将重点介绍线性模型、逻辑回归模型以及为此模型调整的常用超参数。

### 分类的随机搜索

在本节中，我们将探讨声纳数据集上逻辑回归模型的超参数优化。

首先，我们将定义将被优化的模型，并对将不被优化的超参数使用默认值。

```py
...
# define model
model = LogisticRegression()
```

我们将使用[重复分层 k 折叠交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)评估模型配置，重复 3 次，折叠 10 次。

```py
...
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

接下来，我们可以定义搜索空间。

这是一个字典，其中名称是模型的参数，值是从中抽取样本的分布。我们将优化模型的*解算器*、*惩罚*和 *C* 超参数，其中解算器和惩罚类型具有离散分布，而[对数均匀](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html)分布对于 *C* 值从 1e-5 到 100。

对数均匀性对于搜索惩罚值很有用，因为我们经常探索不同数量级的值，至少作为第一步。

```py
...
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
```

接下来，我们可以用所有这些元素定义搜索过程。

重要的是，我们必须通过“ *n_iter* ”参数设置要从搜索空间中抽取的迭代或样本的数量。在这种情况下，我们将它设置为 500。

```py
...
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
```

最后，我们可以执行优化并报告结果。

```py
...
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

将这些联系在一起，完整的示例如下所示。

```py
# random search logistic regression model on the sonar dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

运行该示例可能需要一分钟时间。它很快，因为我们使用了一个小的搜索空间和一个快速模型来拟合和评估。在优化无效配置组合的过程中，您可能会看到一些警告。这些可以放心地忽略。

在运行结束时，会报告获得最佳性能的最佳分数和超参数配置。

鉴于优化过程的随机性，您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到最佳配置实现了大约 78.9%的精度，这是公平的，并且用于实现该分数的*求解器*、*惩罚*和 *C* 超参数的具体值。

```py
Best Score: 0.7897619047619049
Best Hyperparameters: {'C': 4.878363034905756, 'penalty': 'l2', 'solver': 'newton-cg'}
```

接下来，让我们使用网格搜索为声纳数据集找到一个好的模型配置。

### 用于分类的网格搜索

使用网格搜索很像使用随机搜索进行分类。

主要区别在于搜索空间必须是要搜索的离散网格。这意味着我们可以在对数尺度上指定离散值，而不是对 *C* 使用对数均匀分布。

```py
...
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
```

另外， [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 类不需要多次迭代，因为我们只评估网格中超参数的组合。

```py
...
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
```

将这些联系在一起，下面列出了声纳数据集的网格搜索逻辑回归配置的完整示例。

```py
# grid search logistic regression model on the sonar dataset
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

运行该示例可能需要一些时间。它很快，因为我们使用了一个小的搜索空间和一个快速模型来拟合和评估。同样，在优化过程中，您可能会看到一些关于无效配置组合的警告。这些可以放心地忽略。

在运行结束时，会报告获得最佳性能的最佳分数和超参数配置。

鉴于优化过程的随机性，您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到最佳配置实现了大约 78.2%的精度，这也是公平的，并且用于实现该分数的*求解器*、*惩罚*和 *C* 超参数的具体值。有趣的是，结果与通过随机搜索找到的结果非常相似。

```py
Best Score: 0.7828571428571429
Best Hyperparameters: {'C': 1, 'penalty': 'l2', 'solver': 'newton-cg'}
```

## 回归的超参数优化

在本节中，我们将使用超级优化来发现汽车保险数据集的最佳模型配置。

汽车保险数据集是一个标准的机器学习数据集，由 63 行数据组成，包括一个数字输入变量和一个数字目标变量。

使用 3 次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 66 的平均绝对误差(MAE)。一个性能最好的模型可以在大约 28 的相同测试线束上实现 MAE。这提供了此数据集的预期性能范围。

考虑到不同地理区域的索赔数量，数据集包括预测索赔总额(千瑞典克朗)。

*   [车险数据集(auto-insurance.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv)
*   [车险数据集描述(车险.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.names)

不需要下载数据集，我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```py
# summarize the auto insurance dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 63 行数据，1 个输入变量。

```py
(63, 1) (63,)
```

接下来，我们可以使用超参数优化为汽车保险数据集找到一个好的模型配置。

为了简单起见，我们将重点介绍一个线性模型，[线性回归模型](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)以及为该模型调整的常用超参数。

### 回归的随机搜索

为回归配置和使用随机搜索超参数优化过程非常类似于将其用于分类。

在这种情况下，我们将配置线性回归实现的重要超参数，包括*求解器*、*α*、*拟合 _ 截距*、*归一化*。

除了作为惩罚项的“*α*”参数之外，我们将在搜索空间中对所有值使用离散分布，在这种情况下，我们将使用对数均匀分布，就像我们在前面一节中对逻辑回归的“ *C* ”参数所做的那样。

```py
...
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
```

回归与分类的主要区别在于评分方法的选择。

对于回归，性能通常使用误差来衡量，误差被最小化，零代表具有完美技能的模型。scikit-learn 中的超参数优化程序假定分数最大化。因此，每个误差度量的版本被提供为负。

这意味着大的正误差变成大的负误差，好的表现是小的负值接近于零，完美的技巧是零。

当解释结果时，负 MAE 的符号可以忽略。

在这种情况下，我们将意味着绝对误差(MAE)，通过将“*评分*”参数设置为“ *neg_mean_absolute_error* ，可以获得该误差的最大化版本。

```py
...
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)
```

将这些联系在一起，完整的示例如下所示。

```py
# random search linear regression model on the auto insurance dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

运行该示例可能需要一些时间。它很快，因为我们使用了一个小的搜索空间和一个快速模型来拟合和评估。在优化无效配置组合的过程中，您可能会看到一些警告。这些可以放心地忽略。

在运行结束时，会报告获得最佳性能的最佳分数和超参数配置。

鉴于优化过程的随机性，您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到最佳配置实现了大约 29.2 的 MAE，这非常接近模型上的最佳性能。然后，我们可以看到实现这一结果的特定超参数值。

```py
Best Score: -29.23046315344758
Best Hyperparameters: {'alpha': 0.008301451461243866, 'fit_intercept': True, 'normalize': True, 'solver': 'sag'}
```

接下来，让我们使用网格搜索为汽车保险数据集找到一个好的模型配置。

### 回归的网格搜索

作为网格搜索，我们不能定义要采样的分布，而必须定义超参数值的离散网格。因此，我们将把“ *alpha* ”参数指定为一个对数-10 标度的数值范围。

```py
...
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
```

回归的网格搜索需要指定“*评分*”，就像我们对随机搜索所做的那样。

在这种情况下，我们将再次使用负 MAE 评分函数。

```py
...
# define search
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
```

将这些联系在一起，下面列出了汽车保险数据集的网格搜索线性回归配置的完整示例。

```py
# grid search linear regression model on the auto insurance dataset
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
# define search
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
```

运行该示例可能需要一分钟时间。它很快，因为我们使用了一个小的搜索空间和一个快速模型来拟合和评估。同样，在优化过程中，您可能会看到一些关于无效配置组合的警告。这些可以放心地忽略。

在运行结束时，会报告获得最佳性能的最佳分数和超参数配置。

鉴于优化过程的随机性，您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到最佳配置实现了大约 29.2 的 MAE，这与我们在前面部分中使用随机搜索实现的 MAE 几乎相同。有趣的是，超参数也几乎相同，这是很好的证实。

```py
Best Score: -29.275708614337326
Best Hyperparameters: {'alpha': 0.1, 'fit_intercept': True, 'normalize': False, 'solver': 'sag'}
```

## 超参数优化的常见问题

本节讨论一些关于超参数优化的常见问题。

### 如何在随机搜索和网格搜索之间进行选择？

根据你的需要选择方法。我建议从网格开始，如果你有时间的话，做一个随机搜索。

网格搜索适用于已知通常性能良好的超参数值的小型快速搜索。

随机搜索适用于发现新的超参数值或超参数的新组合，通常会带来更好的性能，尽管可能需要更多时间来完成。

### 如何加快超参数优化？

确保将“ *n_jobs* ”参数设置为机器上的核心数量。

之后，更多建议包括:

*   对数据集的较小样本进行评估。
*   探索更小的搜索空间。
*   使用较少的重复和/或折叠进行交叉验证。
*   在更快的机器上执行搜索，例如 AWS EC2。
*   使用评估速度更快的替代模型。

### 如何选择超参数进行搜索？

大多数算法都有一个对搜索过程影响最大的超参数子集。

这些都在算法的大多数描述中列出。例如，以下是一些算法及其最重要的超参数:

*   [调整分类机器学习算法的超参数](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)

如果您不确定:

*   复习使用该算法获得想法的论文。
*   查看应用编程接口和算法文档以获得想法。
*   搜索所有超参数。

### 如何使用性能最佳的超参数？

定义一个新模型，并将模型的超参数值设置为通过搜索找到的值。

然后在所有可用数据上拟合模型，并使用模型开始对新数据进行预测。

这叫做准备最终模型。在此查看更多信息:

*   [如何训练最终的机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)

### 如何做预测？

首先，拟合最终模型(上一个问题)。

然后调用 *predict()* 函数进行预测。

有关使用最终模型进行预测的示例，请参见教程:

*   [如何使用 scikit 进行预测-学习](https://machinelearningmastery.com/make-predictions-scikit-learn/)

**关于超参数优化，还有问题吗？**
在下面的评论里告诉我。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [参数和超参数有什么区别？](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
*   [标准分类和回归机器学习数据集的结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [调整分类机器学习算法的超参数](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)
*   [如何训练最终的机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)
*   [如何使用 scikit 进行预测-学习](https://machinelearningmastery.com/make-predictions-scikit-learn/)

### 蜜蜂

*   [调整估计器的超参数，scikit-learn 文档](https://scikit-learn.org/stable/modules/grid_search.html)。
*   [sklearn.model_selection。GridSearchCV API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 。
*   [硬化. model_selection。random mizedsearchv API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)。

### 文章

*   [超参数优化，维基百科](https://en.wikipedia.org/wiki/Hyperparameter_optimization)。

## 摘要

在本教程中，您在 Python 中发现了机器学习的超参数优化。

具体来说，您了解到:

*   需要超参数优化来充分利用机器学习模型。
*   如何为分类任务配置随机和网格搜索超参数优化？
*   如何为回归任务配置随机和网格搜索超参数优化？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。