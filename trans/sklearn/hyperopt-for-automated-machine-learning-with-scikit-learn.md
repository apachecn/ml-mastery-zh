# 通过 Scikit-Learn 实现自动化机器学习的 HyperOpt】

> 原文:[https://machinelearning master . com/hyperopt-for-automatic-machine-learning-with-sci kit-learn/](https://machinelearningmastery.com/hyperopt-for-automated-machine-learning-with-scikit-learn/)

自动机器学习(AutoML)指的是在很少用户参与的情况下，自动发现用于预测建模任务的性能良好的模型的技术。

HyperOpt 是一个面向大规模 AutoML 的开源库，HyperOpt-Sklearn 是 HyperOpt 的包装器，支持 AutoML 与 HyperOpt 一起用于流行的 Scikit-Learn 机器学习库，包括一套数据准备转换以及分类和回归算法。

在本教程中，您将发现如何使用 HyperOpt 通过 Python 中的 Scikit-Learn 进行自动机器学习。

完成本教程后，您将知道:

*   Hyperopt-Sklearn 是一个面向 AutoML 的开源库，具有 scikit-learn 数据准备和机器学习模型。
*   如何使用 Hyperopt-Sklearn 为分类任务自动发现表现最好的模型。
*   如何使用 Hyperopt-Sklearn 为回归任务自动发现表现最好的模型。

我们开始吧。

![HyperOpt for Automated Machine Learning With Scikit-Learn](img/2ffc96e25d2fef157cb913519166810c.png)

尼尔·威廉森为自动化机器学习拍摄的超级照片，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  超选择性和超选择性硬化
2.  如何安装和使用 HyperOpt-Sklearn
3.  超选择性硬化用于分类
4.  回归超点-Sklearn

## 超选择性和超选择性硬化

[HyperOpt](https://hyperopt.github.io/hyperopt/) 是由 [James Bergstra](https://www.linkedin.com/in/james-bergstra) 开发的用于贝叶斯优化的开源 Python 库。

它专为具有数百个参数的模型的大规模优化而设计，并允许优化过程跨多个内核和多台机器进行扩展。

该库明确用于优化机器学习管道，包括数据准备、模型选择和模型超参数。

> 我们的方法是展示如何根据超参数计算性能指标(例如，验证示例的分类精度)的底层表达式图，这些超参数不仅控制如何应用各个处理步骤，甚至还控制包括哪些处理步骤。

——[做模型搜索的科学:视觉架构的数百维超参数优化](https://dl.acm.org/doi/10.5555/3042817.3042832)，2013。

HyperOpt 很难直接使用，需要仔细指定优化程序和搜索空间。

HyperOpt 的一个扩展被创建为 [HyperOpt-Sklearn](https://hyperopt.github.io/hyperopt-sklearn/) ，它允许 HyperOpt 过程被应用于流行的 [Scikit-Learn 开源机器学习库](https://scikit-learn.org/stable/)提供的数据准备和机器学习模型。

HyperOpt-Sklearn 包装了 HyperOpt 库，允许自动搜索数据准备方法、机器学习算法以及用于分类和回归任务的模型超参数。

> ……我们介绍 Hyperopt-Sklearn:一个为 Python 和 scikit-learn 的用户带来自动算法配置好处的项目。Hyperopt-Sklearn 使用 Hyperopt 来描述 Scikit-Learn 组件的可能配置的搜索空间，包括预处理和分类模块。

——[Hyperopt-Sklearn:sci kit-Learn 的自动超参数配置](https://conference.scipy.org/proceedings/scipy2014/pdfs/komer.pdf)，2014。

现在我们已经熟悉了 HyperOpt 和 HyperOpt-Sklearn，下面我们来看看如何使用 HyperOpt-Sklearn。

## 如何安装和使用 HyperOpt-Sklearn

第一步是安装 HyperOpt 库。

这可以通过使用 pip 包管理器来实现，如下所示:

```
sudo pip install hyperopt
```

安装后，我们可以通过键入以下命令来确认安装成功并检查库的版本:

```
sudo pip show hyperopt
```

这将总结 HyperOpt 的安装版本，确认正在使用现代版本。

```
Name: hyperopt
Version: 0.2.3
Summary: Distributed Asynchronous Hyperparameter Optimization
Home-page: http://hyperopt.github.com/hyperopt/
Author: James Bergstra
Author-email: james.bergstra@gmail.com
License: BSD
Location: ...
Requires: tqdm, six, networkx, future, scipy, cloudpickle, numpy
Required-by:
```

接下来，我们必须安装 HyperOpt-Sklearn 库。

这也可以使用 pip 安装，尽管我们必须通过克隆存储库并从本地文件运行安装来手动执行此操作，如下所示:

```
git clone git@github.com:hyperopt/hyperopt-sklearn.git
cd hyperopt-sklearn
sudo pip install .
cd ..
```

同样，我们可以通过使用以下命令检查版本号来确认安装成功:

```
sudo pip show hpsklearn
```

这将总结 HyperOpt-Sklearn 的安装版本，确认正在使用现代版本。

```
Name: hpsklearn
Version: 0.0.3
Summary: Hyperparameter Optimization for sklearn
Home-page: http://hyperopt.github.com/hyperopt-sklearn/
Author: James Bergstra
Author-email: anon@anon.com
License: BSD
Location: ...
Requires: nose, scikit-learn, numpy, scipy, hyperopt
Required-by:
```

现在已经安装了所需的库，我们可以查看 HyperOpt-Sklearn API 了。

使用 HyperOpt-Sklearn 很简单。搜索过程是通过创建和配置 HyperoptEstimator 类的实例来定义的。

用于搜索的算法可以通过“ *algo* ”参数指定，搜索中执行的评估数量通过“ *max_evals* ”参数指定，并且可以通过“ *trial_timeout* ”参数对评估每个管道施加限制。

```
...
# define search
model = HyperoptEstimator(..., algo=tpe.suggest, max_evals=50, trial_timeout=120)
```

有许多不同的优化算法，包括:

*   随机搜索
*   Parzen 估计量树
*   热处理
*   树
*   高斯过程树

Parzen 估值器的“*树”是一个很好的默认值，你可以在论文“[超参数优化算法](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)中了解更多算法类型。[PDF]"*

对于分类任务，*分类器*参数指定模型的搜索空间，对于回归，*回归器*参数指定模型的搜索空间，两者都可以设置为使用库提供的预定义模型列表，例如*any _ 分类器*、*any _ 回归器*。

同样，数据准备的搜索空间是通过“*预处理*”参数指定的，也可以通过“*any _ premization*使用预定义的预处理步骤列表。

```
...
# define search
model = HyperoptEstimator(classifier=any_classifier('cla'), preprocessing=any_preprocessing('pre'), ...)
```

有关搜索的其他参数的更多信息，您可以直接查看该类的源代码:

*   [超预测器类的参数](https://github.com/hyperopt/hyperopt-sklearn/blob/master/hpsklearn/estimator.py#L429)

一旦定义了搜索，就可以通过调用 *fit()* 函数来执行。

```
...
# perform the search
model.fit(X_train, y_train)
```

在运行结束时，通过调用 *score()* 函数，可以在新数据上评估性能最佳的模型。

```
...
# summarize performance
acc = model.score(X_test, y_test)
print("Accuracy: %.3f" % acc)
```

最后，我们可以通过 *best_model()* 函数检索在训练数据集上表现最好的变换、模型和模型配置的*管道*。

```
...
# summarize the best model
print(model.best_model())
```

现在我们已经熟悉了这个应用编程接口，让我们来看看一些工作示例。

## 超选择性硬化用于分类

在本节中，我们将使用 HyperOpt-Sklearn 来发现声纳数据集的模型。

声纳数据集是一个标准的机器学习数据集，由 208 行数据组成，包含 60 个数字输入变量和一个具有两个类值的目标变量，例如二进制分类。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以达到大约 53%的准确率。一个性能最好的模型在同样的测试设备上可以达到 88%的准确率。这提供了此数据集的预期性能范围。

该数据集包括预测声纳回波显示的是岩石还是模拟地雷。

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集描述(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```
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

```
(208, 60) (208,)
```

接下来，让我们使用 HyperOpt-Sklearn 为声纳数据集找到一个好的模型。

我们可以执行一些基本的数据准备，包括将目标字符串转换为类标签，然后将数据集拆分为训练集和测试集。

```
...
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

接下来，我们可以定义搜索过程。我们将探索该库可用的所有分类算法和所有数据转换，并使用“超参数优化的算法”中描述的 TPE 或 Parzen 估计树搜索算法

搜索将评估 50 条管道，并将每次评估限制在 30 秒内。

```
...
# define search
model = HyperoptEstimator(classifier=any_classifier('cla'), preprocessing=any_preprocessing('pre'), algo=tpe.suggest, max_evals=50, trial_timeout=30)
```

然后我们开始搜索。

```
...
# perform the search
model.fit(X_train, y_train)
```

在运行结束时，我们将报告模型在保持数据集上的性能，并总结性能最好的管道。

```
...
# summarize performance
acc = model.score(X_test, y_test)
print("Accuracy: %.3f" % acc)
# summarize the best model
print(model.best_model())
```

将这些联系在一起，完整的示例如下所示。

```
# example of hyperopt-sklearn for the sonar classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define search
model = HyperoptEstimator(classifier=any_classifier('cla'), preprocessing=any_preprocessing('pre'), algo=tpe.suggest, max_evals=50, trial_timeout=30)
# perform the search
model.fit(X_train, y_train)
# summarize performance
acc = model.score(X_test, y_test)
print("Accuracy: %.3f" % acc)
# summarize the best model
print(model.best_model())
```

运行该示例可能需要几分钟时间。

将报告搜索进度，您将看到一些可以安全忽略的警告。

运行结束时，在保持数据集上评估性能最佳的模型，并打印发现的管道供以后使用。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到选择的模型在保持测试集上达到了大约 85.5%的准确率。管道涉及没有预处理的梯度增强模型。

```
Accuracy: 0.855
{'learner': GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.009132299586303643, loss='deviance',
                           max_depth=None, max_features='sqrt',
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=342, n_iter_no_change=None,
                           presort='auto', random_state=2,
                           subsample=0.6844206624548879, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

然后可以直接使用打印的模型，例如将代码复制粘贴到另一个项目中。

接下来，让我们看一下使用 HyperOpt-Sklearn 解决回归预测建模问题。

## 回归超点-Sklearn

在本节中，我们将使用 HyperOpt-Sklearn 来发现房屋数据集的模型。

外壳数据集是一个标准的机器学习数据集，由 506 行数据组成，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的平均绝对误差(MAE)。一个性能最好的模型可以在大约 1.9 的相同测试线束上实现 MAE。这提供了此数据集的预期性能范围。

该数据集包括预测美国波士顿市郊住宅区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```
# summarize the auto insurance dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 63 行数据带有一个输入变量。

```
(208, 60), (208,)
```

接下来，我们可以使用 HyperOpt-Sklearn 为汽车保险数据集找到一个好的模型。

使用 HyperOpt-Sklearn 进行回归与使用它进行分类相同，只是必须指定“*回归器*”参数。

在这种情况下，我们想要优化 MAE，因此，我们将把“ *loss_fn* ”参数设置为 scikit-learn 库提供的 *mean_absolute_error()* 函数。

```
...
# define search
model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('pre'), loss_fn=mean_absolute_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
```

将这些联系在一起，完整的示例如下所示。

```
# example of hyperopt-sklearn for the housing regression dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
from hyperopt import tpe
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
data = data.astype('float32')
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define search
model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('pre'), loss_fn=mean_absolute_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
# perform the search
model.fit(X_train, y_train)
# summarize performance
mae = model.score(X_test, y_test)
print("MAE: %.3f" % mae)
# summarize the best model
print(model.best_model())
```

运行该示例可能需要几分钟时间。

将报告搜索进度，您将看到一些可以安全忽略的警告。

在运行结束时，在保持数据集上评估性能最佳的模型，并打印发现的管道供以后使用。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到选择的模型在保持测试集上实现了大约 0.883 的 MAE，这看起来很巧妙。管道包含一个没有预处理的*xgbreversor*模型。

**注意**:要搜索使用 xboost，必须安装[xboost 库](https://machinelearningmastery.com/install-xgboost-python-macos/)。

```
MAE: 0.883
{'learner': XGBRegressor(base_score=0.5, booster='gbtree',
             colsample_bylevel=0.5843250948679669, colsample_bynode=1,
             colsample_bytree=0.6635160670570662, gamma=6.923399395303031e-05,
             importance_type='gain', learning_rate=0.07021104887683309,
             max_delta_step=0, max_depth=3, min_child_weight=5, missing=nan,
             n_estimators=4000, n_jobs=1, nthread=None, objective='reg:linear',
             random_state=0, reg_alpha=0.5690202874759704,
             reg_lambda=3.3098341637038, scale_pos_weight=1, seed=1,
             silent=None, subsample=0.7194797262656784, verbosity=1), 'preprocs': (), 'ex_preprocs': ()}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [做模型搜索的科学:视觉架构的数百维超参数优化](https://dl.acm.org/doi/10.5555/3042817.3042832)，2013。
*   [Hyperopt GitHub 项目](https://github.com/hyperopt/hyperopt)。
*   [超选项主页](https://hyperopt.github.io/hyperopt/)。
*   [超选择性硬化 GitHub 项目](https://github.com/hyperopt/hyperopt-sklearn)。
*   超选择性硬化主页。
*   [Hyperopt-Sklearn:sci kit-Learn 的自动超参数配置](https://conference.scipy.org/proceedings/scipy2014/pdfs/komer.pdf)，2014。

## 摘要

在本教程中，您发现了如何使用 Python 中的 Scikit-Learn 将 HyperOpt 用于自动机器学习。

具体来说，您了解到:

*   Hyperopt-Sklearn 是一个面向 AutoML 的开源库，具有 scikit-learn 数据准备和机器学习模型。
*   如何使用 Hyperopt-Sklearn 为分类任务自动发现表现最好的模型。
*   如何使用 Hyperopt-Sklearn 为回归任务自动发现表现最好的模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。