# sci kit-优化机器学习中的超参数调整

> 原文：<https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/>

最后更新于 2020 年 11 月 6 日

超参数优化指的是执行搜索，以便发现一组特定的模型配置参数，从而在特定数据集上获得模型的最佳性能。

有许多方法可以执行超参数优化，尽管现代方法，如贝叶斯优化，是快速有效的。 **Scikit-Optimize 库**是一个开源 Python 库，它提供了贝叶斯优化的实现，可用于调整 scikit-Learn Python 库中机器学习模型的超参数。

您可以很容易地使用 Scikit-Optimize 库在下一个机器学习项目中调整模型。

在本教程中，您将发现如何使用 Scikit-Optimize 库来使用贝叶斯优化进行超参数调整。

完成本教程后，您将知道:

*   Scikit-Optimize 为贝叶斯优化提供了一个通用工具包，可用于超参数调整。
*   如何手动使用 Scikit-Optimize 库来调整机器学习模型的超参数。
*   如何使用内置的 BayesSearchCV 类执行模型超参数调优。

我们开始吧。

*   **2020 年 11 月更新**:更新了中断的 API 链接，因为 skopt 网站发生了变化。

![Scikit-Optimize for Hyperparameter Tuning in Machine Learning](img/5acb0703652cf41e44b2db767c6753f1.png)

scikit-优化机器学习中的超参数调整
图片由[丹·内维尔](https://flickr.com/photos/dnevill/44893231465/)提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  sci kit-优化
2.  机器学习数据集和模型
3.  手动调整算法超参数
4.  自动调整算法超参数

## sci kit-优化

Scikit-Optimize，简称 skopt，是一个用于执行优化任务的开源 Python 库。

它提供了有效的优化算法，如贝叶斯优化，并可用于寻找任意成本函数的最小值或最大值。

贝叶斯优化提供了一种基于贝叶斯定理的有原则的技术，用于指导高效且有效的全局优化问题的搜索。它的工作原理是建立一个目标函数的概率模型，称为替代函数，然后在选择候选样本对真实目标函数进行评估之前，使用获取函数对其进行有效搜索。

有关贝叶斯优化主题的更多信息，请参见教程:

*   [如何在 Python 中从头实现贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)

重要的是，该库为调整 scikit-learn 库提供的机器学习算法的超参数提供了支持，即所谓的超参数优化。因此，它为效率较低的超参数优化过程(如网格搜索和随机搜索)提供了一种有效的替代方案。

scikit-optimize 库可以使用 pip 安装，如下所示:

```py
sudo pip install scikit-optimize
```

安装后，我们可以导入库并打印版本号，以确认库安装成功并可以访问。

下面列出了完整的示例。

```py
# report scikit-optimize version number
import skopt
print('skopt %s' % skopt.__version__)
```

运行该示例会报告 scikit-optimize 的当前安装版本号。

您的版本号应该相同或更高。

```py
skopt 0.7.2
```

有关更多安装说明，请参见文档:

*   [Scikit-优化安装说明](https://scikit-optimize.github.io/stable/install.html)

现在我们已经熟悉了什么是 Scikit-Optimize 以及如何安装它，让我们来探索如何使用它来调整机器学习模型的超参数。

## 机器学习数据集和模型

首先，让我们选择一个标准数据集和一个模型来处理它。

我们将使用电离层机器学习数据集。这是一个标准的机器学习数据集，包括 351 行数据，其中有三个数字输入变量和一个目标变量，目标变量有两个类值，例如二进制分类。

使用带有三次重复的[重复分层 10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)的测试工具，一个简单的模型可以达到大约 64%的准确率。一个性能最好的模型可以在同样的测试设备上达到大约 94%的精确度。这提供了此数据集的预期性能范围。

该数据集包括预测电离层测量值是否表明特定结构。

您可以在此了解有关数据集的更多信息:

*   [电离层数据集(电离层. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv)
*   [电离层数据集描述(电离层.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```py
# summarize the ionosphere dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 351 行数据，34 个输入变量。

```py
(351, 34) (351,)
```

我们可以使用重复的分层交叉验证在这个数据集上评估[支持向量机](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (SVM)模型。

我们可以报告数据集上所有折叠和重复的平均模型性能，这将为后面章节中执行的模型超参数调整提供参考。

下面列出了完整的示例。

```py
# evaluate an svm for the ionosphere dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# define model model
model = SVC()
# define test harness
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
m_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

运行该示例首先加载和准备数据集，然后对数据集评估 SVM 模型。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到具有默认超参数的 SVM 实现了大约 93.7%的平均分类准确率，这是熟练的，并且接近 94%的问题上的最高性能。

```py
(351, 34) (351,)
Accuracy: 0.937 (0.038)
```

接下来，让我们看看是否可以通过使用 scikit-optimize 库调整模型超参数来提高性能。

## 手动调整算法超参数

Scikit-Optimize 库可用于调整机器学习模型的超参数。

我们可以通过使用库的贝叶斯优化功能来手动实现这一点。

这要求我们首先定义一个搜索空间。在这种情况下，这将是我们希望调整的模型的超参数，以及每个超参数的范围。

我们将调整 SVM 模型的以下超参数:

*   **C** ，正则化参数。
*   **内核**，模型中使用的内核类型。
*   **次**，用于多项式核。
*   **γ**，用于大多数其他内核。

对于数值超参数 *C* 和*γ*，我们将定义一个对数标度，在 1e-6 和 100 的小值之间进行搜索。*度*是一个整数，我们将搜索 1 到 5 之间的值。最后，*内核*是一个具有特定命名值的类别变量。

我们可以为这四个超参数定义搜索空间，它们是来自 skopt 库的数据类型列表，如下所示:

```py
...
# define the space of hyperparameters to search
search_space = list()
search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
search_space.append(Integer(1, 5, name='degree'))
search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))
```

请注意为每个参数指定的数据类型、范围和超参数名称。

然后，我们可以定义一个将由搜索过程调用的函数。这是优化过程稍后期望的函数，它获取模型和模型的特定超参数集，对其进行评估，并返回超参数集的分数。

在我们的案例中，我们希望在电离层数据集上使用重复的分层 10 倍交叉验证来评估模型。我们希望最大化分类精确率，例如，找到给出最佳精确率的模型超参数集。默认情况下，该过程最小化从该函数返回的分数，因此，我们将返回 1 减去准确性，例如，完美技能将是(1–准确性)或 0.0，最差技能将是 1.0。

下面的 *evaluate_model()* 函数实现了这一点，并获取了一组特定的超参数。

```py
# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
	# configure the model with specific hyperparameters
	model = SVC()
	model.set_params(**params)
	# define test harness
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# calculate 5-fold cross validation
	result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
	# calculate the mean of the scores
	estimate = mean(result)
	# convert from a maximizing score to a minimizing score
	return 1.0 - estimate
```

接下来，我们可以通过调用*gp _ minimum()*函数来执行搜索，并传递要调用的函数的名称来评估每个模型和要优化的搜索空间。

```py
...
# perform optimization
result = gp_minimize(evaluate_model, search_space)
```

该过程将一直运行，直到它收敛并返回结果。

结果对象包含许多细节，但重要的是，我们可以访问最佳性能配置的分数和最佳成形模型使用的超参数。

```py
...
# summarizing finding:
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: %s' % (result.x))
```

下面列出了手动调整电离层数据集上 SVM 超参数的完整示例。

```py
# manually tune svm model hyperparameters using skopt on the ionosphere dataset
from numpy import mean
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# define the space of hyperparameters to search
search_space = list()
search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
search_space.append(Integer(1, 5, name='degree'))
search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
	# configure the model with specific hyperparameters
	model = SVC()
	model.set_params(**params)
	# define test harness
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# calculate 5-fold cross validation
	result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
	# calculate the mean of the scores
	estimate = mean(result)
	# convert from a maximizing score to a minimizing score
	return 1.0 - estimate

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# perform optimization
result = gp_minimize(evaluate_model, search_space)
# summarizing finding:
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: %s' % (result.x))
```

运行该示例可能需要一些时间，具体取决于机器的速度。

您可能会看到一些可以安全忽略的警告消息，例如:

```py
UserWarning: The objective has been evaluated at this point before.
```

运行结束时，会报告性能最佳的配置。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，按照搜索空间列表的顺序报告的配置是适度的 *C* 值、径向基函数*核*、2 的*度*(被径向基函数核忽略)和适度的*γ*值。

重要的是，我们可以看到这个模型的技能大约是 94.7%，这是一个表现最好的模型

```py
(351, 34) (351,)
Best Accuracy: 0.948
Best Parameters: [1.2852670137769258, 'rbf', 2, 0.18178016885627174]
```

这不是使用 Scikit-Optimize 库进行超参数调整的唯一方法。在下一节中，我们可以看到一种更自动化的方法。

## 自动调整算法超参数

Scikit-Learn 机器学习库提供了调整模型超参数的工具。

具体来说，它提供了 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 和[随机化搜索 CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 类，这些类采用一个模型、一个搜索空间和一个交叉验证配置。

这些类的好处是搜索过程是自动执行的，只需要最少的配置。

类似地，Scikit-Optimize 库提供了一个类似的界面，用于通过[Bayesarccv](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html)类执行模型超参数的贝叶斯优化。

这个类可以用与 Scikit-Learn 等价类相同的方式使用。

首先，搜索空间必须定义为一个字典，其中超参数名称用作关键字，变量的范围用作值。

```py
...
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
```

然后，我们可以定义*Bayesarccv*配置，采用我们希望评估的模型、超参数搜索空间和交叉验证配置。

```py
...
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
```

然后，我们可以执行搜索，并在最后报告最佳结果和配置。

```py
...
# perform the search
search.fit(X, y)
# report the best result
print(search.best_score_)
print(search.best_params_)
```

将这些联系在一起，下面列出了使用电离层数据集上的 BayesSearchCV 类自动调整 SVM 超参数的完整示例。

```py
# automatic svm hyperparameter tuning using skopt for the ionosphere dataset
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X, y)
# report the best result
print(search.best_score_)
print(search.best_params_)
```

运行该示例可能需要一些时间，具体取决于机器的速度。

您可能会看到一些可以安全忽略的警告消息，例如:

```py
UserWarning: The objective has been evaluated at this point before.
```

运行结束时，会报告性能最佳的配置。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，该模型的性能高于表现最好的模型，平均分类准确率约为 95.2%。

搜索发现了一个大的 *C* 值、一个径向基函数*核*和一个小的*γ*值。

```py
(351, 34) (351,)
0.9525166191832859
OrderedDict([('C', 4.8722263953328735), ('degree', 4), ('gamma', 0.09805881007239009), ('kernel', 'rbf')])
```

这提供了一个模板，您可以使用它来调整机器学习项目中的超参数。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [标准分类和回归机器学习数据集的结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [如何在 Python 中从头实现贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)

### 蜜蜂

*   [Scikit-优化主页](https://scikit-optimize.github.io/)。
*   [Scikit-优化 API](https://scikit-optimize.github.io/stable/modules/classes.html) 。
*   [Scikit-优化安装说明](https://scikit-optimize.github.io/stable/install.html)

## 摘要

在本教程中，您发现了如何使用 Scikit-Optimize 库将贝叶斯优化用于超参数调整。

具体来说，您了解到:

*   Scikit-Optimize 为贝叶斯优化提供了一个通用工具包，可用于超参数调整。
*   如何手动使用 Scikit-Optimize 库来调整机器学习模型的超参数。
*   如何使用内置的 BayesSearchCV 类执行模型超参数调优。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。