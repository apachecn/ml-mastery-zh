# Python 自动机器学习（AutoML）库

> 原文：<https://machinelearningmastery.com/automl-libraries-for-python/>

AutoML 提供了一些工具，可以在很少用户干预的情况下，为数据集自动发现良好的机器学习模型管道。

对于刚接触机器学习的领域专家或希望快速获得预测建模任务良好结果的机器学习从业者来说，这是理想的选择。

开源库可用于使用 Python 中流行的机器学习库的 AutoML 方法，例如 Sklearn 机器学习库。

在本教程中，您将发现如何在 Python 中使用顶级开源 AutoML 库进行 Sklearn。

完成本教程后，您将知道:

*   AutoML 是为预测建模任务自动快速发现表现良好的机器学习模型管道的技术。
*   Sklearn 最受欢迎的三个自动库是 Hyperopt-Sklearn、Auto-Sklearn 和 TPOT。
*   如何在 Python 中使用 AutoML 库为预测建模任务发现表现良好的模型。

我们开始吧。

![Automated Machine Learning (AutoML) Libraries for Python](img/1bec0c962c0c0698eddd8cdadb25b94f.png)

Python 的自动机器学习(AutoML)库
图片由 [Michael Coghlan](https://flickr.com/photos/mikecogh/43679370712/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  自动机器学习
2.  汽车硬化
3.  基于树的管道优化工具(TPOT)
4.  超选择性硬化

## 自动机器学习

自动机器学习，简称 AutoML，涉及为预测建模任务自动选择数据准备、机器学习模型和模型超参数。

它指的是允许半复杂的机器学习实践者和非专家快速地为他们的机器学习任务发现一个好的预测模型管道的技术，除了提供数据集之外，几乎没有干预。

> …用户只需提供数据，AutoML 系统就会自动确定最适合该特定应用的方法。因此，AutoML 使对应用机器学习感兴趣但没有资源详细了解机器学习背后技术的领域科学家可以使用最先进的机器学习方法。

—第九页，[自动化机器学习:方法、系统、挑战](https://amzn.to/2w2gVf4s)，2019。

该方法的核心是定义一个大的分层优化问题，除了模型的超参数之外，还包括识别数据转换和机器学习模型本身。

现在，许多公司都提供 AutoML 即服务，其中数据集被上传，模型管道可以通过 web 服务(即 MLaaS)下载或托管和使用。流行的例子包括谷歌、微软和亚马逊提供的服务。

此外，还提供实现 AutoML 技术的开源库，重点关注搜索空间中使用的特定数据转换、模型和超参数，以及用于导航或优化可能性搜索空间的算法类型，其中[贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)版本最为常见。

有许多开源的 AutoML 库，尽管在本教程中，我们将重点介绍可以与流行的 Sklearn Python 机器学习库结合使用的最佳库。

它们是:Hyperopt-Sklearn、Auto-Sklearn 和 TPOT。

**我错过了你最喜欢的 Sklearn 的 AutoML 库了吗？**
在下面的评论里告诉我。

我们将仔细研究每一个，为您评估和考虑哪个库可能适合您的项目提供基础。

## 汽车硬化

Auto-Sklearn 是一个面向 AutoML 的开源 Python 库，使用 Sklearn 机器学习库中的机器学习模型。

它是由[马提亚斯·福雷尔](https://ml.informatik.uni-freiburg.de/people/feurer/index.html)等人开发的，并在他们 2015 年发表的题为“[高效稳健的自动机器学习](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning)的论文中进行了描述

> ……我们介绍了一个基于 Sklearn 的健壮的新 AutoML 系统(使用 15 个分类器、14 种特征预处理方法和 4 种数据预处理方法，产生了一个包含 110 个超参数的结构化假设空间)。

——[高效稳健的自动化机器学习](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning)，2015。

第一步是安装 Auto-Sklearn 库，这可以使用 pip 实现，如下所示:

```py
sudo pip install autosklearn
```

安装后，我们可以导入库并打印版本号，以确认安装成功:

```py
# print autosklearn version
import autosklearn
print('autosklearn: %s' % autosklearn.__version__)
```

运行该示例会打印版本号。您的版本号应该相同或更高。

```py
autosklearn: 0.6.0
```

接下来，我们可以演示在综合分类任务中使用 Auto-Sklearn。

我们可以定义一个自动学习分类器类来控制搜索，并将其配置为运行两分钟(120 秒)，并杀死任何需要 30 秒以上评估的单个模型。在运行结束时，我们可以报告搜索的统计数据，并在保持数据集上评估表现最佳的模型。

下面列出了完整的示例。

```py
# example of auto-sklearn for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
# define dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define search
model = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=8)
# perform the search
model.fit(X_train, y_train)
# summarize
print(model.sprint_statistics())
# evaluate best model
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Accuracy: %.3f" % acc)
```

考虑到我们对运行施加的硬性限制，运行该示例大约需要两分钟。

运行结束时，会打印一份摘要，显示评估了 599 个模型，最终模型的估计表现为 95.6%。

```py
auto-sklearn results:
Dataset name: 771625f7c0142be6ac52bcd108459927
Metric: accuracy
Best validation score: 0.956522
Number of target algorithm runs: 653
Number of successful target algorithm runs: 599
Number of crashed target algorithm runs: 54
Number of target algorithms that exceeded the time limit: 0
Number of target algorithms that exceeded the memory limit: 0
```

然后，我们在保持数据集上评估模型，发现分类准确率达到了 97%，这是相当巧妙的。

```py
Accuracy: 0.970
```

有关自动 Sklearn 库的更多信息，请参见:

*   [自硬化主页](https://automl.github.io/auto-sklearn/master/)。
*   [自硬化 GitHub 项目](https://github.com/automl/auto-sklearn)。

## 基于树的管道优化工具(TPOT)

基于树的管道优化工具，简称 [TPOT](https://epistasislab.github.io/tpot/) ，是一个用于自动机器学习的 Python 库。

TPOT 使用基于树的结构来表示预测建模问题的模型管道，包括数据准备和建模算法以及模型超参数。

> …一种称为基于树的管道优化工具(TPOT)的进化算法，可自动设计和优化机器学习管道。

——[评估用于自动化数据科学的基于树的管道优化工具](https://dl.acm.org/doi/10.1145/2908812.2908918)，2016 年。

第一步是安装 TPOT 图书馆，这可以使用 pip 实现，如下所示:

```py
pip install tpot
```

安装后，我们可以导入库并打印版本号，以确认安装成功:

```py
# check tpot version
import tpot
print('tpot: %s' % tpot.__version__)
```

运行该示例会打印版本号。您的版本号应该相同或更高。

```py
tpot: 0.11.1
```

接下来，我们可以演示在综合分类任务中使用 TPOT。

这包括为进化搜索配置一个具有种群规模和世代数的 TPOTClassifier 实例，以及用于评估模型的交叉验证过程和度量。然后，该算法将运行搜索过程，并将发现的最佳模型管道保存到文件中。

下面列出了完整的示例。

```py
# example of tpot for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
# define dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(X, y)
# export the best model
model.export('tpot_best_model.py')
```

运行该示例可能需要几分钟时间，您将在命令行上看到一个进度条。

高表现模型的准确性将在过程中报告。

考虑到搜索过程的随机性，您的具体结果会有所不同。

```py
Generation 1 - Current best internal CV score: 0.9166666666666666
Generation 2 - Current best internal CV score: 0.9166666666666666
Generation 3 - Current best internal CV score: 0.9266666666666666
Generation 4 - Current best internal CV score: 0.9266666666666666
Generation 5 - Current best internal CV score: 0.9266666666666666

Best pipeline: ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=gini, max_features=0.35000000000000003, min_samples_leaf=2, min_samples_split=6, n_estimators=100)
```

在这种情况下，我们可以看到表现最好的管道实现了大约 92.6%的平均准确率。

然后，表现最好的管道被保存到名为“ *tpot_best_model.py* ”的文件中。

打开这个文件，您可以看到有一些用于加载数据集和拟合管道的通用代码。下面列出了一个例子。

```py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=1)

# Average CV score on the training set was: 0.9266666666666666
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=2, min_samples_split=6, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

然后，您可以检索用于创建模型管道的代码，并将其集成到项目中。

有关 TPOT 的更多信息，请参见以下资源:

*   [评估用于自动化数据科学的基于树的管道优化工具](https://dl.acm.org/doi/10.1145/2908812.2908918)，2016 年。
*   [TPOT 文件](https://epistasislab.github.io/tpot/)。
*   [TPOT GitHub 项目](https://github.com/EpistasisLab/tpot)。

## 超选择性硬化

[HyperOpt](https://hyperopt.github.io/hyperopt/) 是由 [James Bergstra](https://www.linkedin.com/in/james-bergstra) 开发的用于贝叶斯优化的开源 Python 库。

它专为具有数百个参数的模型的大规模优化而设计，并允许优化过程跨多个内核和多台机器进行扩展。

HyperOpt-Sklearn 包装了 HyperOpt 库，允许自动搜索数据准备方法、机器学习算法以及用于分类和回归任务的模型超参数。

> ……我们介绍 Hyperopt-Sklearn:一个为 Python 和 Sklearn 的用户带来自动算法配置好处的项目。Hyperopt-Sklearn 使用 Hyperopt 来描述 Sklearn 组件的可能配置的搜索空间，包括预处理和分类模块。

——[Hyperopt-Sklearn:sci kit-Learn 的自动超参数配置](https://conference.scipy.org/proceedings/scipy2014/pdfs/komer.pdf)，2014。

现在我们已经熟悉了 HyperOpt 和 HyperOpt-Sklearn，下面我们来看看如何使用 HyperOpt-Sklearn。

第一步是安装 HyperOpt 库。

这可以通过使用 pip 包管理器来实现，如下所示:

```py
sudo pip install hyperopt
```

接下来，我们必须安装 HyperOpt-Sklearn 库。

这也可以使用 pip 安装，尽管我们必须通过克隆存储库并从本地文件运行安装来手动执行此操作，如下所示:

```py
git clone git@github.com:hyperopt/hyperopt-sklearn.git
cd hyperopt-sklearn
sudo pip install .
cd ..
```

我们可以通过使用以下命令检查版本号来确认安装成功:

```py
sudo pip show hpsklearn
```

这将总结 HyperOpt-Sklearn 的安装版本，确认正在使用现代版本。

```py
Name: hpsklearn
Version: 0.0.3
Summary: Hyperparameter Optimization for sklearn
Home-page: http://hyperopt.github.com/hyperopt-sklearn/
Author: James Bergstra
Author-email: anon@anon.com
License: BSD
Location: ...
Requires: nose, Sklearn, numpy, scipy, hyperopt
Required-by:
```

接下来，我们可以演示如何在综合分类任务中使用 Hyperopt-Sklearn。

我们可以配置一个运行搜索的 HyperoptEstimator 实例，包括要在搜索空间中考虑的分类器、预处理步骤和要使用的搜索算法。在这种情况下，我们将使用 TPE，或 Parzen 估计树，并执行 50 次评估。

在搜索结束时，将评估和总结表现最佳的模型管道。

下面列出了完整的示例。

```py
# example of hyperopt-sklearn for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe
# define dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)
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

运行结束时，在保持数据集上评估表现最佳的模型，并打印发现的管道供以后使用。

鉴于学习算法和搜索过程的随机性，您的具体结果可能会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到选择的模型在保持测试集上达到了大约 84.8%的准确率。管道涉及一个没有预处理的 SGDClassifier 模型。

```py
Accuracy: 0.848
{'learner': SGDClassifier(alpha=0.0012253733891387925, average=False,
              class_weight='balanced', early_stopping=False, epsilon=0.1,
              eta0=0.0002555872679483392, fit_intercept=True,
              l1_ratio=0.628343459087075, learning_rate='optimal',
              loss='perceptron', max_iter=64710625.0, n_iter_no_change=5,
              n_jobs=1, penalty='l2', power_t=0.42312829309173644,
              random_state=1, shuffle=True, tol=0.0005437535215080966,
              validation_fraction=0.1, verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

然后可以直接使用打印的模型，例如将代码复制粘贴到另一个项目中。

有关 Hyperopt-Sklearn 的更多信息，请参见:

*   超选择性硬化主页。
*   [超选择性硬化 GitHub 项目](https://github.com/hyperopt/hyperopt-sklearn)。

## 摘要

在本教程中，您发现了如何在 Python 中使用顶级开源 AutoML 库进行 Sklearn。

具体来说，您了解到:

*   AutoML 是为预测建模任务自动快速发现表现良好的机器学习模型管道的技术。
*   Sklearn 最受欢迎的三个自动库是 Hyperopt-Sklearn、Auto-Sklearn 和 TPOT。
*   如何在 Python 中使用 AutoML 库为预测建模任务发现表现良好的模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。