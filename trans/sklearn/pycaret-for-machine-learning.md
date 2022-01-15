# 机器学习用 PyCaret 的温和介绍

> 原文:[https://machinelearning master . com/pycaret-for-machine-learning/](https://machinelearningmastery.com/pycaret-for-machine-learning/)

**PyCaret** 是一个 Python 开源机器学习库，旨在使在机器学习项目中执行标准任务变得容易。

它是 R 语言中 Caret 机器学习包的 Python 版本，之所以受欢迎，是因为它允许只需几行代码就可以在给定的数据集上评估、比较和调整模型。

PyCaret 库提供了这些功能，允许 Python 中的机器学习从业者通过单个函数调用在分类或回归数据集上抽查一套标准的机器学习算法。

在本教程中，您将发现用于机器学习的 PyCaret Python 开源库。

完成本教程后，您将知道:

*   PyCaret 是 r .中流行且广泛使用的 Caret 机器学习包的 Python 版本
*   如何使用 PyCaret 轻松评估和比较数据集上的标准机器学习模型。
*   如何使用 PyCaret 轻松调整性能良好的机器学习模型的超参数。

我们开始吧。

![A Gentle Introduction to PyCaret for Machine Learning](img/1a1cf1efcddf6821637a39da739adfca.png)

机器学习用 PyCaret 的温和介绍
托马斯摄，版权所有。

## 教程概述

本教程分为四个部分；它们是:

1.  什么是 PyCaret？
2.  声纳数据集
3.  比较机器学习模型
4.  调整机器学习模型

## 什么是 PyCaret？

[PyCaret](https://pycaret.org/) 是一个开源的 Python 机器学习库，灵感来自于 [caret R 包](https://topepo.github.io/caret/)。

脱字符号包的目标是自动化评估和比较分类和回归的机器学习算法的主要步骤。该库的主要好处是，只需很少的代码行和很少的手动配置就可以完成很多工作。PyCaret 库将这些功能带到了 Python 中。

> PyCaret 是 Python 中的一个开源、低代码的机器学习库，旨在减少从假设到洞察的周期时间。它非常适合经验丰富的数据科学家，他们希望通过在工作流中使用 PyCaret 来提高 ML 实验的生产率，或者适合公民数据科学家和那些对数据科学不太了解或没有编码背景的人。

——[pycaret 主页](https://pycaret.org/)

PyCaret 库自动化了机器学习项目的许多步骤，例如:

*   定义要执行的数据转换(*设置()*)
*   评估和比较标准模型( *compare_models()* )
*   调谐模型超参数( *tune_model()* )

以及更多不限于创建集成、保存模型和部署模型的功能。

PyCaret 库有大量使用该应用编程接口的文档；您可以从这里开始:

*   [PyCaret 主页](https://pycaret.org/)

在本教程中，我们不会探索该库的所有功能；相反，我们将专注于简单的机器学习模型比较和超参数调整。

您可以使用 Python 包管理器安装 PyCaret，例如 pip。例如:

```py
pip install pycaret
```

安装后，我们可以通过打印已安装的版本来确认库在您的开发环境中可用并且工作正常。

```py
# check pycaret version
import pycaret
print('PyCaret: %s' % pycaret.__version__)
```

运行该示例将加载 PyCaret 库并打印安装的版本号。

您的版本号应该相同或更高。

```py
PyCaret: 2.0.0
```

如果您需要为您的系统安装 PyCaret 的帮助，您可以在此处查看安装说明:

*   [PyCaret 安装说明](https://pycaret.org/install)

现在我们已经熟悉了 PyCaret 是什么，让我们来探索如何在机器学习项目中使用它。

## 声纳数据集

我们将使用声纳标准二进制分类数据集。您可以在这里了解更多信息:

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集详细信息(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

我们可以直接从网址下载数据集，并将其作为熊猫数据框架加载。

```py
...
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize the shape of the dataset
print(df.shape)
```

PyCaret 似乎要求数据集有列名，而我们的数据集没有列名，所以我们可以直接将列号设置为列名。

```py
...
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
```

最后，我们可以总结前几行数据。

```py
...
# summarize the first few rows of data
print(df.head())
```

将这些联系在一起，下面列出了加载和总结声纳数据集的完整示例。

```py
# load the sonar dataset
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize the shape of the dataset
print(df.shape)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# summarize the first few rows of data
print(df.head())
```

运行该示例首先加载数据集并报告形状，显示它有 208 行和 61 列。

然后打印前五行，显示输入变量都是数字，目标变量是列“60”并带有字符串标签。

```py
(208, 61)
0 1 2 3 4 ... 56 57 58 59 60
0 0.0200 0.0371 0.0428 0.0207 0.0954 ... 0.0180 0.0084 0.0090 0.0032 R
1 0.0453 0.0523 0.0843 0.0689 0.1183 ... 0.0140 0.0049 0.0052 0.0044 R
2 0.0262 0.0582 0.1099 0.1083 0.0974 ... 0.0316 0.0164 0.0095 0.0078 R
3 0.0100 0.0171 0.0623 0.0205 0.0205 ... 0.0050 0.0044 0.0040 0.0117 R
4 0.0762 0.0666 0.0481 0.0394 0.0590 ... 0.0072 0.0048 0.0107 0.0094 R
```

接下来，我们可以使用 PyCaret 来评估和比较一套标准的机器学习算法，以快速发现什么在这个数据集上运行良好。

## 用于比较机器学习模型的 PyCaret

在本节中，我们将评估和比较标准机器学习模型在声纳分类数据集上的性能。

首先，我们必须通过 [setup()函数](https://pycaret.org/classification/)用 PyCaret 库设置数据集。这要求我们提供 Pandas DataFrame，并指定包含目标变量的列的名称。

*setup()* 功能还允许您配置简单的数据准备，例如缩放、幂变换、缺失数据处理和 PCA 变换。

我们将指定数据、目标变量，并关闭 HTML 输出、详细输出和用户反馈请求。

```py
...
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
```

接下来，我们可以通过调用 *compare_models()* 函数来比较标准机器学习模型。

默认情况下，它将使用 10 倍交叉验证来评估模型，根据分类精度对结果进行排序，并返回单个最佳模型。

这些都是不错的违约，我们不需要改变什么。

```py
...
# evaluate models and compare models
best = compare_models()
```

调用 *compare_models()* 函数还会报告一个结果表，该表总结了所有被评估的模型及其性能。

最后，我们可以报告性能最佳的模型及其配置。

将这些联系在一起，下面列出了在声纳分类数据集上评估一套标准模型的完整示例。

```py
# compare machine learning algorithms on the sonar classification dataset
from pandas import read_csv
from pycaret.classification import setup
from pycaret.classification import compare_models
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# evaluate models and compare models
best = compare_models()
# report the best model
print(best)
```

运行该示例将加载数据集，配置 PyCaret 库，评估一套标准模型，并报告为数据集找到的最佳模型。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到“*额外树分类器*”在数据集上具有最好的准确性，得分约为 86.95%。

然后我们可以看到所使用的模型的配置，看起来它使用了默认的超参数值。

```py
                              Model  Accuracy     AUC  Recall   Prec.      F1  \
0            Extra Trees Classifier    0.8695  0.9497  0.8571  0.8778  0.8631
1               CatBoost Classifier    0.8695  0.9548  0.8143  0.9177  0.8508
2   Light Gradient Boosting Machine    0.8219  0.9096  0.8000  0.8327  0.8012
3      Gradient Boosting Classifier    0.8010  0.8801  0.7690  0.8110  0.7805
4              Ada Boost Classifier    0.8000  0.8474  0.7952  0.8071  0.7890
5            K Neighbors Classifier    0.7995  0.8613  0.7405  0.8276  0.7773
6         Extreme Gradient Boosting    0.7995  0.8934  0.7833  0.8095  0.7802
7          Random Forest Classifier    0.7662  0.8778  0.6976  0.8024  0.7345
8          Decision Tree Classifier    0.7533  0.7524  0.7119  0.7655  0.7213
9                  Ridge Classifier    0.7448  0.0000  0.6952  0.7574  0.7135
10                      Naive Bayes    0.7214  0.8159  0.8286  0.6700  0.7308
11              SVM - Linear Kernel    0.7181  0.0000  0.6286  0.7146  0.6309
12              Logistic Regression    0.7100  0.8104  0.6357  0.7263  0.6634
13     Linear Discriminant Analysis    0.6924  0.7510  0.6667  0.6762  0.6628
14  Quadratic Discriminant Analysis    0.5800  0.6308  0.1095  0.5000  0.1750

     Kappa     MCC  TT (Sec)
0   0.7383  0.7446    0.1415
1   0.7368  0.7552    1.9930
2   0.6410  0.6581    0.0134
3   0.5989  0.6090    0.1413
4   0.5979  0.6123    0.0726
5   0.5957  0.6038    0.0019
6   0.5970  0.6132    0.0287
7   0.5277  0.5438    0.1107
8   0.5028  0.5192    0.0035
9   0.4870  0.5003    0.0030
10  0.4488  0.4752    0.0019
11  0.4235  0.4609    0.0024
12  0.4143  0.4285    0.0059
13  0.3825  0.3927    0.0034
14  0.1172  0.1792    0.0033
ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                     oob_score=False, random_state=2728, verbose=0,
                     warm_start=False)
```

我们可以直接使用这种配置，在整个数据集上拟合一个模型，并使用它对新数据进行预测。

我们还可以使用结果表来了解在数据集上表现良好的模型类型，在本例中是决策树的集合。

现在我们已经熟悉了如何使用 PyCaret 比较机器学习模型，让我们看看如何使用该库来调整模型超参数。

## 调整机器学习模型

在本节中，我们将在声纳分类数据集上调整机器学习模型的超参数。

我们必须像以前比较模型时那样加载和设置数据集。

```py
...
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
```

我们可以使用 PyCaret 库中的 *tune_model()* 函数来调整模型超参数。

该函数将模型的一个实例作为输入进行调整，并知道自动调整哪些超参数。执行模型超参数的随机搜索，并且可以通过“ *n_iter* ”参数控制评估的总数。

默认情况下，该功能将优化“*精度*”，并将使用 10 倍交叉验证来评估每个配置的性能，尽管这个合理的默认配置可以更改。

我们可以对额外的树分类器执行如下随机搜索:

```py
...
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200)
```

该函数将返回性能最佳的模型，可直接使用或打印该模型来确定所选的超参数。

它还将打印 k 折叠交叉验证中最佳配置的结果表(例如 10 次折叠)。

将这些联系在一起，下面列出了在声纳数据集上调整额外树分类器的超参数的完整示例。

```py
# tune model hyperparameters on the sonar classification dataset
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from pycaret.classification import setup
from pycaret.classification import tune_model
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200, choose_better=True)
# report the best model
print(best)
```

运行该示例首先加载数据集并配置 PyCaret 库。

然后执行网格搜索，报告跨 10 倍交叉验证的最佳配置性能和平均精度。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，随机搜索找到了一个准确率约为 75.29%的配置，这并不比上一部分的默认配置好，后者的得分约为 86.95%。

```py
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
0       0.8667  1.0000  1.0000  0.7778  0.8750  0.7368  0.7638
1       0.6667  0.8393  0.4286  0.7500  0.5455  0.3119  0.3425
2       0.6667  0.8036  0.2857  1.0000  0.4444  0.2991  0.4193
3       0.7333  0.7321  0.4286  1.0000  0.6000  0.4444  0.5345
4       0.6667  0.5714  0.2857  1.0000  0.4444  0.2991  0.4193
5       0.8571  0.8750  0.6667  1.0000  0.8000  0.6957  0.7303
6       0.8571  0.9583  0.6667  1.0000  0.8000  0.6957  0.7303
7       0.7857  0.8776  0.5714  1.0000  0.7273  0.5714  0.6325
8       0.6429  0.7959  0.2857  1.0000  0.4444  0.2857  0.4082
9       0.7857  0.8163  0.5714  1.0000  0.7273  0.5714  0.6325
Mean    0.7529  0.8270  0.5190  0.9528  0.6408  0.4911  0.5613
SD      0.0846  0.1132  0.2145  0.0946  0.1571  0.1753  0.1485
ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=1, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=4, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=120,
                     n_jobs=None, oob_score=False, random_state=None, verbose=0,
                     warm_start=False)
```

我们可以通过指定 *tune_model()* 函数搜索哪些超参数和搜索哪些范围来改进网格搜索。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [PyCaret 主页](https://pycaret.org/)
*   [R 脱字符号包主页](https://topepo.github.io/caret/)
*   [PyCaret 安装说明](https://pycaret.org/install)

## 摘要

在本教程中，您发现了用于机器学习的 PyCaret Python 开源库。

具体来说，您了解到:

*   PyCaret 是 r .中流行且广泛使用的 Caret 机器学习包的 Python 版本
*   如何使用 PyCaret 轻松评估和比较数据集上的标准机器学习模型。
*   如何使用 PyCaret 轻松调整性能良好的机器学习模型的超参数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。