# xboost 损失函数的温和介绍

> 原文:[https://machinelearningmastery.com/xgboost-loss-functions/](https://machinelearningmastery.com/xgboost-loss-functions/)

最后更新于 2021 年 4 月 14 日

**XGBoost** 是梯度增强集成算法的一个强大且流行的实现。

配置 XGBoost 模型的一个重要方面是选择在模型训练期间最小化的损失函数。

**损失函数**必须与预测建模问题类型相匹配，同样，我们必须基于具有深度学习神经网络的问题类型选择合适的损失函数。

在本教程中，您将发现如何为 XGBoost 集成模型配置损失函数。

完成本教程后，您将知道:

*   指定训练 XGBoost 集成时使用的损失函数是一个关键步骤，很像神经网络。
*   如何为二类和多类分类任务配置 XGBoost 损失函数？
*   如何为回归预测建模任务配置 XGBoost 损失函数？

我们开始吧。

![A Gentle Introduction to XGBoost Loss Functions](img/2b4b5799a199c9407f8ab6aabc6d1632.png)

XGBoost 损失函数的温和介绍
图片由[凯文·雷塞](https://www.flickr.com/photos/129440207@N08/49980056918/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  扩展和损失函数
2.  分类损失
3.  回归的最大损失

## 扩展和损失函数

极限梯度增强，简称 XGBoost，是梯度增强算法的一个高效开源实现。因此，XGBoost 是一个算法、一个开源项目和一个 Python 库。

它最初是由陈天琪开发的，并由陈和卡洛斯·盖斯特林在他们 2016 年的论文《XGBoost:一个可扩展的树提升系统》中描述

它被设计为既有计算效率(例如，执行速度快)，又非常有效，可能比其他开源实现更有效。

XGBoost 支持一系列不同的预测建模问题，最显著的是分类和回归。

XGBoost 是通过最小化目标函数对数据集的损失来训练的。因此，损失函数的选择是一个关键的超参数，直接与要解决的问题类型相关联，很像深度学习神经网络。

该实现允许通过“*目标*”超参数指定目标函数，并且使用适用于大多数情况的合理默认值。

然而，对于在训练 XGBoost 模型时使用什么损失函数，初学者仍有一些困惑。

在本教程中，我们将详细了解如何为 XGBoost 配置损失函数。

在我们开始之前，让我们开始设置。

xboost 可以作为一个独立的库安装，并且可以使用 scikit-learn API 开发一个 xboost 模型。

第一步是安装尚未安装的 XGBoost 库。这可以在大多数平台上使用 pip python 包管理器来实现；例如:

```py
sudo pip install xgboost
```

然后，您可以通过运行以下脚本来确认 XGBoost 库安装正确，并且可以使用。

```py
# check xgboost version
import xgboost
print(xgboost.__version__)
```

运行该脚本将打印您安装的 XGBoost 库的版本。

您的版本应该相同或更高。如果没有，您必须升级 XGBoost 库的版本。

```py
1.1.1
```

您可能对最新版本的库有问题。这不是你的错。

有时，库的最新版本会带来额外的要求，或者可能不太稳定。

如果您在尝试运行上述脚本时确实有错误，我建议降级到 1.0.1 版(或更低版本)。这可以通过指定要安装到 pip 命令的版本来实现，如下所示:

```py
sudo pip install xgboost==1.0.1
```

如果您看到一条警告消息，您可以暂时忽略它。例如，下面是一个警告消息示例，您可能会看到它，但可以忽略它:

```py
FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
```

如果您需要开发环境的特定说明，请参阅教程:

*   [XGBoost 安装指南](https://xgboost.readthedocs.io/en/latest/build.html)

尽管我们将通过 scikit-learn 包装类使用这个方法:[xgbreversor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)和 [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) ，但是 XGBoost 库有自己的自定义 API。这将允许我们使用 scikit-learn 机器学习库中的全套工具来准备数据和评估模型。

这两个模型以相同的方式运行，并采用相同的参数来影响决策树的创建和添加。

有关如何在 scikit-learn 中使用 XGBoost API 的更多信息，请参见教程:

*   [Python 中的极限梯度增强(XGBoost)集成](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)

接下来，让我们仔细看看如何在分类问题上为 XGBoost 配置损失函数。

## 分类损失

分类任务包括给定一个输入样本，预测每个可能类别的标签或概率。

具有互斥标签的分类任务主要有两种类型:具有两个类别标签的二进制分类和具有两个以上类别标签的多类别分类。

*   **二进制分类**:有两个类标签的分类任务。
*   **多类分类**:两个以上类标签的分类任务。

有关不同类型分类任务的更多信息，请参见教程:

*   [机器学习中的 4 类分类任务](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)

XGBoost 为这些问题类型提供了损失函数。

在机器学习中，典型的做法是训练一个模型来预测概率任务的类成员概率，如果任务需要清晰的类标签来对预测的概率进行后处理(例如，使用 [argmax](https://machinelearningmastery.com/argmax-in-machine-learning/) )。

这种方法在训练深度学习神经网络进行分类时使用，在使用 XGBoost 进行分类时也推荐使用。

用于预测二元分类问题概率的损失函数为“*二元:logistic* ”，用于预测多类问题类概率的损失函数为“*多元:softprob* ”。

*   “*二元:logistic*”:XGBoost 损失函数为二元分类。
*   " *multi:softprob* ": XGBoost 损失函数进行多类分类。

这些字符串值可以在配置 XGBClassifier 模型时通过“*目标”*”超参数来指定。

例如，对于二进制分类:

```py
...
# define the model for binary classification
model = XGBClassifier(objective='binary:logistic')
```

对于多类分类:

```py
...
# define the model for multi-class classification
model = XGBClassifier(objective='multi:softprob')
```

重要的是，如果您没有指定“*目标”*超参数， *XGBClassifier* 将根据训练期间提供的数据自动选择这些损失函数之一。

我们可以用一个具体的例子来说明这一点。

下面的例子创建了一个合成的二进制分类数据集，用默认的超参数在数据集上拟合一个 *XGBClassifier* ，然后打印模型目标配置。

```py
# example of automatically choosing the loss function for binary classification
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到模型自动选择一个损失函数进行二元分类。

```py
binary:logistic
```

或者，我们可以指定目标并拟合模型，确认使用了损失函数。

```py
# example of manually specifying the loss function for binary classification
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define the model
model = XGBClassifier(objective='binary:logistic')
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到用于指定二元分类损失函数的模型。

```py
binary:logistic
```

让我们在具有两个以上类的数据集上重复这个例子。在这种情况下，三个类。

下面列出了完整的示例。

```py
# example of automatically choosing the loss function for multi-class classification
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到模型自动选择了一个损失函数进行多类分类。

```py
multi:softprob
```

或者，我们可以手动指定损失函数，并确认它用于训练模型。

```py
# example of manually specifying the loss function for multi-class classification
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# define the model
model = XGBClassifier(objective="multi:softprob")
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到用于为多类分类指定损失函数的模型。

```py
multi:softprob
```

最后，还有其他损失函数可以用来分类，包括:“*二进制:logitraw* ”和“*二进制:铰链*”用于二进制分类，以及“*多:softmax* ”用于多类分类。

您可以在这里看到完整的列表:

*   [学习任务参数:客观。](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)

接下来，让我们看看用于回归的 XGBoost 损失函数。

## 回归的最大损失

回归指的是预测建模问题，其中给定输入样本，预测数值。

虽然预测概率听起来像回归问题(即概率是数值)，但它通常不被认为是回归型预测建模问题。

预测数值时使用的 XGBoost 目标函数是“ *reg:平方误差*”损失函数。

*   *“reg:平方误差”*:回归预测建模问题的损失函数。

在配置您的*xgbreversor*模型时，可以通过“*目标*”超参数指定该字符串值。

例如:

```py
...
# define the model for regression
model = XGBRegressor(objective='reg:squarederror')
```

重要的是，如果您没有指定“*目标”*超参数，*xgbrejector*会自动为您选择该目标函数。

我们可以用一个具体的例子来说明这一点。

下面的示例创建了一个合成回归数据集，在数据集上拟合一个*xgbreturnor*，然后打印模型目标配置。

```py
# example of automatically choosing the loss function for regression
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = XGBRegressor()
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到模型自动选择一个损失函数进行回归。

```py
reg:squarederror
```

或者，我们可以指定目标并拟合模型，确认使用了损失函数。

```py
# example of manually specifying the loss function for regression
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = XGBRegressor(objective='reg:squarederror')
# fit the model
model.fit(X, y)
# summarize the model loss function
print(model.objective)
```

运行该示例使模型适合数据集，并打印损失函数配置。

我们可以看到模型使用了指定的损失函数进行回归。

```py
reg:squarederror
```

最后，还有其他可以用于回归的损失函数，包括:“ *reg:squaredlogerror* ”、“ *reg:logistic* ”、“*reg:pseuhubererror*”、“ *reg:gamma* ”和“ *reg:tweedie* ”。

您可以在这里看到完整的列表:

*   [学习任务参数:目标](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [Python 中的极限梯度增强(XGBoost)集成](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)
*   [使用 Scikit-Learn、XGBoost、LightGBM 和 CatBoost 进行梯度增强](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
*   [机器学习中的 4 类分类任务](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)

### 蜜蜂

*   [xgboost。xgbcclassifier API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)。
*   [xboost。xgbreversor API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)。
*   [学习任务参数:目标](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)。

## 摘要

在本教程中，您发现了如何为 XGBoost 集成模型配置损失函数。

具体来说，您了解到:

*   像神经网络一样，指定训练 XGBoost 集成时使用的损失函数是一个关键步骤。
*   如何为二类和多类分类任务配置 XGBoost 损失函数？
*   如何为回归预测建模任务配置 XGBoost 损失函数？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。