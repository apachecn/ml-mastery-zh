# 如何修复 scikit 中的未来学习消息-学习

> 原文：<https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/>

最后更新于 2019 年 8 月 21 日

当代码运行时，通过使用*未来学习*消息报告机器学习的 [scikit-learn 库](https://scikit-learn.org/)即将发生的变化。

警告消息可能会让初学者感到困惑，因为看起来代码有问题或者他们做错了什么。警告消息也不利于操作代码，因为它们会掩盖错误和程序输出。

有许多方法可以处理警告消息，包括忽略消息、抑制警告和修复代码。

在本教程中，您将发现 scikit-learn API 中的 FutureWarning 消息，以及如何在自己的机器学习项目中处理这些消息。

完成本教程后，您将知道:

*   未来学习消息旨在通知您即将对 scikit-learn API 中的参数默认值进行的更改。
*   未来学习消息可以被忽略或抑制，因为它们不会停止程序的执行。
*   未来学习消息的示例，以及如何解释消息并更改代码以应对即将到来的更改。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Fix FutureWarning Messages in scikit-learn](img/5bd55ebd4349bbcac5f7be88e21fd54c.png)

如何修复 scikit 中的未来学习消息-学习
图片由 [a.dombrowski](https://www.flickr.com/photos/adombrowski/5914604908/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  未来家具的问题
2.  如何抑制未来的入侵
3.  如何修复未来的家具
4.  未来学习建议

## 未来家具的问题

scikit-learn 库是一个开源库，为数据准备和机器学习算法提供工具。

这是一个广泛使用和不断更新的图书馆。

像许多积极维护的软件库一样，API 经常随着时间的推移而变化。这可能是因为发现了更好的实践或者首选的使用模式发生了变化。

scikit-learn API 中提供的大多数函数都有一个或多个参数，可以让您自定义函数的行为。许多参数都有合理的默认值，因此您不必为参数指定值。当您刚开始使用机器学习或 scikit-learn，并且不知道每个论点有什么影响时，这尤其有用。

随着时间的推移，对 scikit-learn API 的更改通常表现为对函数参数的合理默认值的更改。这种类型的更改通常不会立即执行；相反，他们是有计划的。

例如，如果您的代码是为 scikit-learn 库的早期版本编写的，并且依赖于函数参数的默认值，并且 API 的后续版本计划更改该默认值，则 API 会提醒您即将到来的更改。

每次运行代码时，此警报都会以警告消息的形式出现。具体来说，在标准错误(例如命令行)上报告“*未来预警*”。

这是 API 和项目的一个有用特性，是为您的利益而设计的。它允许您为库的下一个主要版本更改代码，以保留旧的行为(为参数指定一个值)或采用新的行为(不更改代码)。

运行时报告警告的 Python 脚本可能会令人沮丧。

*   **对于初学者**来说，可能会觉得代码工作不正常，可能是自己做错了什么。
*   **对于专业人士**来说，是一个程序需要更新的标志。

在这两种情况下，警告消息可能会掩盖真实的错误消息或程序输出。

## 如何抑制未来的入侵

警告消息不是错误消息。

因此，程序报告的警告消息，如*未来预警*，不会停止程序的执行。警告信息将被报告，程序将继续执行。

因此，如果您愿意，可以在每次执行代码时忽略该警告。

也可以通过编程方式忽略警告消息。这可以通过在程序运行时隐藏警告消息来实现。

这可以通过显式配置 Python 警告系统来忽略特定类型的警告消息来实现，例如忽略所有未来警告，或者更一般地，忽略所有警告。

这可以通过在代码周围添加以下您知道会生成警告的块来实现:

```py
# run block of code and catch warnings
with warnings.catch_warnings():
	# ignore all caught warnings
	warnings.filterwarnings("ignore")
	# execute code that will generate warnings
	...
```

或者，如果您有一个非常简单的平面脚本(没有函数或块)，您可以通过在文件顶部添加两行来抑制所有的未来预览:

```py
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
```

要了解 Python 中抑制的更多信息，请参见:

*   [Python 警告控制 API](https://docs.python.org/3/library/warnings.html)

## 如何修复未来的家具

或者，您可以更改代码来解决 scikit-learn API 报告的更改。

通常，警告消息本身会指示您更改的性质以及如何更改代码来处理警告。

尽管如此，让我们看几个你可能会遇到并正在努力解决的未来家具的例子。

本节中的示例是用 scikit-learn 版本 0.20.2 开发的。您可以通过运行以下代码来检查 scikit-learn 版本:

```py
# check scikit-learn version
import sklearn
print('sklearn: %s' % sklearn.__version__)
```

您将看到如下输出:

```py
sklearn: 0.20.2
```

随着 scikit-learn 新版本的发布，所报告的警告消息的性质将会改变，并将采用新的默认值。

因此，尽管下面的例子是特定于 scikit-learn 版本的，但是诊断和解决每一个 API 变更本质的方法为处理未来的变更提供了很好的例子。

### 物流出口的未来学习

[后勤导出算法](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)最近对默认参数值进行了两次更改，导致了未来预警消息。

第一个与寻找系数的解算器有关，第二个与如何使用模型进行多类分类有关。让我们用代码示例来看看每一个。

#### 对求解器的更改

下面的示例将生成一个关于物流配送使用的求解器参数的未来学习。

```py
# example of LogisticRegression that generates a FutureWarning
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
# create and configure model
model = LogisticRegression()
# fit model
model.fit(X, y)
```

运行该示例会导致以下警告消息:

```py
FutureWarning: Default solver will be changed to 'lbfgs' in 0.22\. Specify a solver to silence this warning.
```

这个问题涉及到对“*求解器*参数的更改，该参数过去默认为“ *liblinear* ，在未来版本中将更改为默认为“ *lbfgs* ”。您现在必须指定“*求解器*参数。

要保持旧的行为，可以按如下方式指定参数:

```py
# create and configure model
model = LogisticRegression(solver='liblinear')
```

为了支持新的行为(推荐)，可以按如下方式指定参数:

```py
# create and configure model
model = LogisticRegression(solver='lbfgs')
```

#### 对多类的更改

下面的示例将生成一个关于物流出口使用的“*多类*参数的未来警告。

```py
# example of LogisticRegression that generates a FutureWarning
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
# prepare dataset
X, y = make_blobs(n_samples=100, centers=3, n_features=2)
# create and configure model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
```

运行该示例会导致以下警告消息:

```py
FutureWarning: Default multi_class will be changed to 'auto' in 0.22\. Specify the multi_class option to silence this warning.
```

该警告消息仅影响多类分类问题的逻辑回归的使用，而不是该方法所设计的二元分类问题。

“ *multi_class* ”参数的默认值从“ *ovr* 更改为“ *auto* ”。

要保持旧的行为，可以按如下方式指定参数:

```py
# create and configure model
model = LogisticRegression(solver='lbfgs', multi_class='ovr')
```

为了支持新的行为(推荐)，可以按如下方式指定参数:

```py
# create and configure model
model = LogisticRegression(solver='lbfgs', multi_class='auto')
```

### SVM 的未来教育

支持向量机实现最近对“ *gamma* ”参数进行了更改，导致了一条警告消息，特别是 [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) 和 [SVR 类](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)。

下面的示例将生成一个关于 SVC 使用的“ *gamma* ”参数的未来警告，但同样适用于 SVR。

```py
# example of SVC that generates a FutureWarning
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
# create and configure model
model = SVC()
# fit model
model.fit(X, y)
```

运行此示例将生成以下警告消息:

```py
FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
```

此警告消息报告“*伽玛*”参数的默认值正在从当前值“*自动*更改为新默认值“*刻度*”。

gamma 参数只影响使用径向基函数、多项式或 Sigmoid 核的 SVM 模型。

该参数控制算法中使用的“*γ*系数的值，如果不指定值，将使用启发式算法来指定值。该警告是关于默认计算方式的更改。

要保持旧的行为，可以按如下方式指定参数:

```py
# create and configure model
model = SVC(gamma='auto')
```

为了支持新的行为(推荐)，可以按如下方式指定参数:

```py
# create and configure model
model = SVC(gamma='scale')
```

### 决策树集成算法的未来学习

基于决策树的集成算法将改变集成中使用的子模型或树的数量，这些子模型或树由“*n _ estimates*参数控制。

这会影响模型的随机森林和用于分类和回归的额外树，特别是类:*随机森林分类器*、*随机森林回归器*、*提取森林分类器*、*提取森林回归器*和*随机森林分类器*。

下面的例子将生成一个关于随机森林分类器使用的“*n _ estimates”*参数的未来警告，但同样适用于随机森林回归器和额外的树类。

```py
# example of RandomForestClassifier that generates a FutureWarning
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
# create and configure model
model = RandomForestClassifier()
# fit model
model.fit(X, y)
```

运行此示例将生成以下警告消息:

```py
FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
```

此警告消息报告子模型的数量正在从 10 个增加到 100 个，这可能是因为计算机变得越来越快，10 个非常小，甚至 100 个也很小。

要保持旧的行为，可以按如下方式指定参数:

```py
# create and configure model
model = RandomForestClassifier(n_estimators=10)
```

为了支持新的行为(推荐)，可以按如下方式指定参数:

```py
# create and configure model
model = RandomForestClassifier(n_estimators=100)
```

### 更多未来警告？

您是否在为一个未涵盖的未来学习而奋斗？

请在下面的评论中告诉我，我会尽我所能提供帮助。

## 未来学习建议

一般来说，我不建议忽略或抑制警告消息。

忽略警告消息意味着该消息可能会掩盖真正的错误或程序输出，并且 API 未来的更改可能会对您的程序产生负面影响，除非您已经考虑过它们。

抑制警告可能是 R&D 工作的快速解决方案，但不应在生产系统中使用。比简单地忽略消息更糟糕的是，抑制警告也可能抑制来自其他 API 的消息。

相反，我建议您修复软件中的警告消息。

### 您应该如何更改代码？

一般来说，我建议几乎总是采用应用编程接口的新行为，例如新的默认值，除非您明确依赖函数的先前行为。

对于长期运行的操作代码或生产代码，显式指定所有函数参数而不使用默认值可能是一个好主意，因为它们可能会在未来发生变化。

我还建议您保持 scikit-learn 库的最新状态，并跟踪每个新版本中对 API 的更改。

最简单的方法是查看每个版本的发行说明，如下所示:

*   [scikit-学习发布历史](https://scikit-learn.org/stable/whats_new.html)

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [Python 警告控制 API](https://docs.python.org/3/library/warnings.html)
*   [sklearn.linear_model。物流配送应用编程接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
*   [硬化. svm.SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
*   [硬化. svm.SVR API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
*   [scikit-学习发布历史](https://scikit-learn.org/stable/whats_new.html)

## 摘要

在本教程中，您发现了 scikit-learn API 中的 FutureWarning 消息，以及如何在自己的机器学习项目中处理这些消息。

具体来说，您了解到:

*   未来学习消息旨在通知您即将对 scikit-learn API 中的参数默认值进行的更改。
*   未来学习消息可以被忽略或抑制，因为它们不会停止程序的执行。
*   未来学习消息的示例，以及如何解释消息并更改代码以应对即将到来的更改。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。