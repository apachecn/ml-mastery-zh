# 机器学习中的 Python 生态系统

> 原文： [https://machinelearningmastery.com/python-ecosystem-machine-learning/](https://machinelearningmastery.com/python-ecosystem-machine-learning/)

Python 生态系统正在发展，可能成为机器学习的主要平台。

采用 Python 进行机器学习的主要原因是因为它是一种通用编程语言，可以用于研究和开发以及生产。

在这篇文章中，您将发现用于机器学习的 Python 生态系统。

![Python Ecosystem for Machine Learning](img/6722b29a5e9b23802bc7fcaed7f651eb.jpg)

用于机器学习的 Python 生态系统
照片由 [Stewart Black](https://www.flickr.com/photos/s2ublack/6678407353/) 拍摄，保留一些权利。

## 蟒蛇

[Python](https://www.python.org/) 是一种通用的解释型编程语言。它易于学习和使用，主要是因为语言侧重于可读性。

Python 的 [Zen](https://en.wikipedia.org/wiki/Zen_of_Python) 中包含了 Python 的哲学，其中包括以下短语：

*   美丽胜过丑陋。
*   显式优于隐式。
*   简单比复杂更好。
*   复杂比复杂更好。
*   Flat 优于嵌套。
*   稀疏优于密集。
*   可读性很重要。

您可以通过键入以下内容在 Python 环境中查看 Python 的完整 Zen：

```
import this
```

它是一种流行语言，一直出现在 StackOverflow 调查中的前 10 种编程语言中（例如 [2015 年调查结果](http://stackoverflow.com/research/developer-survey-2015)）。它是一种动态语言，非常适合交互式开发和快速原型设计，具有支持大型应用程序开发的能力。

它还广泛用于机器学习和数据科学，因为它具有出色的库支持，并且因为它是一种通用编程语言（与 R 或 Matlab 不同）。例如，请参阅 2011 年 [Kaggle 平台调查结果](http://blog.kaggle.com/2011/11/27/kagglers-favorite-tools/)和 [KDD Nuggets 2015 工具调查结果](http://www.kdnuggets.com/polls/2015/analytics-data-mining-data-science-software-used.html)的结果。

这是一个简单而非常重要的考虑因素。

这意味着您可以使用您在操作中使用的相同编程语言执行研究和开发（确定要使用的模型）。大大简化了从开发到运营的过渡。

## SciPy 的

[SciPy](https://en.wikipedia.org/wiki/SciPy) 是一个用于数学，科学和工程的 Python 库生态系统。它是 Python 的附加组件，您需要进行机器学习。

SciPy 生态系统由以下与机器学习相关的核心模块组成：

*   [NumPy](http://www.numpy.org/) ：SciPy 的基础，可让您有效地处理数组中的数据。
*   [Matplotlib](http://matplotlib.org/) ：允许您从数据创建二维图表和图表。
*   [pandas](http://pandas.pydata.org/) ：用于组织和分析数据的工具和数据结构。

要在 Python 中进行机器学习，您必须安装并熟悉 SciPy。特别：

*   您将使用 Pandas 加载探索并更好地了解您的数据。
*   您将使用 Matplotlib（以及其他框架中的 Matplotlib 包装器）来创建数据的图表和图表。
*   您将把数据准备为 NumPy 数组，以便在机器学习算法中进行建模。

您可以在帖子中了解更多关于熊猫的信息[使用 Pandas](http://machinelearningmastery.com/prepare-data-for-machine-learning-in-python-with-pandas/) 和[使用 Pandas](http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/) [快速和脏数据分析为 Python 机器学习准备数据。](http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/)

## scikit 学习

[scikit-learn](http://scikit-learn.org/) 库是如何在 python 中开发和练习机器学习的。

它建立在 SciPy 生态系统的基础之上。名称“`scikit`”表明它是一个 SciPy 插件或工具包。您可以查看[完整的 SciKits](http://scikits.appspot.com/scikits) 列表。

该库的重点是用于分类，回归，聚类等的机器学习算法。它还为相关任务提供工具，例如评估模型，调整参数和预处理数据。

与 Python 和 SciPy 一样，scikit-learn 是开源的，在 BSD 许可下可以商业使用。这意味着您可以了解机器学习，开发模型并将它们投入到具有相同生态系统和代码的操作中。使用 scikit-learn 的一个有力理由。

您可以在帖子 [A scntle introduction to scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/) 中了解更多关于 scikit-learn 的信息。

## Python 生态系统安装

有多种方法可以安装 Python 生态系统用于机器学习。在本节中，我们将介绍如何安装用于机器学习的 Python 生态系统。

### 如何安装 Python

第一步是安装 Python。我更喜欢使用和推荐 Python 2.7。

这将特定于您的平台。有关说明，请参阅 [Python 初学者指南](https://wiki.python.org/moin/BeginnersGuide)中的[下载 Python](https://wiki.python.org/moin/BeginnersGuide/Download) 。

安装完成后，您可以确认安装是否成功。打开命令行并键入：

```
python --version
```

您应该看到如下响应：

```
Python 2.7.11
```

### 如何安装 SciPy

安装 SciPy 的方法有很多种。例如，两种流行的方法是在您的平台上使用包管理（例如，RedHat 上的 yum 或 OS X 上的 macport）或使用像 pip 这样的 Python 包管理工具。

SciPy 文档非常出色，涵盖了页面上许多不同平台的操作说明[安装 SciPy Stack](http://scipy.org/install.html) 。

安装 SciPy 时，请确保至少安装以下软件包：

*   SciPy 的
*   numpy 的
*   matplotlib
*   大熊猫

安装后，您可以确认安装是否成功。通过在命令行键入“`python`”打开 python 交互式环境，然后键入并运行以下 python 代码以打印已安装库的版本。

```
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
```

在发布时我的工作站上看到以下输出。

```
scipy: 0.17.0
numpy: 1.10.4
matplotlib: 1.5.1
pandas: 0.17.1
```

你看到什么输出？在评论中发布。

如果您有错误，可能需要查阅适用于您的平台的文档。

### 如何安装 scikit-learn

我建议您使用相同的方法安装 scikit-learn，就像您以前安装 SciPy 一样。

有[安装 scikit-learn](http://scikit-learn.org/stable/install.html) 的说明，但它们仅限于使用 Python [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) 和 [conda](http://conda.pydata.org/docs/) 包管理器。

与 SciPy 一样，您可以确认 scikit-learn 已成功安装。启动 Python 交互式环境，键​​入并运行以下代码。

```
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```

它将打印安装的 scikit-learn 库的版本。在我的工作站上，我看到以下输出：

```
sklearn: 0.17.1
```

### 如何安装生态系统：一种更简单的方法

如果您对在计算机上安装软件没有信心，可以选择更方便的选项。

有一个名为 Anaconda 的发行版，您可以[免费下载和安装](https://www.continuum.io/downloads)。

它支持 Microsoft Windows，Mac OS X 和 Linux 三个主要平台。

它包括 Python，SciPy 和 scikit-learn。使用 Python 环境学习，练习和使用机器学习所需的一切。

## 摘要

在这篇文章中，您发现了用于机器学习的 Python 生态系统。

你了解到：

*   Python 和它越来越多地用于机器学习。
*   SciPy 及其为 NumPy，Matplotlib 和 Pandas 提供的功能。
*   scikit-learn 提供所有机器学习算法。

您还学习了如何在工作站上安装用于机器学习的 Python 生态系统。

你对机器学习的 Python 或这篇文章有什么疑问吗？在评论中提出您的问题，我会尽力回答。