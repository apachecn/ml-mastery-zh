# 如何用 Python 计算偏差方差权衡

> 原文：<https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/>

最后更新于 2020 年 8 月 26 日

机器学习模型的表现可以用模型的**偏差**和**方差**来表征。

具有高偏差的模型会对数据集中将输入映射到输出的未知基础函数的形式做出强有力的假设，例如线性回归。具有高方差的模型高度依赖于训练数据集的细节，例如未运行的决策树。我们希望模型具有低偏差和低方差，尽管在这两个关注点之间经常存在权衡。

偏差-方差权衡对于选择和配置模型是一个有用的概念化，尽管通常不能直接计算，因为它需要问题领域的全面知识，而我们没有。然而，在某些情况下，我们可以估计模型的误差，并将误差分成偏差和方差分量，这可以提供对给定模型行为的洞察。

在本教程中，您将发现如何计算机器学习模型的偏差和方差。

完成本教程后，您将知道:

*   模型误差由模型方差、模型偏差和不可约误差组成。
*   我们寻求具有低偏差和方差的模型，尽管通常减少一个会导致另一个上升。
*   如何将均方误差分解为模型偏差和方差项？

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Calculate the Bias-Variance Trade-off in Python](img/fc72b5dec68b97349eaa9da074280245.png)

如何计算 Python 中的偏差-方差权衡
图片由 [Nathalie](https://flickr.com/photos/nathalie-photos/38618514704/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  偏差、方差和不可约误差
2.  偏差-方差权衡
3.  计算偏差和方差

## 偏差、方差和不可约误差

考虑一个机器学习模型，它对预测建模任务进行预测，例如回归或分类。

模型在任务上的表现可以用所有未用于训练模型的例子上的预测误差来描述。我们称之为模型误差。

*   错误(型号)

模型误差可以分解为三个误差源:模型的**方差**、模型的**偏差**、数据中**不可约误差**的方差。

*   误差(模型)=方差(模型)+偏差(模型)+方差(不可约误差)

让我们仔细看看这三个术语。

### 模型偏差

偏差是模型能够捕捉输入和输出之间映射函数的程度的度量。

它抓住了模型的刚性:模型关于输入和输出之间映射的功能形式的假设的强度。

> 这反映了模型的功能形式能够多么接近预测因子和结果之间的真实关系。

—第 97 页，[应用预测建模](https://amzn.to/3a7Yzrc)，2013 年。

当偏差与预测建模问题的真实但未知的底层映射函数相匹配时，具有高偏差的模型是有帮助的。然而，当问题的函数形式与模型的假设不匹配时，具有大偏差的模型将完全无用，例如，假设数据具有高度非线性关系的线性关系。

*   **低偏差**:关于输入到输出映射的函数形式的弱假设。
*   **高偏差**:关于输入到输出映射的函数形式的强假设。

偏见总是积极的。

### 模型方差

模型的方差是当模型适合不同的训练数据时，模型表现的变化量。

它捕捉数据对模型的具体影响。

> 方差是指如果我们使用不同的训练数据集进行估计，那么[模型]将会改变的量。

—第 34 页，[R](https://amzn.to/2RC7ElX)中应用的统计学习介绍，2014。

具有高方差的模型会随着训练数据集的微小变化而发生很大变化。相反，低方差模型对训练数据集的变化很小，甚至很大。

*   **低方差**:随着训练数据集的变化，模型的微小变化。
*   **高方差**:随着训练数据集的变化，模型的变化很大。

方差总是正的。

### 不可约误差

总的来说，模型的误差由可约误差和不可约误差组成。

*   模型误差=可约误差+不可约误差

可减少的误差是我们可以改进的地方。这是当模型在训练数据集上学习时我们减少的数量，我们试图使这个数字尽可能接近零。

不可约误差是我们无法用模型或任何模型消除的误差。

误差是由我们无法控制的因素造成的，比如观测中的统计噪声。

> ……通常称为“不可约噪声”，无法通过建模消除。

—第 97 页，[应用预测建模](https://amzn.to/3a7Yzrc)，2013 年。

因此，尽管我们可能能够将可约误差压缩到非常小的接近零的值，或者在某些情况下甚至为零，但我们也会有一些不可约误差。它定义了问题表现的下限。

> 重要的是要记住，不可约误差总是为我们对 y 的预测精确率提供一个上限。这个上限在实践中几乎总是未知的。

—第 19 页，[R](https://amzn.to/2RC7ElX)中应用的统计学习介绍，2014。

这提醒我们，没有一种模式是完美的。

## 偏差-方差权衡

模型表现的偏差和方差是有联系的。

理想情况下，我们更喜欢具有低偏差和低方差的模型，尽管在实践中，这非常具有挑战性。事实上，对于给定的预测建模问题，这可以被描述为应用机器学习的目标，

通过增加方差可以很容易地减少偏差。相反，通过增加偏差可以很容易地降低方差。

> 这被称为权衡，因为很容易获得偏差极低但方差高的方法……或偏差极低但方差高的方法……

—第 36 页，[R](https://amzn.to/2RC7ElX)中应用的统计学习介绍，2014。

这种关系通常被称为**偏差-方差权衡**。它是思考如何选择模型和模型配置的概念框架。

我们可以根据模型的偏差或方差来选择模型。简单模型，如线性回归和逻辑回归，通常具有高偏差和低方差。复杂的模型，如随机森林，一般具有低偏差但高方差。

我们也可以根据模型配置对模型偏差和方差的影响来选择模型配置。k 近邻中的 k 超参数控制偏差方差的权衡。小值(如 k=1)导致低偏差和高方差，而大 k 值(如 k=21)导致高偏差和低方差。

高偏差并不总是不好的，高方差也不总是不好的，但它们会导致糟糕的结果。

我们经常必须测试一套不同的模型和模型配置，以便发现什么最适合给定的数据集。一个有很大偏差的模型可能过于僵化，并且忽视了这个问题。相反，大的差异可能会使问题复杂化。

我们可以决定增加偏差或方差，只要它降低模型误差的总体估计。

## 计算偏差和方差

我一直在问这个问题:

> 如何在数据集上计算算法的偏差方差权衡？

从技术上讲，我们无法进行这种计算。

我们无法计算预测建模问题的实际偏差和方差。

这是因为我们不知道预测建模问题的真实映射函数。

相反，我们使用偏差、方差、不可约误差和偏差-方差权衡作为工具来帮助选择模型、配置模型和解释结果。

> 在现实生活中 f 不被观察到的情况下，通常不可能明确计算统计学习方法的测试均方误差、偏差或方差。然而，人们应该始终牢记偏差-方差权衡。

—第 36 页，[R](https://amzn.to/2RC7ElX)中应用的统计学习介绍，2014。

即使偏差-方差权衡是一个概念工具，我们也可以在某些情况下对其进行估计。

由[塞巴斯蒂安·拉什卡](https://sebastianraschka.com/)创建的[扩展库](https://rasbt.github.io/mlxtend/)提供了 [bias_variance_decomp()函数](https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)，该函数可以在多个自举样本上估计模型的偏差和方差。

首先，您必须安装 mlxtend 库；例如:

```py
sudo pip install mlxtend
```

以下示例通过 URL 直接加载[波士顿住房数据集](https://github.com/jbrownlee/Datasets/blob/master/housing.names)，将其拆分为训练集和测试集，然后估计线性回归的均方误差(MSE)以及 200 多个自举样本的模型误差的偏差和方差。

```py
# estimate the bias and variance for a regression model
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# separate into inputs and outputs
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define the model
model = LinearRegression()
# estimate bias and variance
mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
```

运行该示例会报告估计误差以及模型误差的估计偏差和方差。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型具有高偏差和低方差。鉴于我们使用的是线性回归模型，这是可以预期的。我们还可以看到，估计的均值和方差之和等于模型的估计误差，例如 20.726 + 1.761 = 22.487。

```py
MSE: 22.487
Bias: 20.726
Variance: 1.761
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习中偏差-方差权衡的温和介绍](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)

### 书

*   [R](https://amzn.to/2RC7ElX)中应用的统计学习导论，2014。
*   [应用预测建模](https://amzn.to/3a7Yzrc)，2013。

### 文章

*   [偏差–方差权衡，维基百科](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)。
*   [mlxe tend library](https://rasbt.github.io/mlxtend/)。
*   [偏差-方差分解，mlextend](https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)。

## 摘要

在本教程中，您发现了如何计算机器学习模型的偏差和方差。

具体来说，您了解到:

*   模型误差由模型方差、模型偏差和不可约误差组成。
*   我们寻求具有低偏差和方差的模型，尽管通常减少一个会导致另一个上升。
*   如何将均方误差分解为模型偏差和方差项？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。