# 统计假设检验的关键值以及如何在Python中计算它们

> 原文： [https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/](https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/)

如果不是标准的话，使用p值来解释统计假设检验的结果是常见的。

并非所有统计测试的实现都返回p值。在某些情况下，您必须使用替代方法，例如关键值。此外，在估计群体观测的预期间隔时使用临界值，例如在公差间隔中。

在本教程中，您将发现关键值，它们重要的原因，它们的使用方式以及如何使用SciPy在Python中计算它们。

完成本教程后，您将了解：

*   统计假设检验的例子及其可以计算和使用临界值的分布。
*   如何在单尾和双尾统计假设检验中使用准确的临界值。
*   如何计算Gaussian，Student's t和Chi-Squared分布的临界值。

让我们开始吧。

![A Gentle Introduction to Critical Values for Statistical Hypothesis Testing](img/baf3960287a2946722e201699c42252f.jpg)

统计假设检验关键值的温和介绍
[Steve Bittinger](https://www.flickr.com/photos/sbittinger/16122347361/) 的照片，保留一些权利。

## 教程概述

本教程分为4个部分;他们是：

1.  为什么我们需要关键价值观？
2.  什么是关键价值？
3.  如何使用关键值
4.  如何计算临界值

## 为什么我们需要关键价值观？

许多统计假设检验返回用于解释检验结果的p值。

某些测试不返回p值，需要另一种方法来直接解释计算的测试统计量。

通过统计假设检验计算的统计量可以使用来自检验统计量分布的临界值来解释。

统计假设检验的一些例子及其可以计算临界值的分布如下：

*   **Z-Test** ：高斯分布。
*   **学生t检验**：学生t分布。
*   **Chi-Squared Test** ：Chi-Squared分布。
*   **ANOVA** ：F分布。

在为分布中的预期（或意外）观察定义间隔时，也使用临界值。在量化估计的统计量或间隔（例如置信区间和容差区间）的不确定性时，计算和使用临界值可能是合适的。

## 什么是关键价值？

[临界值](https://en.wikipedia.org/wiki/Critical_value)是在人口分布和概率的背景下定义的。

来自群体的观察值，其具有等于或小于具有给定概率的临界值的值。

我们可以用数学方式表达如下：

```py
Pr[X <= critical value] = probability
```

其中 _Pr_ 是概率的计算， _X_ 是来自群体的观察， _critica_value_ 是计算的临界值，_概率_是选择概率。

使用数学函数计算临界值，其中概率作为参数提供。对于大多数常见分布，无法通过分析计算该值;相反，它必须使用数值方法估算。从历史上看，在统计教科书的附录中提供预先计算的临界值表是常见的，以供参考。

临界值用于统计显着性检验。概率通常表示为重要性，表示为小写希腊字母alpha（a），它是反转概率。

```py
probability = 1 - alpha
```

计算临界值时使用标准alpha值，由于历史原因选择并且由于一致性原因而不断使用。这些alpha值包括：

*   1％（alpha = 0.01）
*   5％（alpha = 0.05）
*   10％（alpha = 0.10）

临界值提供了将统计假设检验解释为 [p值](https://en.wikipedia.org/wiki/P-value)的替代和等效方法。

## 如何使用关键值

计算的临界值用作解释统计检验结果的阈值。

超过临界值的群体中的观察值通常被称为“_关键区域_”或“_拒绝区域_”。

> 临界值：在表中出现的用于指定统计测试的值，指示可以拒绝原假设的计算值（计算的统计量落在拒绝区域中）。

- 第265页，[研究方法手册：社会科学从业者和学生指南](http://amzn.to/2G4vG4k)，2003年。

统计检验可以是[单尾或双尾](https://en.wikipedia.org/wiki/One-_and_two-tailed_tests)。

### 单尾测试

单尾测试具有单个临界值，例如分布的左侧或右侧。

通常，单尾测试在非对称分布（例如Chi-Squared分布）的分布右侧具有临界值。

将统计值与计算的临界值进行比较。如果统计量小于或等于临界值，则统计检验的零假设未被拒绝。否则它被拒绝。

我们可以总结如下解释：

*   **测试统计＆lt;临界值**：未能拒绝统计检验的零假设。
*   **测试统计=＆gt;临界值**：拒绝统计检验的零假设。

### 双尾测试

双尾检验具有两个临界值，分布的每一侧有一个，通常假设它们是对称的（例如高斯分布和学生t分布）。

使用双尾测试时，计算临界值时使用的显着性水平（或alpha）必须除以2.临界值将在分布的每一侧使用此alpha的一部分。

为了使这个具体，请考虑5％的alpha。这将被分割为在分布的任一侧给出2.5％的两个α值，其中接收区域在分布的中间为95％。

我们可以将每个临界值分别称为分布左右的下临界值和上临界值。测试统计值大于或等于下临界值且小于或等于上临界值表示未能拒绝原假设。而测试统计值小于下临界值且大于上临界值表示拒绝测试的零假设。

We can summarize this interpretation as follows:

*   **降低CR＆lt;测试统计＆lt;上CR** ：未拒绝统计检验的零假设。
*   **测试统计＆lt; =下CR OR测试统计＆gt; =上CR** ：拒绝统计测试的原假设。

如果测试统计量的分布在零均值附近对称，那么我们可以通过将检验统计量的绝对值（正值）与上临界值进行比较来快速检查。

*   **|测试统计| ＆LT;上临界值**：未拒绝统计检验的零假设。

哪里 _|测试统计|_ 是计算的检验统计量的绝对值。

## 如何计算临界值

密度函数返回分布中观察的概率。回想一下PDF和CDF的定义如下：

*   **概率密度函数（PDF）**：返回具有分布中特定值的观测值的概率。
*   **累积密度函数（CDF）**：返回观察的概率等于或小于分布中的特定值。

为了计算临界值，我们需要一个函数，给定概率（或重要性），将从分布中返回观测值。

具体而言，我们需要累积密度函数的倒数，在给定概率的情况下，给出小于或等于概率的观测值。这称为百分点函数（PPF），或更一般地称为[分位数函数](https://en.wikipedia.org/wiki/Quantile_function)。

*   **百分点函数（PPF）**：返回所提供概率的观测值，该概率小于或等于分布中提供的概率。

具体而言，来自分布的值将等于或小于具有指定概率的PPF返回的值。

让我们通过三个分布来具体化，通常需要通过三个分布来计算临界值。即，高斯分布，学生t分布和卡方分布。

我们可以使用给定分布上的 _ppf（）_函数计算SciPy中的百分点函数。还应该注意的是，您还可以使用SciPy中名为 _isf（）_的逆生存函数来计算 _ppf（）_。这是提到的，因为您可能会在第三方代码中看到这种替代方法的使用。

## 高斯临界值

下面的示例计算标准高斯分布上95％的百分点函数。

```py
# Gaussian Percent Point Function
from scipy.stats import norm
# define probability
p = 0.95
# retrieve value <= probability
value = norm.ppf(p)
print(value)
# confirm with cdf
p = norm.cdf(value)
print(p)
```

运行该示例首先打印标记95％或更少的观察值的值，该值来自约1.65的分布。然后通过从CDF检索观察的概率来确认该值，CDF按预期返回95％。

我们可以看到，1.65的值与我们对 [68-95-99.7规则](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)中覆盖95％分布的平均值的标准偏差数量的预期一致。

```py
1.6448536269514722
0.95
```

### 学生的关键价值观

下面的示例计算标准学生t分布的95％的百分点函数，具有10个自由度。

```py
# Student t-distribution Percent Point Function
from scipy.stats import t
# define probability
p = 0.95
df = 10
# retrieve value <= probability
value = t.ppf(p, df)
print(value)
# confirm with cdf
p = t.cdf(value, df)
print(p)
```

运行该示例将返回大约1.812或更小的值，该值覆盖所选分布中95％的观察值。然后通过CDF确认该值的概率（具有较小的舍入误差）。

```py
1.8124611228107335
0.949999999999923
```

### 卡方临界值

下面的示例计算了具有10个自由度的标准Chi-Squared分布的95％的百分点函数。

```py
# Chi-Squared Percent Point Function
from scipy.stats import chi2
# define probability
p = 0.95
df = 10
# retrieve value <= probability
value = chi2.ppf(p, df)
print(value)
# confirm with cdf
p = chi2.cdf(value, df)
print(p)
```

首先运行该示例计算的值为18.3或更小，覆盖了分布中95％的观测值。通过将其用作CDF的输入来确认该观察的概率。

```py
18.307038053275143
0.95
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [研究方法手册：社会科学从业者和学生指南](http://amzn.to/2G4vG4k)，2003年。

### API

*   [scipy.stats.norm API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
*   [scipy.stats.t API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
*   [scipy.stats.chi2 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)

### 用品

*   [维基百科的重要价值](https://en.wikipedia.org/wiki/Critical_value)
*   维基百科上的 [P值](https://en.wikipedia.org/wiki/P-value)
*   [维基百科的单尾和双尾测试](https://en.wikipedia.org/wiki/One-_and_two-tailed_tests)
*   维基百科上的[分位数函数](https://en.wikipedia.org/wiki/Quantile_function)
*   维基百科上的 [68-95-99.7规则](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)

### 摘要

在本教程中，您发现了关键值，它们重要的原因，它们的使用方式以及如何使用SciPy在Python中计算它们。

具体来说，你学到了：

*   统计假设检验的例子及其可以计算和使用临界值的分布。
*   如何在单尾和双尾统计假设检验中使用准确的临界值。
*   如何计算Gaussian，Student's t和Chi-Squared分布的临界值。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。