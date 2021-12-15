# 如何在Python中从零开始编写T检验

> 原文： [https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/](https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/)

也许最广泛使用的统计假设检验之一是T检验。

因为有一天你可能会自己使用这个测试，所以深入了解测试的工作原理非常重要。作为开发人员，通过从零开始实现假设检验，可以最好地实现这种理解。

在本教程中，您将了解如何在Python中从零开始实现Student's t检验统计假设检验。

完成本教程后，您将了解：

*   学生的t检验将评论是否可能观察到两个样本，因为样本来自同一人群。
*   如何从零开始实现T检验两个独立样本。
*   如何从零开始对两个相关样本实现配对T检验。

让我们开始吧。

![How to Code the Student's t-Test from Scratch in Python](img/18c211273ce8051c7807ef2e09b805d7.jpg)

如何在Python中从零开始编写T检验
照片由 [n1d](https://www.flickr.com/photos/62400641@N07/33385804523/) ，保留一些权利。

## 教程概述

本教程分为三个部分;他们是：

1.  学生的t-测试
2.  学生对独立样本的t检验
3.  学生对依赖样本的t检验

## 学生的t-测试

[T检验](https://en.wikipedia.org/wiki/Student%27s_t-test)是一项统计假设检验，用于检验是否预期两个样本来自同一人群。

它以William Gosset使用的化名“`Student`”命名，他开发了该测试。

测试通过检查来自两个样品的平均值来确定它们是否彼此显着不同。它通过计算均值之间差异的标准误差来做到这一点，如果两个样本具有相同的均值（零假设），可以解释为差异的可能性。

通过将其与来自t分布的临界值进行比较，可以解释通过测试计算的t统计量。可以使用自由度和百分点函数（PPF）的显着性水平来计算临界值。

我们可以在双尾检验中解释统计值，这意味着如果我们拒绝零假设，那可能是因为第一个均值小于或大于第二个均值。为此，我们可以计算检验统计量的绝对值，并将其与正（右尾）临界值进行比较，如下所示：

*   **如果abs（t-statistic）＆lt; =临界值**：接受平均值相等的零假设。
*   **如果abs（t-statistic）&gt;临界值**：拒绝平均值相等的零假设。

我们还可以使用t分布的累积分布函数（CDF）来检索观察t统计量的绝对值的累积概率，以便计算p值。然后可以将p值与选择的显着性水平（α）（例如0.05）进行比较，以确定是否可以拒绝原假设：

*   **如果p&gt; alpha** ：接受平均值相等的零假设。
*   **如果p &lt;= alpha** ：拒绝零假设，即平均值相等。

在处理样本的平均值时，测试假设两个样本都是从高斯分布中提取的。该测试还假设样本具有相同的方差和相同的大小，尽管如果这些假设不成立，则对测试进行校正。例如，参见 [Welch的t检验](https://en.wikipedia.org/wiki/Welch%27s_t-test)。

Student's t-test有两个主要版本：

*   **独立样本**。两个样本不相关的情况。
*   **相关样本**。样本相关的情况，例如对同一群体的重复测量。也称为配对测试。

独立和依赖学生的t检验分别通过 [ttest_ind（）](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)和 [ttest_rel（）](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) SciPy函数在Python中提供。

**注意**：我建议使用这些SciPy函数计算应用程序的Student t检验，如果它们合适的话。库实现将更快，更不容易出错。我只建议您出于学习目的自行实现测试，或者在需要修改测试版本的情况下。

我们将使用SciPy函数来确认我们自己的测试版本的结果。

请注意，作为参考，本教程中提供的所有计算都直接取自“ [t [ _T测试_”中的“ _T Tests_](https://amzn.to/2J2Qibd)”，第三版，2010年。我提到这一点是因为您可能会看到具有不同形式的方程式，具体取决于您使用的参考文本。

## 学生对独立样本的t检验

我们将从最常见的T检验形式开始：我们比较两个独立样本的平均值的情况。

### 计算

两个独立样本的t统计量的计算如下：

```py
t = observed difference between sample means / standard error of the difference between the means
```

要么

```py
t = (mean(X1) - mean(X2)) / sed
```

其中`X1`和`X2`是第一和第二数据样本而`sed`是平均值之差的标准误差。

平均值之间差异的标准误差可以计算如下：

```py
sed = sqrt(se1^2 + se2^2)
```

其中`se1`和`se2`是第一和第二数据集的标准误差。

样本的标准误差可以计算为：

```py
se = std / sqrt(n)
```

当`se`是样品的标准误差时，`std`是样品标准偏差，`n`是样品中的观察数。

这些计算做出以下假设：

*   样本是从高斯分布中提取的。
*   每个样本的大小大致相等。
*   样本具有相同的方差。

### 履行

我们可以使用Python标准库，NumPy和SciPy中的函数轻松实现这些方程。

假设我们的两个数据样本存储在变量`data1`和`data2`中。

我们可以从计算这些样本的平均值开始，如下所示：

```py
# calculate means
mean1, mean2 = mean(data1), mean(data2)
```

我们在那里一半。

现在我们需要计算标准误差。

我们可以手动完成，首先计算样本标准偏差：

```py
# calculate sample standard deviations
std1, std2 = std(data1, ddof=1), std(data2, ddof=1)
```

然后是标准错误：

```py
# calculate standard errors
n1, n2 = len(data1), len(data2)
se1, se2 = std1/sqrt(n1), std2/sqrt(n2)
```

或者，我们可以使用`sem()`SciPy函数直接计算标准误差。

```py
# calculate standard errors
se1, se2 = sem(data1), sem(data2)
```

我们可以使用样本的标准误差来计算样本之间差异的“_标准误差”：_

```py
# standard error on the difference between the samples
sed = sqrt(se1**2.0 + se2**2.0)
```

我们现在可以计算t统计量：

```py
# calculate the t statistic
t_stat = (mean1 - mean2) / sed
```

我们还可以计算一些其他值来帮助解释和呈现统计量。

测试的自由度数计算为两个样本中观察值的总和减去2。

```py
# degrees of freedom
df = n1 + n2 - 2
```

对于给定的显着性水平，可以使用百分点函数（PPF）计算临界值，例如0.05（95％置信度）。

此功能可用于SciPy中的t分发，如下所示：

```py
# calculate the critical value
alpha = 0.05
cv = t.ppf(1.0 - alpha, df)
```

可以使用t分布上的累积分布函数来计算p值，再次在SciPy中。

```py
# calculate the p-value
p = (1 - t.cdf(abs(t_stat), df)) * 2
```

在这里，我们假设一个双尾分布，其中零假设的拒绝可以解释为第一个均值小于或大于第二个均值。

我们可以将所有这些部分组合成一个简单的函数来计算两个独立样本的t检验：

```py
# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p
```

### 工作示例

在本节中，我们将计算一些合成数据样本的t检验。

首先，让我们生成两个100高斯随机数的样本，其方差相同，分别为50和51。我们期望测试拒绝原假设并找出样本之间的显着差异：

```py
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
```

我们可以使用内置的SciPy函数 _ttest_ind（）_计算这些样本的t检验。这将为我们提供t统计值和要比较的p值，以确保我们正确地实现了测试。

下面列出了完整的示例。

```py
# Student's t-test for independent samples
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = ttest_ind(data1, data2)
print('t=%.3f, p=%.3f' % (stat, p))
```

运行该示例，我们可以看到t统计值和p值。

我们将使用这些作为我们对这些数据进行测试的预期值。

```py
t=-2.262, p=0.025
```

我们现在可以使用上一节中定义的函数对相同的数据应用我们自己的实现。

该函数将返回t统计值和临界值。我们可以使用临界值来解释t统计量，以查看测试的结果是否显着，并且确实手段与我们预期的不同。

```py
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
```

该函数还返回p值。我们可以使用α来解释p值，例如0.05，以确定测试的结果是否显着，并且确实手段与我们预期的不同。

```py
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
```

我们希望这两种解释始终匹配。

The complete example is listed below.

```py
# t-test for independent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
```

首先运行该示例计算测试。

打印测试结果，包括t统计量，自由度，临界值和p值。

我们可以看到t统计量和p值都与SciPy函数的输出相匹配。测试似乎正确实现。

然后使用t统计量和p值来解释测试结果。我们发现，正如我们所期望的那样，有足够的证据可以拒绝零假设，发现样本均值可能不同。

```py
t=-2.262, df=198, cv=1.653, p=0.025
Reject the null hypothesis that the means are equal.
Reject the null hypothesis that the means are equal.
```

## 学生对依赖样本的t检验

我们现在可以看一下计算依赖样本的T检验的情况。

在这种情况下，我们收集来自种群的样本的一些观察结果，然后应用一些处理，然后从同一样本收集观察结果。

结果是两个相同大小的样本，其中每个样本中的观察结果是相关的或配对的。

依赖样本的t检验称为配对T检验。

### Calculation

配对T检验的计算与独立样本的情况类似。

主要区别在于分母的计算。

```py
t = (mean(X1) - mean(X2)) / sed
```

Where`X1`and`X2`are the first and second data samples and`sed`is the standard error of the difference between the means.

这里，`sed`计算如下：

```py
sed = sd / sqrt(n)
```

其中`sd`是依赖样本平均值与_之间的差异的标准偏差n_ 是配对观察的总数（例如每个样本的大小）。

`sd`的计算首先需要计算样本之间的平方差之和：

```py
d1 = sum (X1[i] - X2[i])^2 for i in n
```

它还需要样本之间（非平方）差异的总和：

```py
d2 = sum (X1[i] - X2[i]) for i in n
```

然后我们可以将sd计算为：

```py
sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
```

而已。

### Implementation

我们可以直接在Python中实现配对Student's t-test的计算。

第一步是计算每个样本的平均值。

```py
# calculate means
mean1, mean2 = mean(data1), mean(data2)
```

接下来，我们将需要对的数量（`n`）。我们将在几个不同的计算中使用它。

```py
# number of paired samples
n = len(data1)
```

接下来，我们必须计算样本之间的平方差的总和，以及总和差异。

```py
# sum squared difference between observations
d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
# sum difference between observations
d2 = sum([data1[i]-data2[i] for i in range(n)])
```

我们现在可以计算平均值之差的标准差。

```py
# standard deviation of the difference between means
sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
```

然后用它来计算平均值之间差异的标准误差。

```py
# standard error of the difference between the means
sed = sd / sqrt(n)
```

最后，我们拥有计算t统计量所需的一切。

```py
# calculate the t statistic
t_stat = (mean1 - mean2) / sed
```

此实现与独立样本实现之间唯一的其他关键区别是计算自由度的数量。

```py
# degrees of freedom
df = n - 1
```

和以前一样，我们可以将所有这些结合在一起成为可重用的功能。该函数将采用两个配对样本和显着性水平（alpha）并计算t统计量，自由度数，临界值和p值。

完整的功能如下所列。

```py
# function for calculating the t-test for two dependent samples
def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p
```

### Worked Example

在本节中，我们将在工作示例中使用与独立Student's t检验相同的数据集。

数据样本没有配对，但我们会假装它们。我们希望测试拒绝原假设并找出样本之间的显着差异。

```py
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
```

和以前一样，我们可以使用SciPy函数评估测试问题，以计算配对t检验。在这种情况下， _ttest_rel（）_功能。

The complete example is listed below.

```py
# Paired Student's t-test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_rel
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# compare samples
stat, p = ttest_rel(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

运行该示例计算并打印t统计量和p值。

我们将使用这些值来验证我们自己的配对t检验函数的计算。

```py
Statistics=-2.372, p=0.020
```

我们现在可以测试我们自己的配对T检验的实现。

下面列出了完整的示例，包括已开发的函数和函数结果的解释。

```py
# t-test for dependent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import t

# function for calculating the t-test for two dependent samples
def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

# seed the random number generator
seed(1)
# generate two independent samples (pretend they are dependent)
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = dependent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')
```

运行该示例计算样本问题的配对t检验。

计算出的t统计量和p值与我们对SciPy库实现的期望相匹配。这表明实现是正确的。

具有临界值的t检验统计量和具有显着性水平的p值的解释都发现了显着的结果，拒绝了平均值相等的零假设。

```py
t=-2.372, df=99, cv=1.660, p=0.020
Reject the null hypothesis that the means are equal.
Reject the null hypothesis that the means are equal.
```

### 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   将每个测试应用于您自己设计的样本问题。
*   更新独立测试并为具有不同方差和样本大小的样本添加校正。
*   对SciPy库中实现的其中一个测试执行代码审查，并总结实现细节的差异。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [普通英语统计](https://amzn.to/2J2Qibd)，第三版，2010年。

### API

*   [scipy.stats.ttest_ind API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
*   [scipy.stats.ttest_rel API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
*   [scipy.stats.sem API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html)
*   [scipy.stats.t API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)

### 用品

*   [维基百科上的T检验](https://en.wikipedia.org/wiki/Student%27s_t-test)
*   [韦尔奇在维基百科上的t检验](https://en.wikipedia.org/wiki/Welch%27s_t-test)

## 摘要

在本教程中，您了解了如何在Python中从零开始实现Student's t检验统计假设检验。

具体来说，你学到了：

*   学生的t检验将评论是否可能观察到两个样本，因为样本来自同一人群。
*   如何从零开始实现T检验两个独立样本。
*   如何从零开始对两个相关样本实现配对T检验。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。