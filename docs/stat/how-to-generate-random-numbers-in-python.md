# 如何在Python中生成随机数

> 原文： [https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)

随机性的使用是机器学习算法的配置和评估的重要部分。

从人工神经网络中的权重的随机初始化，到将数据分成随机训练和测试集，到随机梯度下降中的训练数据集的随机改组，生成随机数和利用随机性是必需的技能。

在本教程中，您将了解如何在Python中生成和使用随机数。

完成本教程后，您将了解：

*   可以通过使用伪随机数生成器在程序中应用该随机性。
*   如何通过Python标准库生成随机数并使用随机性。
*   如何通过NumPy库生成随机数组。

让我们开始吧。

![How to Generate Random Numbers in Python](img/e6b0c44ac6567261c267d3920a3ef41c.jpg)

如何在Python中生成随机数
照片来自 [Harold Litwiler](https://www.flickr.com/photos/a_little_brighter/15908996592/) ，保留一些权利。

## 教程概述

本教程分为3个部分;他们是：

1.  伪随机数生成器
2.  Python的随机数
3.  NumPy的随机数

## 1.伪随机数发生器

我们注入到我们的程序和算法中的[随机性](https://en.wikipedia.org/wiki/Randomness)的来源是一种称为伪随机数生成器的数学技巧。

随机数生成器是从真实的随机源生成随机数的系统。经常是物理的东西，比如盖革计数器，结果变成随机数。我们在机器学习中不需要真正的随机性。相反，我们可以使用[伪随机性](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)。伪随机性是看起来接近随机的数字样本，但是使用确定性过程生成。

随机值改组数据和初始化系数使用伪随机数生成器。这些小程序通常是一个可以调用的函数，它将返回一个随机数。再次调用，他们将返回一个新的随机数。包装函数通常也可用，允许您将随机性作为整数，浮点，特定分布，特定范围内等等。

数字按顺序生成。序列是确定性的，并以初始数字播种。如果您没有显式地为伪随机数生成器设定种子，那么它可以使用当前系统时间（以秒或毫秒为单位）作为种子。

种子的价值无关紧要。选择你想要的任何东西重要的是，该过程的相同种子将导致相同的随机数序列。

让我们通过一些例子来具体化。

## 2.使用Python的随机数

Python标准库提供了一个名为 [random](https://docs.python.org/3/library/random.html) 的模块，它提供了一组用于生成随机数的函数。

Python使用一种流行且强大的伪随机数生成器，称为 [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) 。

在本节中，我们将介绍使用标准Python API生成和使用随机数和随机性的一些用例。

### 种子随机数发生器

伪随机数发生器是一种生成几乎随机数序列的数学函数。

它需要一个参数来启动序列，称为种子。该函数是确定性的，意味着给定相同的种子，它每次都会产生相同的数字序列。种子的选择无关紧要。

_seed（）_函数将为伪随机数生成器播种，将整数值作为参数，例如1或7.如果在使用randomness之前未调用seed（）函数，则默认为使用epoch（1970）中的当前系统时间（以毫秒为单位）。

下面的示例演示了对伪随机数生成器进行播种，生成一些随机数，并显示重新生成生成器将导致生成相同的数字序列。

```py
# seed the pseudorandom number generator
from random import seed
from random import random
# seed random number generator
seed(1)
# generate some random numbers
print(random(), random(), random())
# reset the seed
seed(1)
# generate some random numbers
print(random(), random(), random())
```

运行示例为伪随机数生成器播种值为1，生成3个随机数，重新生成生成器，并显示生成相同的三个随机数。

```py
0.13436424411240122 0.8474337369372327 0.763774618976614
0.13436424411240122 0.8474337369372327 0.763774618976614
```

通过设置种子来控制随机性可能很有用，以确保您的代码每次都产生相同的结果，例如在生产模型中。

对于运行实验，其中使用随机化来控制混杂变量，可以对每个实验运行使用不同的种子。

### 随机浮点值

可以使用 _random（）_函数生成随机浮点值。值将在0和1之间的范围内生成，特别是在区间[0,1）中。

值来自均匀分布，意味着每个值具有相同的绘制机会。

以下示例生成10个随机浮点值。

```py
# generate random floating point values
from random import seed
from random import random
# seed random number generator
seed(1)
# generate random numbers between 0-1
for _ in range(10):
	value = random()
	print(value)
```

运行该示例会生成并打印每个随机浮点值。

```py
0.13436424411240122
0.8474337369372327
0.763774618976614
0.2550690257394217
0.49543508709194095
0.4494910647887381
0.651592972722763
0.7887233511355132
0.0938595867742349
0.02834747652200631
```

浮点值可以通过将它们乘以新范围的大小并添加最小值来重新调整到所需范围，如下所示：

```py
scaled value = min + (value * (max - min))
```

其中`min`和`max`分别是所需范围的最小值和最大值，_值_是0到1范围内随机生成的浮点值。

### 随机整数值

可以使用 _randint（）_函数生成随机整数值。

此函数有两个参数：生成的整数值的范围的开始和结束。随机整数在范围值的开始和结束范围内生成，包括范围值的开始和结束，特别是在区间[start，end]中。随机值来自均匀分布。

下面的示例生成10个10到10之间的随机整数值。

```py
# generate random integer values
from random import seed
from random import randint
# seed random number generator
seed(1)
# generate some integers
for _ in range(10):
	value = randint(0, 10)
	print(value)
```

运行该示例会生成并打印10个随机整数值。

```py
2
9
1
4
1
7
7
7
10
6
```

### 随机高斯值

可以使用 _gauss（）_函数从高斯分布中绘制随机浮点值。

此函数采用两个参数，这些参数对应于控制分布大小的参数，特别是平均值和标准偏差。

下面的示例生成从高斯分布绘制的10个随机值，平均值为0.0，标准差为1.0。

请注意，这些参数不是值的界限，并且值的扩展将由分布的钟形控制，在这种情况下，比例可能高于和低于0.0。

```py
# generate random Gaussian values
from random import seed
from random import gauss
# seed random number generator
seed(1)
# generate some Gaussian values
for _ in range(10):
	value = gauss(0, 1)
	print(value)
```

运行该示例生成并打印10个高斯随机值。

```py
1.2881847531554629
1.449445608699771
0.06633580893826191
-0.7645436509716318
-1.0921732151041414
0.03133451683171687
-1.022103170010873
-1.4368294451025299
0.19931197648375384
0.13337460465860485
```

### 从列表中随机选择

随机数可用于从列表中随机选择项目。

例如，如果列表有10个索引在0到9之间的项目，那么您可以生成0到9之间的随机整数，并使用它从列表中随机选择一个项目。 _choice（）_函数为您实现此行为。选择是以均匀的可能性进行的。

下面的示例生成一个包含20个整数的列表，并给出了从列表中选择一个随机项的五个示例。

```py
# choose a random element from a list
from random import seed
from random import choice
# seed random number generator
seed(1)
# prepare a sequence
sequence = [i for i in range(20)]
print(sequence)
# make choices from the sequence
for _ in range(5):
	selection = choice(sequence)
	print(selection)
```

首先运行该示例打印整数值列表，然后是从列表中选择和打印随机值的五个示例。

```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
4
18
2
8
3
```

### 列表中的随机子样本

我们可能有兴趣重复从列表中随机选择项目以创建随机选择的子集。

重要的是，一旦从列表中选择了一个项目并将其添加到子集中，就不应再次添加它。这被称为无需替换的选择，因为一旦为子集选择了列表中的项目，它就不会被添加回原始列表（即，不能用于重新选择）。

_sample（）_函数提供了此行为，该函数从列表中选择随机样本而不进行替换。该函数将列表和子集的大小选为参数。请注意，项目实际上并未从原始列表中删除，只能选择列表的副本。

下面的示例演示如何从20个整数的列表中选择五个项目的子集。

```py
# select a random sample without replacement
from random import seed
from random import sample
# seed random number generator
seed(1)
# prepare a sequence
sequence = [i for i in range(20)]
print(sequence)
# select a subset without replacement
subset = sample(sequence, 5)
print(subset)
```

首先运行该示例打印整数值列表，然后选择并打印随机样本以进行比较。

```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
[4, 18, 2, 8, 3]
```

### 随机随机播放列表

随机性可用于随机播放项目列表，例如洗牌一副牌。

_shuffle（）_函数可用于混洗列表。 shuffle在适当的位置执行，这意味着作为 _shuffle（）_函数的参数提供的列表被混洗，而不是正在制作和返回的列表的混洗副本。

下面的示例演示了随机填充整数值列表。

```py
# randomly shuffle a sequence
from random import seed
from random import shuffle
# seed random number generator
seed(1)
# prepare a sequence
sequence = [i for i in range(20)]
print(sequence)
# randomly shuffle the sequence
shuffle(sequence)
print(sequence)
```

运行该示例首先打印整数列表，然后打印随机洗牌后的相同列表。

```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
[11, 5, 17, 19, 9, 0, 16, 1, 15, 6, 10, 13, 14, 12, 7, 3, 8, 2, 18, 4]
```

## 3\. NumPy的随机数

在机器学习中，您可能正在使用诸如scikit-learn和Keras之类的库。

这些库使用了NumPy，这个库可以非常有效地处理数字的向量和矩阵。

NumPy还有自己的[伪随机数生成器](https://docs.scipy.org/doc/numpy/reference/routines.random.html)和便捷包装函数的实现。

NumPy还实现了Mersenne Twister伪随机数生成器。

让我们看几个生成随机数并使用NumPy数组随机性的例子。

### Seed The Random Number Generator

NumPy伪随机数生成器与Python标准库伪随机数生成器不同。

重要的是，播种Python伪随机数生成器不会影响NumPy伪随机数生成器。它必须单独播种和使用。

_seed（）_函数可用于为NumPy伪随机数生成器播种，取整数作为种子值。

下面的示例演示了如何为生成器设定种子以及如何重新生成生成器将导致生成相同的随机数序列。

```py
# seed the pseudorandom number generator
from numpy.random import seed
from numpy.random import rand
# seed random number generator
seed(1)
# generate some random numbers
print(rand(3))
# reset the seed
seed(1)
# generate some random numbers
print(rand(3))
```

运行示例种子伪随机数生成器，打印一系列随机数，然后重新生成生成器，显示生成完全相同的随机数序列。

```py
[4.17022005e-01 7.20324493e-01 1.14374817e-04]
[4.17022005e-01 7.20324493e-01 1.14374817e-04]
```

### 随机浮点值数组

可以使用 _rand（）_ NumPy函数生成随机浮点值数组。

如果未提供参数，则创建单个随机值，否则可以指定数组的大小。

下面的示例创建一个由均匀分布绘制的10个随机浮点值的数组。

```py
# generate random floating point values
from numpy.random import seed
from numpy.random import rand
# seed random number generator
seed(1)
# generate random numbers between 0-1
values = rand(10)
print(values)
```

运行该示例生成并打印随机浮点值的NumPy数组。

```py
[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01
 1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01
 3.96767474e-01 5.38816734e-01]
```

### 随机整数值数组

可以使用 _randint（）_ NumPy函数生成随机整数数组。

此函数有三个参数，范围的下限，范围的上端，以及要生成的整数值的数量或数组的大小。随机整数将从均匀分布中提取，包括较低的值并排除较高的值，例如，在区间[下，上）。

下面的示例演示了如何生成随机整数数组。

```py
# generate random integer values
from numpy.random import seed
from numpy.random import randint
# seed random number generator
seed(1)
# generate some integers
values = randint(0, 10, 20)
print(values)
```

运行该示例将生成并打印一个包含0到10之间的20个随机整数值的数组。

```py
[5 8 9 5 0 0 1 7 6 9 2 4 5 2 4 2 4 7 7 9]
```

### 随机高斯值数组

可以使用 _randn（）_ NumPy函数生成随机高斯值的数组。

此函数使用单个参数来指定结果数组的大小。高斯值是从标准高斯分布中提取的;这是一个平均值为0.0且标准差为1.0的分布。

下面的示例显示了如何生成随机高斯值数组。

```py
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn
# seed random number generator
seed(1)
# generate some Gaussian values
values = randn(10)
print(values)
```

运行该示例生成并打印来自标准高斯分布的10个随机值的数组。

```py
[ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
  1.74481176 -0.7612069   0.3190391  -0.24937038]
```

来自标准高斯分布的值可以通过将该值乘以标准偏差并且从期望的缩放分布中添加平均值来缩放。例如：

```py
scaled value = mean + value * stdev
```

其中_表示_和`stdev`是所需缩放高斯分布的平均值和标准偏差，_值_是来自标准高斯分布的随机生成值。

### Shuffle NumPy数组

NumPy数组可以使用 _shuffle（）_ NumPy函数随机混洗。

下面的示例演示了如何随机播放NumPy数组。

```py
# randomly shuffle a sequence
from numpy.random import seed
from numpy.random import shuffle
# seed random number generator
seed(1)
# prepare a sequence
sequence = [i for i in range(20)]
print(sequence)
# randomly shuffle the sequence
shuffle(sequence)
print(sequence)
```

首先运行该示例生成一个包含20个整数值的列表，然后随机播放并打印混洗数组。

```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
[3, 16, 6, 10, 2, 14, 4, 17, 7, 1, 13, 0, 19, 18, 9, 15, 8, 12, 11, 5]
```

### 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [在机器学习中拥抱随机性](https://machinelearningmastery.com/randomness-in-machine-learning/)
*   [random - 生成伪随机数](https://docs.python.org/3/library/random.html)
*   [NumPy](https://docs.scipy.org/doc/numpy/reference/routines.random.html) 中的随机抽样
*   [维基百科上的伪随机数发生器](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)

## 摘要

在本教程中，您了解了如何在Python中生成和使用随机数。

具体来说，你学到了：

*   可以通过使用伪随机数生成器在程序中应用该随机性。
*   如何通过Python标准库生成随机数并使用随机性。
*   如何通过NumPy库生成随机数组。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。