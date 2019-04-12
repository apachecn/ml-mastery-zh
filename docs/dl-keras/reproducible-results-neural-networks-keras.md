# 如何使用 Keras 获得可重现的结果

> 原文： [https://machinelearningmastery.com/reproducible-results-neural-networks-keras/](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)

神经网络算法是随机的。

这意味着它们利用随机性，例如初始化为随机权重，反过来，在相同数据上训练的相同网络可以产生不同的结果。

这可能会让初学者感到困惑，因为算法似乎不稳定，实际上它们是设计的。随机初始化允许网络学习正在学习的函数的良好近似。

然而，有时候每次在相同的数据上训练相同的网络时，您需要完全相同的结果。比如教程，或者可能是操作上的。

在本教程中，您将了解如何为随机数生成器设定种子，以便每次都可以从同一网络中获取相同数据的相同结果。

让我们开始吧。

![How to Get Reproducible Results from Neural Networks with Keras](img/7c82a23fe9ca5ce16ae054ff7260f3f0.png)

如何从 Keras 的神经网络获得可重现的结果
照片由 [Samuel John](https://www.flickr.com/photos/samueljohn/6129216625/) ，保留一些权利。

## 教程概述

本教程分为 6 个部分。他们是：

1.  为什么每次都会得到不同的结果？
2.  证明不同的结果
3.  解决方案
4.  种子随机数与 Theano 后端
5.  种子随机数与 TensorFlow 后端
6.  如果我仍然得到不同的结果怎么办？

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

本教程假设您使用 TensorFlow（v1.1.0 +）或 Theano（v0.9 +）后端安装了 Keras（v2.0.3 +）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 为什么每次都会得到不同的结果？

这是我从初学者到神经网络和深度学习领域的常见问题。

这种误解也可能以下列问题的形式出现：

*   _ 如何获得稳定的结果？_
*   _ 如何获得可重复的结果？_
*   _ 我应该使用什么种子？_

神经网络通过设计使用随机性，以确保它们有效地学习与问题近似的函数。使用随机性是因为这类机器学习算法比没有它更好。

神经网络中最常用的随机形式是网络权重的随机初始化。虽然随机性可用于其他领域，但这里只是一个简短的列表：

*   初始化中的随机性，例如权重。
*   正则化中的随机性，例如dropout。
*   层中的随机性，例如单词嵌入。
*   优化中的随机性，例如随机优化。

这些随机性来源等等意味着当您在完全相同的数据上运行完全相同的神经网络算法时，您可以保证得到不同的结果。

有关随机算法背后原因的更多信息，请参阅帖子：

*   [在机器学习中拥抱随机性](http://machinelearningmastery.com/randomness-in-machine-learning/)

## 证明不同的结果

我们可以用一个小例子来证明神经网络的随机性。

在本节中，我们将开发一个多层感知器模型，以学习从 0.0 到 0.9 增加 0.1 的短序列。给定 0.0，模型必须预测 0.1;给定 0.1，模型必须输出 0.2;等等。

下面列出了准备数据的代码。

```py
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLPfriendly format
values = df.values
X, y = values[:,0], values[:,1]
```

我们将使用一个输入的网络，隐藏层中的 10 个神经元和 1 个输出。网络将使用均方误差丢失函数，并将使用有效的 ADAM 算法进行训练。

网络需要大约 1000 个时代来有效地解决这个问题，但我们只会训练它 100 个时代。这是为了确保我们得到一个在进行预测时出错的模型。

在训练网络之后，我们将对数据集进行预测并打印均方误差。

网络代码如下所示。

```py
# design network
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
# forecast
yhat = model.predict(X, verbose=0)
print(mean_squared_error(y, yhat[:,0]))
```

在该示例中，我们将创建网络 10 次并打印 10 个不同的网络分数。

完整的代码清单如下。

```py
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

# fit MLP to dataset and print error
def fit_model(X, y):
	# design network
	model = Sequential()
	model.add(Dense(10, input_dim=1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
	# forecast
	yhat = model.predict(X, verbose=0)
	print(mean_squared_error(y, yhat[:,0]))

# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:,0], values[:,1]
# repeat experiment
repeats = 10
for _ in range(repeats):
	fit_model(X, y)
```

运行该示例将在每行中打印不同的精度。

您的具体结果会有所不同。下面提供了一个示例输出。

```py
0.0282584265697
0.0457025913022
0.145698137198
0.0873461454407
0.0309397604521
0.046649185173
0.0958450337178
0.0130660263779
0.00625176026631
0.00296055161492
```

## 解决方案

这是两个主要的解决方案。

### 解决方案＃1：重复您的实验

解决此问题的传统和实用方法是多次运行您的网络（30+）并使用统计数据来总结模型的表现，并将您的模型与其他模型进行比较。

我强烈推荐这种方法，但由于某些型号的训练时间很长，因此并不总是可行。

有关此方法的更多信息，请参阅：

*   [如何评估深度学习模型的技巧](http://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

### 解决方案＃2：为随机数生成器播种

或者，另一种解决方案是使用固定种子作为随机数发生器。

使用伪随机数生成器生成随机数。随机数生成器是一种数学函数，它将生成一个长序列的数字，这些数字足够随机用于通用目的，例如机器学习算法。

随机数生成器需要种子来启动进程，并且通常在大多数实现中使用当前时间（以毫秒为单位）作为默认值。这是为了确保每次运行代码时都会生成不同的随机数序列，默认情况下。

也可以使用特定数字（例如“1”）指定此种子，以确保每次运行代码时都生成相同的随机数序列。

只要每次运行代码时，特定的种子值都无关紧要。

设置随机数生成器的具体方法因后端而异，我们将在 Theano 和 TensorFlow 中查看如何执行此操作。

## 种子随机数与 Theano 后端

通常，Keras 从 NumPy 随机数生成器获得随机源。

在大多数情况下，Theano 后端也是如此。

我们可以通过从随机模块调用 [seed（）函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)来播种 NumPy 随机数生成器，如下所示：

```py
from numpy.random import seed
seed(1)
```

导入和调用种子函数最好在代码文件的顶部完成。

这是[最佳实践](https://github.com/fchollet/keras/issues/439)，因为即使在直接使用它们之前，当各种 Keras 或 Theano（或其他）库作为初始化的一部分导入时，可能会使用某些随机性。

我们可以将两行添加到上面示例的顶部并运行两次。

每次运行代码时都应该看到相同的均方误差值列表（可能由于不同机器上的精度而有一些微小的变化），如下所示：

```py
0.169326527063
2.75750621228e-05
0.0183287291562
1.93553737255e-07
0.0549871087449
0.0906326807824
0.00337575114075
0.00414857518259
8.14587362008e-08
0.0522927019639
```

您的结果应与我的匹配（忽略精确度的微小差异）。

## 种子随机数与 TensorFlow 后端

Keras 确实从 NumPy 随机数生成器获得随机源，因此无论您使用的是 Theano 还是 TensorFlow 后端，都必须播种。

必须通过在任何其他导入或其他代码之前调用文件顶部的 [seed（）函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)来播种。

```py
from numpy.random import seed
seed(1)
```

此外，TensorFlow 还有自己的随机数生成器，必须通过在 NumPy 随机数生成器之后立即调用 [set_random_seed（）函数](https://www.tensorflow.org/api_docs/python/tf/set_random_seed)来播种，如下所示：

```py
from tensorflow import set_random_seed
set_random_seed(2)
```

为了清楚起见，代码文件的顶部必须在任何其他文件之前有以下 4 行;

```py
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
```

您可以为两种或不同的种子使用相同的种子。我不认为随机性来源进入不同的过程会产生很大的不同。

将这 4 行添加到上面的示例将允许代码在每次运行时生成相同的结果。您应该看到与下面列出的相同的均方误差值（由于不同机器上的精度，可能会有一些微小的变化）：

```py
0.224045112999
0.00154879478823
0.00387589994044
0.0292376881968
0.00945528404353
0.013305765525
0.0206255228201
0.0359538356108
0.00441943512128
0.298706569397
```

您的结果应与我的匹配（忽略精确度的微小差异）。

## 如果我仍然得到不同的结果怎么办？

要重新迭代，最有效的报告结果和比较模型的方法是多次重复实验（30+）并使用汇总统计。

如果无法做到这一点，您可以通过为代码使用的随机数生成器播种获得 100％可重复的结果。上述解决方案应涵盖大多数情况，但不是全部。

如果您遵循上述说明并仍然在相同数据上获得相同算法的不同结果，该怎么办？

您可能还有其他未考虑的随机性来源。

### 来自第三方库的随机性

也许你的代码使用了一个额外的库，它使用了一个也必须种子的不同随机数生成器。

尝试将代码切换回所需的最低要求（例如，一个数据样本，一个训练时期等）并仔细阅读 API 文档，以便缩小引入随机性的其他第三方库。

### 使用 GPU 的随机性

以上所有示例均假设代码在 CPU 上运行。

当使用 GPU 训练模型时，后端可能被配置为使用复杂的 GPU 库堆栈，其中一些可能会引入他们自己或可能无法解释的随机源。

例如，有一些证据表明，如果您在堆栈中使用 [Nvidia cuDNN](https://developer.nvidia.com/cudnn) ，则[可能会引入其他随机源](https://github.com/fchollet/keras/issues/2479#issuecomment-213987747)并阻止您的结果的完全重现性。

### 来自复杂模型的随机性

由于您的模型的复杂性和训练的平行性，您可能会得到不可重复的结果。

这很可能是由后端库产生的效率引起的，也许是因为无法跨核心使用随机数序列。

我自己没有看到这个，但看到一些 GitHub 问题和 StackOverflow 问题的迹象。

您可以尝试降低模型的复杂性，看看这是否会影响结果的可重复性，只是为了缩小原因。

我建议您阅读后端如何使用随机性，看看是否有任何选项可供您使用。

在 Theano，请参阅：

*   [随机数](http://deeplearning.net/software/theano/sandbox/randomnumbers.html)
*   [友好的随机数](http://deeplearning.net/software/theano/library/tensor/shared_randomstreams.html)
*   [使用随机数](http://deeplearning.net/software/theano/tutorial/examples.html#using-random-numbers)

在 TensorFlow 中，请参阅：

*   [常数，序列和随机值](https://www.tensorflow.org/api_guides/python/constant_op)
*   [tf.set_random_seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed)

此外，请考虑搜索具有相同问题的其他人以获得进一步的洞察力。一些很棒的搜索地点包括：

*   [GrasHub 上的 Keras 问题](https://github.com/fchollet/keras/issues)
*   [The On 对 Github 的问题](https://github.com/Theano/Theano/issues)
*   [GitHub 上的 TensorFlow 问题](https://github.com/tensorflow/tensorflow/issues)
*   [StackOverflow 通用编程 Q＆amp; A](http://stackoverflow.com/)
*   [CrossValidated 机器学习 Q＆amp; A](https://stats.stackexchange.com/)

## 摘要

在本教程中，您了解了如何在 Keras 中获得可重现的神经网络模型结果。

具体来说，你学到了：

*   神经网络在设计上是随机的，并且可以固定随机源以使结果可重复。
*   您可以在 NumPy 和 TensorFlow 中播种随机数生成器，这将使大多数 Keras 代码 100％可重现。
*   在某些情况下，存在其他随机源，并且您有关于如何寻找它们的想法，也许也可以修复它们。

本教程有帮助吗？
在评论中分享您的经验。

您是否仍然可以通过 Keras 获得无法重现的结果？
分享您的经验;也许这里的其他人可以帮忙。