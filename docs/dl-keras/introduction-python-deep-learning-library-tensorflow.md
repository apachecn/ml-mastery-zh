# Python 深度学习库 TensorFlow 简介

> 原文： [https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/](https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/)

TensorFlow 是一个用于 Google 创建和发布的快速数值计算的 Python 库。

它是一个基础库，可用于直接创建深度学习模型，或使用包装库来简化在 TensorFlow 之上构建的过程。

在这篇文章中，您将发现用于深度学习的 TensorFlow 库。

让我们开始吧。

![Introduction to the Python Deep Learning Library TensorFlow](img/fe8f0396c0f9d7b02308150c33abe2da.png)

Python 深度学习库简介 TensorFlow
摄影： [Nicolas Raymond](https://www.flickr.com/photos/82955120@N05/15932303392/) ，保留一些权利。

## 什么是 TensorFlow？

TensorFlow 是一个用于快速数值计算的开源库。

它由 Google 创建并维护，并在 Apache 2.0 开源许可下发布。虽然可以访问底层的 C ++ API，但 API 名义上是用于 Python 编程语言的。

与 Theano 等深度学习中使用的其他数值库不同，TensorFlow 设计用于研究和开发以及生产系统，尤其是谷歌搜索中的 [RankBrain 和有趣的](https://en.wikipedia.org/wiki/RankBrain) [DeepDream 项目](https://en.wikipedia.org/wiki/DeepDream) ]。

它可以在单 CPU 系统，GPU 以及移动设备和数百台机器的大规模分布式系统上运行。

## 如何安装 TensorFlow

如果您已经拥有 Python SciPy 环境，那么 TensorFlow 的安装非常简单。

TensorFlow 适用于 Python 2.7 和 Python 3.3+。您可以按照 TensorFlow 网站上的[下载和设置说明](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)进行操作。通过 PyPI 进行安装可能是最简单的，并且下载和设置网页上有用于 Linux 或 Mac OS X 平台的 pip 命令的特定说明。

如果您愿意，还可以使用 [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) 和[泊坞窗图像](https://www.docker.com/)。

要使用 GPU，只支持 Linux，它需要 Cuda Toolkit。

## 你在 TensorFlow 中的第一个例子

根据有向图的结构中的数据流和操作来描述计算。

*   **节点**：节点执行计算并具有零个或多个输入和输出。在节点之间移动的数据称为张量，它是实数值的多维数组。
*   **Edges** ：该图定义了数据流，分支，循环和状态更新。特殊边缘可用于同步图形中的行为，例如等待完成多个输入的计算。
*   **操作**：一个操作是一个命名的抽象计算，它可以获取输入属性并产生输出属性。例如，您可以定义添加或乘法操作。

### 用 TensorFlow 计算

第一个示例是 [TensorFlow 网站](https://github.com/tensorflow/tensorflow)上的示例的修改版本。它显示了如何使用会话创建会话，定义常量和使用这些常量执行计算。

```py
import tensorflow as tf
sess = tf.Session()
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
```

运行此示例显示：

```py
42
```

### 使用 TensorFlow 进行线性回归

下一个例子来自 [TensorFlow 教程](https://www.tensorflow.org/versions/r0.8/get_started/index.html)的介绍。

此示例显示了如何定义变量（例如 W 和 b）以及作为计算结果的变量（y）。

我们对 TensorFlow 有一定的了解，它将计算的定义和声明与会话中的执行和运行调用分开。

```py
import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]
```

运行此示例将输出以下输出：

```py
(0, array([ 0.2629351], dtype=float32), array([ 0.28697217], dtype=float32))
(20, array([ 0.13929555], dtype=float32), array([ 0.27992988], dtype=float32))
(40, array([ 0.11148042], dtype=float32), array([ 0.2941364], dtype=float32))
(60, array([ 0.10335406], dtype=float32), array([ 0.29828694], dtype=float32))
(80, array([ 0.1009799], dtype=float32), array([ 0.29949954], dtype=float32))
(100, array([ 0.10028629], dtype=float32), array([ 0.2998538], dtype=float32))
(120, array([ 0.10008363], dtype=float32), array([ 0.29995731], dtype=float32))
(140, array([ 0.10002445], dtype=float32), array([ 0.29998752], dtype=float32))
(160, array([ 0.10000713], dtype=float32), array([ 0.29999638], dtype=float32))
(180, array([ 0.10000207], dtype=float32), array([ 0.29999897], dtype=float32))
(200, array([ 0.1000006], dtype=float32), array([ 0.29999971], dtype=float32))
```

您可以在[基本使用指南](https://www.tensorflow.org/versions/r0.8/get_started/basic_usage.html)中了解有关 TensorFlow 机制的更多信息。

## 更多深度学习模型

您的 TensorFlow 安装附带了许多深度学习模型，您可以直接使用它们进行试验。

首先，您需要找出系统上 TensorFlow 的安装位置。例如，您可以使用以下 Python 脚本：

```py
python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```

例如，这可能是：

```py
/usr/lib/python2.7/site-packages/tensorflow
```

切换到此目录并记下 models 子目录。包括许多深度学习模型，包含类似教程的注释，例如：

*   多线程 word2vec 迷你批量跳过克模型。
*   多线程 word2vec unbatched skip-gram 模型。
*   CNN 用于 CIFAR-10 网络。
*   简单，端到端，类似 LeNet-5 的卷积 MNIST 模型示例。
*   具有注意机制的序列到序列模型。

还要检查 examples 目录，因为它包含使用 MNIST 数据集的示例。

在 TensorFlow 主网站上还有一个很棒的[教程列表](https://www.tensorflow.org/versions/r0.8/tutorials/index.html)。它们展示了如何使用不同的网络类型，不同的数据集以及如何以各种不同的方式使用框架。

最后，有 [TensorFlow 游乐场](http://playground.tensorflow.org/)，您可以在 Web 浏览器中试验小型网络。

## TensorFlow 资源

*   [TensorFlow 官方主页](https://www.tensorflow.org/)
*   [GitHub 上的 TensforFlow 项目](https://github.com/tensorflow/tensorflow)
*   [TensorFlow 教程](https://www.tensorflow.org/versions/r0.7/tutorials/index.html)

### 更多资源

*   [关于 Udacity 的 TensorFlow 课程](https://www.udacity.com/course/deep-learning--ud730)
*   [TensorFlow：异构分布式系统上的大规模机器学习](http://download.tensorflow.org/paper/whitepaper2015.pdf)（2015）

## 摘要

在这篇文章中，您发现了用于深度学习的 TensorFlow Python 库。

您了解到它是一个快速数值计算库，专门为大型深度学习模型的开发和评估所需的操作类型而设计。

您对 TensorFlow 或者这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。