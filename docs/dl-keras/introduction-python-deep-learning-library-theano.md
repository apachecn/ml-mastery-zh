# Python 深度学习库 Theano 简介

> 原文： [https://machinelearningmastery.com/introduction-python-deep-learning-library-theano/](https://machinelearningmastery.com/introduction-python-deep-learning-library-theano/)

Theano 是一个用于快速数值计算的 Python 库，可以在 CPU 或 GPU 上运行。

它是 Python 中深度学习的关键基础库，您可以直接使用它来创建深度学习模型或包装库，从而大大简化过程。

在这篇文章中，您将发现 Theano Python 库。

![Introduction to the Python Deep Learning Library Theano](img/97a5a2d7ca50d98d6dd0d2787ea7bd18.png)

Python 深度学习库 Theano
照片由 [Kristoffer Trolle](https://www.flickr.com/photos/kristoffer-trolle/17088729869/) 拍摄，保留一些权利。

## 什么是 Theano？

Theano 是一个根据 BSD 许可发布的开源项目，由加拿大魁北克省蒙特利尔大学（现为 [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html) 的所在地）的 LISA（现为 [MILA](http://mila.umontreal.ca/) ）小组开发。它以[希腊数学家](https://en.wikipedia.org/wiki/Theano_(philosopher))的名字命名。

在它的核心 Theano 是 Python 中数学表达式的编译器。它知道如何使用您的结构并将它们转换为非常有效的代码，使用 NumPy，高效的本机库（如 [BLAS](http://www.netlib.org/blas/) 和本机代码（C ++））在 CPU 或 GPU 上尽可能快地运行。

它使用大量巧妙的代码优化来尽可能地从硬件中获取尽可能多的表现。如果你深入研究代码中数学优化的细节，[请查看这个有趣的列表](http://deeplearning.net/software/theano/optimizations.html#optimizations)。

Theano 表达式的实际语法是象征性的，这可能不适合初学者用于正常的软件开发。具体而言，表达式在抽象意义上定义，编译后实际用于进行计算。

它专门用于处理深度学习中使用的大型神经网络算法所需的计算类型。它是同类中最早的库之一（2007 年开始开发），被认为是深度学习研究和开发的行业标准。

## 如何安装 Theano

Theano 为主要操作系统提供了广泛的安装说明：Windows，OS X 和 Linux。阅读适合您平台的[安装 Theano 指南](http://deeplearning.net/software/theano/install.html)。

Theano 假设使用 [SciPy](https://www.scipy.org/) 工作的 Python 2 或 Python 3 环境。有一些方法可以使安装更容易，例如使用 [Anaconda](https://www.continuum.io/downloads) 在您的机器上快速设置 Python 和 SciPy 以及使用 [Docker 图像](http://deeplearning.net/software/theano/install.html#docker-images)。

使用 Python 和 SciPy 环境，安装 Theano 相对简单。来自 PyPI 使用 pip，例如：

```py
pip install Theano
```

在撰写本文时，Theano 的最后一个正式版本是在 2016 年 3 月 21 日发布的 0.8 版本。

可能会发布新版本，您需要更新以获取任何错误修复和效率改进。您可以使用 pip 升级 Theano，如下所示：

```py
sudo pip install --upgrade --no-deps theano
```

您可能想要使用直接从 Github 检查的最新版本的 Theano。

对于一些使用前沿 API 更改的包装器库，可能需要这样做。您可以直接从 Github 结帐安装 Theano，如下所示：

```py
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

您现在已准备好在 CPU 上运行 Theano，这对于小型模型的开发来说非常好。

大型号可能在 CPU 上运行缓慢。如果你有一个 Nvidia GPU，你可能想要配置 Theano 来使用你的 GPU。阅读[使用适用于 Linux](http://deeplearning.net/software/theano/install.html#using-the-gpu) 或 [Mac OS X 的 GPU 指南设置 Theano 使用 GPU](http://deeplearning.net/software/theano/install.html#gpu-macos) 和[使用 GPU 指南](http://deeplearning.net/software/theano/tutorial/using_gpu.html)如何测试是否可以工作中。

## 简单的 Theano 例子

在本节中，我们将演示一个简单的 Python 脚本，它为您提供了 Theano 的味道。

它取自 [Theano 概览指南](http://deeplearning.net/software/theano/introduction.html)。在这个例子中，我们定义了两个符号浮点变量 _a_ 和 _b_ 。

我们定义了一个使用这些变量 _（c = a + b）_ 的表达式。

然后我们使用 Theano 将这个符号表达式编译成一个函数，我们稍后可以使用它。

最后，我们使用我们的编译表达式，插入一些实际值并使用高效的编译 Theano 代码执行计算。

```py
import theano
from theano import tensor
# declare two symbolic floating-point scalars
a = tensor.dscalar()
b = tensor.dscalar()
# create a simple expression
c = a + b
# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function([a,b], c)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
assert 4.0 == f(1.5, 2.5)
```

运行该示例不提供任何输出。 _1.5 + 2.5 = 4.0_ 的断言是正确的。

这是一个有用的示例，因为它为您提供了如何定义，编译和使用符号表达式的风格。您可以看到这可以扩展到深度学习所需的大向量和矩阵运算。

## Theano 的扩展和包装

如果您不熟悉深度学习，则不必直接使用 Theano。

实际上，我们强烈建议您使用许多流行的 Python 项目之一，这些项目使 Theano 更容易用于深度学习。

这些项目提供 Python 中的数据结构和行为，专门用于快速可靠地创建深度学习模型，同时确保由 Theano 创建和执行快速高效的模型。

库提供的 Theano 语法的数量各不相同。

*   例如， [Lasagne 库](http://lasagne.readthedocs.org/en/latest/)为创建深度学习模型提供了便利类，但仍希望您了解并使用 Theano 语法。这对于知道或愿意学习一点 Theano 的初学者来说也是有益的。
*   另一个例子是 [Keras](http://keras.io/) 完全隐藏 Theano 并提供了一个非常简单的 API 来创建深度学习模型。它很好地隐藏了 Theano，它实际上可以作为另一个流行的基础框架 [TensorFlow](https://www.tensorflow.org/) 的包装器运行。

我强烈建议您直接尝试使用 Theano，然后选择一个包装库来学习和练习深度学习。

有关在 Theano 上构建的库的完整列表，请参阅 Theano Wiki 上的[相关项目指南](https://github.com/Theano/Theano/wiki/Related-projects)。

## 更多 Theano 资源

在 Theano 寻找更多资源？看看下面的一些内容。

*   [Theano 官方主页](http://deeplearning.net/software/theano/)
*   [Theano GitHub 存储库](https://github.com/Theano/Theano/)
*   [Theano：Python 中的 CPU 和 GPU 数学编译器](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf)（2010）（PDF）
*   [在 Theano 上建立的库名单](https://github.com/Theano/Theano/wiki/Related-projects)
*   [Theano 配置选项列表](http://deeplearning.net/software/theano/library/config.html)

### Theano 和深度学习教程

*   [Theano 教程](http://deeplearning.net/software/theano/tutorial/index.html)
*   [Theano 教程的深度学习](http://www.deeplearning.net/tutorial/)

### 获得 Theano 的帮助

*   [Theano 用户 Google Group](http://groups.google.com/group/theano-users?pli=1)

## 摘要

在这篇文章中，您发现了 Theano Python 库，用于高效的数值计算。

您了解到它是一个用于深度学习研究和开发的基础库，它可以直接用于创建深度学习模型，或者通过基于它的便利库（如 Lasagne 和 Keras）。

您对 Theano 或 Python 中的深度学习有任何疑问吗？在评论中提出您的问题，我会尽力回答。