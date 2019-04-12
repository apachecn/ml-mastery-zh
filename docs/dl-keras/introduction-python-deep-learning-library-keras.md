# Python 深度学习库 Keras 简介

> 原文： [https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/](https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/)

Python 中两个为深度学习研究和开发提供基础的顶级数字平台是 Theano 和 TensorFlow。

两者都是非常强大的库，但两者都难以直接用于创建深度学习模型。

在这篇文章中，您将发现 Keras Python 库，它提供了一种在 Theano 或 TensorFlow 上创建一系列深度学习模型的简洁方便的方法。

让我们开始吧。

**2016 年 10 月更新**：更新了 Keras 1.1.0，Theano 0.8.2 和 TensorFlow 0.10.0 的示例。

![Introduction to the Python Deep Learning Library Keras](img/faed449c22db6586e71129af66aef2a5.png)

Python 深度学习库 Keras 简介
照片由 [Dennis Jarvis](https://www.flickr.com/photos/archer10/2216602404/) 拍摄，保留一些权利。

## 什么是 Keras？

Keras 是一个用于深度学习的极简主义 Python 库，可以在 Theano 或 TensorFlow 之上运行。

它的开发旨在使研究和开发尽可能快速简便地实施深度学习模型。

它运行在 Python 2.7 或 3.5 上，并且可以在给定底层框架的情况下在 GPU 和 CPU 上无缝执行。它是在许可的 MIT 许可下发布的。

Keras 由[FrançoisChollet](https://www.linkedin.com/in/fchollet)开发和维护，他是一位 Google 工程师，使用四个指导原则：

*   **模块性**：模型可以理解为单独的序列或图形。深度学习模型的所有关注点都是可以以任意方式组合的离散组件。
*   **极简主义**：该库提供了足够的结果，没有多余的装饰和最大化的可读性。
*   **可扩展性**：新组件有意在框架内轻松添加和使用，供研究人员试用和探索新想法。
*   **Python** ：没有自定义文件格式的单独模型文件。一切都是原生 Python。

## 如何安装 Keras

如果您已经拥有可用的 Python 和 SciPy 环境，那么 Keras 的安装相对简单。

您还必须在系统上安装 Theano 或 TensorFlow。

您可以在此处查看两个平台的安装说明：

*   [Theano 的安装说明](http://deeplearning.net/software/theano/install.html#install)
*   [TensorFlow](https://github.com/tensorflow/tensorflow#download-and-setup) 的安装说明

使用 [PyPI](https://pypi.python.org/pypi) 可以轻松安装 Keras，如下所示：

```py
sudo pip install keras
```

在撰写本文时，Keras 的最新版本是 1.1.0 版。您可以使用以下代码段在命令行上检查您的 Keras 版本：

您可以使用以下代码段在命令行上检查您的 Keras 版本：

```py
python -c "import keras; print keras.__version__"
```

运行上面的脚本你会看到：

```py
1.1.0
```

您可以使用相同的方法升级 Keras 的安装：

```py
sudo pip install --upgrade keras
```

## 针对 Keras 的 Theano 和 TensorFlow 后端

假设您同时安装了 Theano 和 TensorFlow，则可以配置 Keras 使用的[后端](http://keras.io/backend/)。

最简单的方法是在主目录中添加或编辑 Keras 配置文件：

```py
~/.keras/keras.json
```

其格式如下：

```py
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```

在此配置文件中，您可以将“_ 后端 _”属性从“ _tensorflow_ ”（默认值）更改为“ _theano_ ”。然后 Keras 将在下次运行时使用该配置。

您可以在命令行上使用以下代码段确认 Keras 使用的后端：

```py
python -c "from keras import backend; print backend._BACKEND"
```

使用默认配置运行此选项，您将看到：

```py
Using TensorFlow backend.
tensorflow
```

您还可以通过指定 KERAS_BACKEND 环境变量在命令行上指定 Keras 使用的后端，如下所示：

```py
KERAS_BACKEND=theano python -c "from keras import backend; print(backend._BACKEND)"
```

运行此示例打印：

```py
Using Theano backend.
theano
```

## 使用 Keras 构建深度学习模型

Keras 的重点是模型的概念。

主要类型的模型称为序列，它是层的线性堆栈。

您可以按照希望执行计算的顺序创建序列并向其添加层。

一旦定义，您就可以编译模型，该模型利用底层框架来优化模型执行的计算。在此，您可以指定损失函数和要使用的优化程序。

编译后，模型必须适合数据。这可以一次完成一批数据，也可以通过启动整个模型训练制度来完成。这是所有计算发生的地方。

经过训练，您可以使用模型对新数据进行预测。

我们可以总结一下 Keras 深度学习模型的构建如下：

1.  **定义你的模型**。创建序列并添加层。
2.  **编译你的模型**。指定损失函数和优化器。
3.  **适合您的型号**。使用数据执行模型。
4.  **做出预测**。使用该模型生成对新数据的预测。

## Keras 资源

下面的列表提供了一些其他资源，您可以使用它们来了解有关 Keras 的更多信息。

*   [Keras 官方主页](http://keras.io/)（文档）
*   [GitHub 上的 Keras 项目](https://github.com/fchollet/keras)
*   [Keras 用户组](https://groups.google.com/forum/#!forum/keras-users)

您是否正在寻找一个良好的深度学习教程来开始，请看看：

*   [用 Keras 逐步开发 Python 中的第一个神经网络](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

## 摘要

在这篇文章中，您发现了用于深度学习研究和开发的 Keras Python 库。

您发现 Keras 专为极简主义和模块化而设计，允许您快速定义深度学习模型并在 Theano 或 TensorFlow 后端运行它们。

你对 Keras 或这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。