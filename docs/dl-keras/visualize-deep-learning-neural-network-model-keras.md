# 如何在 Keras 中可视化深度学习神经网络模型

> 原文： [https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)

Keras Python 深度学习库提供了可视化和更好地理解您的神经网络模型的工具。

在本教程中，您将发现如何在 Keras 中总结和可视化您的深度学习模型。

完成本教程后，您将了解：

*   如何创建深度学习模型的文本摘要。
*   如何创建深度学习模型的图形图。
*   在 Keras 开发深度学习模型时的最佳实践技巧。

让我们开始吧。

![How to Visualize a Deep Learning Neural Network Model in Keras](img/ae4bbe3728d2ae14a8486fd00fc87a8b.png)

如何在 Keras 中可视化深度学习神经网络模型
照片由 [Ed Dunens](https://www.flickr.com/photos/blachswan/14990404869/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  示例模型
2.  总结模型
3.  可视化模型
4.  最佳实践技巧

## 示例模型

我们可以从 Keras 中定义一个简单的多层感知器模型开始，我们可以将其用作摘要和可视化的主题。

我们将定义的模型有一个输入变量，一个带有两个神经元的隐藏层，以及一个带有一个二进制输出的输出层。

例如：

```py
[1 input] -> [2 neurons] -> [1 output]
```

下面提供了该网络的代码清单。

```py
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## 总结模型

Keras 提供了一种总结模型的方法。

摘要是文本性的，包括以下信息：

*   层和它们在模型中的顺序。
*   每层的输出形状。
*   每层中的参数（权重）数。
*   模型中的参数（权重）总数。

可以通过调用模型上的 _summary（）_ 函数来创建摘要，该函数返回可以打印的字符串。

下面是打印已创建模型摘要的更新示例。

```py
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

运行此示例将打印下表。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 2)                 4
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 7
Trainable params: 7
Non-trainable params: 0
_________________________________________________________________
```

我们可以清楚地看到每层的输出形状和权重数量。

## 可视化模型

摘要对于简单模型很有用，但对于具有多个输入或输出的模型可能会造成混淆。

Keras 还提供了创建网络神经网络图的功能，可以使更复杂的模型更容易理解。

Keras 中的 _plot_model（）_ 功能将创建您的网络图。这个函数有一些有用的参数：

*   _ 型号 _ :(必填）您想要绘制的模型。
*   _to_file_ :(必需）要保存绘图的文件的名称。
*   _show_shapes_ :(可选，默认为 _False_ ）是否显示每个层的输出形状。
*   _show_layer_names_ :(可选，默认为 _True_ ）是否显示每个层的名称。

下面是绘制创建模型的更新示例。

注意，该示例假设您已安装 [graphviz 图形库](http://www.graphviz.org/)和 [Python 接口](https://pypi.python.org/pypi/graphviz)。

```py
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

运行该示例将创建文件 _model_plot.png_ ，其中包含已创建模型的图。

![Plot of Neural Network Model Graph](img/cd2ea0cb6ea3f16f73d52c1580d22310.png)

神经网络模型图的绘制

## 最佳实践技巧

我通常建议始终在 Keras 中创建神经网络模型的摘要和图表。

我推荐这个有几个原因：

*   **确认层顺序**。使用顺序 API 以错误的顺序添加层或使用功能 API 错误地将它们连接在一起很容易。图表图可以帮助您确认模型是否按照您预期的方式连接。
*   **确认每层的输出形状**。在定义复杂网络（如卷积和递归神经网络）的输入数据形状时，常常会遇到问题。摘要和图表可以帮助您确认网络的输入形状是否符合您的预期。
*   **确认参数**。一些网络配置可以使用更少的参数，例如在编码器 - 解码器递归神经网络中使用 _TimeDistributed_ 包裹的密集层。查看摘要可以帮助发现使用远远超出预期的参数的情况。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [模型可视化 Keras API](https://keras.io/visualization/)
*   [Graphviz - 图形可视化软件](http://www.graphviz.org/)
*   [Graphviz 的简单 Python 接口](https://pypi.python.org/pypi/graphviz)

## 摘要

在本教程中，您了解了如何在 Keras 中总结和可视化您的深度学习模型。

具体来说，你学到了：

*   如何创建深度学习模型的文本摘要。
*   如何创建深度学习模型的图形图。
*   在 Keras 开发深度学习模型时的最佳实践技巧。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。