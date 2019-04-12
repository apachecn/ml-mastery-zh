# Keras 中神经网络模型的 5 步生命周期

> 原文： [https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/](https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)

使用 Keras 在 Python 中创建和评估深度学习神经网络非常容易，但您必须遵循严格的模型生命周期。

在这篇文章中，您将发现在 Keras 中创建，训练和评估深度学习神经网络的逐步生命周期，以及如何使用训练有素的模型进行预测。

阅读这篇文章后你会知道：

*   如何在 Keras 中定义，编译，拟合和评估深度学习神经网络。
*   如何为回归和分类预测建模问题选择标准默认值。
*   如何将它们结合在一起，在 Keras 开发和运行您的第一个多层感知器网络。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Deep Learning Neural Network Life-Cycle in Keras](img/ba26805ccbbb3318546dbf2663ddbef9.png)

Keras 的深度学习神经网络生命周期
[Martin Stitchener](https://www.flickr.com/photos/dxhawk/6842278135/) 的照片，保留一些权利。

## 概观

下面概述了我们将要研究的 Keras 神经网络模型生命周期的 5 个步骤。

1.  定义网络。
2.  编译网络。
3.  适合网络。
4.  评估网络。
5.  作出预测。

![5 Step Life-Cycle for Neural Network Models in Keras](img/2996eabdf1f9d9a0bc2b5e1c62d6b4e5.png)

Keras 中神经网络模型的 5 步生命周期

## 步骤 1.定义网络

第一步是定义您的神经网络。

神经网络在 Keras 中定义为层序列。这些层的容器是 Sequential 类。

第一步是创建 Sequential 类的实例。然后，您可以创建层并按照它们应连接的顺序添加它们。

例如，我们可以分两步完成：

```py
model = Sequential()
model.add(Dense(2))
```

但是我们也可以通过创建一个层数组并将其传递给 Sequential 的构造函数来一步完成。

```py
layers = [Dense(2)]
model = Sequential(layers)
```

网络中的第一层必须定义预期的输入数量。指定它的方式可能因网络类型而异，但对于 Multilayer Perceptron 模型，这由 input_dim 属性指定。

例如，一个小的多层感知器模型，在可见层中有 2 个输入，隐藏层中有 5 个神经元，输出层中有一个神经元，可以定义为：

```py
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(1))
```

将序列模型视为管道，将原始数据输入底部，并将预测输出到顶部。

这在 Keras 中是一个有用的概念，因为传统上与层相关的关注点也可以拆分并作为单独的层添加，清楚地显示它们在从输入到预测的数据转换中的作用。例如，可以提取转换来自层中每个神经元的求和信号的激活函数，并将其作为称为激活的层状对象添加到 Sequential 中。

```py
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```

激活函数的选择对于输出层是最重要的，因为它将定义预测将采用的格式。

例如，下面是一些常见的预测建模问题类型以及可以在输出层中使用的结构和标准激活函数：

*   **回归**：线性激活函数或'线性'和与输出数匹配的神经元数。
*   **二元分类（2 级）**：Logistic 激活函数或'sigmoid'和一个神经元输出层。
*   **多类分类（＆gt; 2 类）**：假设单热编码输出模式，Softmax 激活函数或'softmax'和每类值一个输出神经元。

## 第 2 步。编译网络

一旦我们定义了网络，我们就必须编译它。

编译是一个效率步骤。它将我们定义的简单层序列转换为高效的矩阵变换系列，其格式应在 GPU 或 CPU 上执行，具体取决于 Keras 的配置方式。

将编译视为网络的预计算步骤。

定义模型后始终需要编译。这包括在使用优化方案训练之前以及从保存文件加载一组预先训练的权重之前。原因是编译步骤准备了网络的有效表示，这也是对硬件进行预测所必需的。

编译需要指定许多参数，专门用于训练您的网络。具体地，用于训练网络的优化算法和用于评估由优化算法最小化的网络的损失函数。

例如，下面是编译定义模型并指定随机梯度下降（sgd）优化算法和均方误差（mse）损失函数的情况，用于回归类型问题。

```py
model.compile(optimizer='sgd', loss='mse')
```

预测建模问题的类型对可以使用的损失函数的类型施加约束。

例如，下面是不同预测模型类型的一些标准损失函数：

*   **回归**：均值平方误差或' _mse_ '。
*   **二元分类（2 类）**：对数损失，也称为交叉熵或' _binary_crossentropy_ '。
*   **多类分类（＆gt; 2 类）**：多类对数损失或'_ 分类 _ 交响曲 _'。

您可以查看 Keras 支持的[损失函数套件。](http://keras.io/objectives/)

最常见的优化算法是随机梯度下降，但 Keras 还支持其他最先进的优化算法的[套件。](http://keras.io/optimizers/)

也许最常用的优化算法因为它们通常具有更好的表现：

*   **随机梯度下降**或' _sgd_ '，需要调整学习速度和动量。
*   **ADAM** 或' _adam_ '需要调整学习率。
*   **RMSprop** 或' _rmsprop_ '需要调整学习率。

最后，除了损失函数之外，您还可以指定在拟合模型时收集的度量标准。通常，要收集的最有用的附加度量标准是分类问题的准确性。要收集的度量标准由数组中的名称指定。

例如：

```py
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
```

## 步骤 3.适合网络

一旦网络被编译，它就可以适合，这意味着在训练数据集上调整权重。

安装网络需要指定训练数据，输入模式矩阵 X 和匹配输出模式 y 的阵列。

使用反向传播算法训练网络，并根据编译模型时指定的优化算法和损失函数进行优化。

反向传播算法要求网络训练指定数量的时期或暴露于训练数据集。

每个迭代可以被划分为称为批次的输入 - 输出模式对的组。这定义了在一个迭代内更新权重之前网络所暴露的模式数。它也是一种效率优化，确保一次不会将太多输入模式加载到内存中。

拟合网络的最小例子如下：

```py
history = model.fit(X, y, batch_size=10, epochs=100)
```

适合后，将返回历史对象，该对象提供训练期间模型表现的摘要。这包括损失和编译模型时指定的任何其他指标，记录每个迭代。

## 第 4 步。评估网络

一旦网络被训练，就可以对其进行评估。

可以在训练数据上评估网络，但是这不会提供作为预测模型的网络表现的有用指示，因为它之前已经看到了所有这些数据。

我们可以在测试期间看不到的单独数据集上评估网络的表现。这将提供对网络表现的估计，以便对未来看不见的数据进行预测。

该模型评估所有测试模式的损失，以及编译模型时指定的任何其他指标，如分类准确性。返回评估指标列表。

例如，对于使用精度度量编制的模型，我们可以在新数据集上对其进行评估，如下所示：

```py
loss, accuracy = model.evaluate(X, y)
```

## 第 5 步。做出预测

最后，一旦我们对拟合模型的表现感到满意，我们就可以用它来预测新数据。

这就像使用新输入模式数组调用模型上的 predict（）函数一样简单。

例如：

```py
predictions = model.predict(x)
```

预测将以网络输出层提供的格式返回。

在回归问题的情况下，这些预测可以是直接问题的格式，由线性激活函数提供。

对于二元分类问题，预测可以是第一类的概率数组，其可以通过舍入转换为 1 或 0。

对于多类分类问题，结果可以是概率数组的形式（假设一个热编码输出变量），可能需要使用 [argmax 函数](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)将其转换为单个类输出预测。

## 端到端工作示例

让我们将所有这些与一个小例子结合起来。

这个例子将使用皮马印第安人发病的糖尿病二元分类问题，即可以从 UCI 机器学习库下载[（更新：](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

该问题有 8 个输入变量和一个输出类变量，其整数值为 0 和 1。

我们将构建一个多层感知器神经网络，在可见层中有 8 个输入，隐藏层中有 12 个神经元，具有整流器激活功能，输出层中有 1 个神经元具有 S 形激活功能。

我们将使用 ADAM 优化算法和对数损失函数对批量大小为 10 的 100 个时期进行网络训练。

一旦适合，我们将评估训练数据的模型，然后对训练数据进行独立预测。这是为了简洁起见，通常我们会在单独的测试数据集上评估模型并对新数据进行预测。

完整的代码清单如下。

```py
# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# load and prepare the dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
# 1\. define the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2\. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3\. fit the network
history = model.fit(X, Y, epochs=100, batch_size=10)
# 4\. evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5\. make predictions
probabilities = model.predict(X)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))
```

运行此示例将生成以下输出：

```py
...
768/768 [==============================] - 0s - loss: 0.5219 - acc: 0.7591
Epoch 99/100
768/768 [==============================] - 0s - loss: 0.5250 - acc: 0.7474
Epoch 100/100
768/768 [==============================] - 0s - loss: 0.5416 - acc: 0.7331
 32/768 [>.............................] - ETA: 0s
Loss: 0.51, Accuracy: 74.87%
Prediction Accuracy: 74.87%
```

## 摘要

在这篇文章中，您使用 Keras 库发现了深度学习神经网络的 5 步生命周期。

具体来说，你学到了：

*   如何在 Keras 中为神经网络定义，编译，拟合，评估和预测。
*   如何为分类和回归问题选择激活函数和输出层配置。
*   如何在 Keras 开发和运行您的第一个多层感知器模型。

您对 Keras 中的神经网络模型有任何疑问吗？在评论中提出您的问题，我会尽力回答。