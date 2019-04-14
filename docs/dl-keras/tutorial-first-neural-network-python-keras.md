# 用 Keras 逐步开发 Python 中的第一个神经网络

> 原文： [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

Keras 是一个功能强大且易于使用的 Python 库，用于开发和评估[深度学习](http://machinelearningmastery.com/what-is-deep-learning/)模型。

它包含了高效的数值计算库 Theano 和 TensorFlow，允许您在几行代码中定义和训练神经网络模型。

在这篇文章中，您将了解如何使用 Keras 在 Python 中创建第一个神经网络模型。

让我们开始吧。

*   **2017 年 2 月更新**：更新了预测示例，因此在 Python 2 和 Python 3 中可以进行舍入。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Tour of Deep Learning Algorithms](img/b146789c05a7cd65c9b0afcfc33dcd5e.png)

使用 Keras 逐步开发 Python 中的第一个神经网络
Phil Whitehouse 的照片，保留一些权利。

## 教程概述

不需要很多代码，但我们会慢慢跨过它，以便您知道将来如何创建自己的模型。

您将在本教程中介绍的步骤如下：

1.  加载数据。
2.  定义模型。
3.  编译模型。
4.  适合模型。
5.  评估模型。
6.  把它绑在一起。

本教程有一些要求：

1.  您已安装并配置了 Python 2 或 3。
2.  您已安装并配置了 SciPy（包括 NumPy）。
3.  您安装并配置了 Keras 和后端（Theano 或 TensorFlow）。

如果您需要有关环境的帮助，请参阅教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

创建一个名为 **keras_first_network.py** 的新文件，然后在您输入时将代码输入或复制并粘贴到文件中。

## 1.加载数据

每当我们使用使用随机过程（例如随机数）的机器学习算法时，最好设置随机数种子。

这样您就可以反复运行相同的代码并获得相同的结果。如果您需要演示结果，使用相同的随机源比较算法或调试代码的一部分，这非常有用。

您可以使用您喜欢的任何种子初始化随机数生成器，例如：

```py
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
```

现在我们可以加载我们的数据。

在本教程中，我们将使用 Pima Indians 糖尿病数据集。这是来自 UCI 机器学习库的标准机器学习数据集。它描述了皮马印第安人的患者病历数据，以及他们是否在五年内患有糖尿病。

因此，它是二元分类问题（糖尿病发作为 1 或不为 0）。描述每个患者的所有输入变量都是数字的。这使得它可以直接用于期望数字输入和输出值的神经网络，是我们在 Keras 的第一个神经网络的理想选择。

*   [数据集文件](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
*   [数据集详情](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)

下载数据集并将其放在本地工作目录中，与 python 文件相同。使用文件名保存：

```py
pima-indians-diabetes.csv
```

您现在可以使用 NumPy 函数 **loadtxt（）**直接加载文件。有八个输入变量和一个输出变量（最后一列）。加载后，我们可以将数据集拆分为输入变量（X）和输出类变量（Y）。

```py
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
```

我们初始化了随机数生成器，以确保我们的结果可重现并加载我们的数据。我们现在准备定义我们的神经网络模型。

请注意，数据集有 9 列，范围 0：8 将选择 0 到 7 之间的列，在索引 8 之前停止。如果这对您来说是新的，那么您可以在此帖子中了解有关数组切片和范围的更多信息：

*   [如何在 Python 中为机器学习索引，切片和重塑 NumPy 数组](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)

## 2.定义模型

Keras 中的模型被定义为层序列。

我们创建一个 Sequential 模型并一次添加一个层，直到我们对网络拓扑感到满意为止。

要做的第一件事是确保输入层具有正确数量的输入。当使用 **input_dim** 参数创建第一层并为 8 个输入变量将其设置为 8 时，可以指定此项。

我们如何知道层数及其类型？

这是一个非常难的问题。我们可以使用启发式方法，通常通过试验和错误实验的过程找到最好的网络结构。通常，如果有任何帮助，您需要一个足够大的网络来捕获问题的结构。

在此示例中，我们将使用具有三个层的完全连接的网络结构。

完全连接的层使用 Dense 类定义。我们可以指定层中神经元的数量作为第一个参数，初始化方法作为 **init** 指定第二个参数，并使用**激活**参数指定激活函数。

在这种情况下，我们将网络权重初始化为从均匀分布（' **uniform** '）生成的小随机数，在这种情况下介于 0 和 0.05 之间，因为这是 Keras 中的默认均匀权重初始化。对于从高斯分布产生的小随机数，另一种传统的替代方案是'**正常'**。

我们将在前两层使用[整流器](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))（' **relu** '）激活函数，在输出层使用 sigmoid 函数。过去，所有层都优先选择 sigmoid 和 tanh 激活函数。目前，使用整流器激活功能可以获得更好的表现。我们在输出层使用 sigmoid 来确保我们的网络输出介于 0 和 1 之间，并且很容易映射到 1 级概率或者使用默认阈值 0.5 捕捉到任一类的硬分类。

我们可以通过添加每一层将它们拼凑在一起。第一层有 12 个神经元，需要 8 个输入变量。第二个隐藏层有 8 个神经元，最后，输出层有 1 个神经元来预测类别（是否发生糖尿病）。

```py
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## 3.编译模型

既然定义了模型，我们就可以编译它。

编译模型使用封面下的高效数字库（所谓的后端），如 Theano 或 TensorFlow。后端自动选择最佳方式来表示网络以进行训练并使预测在硬件上运行，例如 CPU 或 GPU 甚至分布式。

编译时，我们必须指定训练网络时所需的一些其他属性。记住训练网络意味着找到最佳权重集来预测这个问题。

我们必须指定用于评估一组权重的损失函数，用于搜索网络的不同权重的优化器以及我们希望在训练期间收集和报告的任何可选指标。

在这种情况下，我们将使用对数损失，对于二元分类问题，在 Keras 中定义为“ **binary_crossentropy** ”。我们还将使用有效的梯度下降算法“ **adam** ”，因为它是一个有效的默认值。在“ [Adam：随机优化方法](http://arxiv.org/abs/1412.6980)”一文中了解有关 Adam 优化算法的更多信息。

最后，因为它是一个分类问题，我们将收集并报告分类准确度作为指标。

```py
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## 4.适合模型

我们已经定义了我们的模型并将其编译为高效计算。

现在是时候在一些数据上执行模型了。

我们可以通过调用模型上的 **fit（）**函数来训练或拟合我们的加载数据模型。

训练过程将通过名为 epochs 的数据集进行固定次数的迭代，我们必须使用 **nepochs** 参数指定。我们还可以设置在执行网络中的权重更新之前评估的实例数，称为批量大小并使用 **batch_size** 参数进行设置。

对于这个问题，我们将运行少量迭代（150）并使用相对较小的批量大小 10.再次，这些可以通过试验和错误通过实验选择。

```py
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
```

这是工作在 CPU 或 GPU 上发生的地方。

此示例不需要 GPU，但如果您对如何在云中廉价地在 GPU 硬件上运行大型模型感兴趣，请参阅此帖子：

*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 5.评估模型

我们已经在整个数据集上训练了神经网络，我们可以在同一数据集上评估网络的表现。

这只会让我们了解我们对数据集建模的程度（例如训练精度），但不知道算法在新数据上的表现如何。我们这样做是为了简化，但理想情况下，您可以将数据分成训练和测试数据集，以便对模型进行训练和评估。

您可以使用模型上的 **evaluate（）**函数在训练数据集上评估模型，并将其传递给用于训练模型的相同输入和输出。

这将为每个输入和输出对生成预测并收集分数，包括平均损失和您配置的任何指标，例如准确性。

```py
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

## 6.将它们结合在一起

您刚刚看到了如何在 Keras 中轻松创建第一个神经网络模型。

让我们将它们组合成一个完整的代码示例。

```py
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

运行此示例，您应该看到 150 个迭代中的每个迭代记录每个历史记录的丢失和准确性的消息，然后对训练数据集上的训练模型进行最终评估。

在带有 Theano 后端的 CPU 上运行的工作站上执行大约需要 10 秒钟。

```py
...
Epoch 145/150
768/768 [==============================] - 0s - loss: 0.5105 - acc: 0.7396
Epoch 146/150
768/768 [==============================] - 0s - loss: 0.4900 - acc: 0.7591
Epoch 147/150
768/768 [==============================] - 0s - loss: 0.4939 - acc: 0.7565
Epoch 148/150
768/768 [==============================] - 0s - loss: 0.4766 - acc: 0.7773
Epoch 149/150
768/768 [==============================] - 0s - loss: 0.4883 - acc: 0.7591
Epoch 150/150
768/768 [==============================] - 0s - loss: 0.4827 - acc: 0.7656
 32/768 [>.............................] - ETA: 0s
acc: 78.26%
```

**注意**：如果您尝试在 IPython 或 Jupyter 笔记本中运行此示例，则可能会出错。原因是训练期间的输出进度条。您可以通过在 **model.fit（）**的调用中设置 **verbose = 0** 来轻松关闭它们。

请注意，您的模型的技能可能会有所不同。

神经网络是一种随机算法，意味着相同数据上的相同算法可以训练具有不同技能的不同模型。这是一个功能，而不是一个 bug。您可以在帖子中了解更多相关信息：

*   [在机器学习中拥抱随机性](https://machinelearningmastery.com/randomness-in-machine-learning/)

我们确实尝试修复随机种子以确保您和我获得相同的模型，因此得到相同的结果，但这并不总是适用于所有系统。我在这里写了更多关于使用 Keras 模型再现结果的[问题](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)。

## 7.奖金：做出预测

我被问到的头号问题是：

> 在训练我的模型后，如何使用它来预测新数据？

好问题。

我们可以调整上面的示例并使用它来生成训练数据集的预测，假装它是我们以前从未见过的新数据集。

进行预测就像调用 **model.predict（）**一样简单。我们在输出层使用 sigmoid 激活函数，因此预测将在 0 到 1 之间的范围内。我们可以通过舍入它们轻松地将它们转换为这个分类任务的清晰二元预测。

下面列出了为训练数据中的每条记录进行预测的完整示例。

```py
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
```

现在运行此修改示例将打印每个输入模式的预测。如果需要，我们可以直接在我们的应用程序中使用这些预测

```py
[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
```

如果您对使用经过训练的模型进行预测有更多疑问，请参阅此帖子：

*   [如何用 Keras 进行预测](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)

## 摘要

在这篇文章中，您发现了如何使用功能强大的 Keras Python 库创建第一个神经网络模型以进行深度学习。

具体来说，您学习了使用 Keras 创建神经网络或深度学习模型的五个关键步骤，包括：

1.  如何加载数据。
2.  如何在 Keras 中定义神经网络。
3.  如何使用高效的数字后端编译 Keras 模型。
4.  如何训练数据模型。
5.  如何评估数据模型。

您对 Keras 或本教程有任何疑问吗？
在评论中提出您的问题，我会尽力回答。

### 相关教程

您是否正在寻找使用 Python 和 Keras 的更多深度学习教程？

看看其中一些：

*   [Keras 神经网络模型的 5 步生命周期](http://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)
*   [如何使用 Keras 网格搜索 Python 中的深度学习模型的超参数](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
*   [Keras 中深度学习的时间序列预测](http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)
*   [Keras 深度学习库的多类分类教程](http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)
*   [使用 Python 中的 Keras 深度学习库进行回归教程](http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)