# 在 Python 迷你课程中应用深度学习

> 原文： [https://machinelearningmastery.com/applied-deep-learning-in-python-mini-course/](https://machinelearningmastery.com/applied-deep-learning-in-python-mini-course/)

深度学习是一个迷人的研究领域，这些技术在一系列具有挑战性的机器学习问题中取得了世界一流的成果。

深入学习可能很难开始。

您应该使用哪个库以及您应该关注哪些技术？

在这篇文章中，您将学习一个由 14个 部分组成的 Python 深度学习速成课程，其中包含易于使用且功能强大的 Keras 库。

这个迷你课程适用于已经熟悉 SciPy 生态学机器学习的Python机器学习从业者。

让我们现在开始吧。

（**提示**：_ 你可以收藏或者可以打印这个页面，以便于以后重新参考。_）

*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Applied Deep Learning in Python Mini-Course](img/2f778185ceac552e5d1ee21c4cdd45b1.png)


图片由 [darkday](https://www.flickr.com/photos/drainrat/15783392494/) 提供，并保留所属权利

## 这个迷你课程是为谁而准备的

在我们开始之前，让我们确保您处于合适的位置，以下列表提供了有关本课程设计对象的一般指导原则。

如果你没有完全匹配以下特点，请不要惊慌，你可能只需要在某一个领域更新您的知识，以便于开始学习这个课程。

*   **开发人员且知道如何编写一些代码**。这意味着使用 Python 完成任务并了解如何在工作站上设置 SciPy 生态系统（先决条件）对您来说并不是什么大问题。它并不意味着你是一个向导编码器，但它确实意味着你不怕安装软件包和编写脚本。
*   **知道一点机器学习的开发人员**。这意味着您了解机器学习的基础知识，如交叉验证，一些算法和偏差-方差权衡。这并不意味着你是一个机器学习大佬，只是你知道一些专业术语或者知道在哪里查找它们。

这个迷你课程并不是深度学习的教科书。

它将使您从熟悉 Python 的机器学习的开发人员到能够获得学习成果并将深度学习的强大功能应用到您自己的项目的开发人员。

## 迷你课程概述（期待什么）

这个迷你课程分为 14 个部分。

开发者每节课平均需要花费的时间大约是30分钟，课程中的一些知识也许可以很快的学完，而另一些知识则需要花费更多的时间以便于学习的更加深入。

您可以根据自己的需要或快或慢的完成每个部分，强烈推荐您可以在两周的时间内每天花费一些的时间完成一节课的内容。

您将在接下来的 14 节课中讨论的主题如下所示：

*   **第 01 课**：Theano 简介
*   **第 02 课**：TensorFlow 简介
*   **第 03 课**：Keras 简介
*   **第 04 课**：多层感知器中的速成课程
*   **第 05 课**：在 Keras 开发您的第一个神经网络
*   **第 06 课**：使用带 Scikit-Learn 的 Keras 模型
*   **第 07 课**：绘制模型训练历史
*   **第 08 课**：使用检查点在训练期间保存最佳模型
*   **第 09 课**：通过降压正则化减少过度拟合
*   **第 10 课**：通过学习率计划提升绩效
*   **第 11 课**：卷积神经网络中的速成课程
*   **第 12 课**：手写数字识别
*   **第 13 课**：小照片中的物体识别
*   **第 14 课**：通过数据增强改进泛化

这将是一件很有趣的事情。

你将做一些相关工作，包括一些阅读，一些研究和一些编程，你想学习深度学习吗？

（**提示**：_ 这些课程的所有答案都可以在这个博客中使用搜索功能找到。 _）

如有任何问题，请在下面的评论中发布，并在评论中分享您的结果。

坚持下去，不要放弃！

## 第 01 课：Theano 简介

Theano 是一个用于快速数值计算的 Python 库，有助于深度学习模型的开发。

它的核心 Theano 是 Python 中数学表达式的编译器，能够将您的结构转换为使用 NumPy 和高效的本机库的代码，以便在 CPU 或 GPU 上尽可能快地运行。

Theano 表达式的实际语法是符号性的，这对于习惯于普通软件开发的初学者来说可能是不适应的，具体而言，表达式实在抽象意义上定义，编译后用于实际计算。

在本课程中，您的目标是安装 Theano 并编写一个小例子来演示 Theano 程序的符号性质。

例如，您可以使用 pip 安装 Theano，如下所示：

```py
sudo pip install Theano
```

下面列出了一个可以用作起点的 Theano 程序的小例子：

```py
import theano
from theano import tensor
# 定义两个符号性浮点数
a = tensor.dscalar()
b = tensor.dscalar()
# 创建一个简单的表达式
c = a + b
# 将表达式转换为可调用的对象
# 计算输入值并将其赋值给输出变量c
f = theano.function([a,b], c)

result = f(1.5, 2.5)
print(result)
```

您可以在 [Theano 主页](http://deeplearning.net/software/theano/)上了解有关 Theano 的更多信息。

## 课程 02：TensorFlow 简介

TensorFlow 是一个用于 Google 创建和发布的快速数值计算的 Python 库，与 Theano 一样，TensorFlow 旨在用于开发深度学习模型。

在谷歌的支持下，可能会在在谷歌 DeepMind 研究小组的某些生产系统中使用它，它是一个我们不能忽视的平台。

与 Theano 不同，TensorFlow 确实更注重生产，能够在 CPU，GPU 甚至非常大的集群上运行。

在本课程中，您的目标是安装 TensorFlow，熟悉 TensorFlow 程序中使用的符号表达式的语法。

如下所示，您可以使用 pip 安装 TensorFlow：

```py
sudo pip install TensorFlow
```

下面列出了一个可以用作起点的 TensorFlow 程序的小例子：

```py
import tensorflow as tf
# 声明两个符号性浮点变量
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个简单的符号性表达式
add = tf.add(a, b)
# bind 1.5 to ' a ' , 2.5 to ' b ' , and evaluate ' c '
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
```

您可以在 [TensorFlow 主页](https://www.tensorflow.org/)上了解有关 TensorFlow 的更多信息。

## 第 03 课：Keras 简介

Theano 和 TensorFlow 的难点在于它可能需要大量代码来创建非常简单的神经网络模型。

这些库主要是作为研究和开发的平台而设计，而不是应用深度学习的实际问题。

Keras 库通过为 Theano 和 TensorFlow 提供封装来解决这些问题，它提供了简洁的 API，允许您在几行代码中定义和评估深度学习模型。

由于易于使用，并且因为它利用了 Theano 和 TensorFlow 的强大功能，Keras 很快成为应用深度学习的首选库。

Keras 的重点是模型的概念，模型的生命周期可归纳如下：

1.  定义您的模型，创建顺序模型并添加已配置的层；
2.  编译您的模型，指定损失函数和优化器，并在模型上调用`compile()`函数
    函数。
3.  拟合您的模型，通过调用模型上的`fir()`函数，在数据样本上训练模型。
4.  作出预测，通过调用模型上的`evaluate()`或`predict()`等函数，使用该模型生成对新数据的预测。

您的本课目标是安装 Keras。

例如，您可以使用 pip 安装 Keras：

```py
sudo pip install keras
```

现在开始熟悉一下 Keras 库，为即将到来的课程做好准备，我们将实现我们的第一个模型。

您可以在 [Keras 主页](http://keras.io/)上了解有关 Keras 库的更多信息。

## 课程 04：多层感知器中的速成课程

人工神经网络是一个迷人的研究领域，尽管它们刚开始时可能会令人生畏。

人工神经网络领域通常被称为神经网络或多层感知器之后可能是最有用的神经网络类型。

神经网络的构建块是人工神经元，这些是简单的计算单元，其具有加权输入信号并能使用激活函数产生输出信号。

神经元被排列成神经元网络。一行神经元称为层，一个网络可以有多个层，网络中神经元的体系结构通常称为网络拓扑。

神经网络配置完成后，需要在数据集上训练神经网络，神经网络经典且仍然最流行的的训练算法称为随机
梯度下降算法。

![Model of a Simple Neuron](img/498ab2d8740c6a44a78ade60a46c95a9.png)
简单神经元的模型


您的本课目标是熟悉神经网络术语，深入研究神经元，权重，激活函数，学习率等等。

## 第 05 课：在 Keras 开发您的第一个神经网络

Keras 允许您在极少数代码行中开发和评估深度学习模型。

在本课程中，您的目标是使用 Keras 库开发您的第一个神经网络。

您可以使用来自 UCI 机器学习库的标准二进制（两类）分类数据集，如 [Pima Indians 糖尿病](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)或[电离层数据集](https://archive.ics.uci.edu/ml/datasets/Ionosphere)。

编写代码以实现以下目标：

1.  使用 NumPy 或 Pandas 加载数据集；
2.  定义您的神经网络模型并进行编译；
3.  拟合您的神经网络模型；
4.  评估模型在不可见的数据上的性能。

为了给您的学习带来较大帮助，下面是一个完整的工作示例，您可以将其作为您的学习起点。

您可以[将 Pima Indians 数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)下载到您当前的工作目录中，文件名为 _pima-indians-diabetes.csv_ （更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)） 。

```py
# 利用keras创建第一个神经网络
from keras.models import Sequential
from keras.layers import Dense
import numpy
# 设置随机种子
seed = 7
numpy.random.seed(seed)
#加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据及分割为输入变量和输出变量
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
编译模型
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# 拟合网络
model.fit(X, Y, nb_epoch=150, batch_size=10)
# 评估网络
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

现在，根据此示例，您可以在利用不同数据集开发不同的模型或者继续调整此模型。

您可以在[Keras API更多信息](http://keras.io/models/sequential/)上了解有关用于简单模型开发的更多的知识。

## 第 06 课：使用 Scikit-Learn 的 Keras 模型

scikit-learn 库是一个基于 SciPy 构建的 Python 通用机器学习框架。

Scikit-learn 擅长仅在几行代码之内评估模型性能和完成优化模型超参数等任务。

Keras 提供了一个封装类，允许您使用 scikit-learn 的深度学习模型，例如，Keras 中的 KerasClassifier 类的实例可以封装您的深度学习模型，并在 scikit-learn 中用作估计器。

使用 KerasClassifier 类时，必须指定该类可用于定义和编译模型的函数的名称。您还可以将其他参数传递给 KerasClassifier 类的构造函数，之后将传递给 `model.fit()` 调用，例如迭代次数和批量大小等。

在本课程中，您的目标是开发深度学习模型并使用 k 折交叉验证对其进行评估。

例如，您可以定义 KerasClassifier 的实例和自定义函数来创建模型，如下所示：

```py
#  KerasClassifier类中创建模型的函数
def create_model():
	# 创建模型
	model = Sequential()
	...
	# 编译模型
	model.compile(...)
	return model

# 使用scikit-learn创建分类器
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
# 在scikit-learn中使用10折交叉验证评估模型性能
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
```

您可以在 Sciki-Learn API 网页的 [Wrappers 上了解有关使用 Keras 深度学习模型和 scikit-learn 的更多信息。](http://keras.io/scikit-learn-api/)

## 第 07 课：绘制模型训练历史

您可以通过观察模型整个训练期间的性能了解更多关于神经网络模型和深度学习更多的信息。

Keras 提供了在训练深度学习模型时注册回调的功能。

训练所有深度学习模型时注册的默认回调之一是历史回调，它记录每个迭代次数的训练指标，包括损失和精确度（对于分类问题）以及验证数据集的损失和精确度（如果已设置）。

历史对象通过调用`fit()`函数返回，度量标准保存在返回对象的历史成员的字典中。

您本课程的目标是研究历史对象，并绘制模型在训练期间的性能图像。

例如，您可以输出历史对象收集的指标列表，如下所示：

```py
# 历史对象中的所有数据
history = model.fit(...)
print(history.history.keys())
```

您可以在 Keras 中了解有关 [History 对象和回调 API 的更多信息。](http://keras.io/callbacks/#history)

## 第 08 课：使用检查点在训练期间保存最佳模型

应用程序检查点是一种适用于长时间运行过程的容错技术。

Keras 库通过回调 API 提供检查点功能， ModelCheckpoint回调类允许您定义模型权重参数检查点的位置，文件命名规范和创建模型检查点的条件。

如果训练运行过早停止，则检查点可用于跟踪模型权重，也可以跟踪训练期间观察到的最佳模型。

在本课程中，您的目标是使用 Keras 中的 ModelCheckpoint 回调来跟踪训练期间观察到的最佳模型。

您可以定义 ModelCheckpoint，每次观察到改进时，都会将网络权重保存到同一文件中。例如：

```py
from keras.callbacks import ModelCheckpoint
...
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#拟合网络
model.fit(..., callbacks=callbacks_list)
```

了解有关在 Keras 中使用 [ModelCheckpoint 回调的更多信息。](http://keras.io/callbacks/#modelcheckpoint)

## 第 09 课：通过随机失活正则化减少过拟合

过度学习训练数据集是神经网络面临的一个大问题！

随机失活（Dropout） 是一种简单但非常有效的减少丢失的技术，并且已证明在大型深度学习模型中很有用。

随机失活是一种在训练过程中随机选择被忽略的神经元的技术，而这些神经元是随机选择的。这意味着它们对下游神经元激活的贡献在正向通道时暂时消除，并且任何权重参数的更新都不会应用于后向通过的的神经元。

您可以使用 Dropout 层类将随机失活层添加到深度学习模型中。

在本课程中，您的目标是尝试在神经网络的不同节点添加随机dropout，并设置不同的dropout的概率值。

例如，您可以创建一个概率为 20％的随机失活层，并将其添加到您的模型中，如下所示：

```py
from keras.layers import Dropout
...
model.add(Dropout(0.2))
```

你可以在 Keras 中了解更多关于的[dropout.](http://keras.io/layers/core/#dropout)的更多信息。

## 第 10 课： 通过学习率计划提升模型性能

通过使用学习率计划，您通常可以提高模型的性能。

通常称为自适应学习率或退火学习率，随机梯度下降的学习率在训练模型时会发生变化的技术。

Keras 具有基于时间的学习率计划，该表内置于 SGD 类中的随机梯度下降算法的实现中。

构建类时，您可以指定衰减数量，即您的学习率（也是被指定的）每次迭代时减少的数量，当使用学习率衰减数量时，你应该指定您的初始学习率值并考虑增加一个大的动量值，如0.8或者0.9.

您在本课程中的目标是尝试 Keras 内置的基于时间的学习率计划。

例如，您可以指定从 0.1 开始的学习率计划，每次迭代下降 0.0001，如下所示：

```py
from keras.optimizers import SGD
...
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
model.compile(..., optimizer=sgd)
```

您可以在此处了解更多关于 Keras 的 [SGD 课程](http://keras.io/optimizers/#sgd)。

## 第 11 课：卷积神经网络中的速成课程

卷积神经网络是一种强大的人工神经网络技术。

他们通过使用小方块格式的输入数据学习其内部特征表示来期望并保持图像中像素之间的空间关系。

在整个图像中学习和使用特征，允许图像中的物体在场景中移动或平移，并且仍然可以被网络检测到，这就是为什么这种类型的网络对于照片不同方向中的物体识别，数字识别，人脸识别别等非常有用的原因。

卷积神经网络中的三种类型：

1.  **卷积层**: 由过滤器和特征图组成。
2.  **池化层**: 从特征图中下采样激活。
3.  **完全连接层**: 连接于模型的末端，可用于进行预测。

在本课中，您需要熟悉描述卷积神经网络时使用的术语。

这可能需要您自己进行一些研究，但是不要过分担心它们如何工作，只需学习这种网络中使用的各种层的术语和配置。

## 第 12 课：手写数字识别

手写数字识别是一类复杂的计算机视觉分类问题。

MNIST 数据集是用于评估手写数字识别问题的算法的标准问题。它包含可用于训练模型的 60,000 个数字图像，以及可用于评估其性能的 10,000 个图像，如下图所示：

![Example MNIST images](img/256dfb575d54b2eec4be14c906ce2c11.png)

使用卷积神经网络可以在 MNIST 问题上实现现有技术的结果，Keras 使得加载 MNIST 数据集变得更加容易。

在本课程中，您的目标是为 MNIST 问题开发一个非常简单的卷积神经网络，该模型由一个卷积层，一个池化层和一个用来预测的完全连接层组成。

您可以在 Keras 中加载 MNIST 数据集，如下所示：

```py
from keras.datasets import mnist
...
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

将文件下载到您的计算机可能需要一些时间。

>一个小提示：您 Keras [Conv2D](http://keras.io/layers/convolutional/) 作为您的第一个隐藏层，数据的格式为 通道 x 宽 x 高图像数据，其中 MNIST 数据集具有 1 个通道，因为图像是宽度和高度都是28像素的灰度级图像，您可以很容易的在代码中重新改变数据的格式，如下代码所示:

```py
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
```

您还需要对输出类值进行单热编码，Keras 还提供了一个方便的辅助函数来实现：

```py
from keras.utils import np_utils
...
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```

作为最后的提示，这里是一个模型定义示例，您可以将其作为您的学习起点起点：

```py
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(1, 28, 28),
activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## 第 13 课：小照片中的物体识别

物体识别是您的模型必须能够识别出每个小照片中具体物体的问题

深度学习模型使用深度卷积神经网络在该问题中取得最先进的研究结果。

用于评估此类问题模型的最为流行的标准数据集称为 CIFAR-10,它包含 60,000 张小照片，每张照片都是 10 个物体中的一个，如猫，船或飞机等，如下所示：

![Small Sample of CIFAR-10 Images](img/def90f3b9b58bf30829a905ae7da3e0d.png)


与 MNIST 数据集一样，Keras 提供了一个方便的函数，您可以使用它来加载数据集，并在您第一次尝试加载数据集时将其下载到您的计算机，数据集为 163 MB，因此下载可能需要几分钟。

您在本课程中的目标是为 CIFAR-10 数据集开发一个深度卷积神经网络。考虑尝试随机失活和较长的训练时间，我建议您使用重复模式的卷积层和池化层。。

例如，您可以在 Keras 中加载 CIFAR-10 数据集并与卷积神经网络一起使用，如下所示：

```py
from keras.datasets import cifar10
from keras.utils import np_utils
# 加载数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 正则化输出
X_train = X_train.astype('float32') X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# 单热编码输出
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```

## 第 14 课：通过数据扩充改进泛化

使用神经网络和深度学习模型时，需要对数据进行预处理。

更复杂的对象识别任务也会导致越来越多的数据扩充，这是您使用随机翻转和移位修改数据集中的图像的位置的结果，这实质上会使您的训练数据集更大，并且有助于您的模型生成位置和方向信息。

Keras 提供了一个图像增强 API，可以及时在数据集中创建图像的修改版本， [ImageDataGenerator](http://keras.io/preprocessing/image/) 类可用于定义要执行的图像增强操作，这些操作可拟合数据集，然后在训练模型时用于代替数据集。

本课程的目标是使用您在上一课（如 MNIST 或 CIFAR-10）中熟悉的数据集来试验 Keras 图像增强 API。

例如，下面的示例在 MNIST 数据集中创建最多 90 度图像的随机旋转。

```py
# 随机旋转
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# 加载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 重新改变图像的数据格式
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# 将整型数据转换为浮点型数据
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 定义数据预处理
datagen = ImageDataGenerator(rotation_range=90)
# 从数据中拟合参数
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# 创建一个3×3的网格图像
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# 显示突袭iang
	pyplot.show()
	break
```

您可以了解有关 [Keras 图像增强 API](http://keras.io/preprocessing/image/) 的更多信息。

## 深度学习迷你课程评论

恭喜你，你做到了。做得好！

花点时间回顾一下你走了多远：

*   您在 python 中了解了深度学习库，包括可用于深度学习的强大的数值库 Theano 和 TensorFlow 以及易于使用的 Keras 库。
*   您使用 Keras 构建了第一个神经网络，并学习了如何使用 scikit-learn 的深度学习模型以及如何检索和绘制模型的训练历史记录。
*   您了解了更多高级技术，例如随机失活正则化和基于时间的学习率计划，以及如何在 Keras 中使用这些技术。
*   最后，您了解并开发了用于复杂计算机视觉任务的卷积神经网络，并了解了图像数据的增强的相关知识。

不要轻视这一点，你在很短的时间内走了很长的路，这只是您在 Python 中深入学习的旅程的开始。请继续练习和发展你的技能。

你喜欢这个迷你课程吗？你有任何问题或疑点可以发表评论！