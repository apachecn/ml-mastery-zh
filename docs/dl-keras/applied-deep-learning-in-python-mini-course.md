# 在 Python 迷你课程中应用深度学习

> 原文： [https://machinelearningmastery.com/applied-deep-learning-in-python-mini-course/](https://machinelearningmastery.com/applied-deep-learning-in-python-mini-course/)

深度学习是一个迷人的研究领域，这些技术在一系列具有挑战性的机器学习问题中取得了世界一流的成果。

深入学习可能很难开始。

您应该使用哪个库以及您应该关注哪些技术？

在这篇文章中，您将发现一个由 14 部分组成的 Python 深度学习速成课程，其中包含易于使用且功能强大的 Keras 库。

这个迷你课程适用于已经熟悉SciPy生态学机器学习的python机器学习从业者。

让我们现在开始吧。

( _提示：你可以收藏或者打印这个页面，这样你以后可以重新参考这篇文章。_)

*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Applied Deep Learning in Python Mini-Course](img/2f778185ceac552e5d1ee21c4cdd45b1.png)

在 Python 迷你课程中应用深度学习，照片由 [darkday](https://www.flickr.com/photos/drainrat/15783392494/) 提供，并保留所属权利。

## 这个迷你课程是为哪些人准备的？

在我们开始之前，需要确认您的深度学习的相关知识储备，以下列表提供了有关本课程设计对象的一般指导原则：

如果你没有完全达到这些条件，请不要惊慌，你可能只需要在一个或另一个相关领域更新您的知识以便于开始学习。

*   **知道如何编写一些代码的开发人员**，这意味着使用Python完成任务并了解如何在工作站上设置 SciPy 生态系统（先决条件）对您来说并不是什么大问题。它并不意味着你是一个杰出的程序员，但它确实意味着你不怕安装软件包和编写脚本。
*   **知道一点机器学习的开发人员**，这意味着您了解机器学习的基础知识，如交叉验证，一些算法和偏差 - 方差权衡。这并不意味着你是一个机器学习专家，只是你知道一些机器学习术语或知道在哪里查找它们。

这个迷你课程不是深度学习的教科书。

它将使您从了解一点Python机器学习的开发人员变成能够利用深度学习的结果并将深度学习的强大功能带入您自己的项目的开发人员。

## 迷你课程概述（期待什么）

这个迷你课程分为 14 个部分。

普通开发人员每节课大概要花费30分钟，你可能会更快完成一些，而另一些你可能会选择更深入并花更多时间。

您可以根据自己的需要快速或慢速的完成每个部分，强烈推荐您在较为合适的时间安排两周的时间内每天完成一节课。

您将在接下来的 14 节课中讨论的主题如下：

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
*   **第 11 课**：卷积神经网络中的崩溃课程
*   **第 12 课**：手写数字识别
*   **第 13 课**：小照片中的物体识别
*   **第 14 课**：通过数据增强改进泛化

这将是一件很有趣的事情，不过你得做这些工作，包括一些阅读，一些研究和一些编程，你想学习深度学习，对吧？

( _**提示**：这些课程的所有答案都可以使用搜索功能在这个博客上找到。_ )

如有任何问题，请在下面的评论中发布，在评论中分享您的问题，坚持住，不要放弃！

## 第 01 课：Theano 简介

Theano 是一个用于快速数值计算的 Python 库，有助于深度学习模型的开发。

它的核心Theano是Python编写数学表达式的编译器，它能够将你的数据结构转换为使用numpy库和高效的本机库的代码，以便尽可能快的在CPU或者GPU上运行。

Theano表达式的实际语法是符号性的，这对于习惯于普通软件开发的初学者来说可能不太习惯，具体而言，表达式在抽象意义上定义，编译后实际用于进行计算。

在本课程中，您的目标是安装Theano并编写一个小例子来演示Theano程序的符号性。

例如，您可以使用pip工具安装Theano，如下所示：

```py
sudo pip install Theano
```

下面列出了一个可以当作起点学习的`Theano`程序的小例子：

```py
import theano
from theano import tensor
# 声明两个符号性浮点标量
a = tensor.dscalar()
b = tensor.dscalar()
# 创建一个简单的表达式
c = a + b
# 将表达式转换为接受(a,b)的可调用对象,将其值作为输入，并计算c的值
f = theano.function([a,b], c)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
result = f(1.5, 2.5)
print(result)
# 输出： result = array(4.)

```

您可以在 [Theano 主页](http://deeplearning.net/software/theano/)上了解有关 Theano 的更多信息。

## 课程 02：TensorFlow 简介

TensorFlow是 Google 一个用于创建和发布的快速数值计算的Python库,与Theano一样，TensorFlow 旨在用于开发深度学习模型。

在谷歌的支持下，可能会在在谷歌DeepMind研究小组的某些生产系统中使用它，它是一个我们不能忽视的平台.

与Theano不同，TensorFlow确实更注重生产，它能够在CPU，GPU甚至非常大的集群上运行。

在本课程中，您的目标是安装TensorFlow，熟悉TensorFlow程序中使用的符号性的表达式语法。

如下所示，您可以使用pip安装TensorFlow：

```py
sudo pip install TensorFlow
```

下面列出了一个可以当作起点学习的 TensorFlow 程序的小例子：

```py
import tensorflow as tf
# 声明两个符号性浮点标量
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 使用add()函数创建一个简单表达式
add = tf.add(a, b)
# 赋值a=1.5,b=2.5并计算c
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
# 输出：c=4
```

您可以在在 [TensorFlow 主页](https://www.tensorflow.org/)上了解有关 TensorFlow 的更多信息。

## 第 03 课：Keras 简介

Theano 和 TensorFlow 的难点在于它可能需要大量代码来创建甚至非常简单的神经网络模型。

这些库主要是作为研究和开发的平台而设计，而不是应用深度学习的实际问题。

Keras 库通过为 Theano 和 TensorFlow 提供包装器来解决这些问题，它提供了更为简洁的 API，允许您在几行代码中定义和评估深度学习模型。

由于易于学习和使用，并且因为它利用了 Theano 和 TensorFlow 的强大功能，Keras 很快成为应用深度学习的首选库。

Keras 的重点是模型的概念，模型的生命周期可归纳如下：

1.  定义您的模型，创建顺序模型并添加已配置的层。
2.  编译您的模型，指定损失函数和优化器，并在模型上调用`compile()`函数。
    函数。
3.  拟合你的模型，通过调用模型上的`fit()`函数，在简单的数据样本上训练模型。
4.  做出预测，通过调用模型上的`evaluate()`或`predict()`等函数，使用该模型生成对新数据的预测。

您的本课目标是安装 Keras.

例如，您可以使用 pip 安装 Keras：

```py
sudo pip install keras
```
自己可以开始先熟悉Keras库，为即将到来的课程做好准备，我们很快将会实现我们的第一个模型。

您可以在 [Keras 主页](http://keras.io/)上了解有关 Keras 库的更多信息。

## 课程 04：多层感知器中的速成课程

人工神经网络是一个迷人的研究领域，尽管它们刚开始时可能会令人生畏。

人工神经网络的领域通常被称为神经元网络或多层感知器之后可能是最有用的神经网络类型。

神经网络的构建块是人工神经元，这是一些简单的计算单元，其具有加权输入信号并使用激活功能产生输出信号的单元。

多个神经元可以组成神经元网络，一行神经元称为一层，一个网络可以有多个层，网络中的神经元的体系结构通常称为网络拓扑。

神经网络一旦配置完成后，神经网络需要在你的数据集上进行训练，神经网络经典且仍然主流的训练算法是随机梯度下降算法。

![Model of a Simple Neuron](img/498ab2d8740c6a44a78ade60a46c95a9.png)

简单神经元的模型如上图所示

您的本课目标是熟悉神经网络术语,深入研究神经元，权重，激活函数，学习率等。

## 第 05 课：在 Keras 开发您的第一个神经网络

Keras 允许您在极少数代码开发和评估深度学习模型。

在本课程中，您的目标是使用 Keras 库开发您的第一个神经网络。

>您可以使用来自于 UCI 机器学习库的标准二元（两类）分类数据集，如 [Pima Indians 糖尿病](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)或[电离层数据集](https://archive.ics.uci.edu/ml/datasets/Ionosphere)。

编写代码并实现以下目标：

1.  使用 NumPy 或 Pandas 加载数据集。
2.  定义您的神经网络模型并进行编译。
3.  使模型适合数据集。
4.  估计模型在看不见的数据上的表现。

为了给您一个巨大的助力，下面是一个完整的工作示例，您可以将其当作学习的起点。

>您可以将[ Pima Indians 数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)下载到您当前的工作目录中，文件名为 _pima-indians-diabetes.csv_ （更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)） 。

```py
# 使用Keras创建第一个神经网络
from keras.models import Sequential
from keras.layers import Dense
import numpy
# 固定随机种子的重现性
seed = 7
numpy.random.seed(seed)
# 加载数据
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据的输入变量X和输出变量Y进行分离
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
#具有12个隐藏神经单元，输入是一个8维的向量
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# 拟合模型
# 迭代150次，每次的输入的批大小为10
model.fit(X, Y, nb_epoch=150, batch_size=10)
# 评估模型
scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#输出： acc：77，21%
```

现在，您可以在不同的数据集上开发自己的模型，或者调整此示例。

了解有关用于简单模型开发的 [Keras API 的更多信息](http://keras.io/models/sequential/)。

## 第 06 课：使用 Scikit-Learn 的 Keras 模型

scikit-learn 库是一个基于 SciPy 构建的 Python 通用机器学习框架。

Scikit-learn 擅长仅在几行代码中实现评估模型表现和优化模型超参数等任务。

Keras 提供了一个封装类，允许您通过scikit-learn 使用深度学习模型，例如，Keras 中的 KerasClassifier 类的实例可以封装您的深度学习模型，并在 scikit-learn 中用作评估器。

使用 KerasClassifier 类时，必须指定该类可用于定义和编译模型的函数的名称，您还可以将其他参数传递给 KerasClassifier 类的构造函数，稍后将传递给 `model.fit()` 调用，例如迭代次数和批大小等。

在本课程中，您的目标是开发深度学习模型并使用 k 折交叉验证对其进行评估。

例如，您可以定义 KerasClassifier 的实例和自定义函数来创建模型，如下所示：

```py
# KerasClassifier分类器所必需的创造模型的函数
def create_model():
	# 创建模型
	model = Sequential()
	...
	# 编译模型
	model.compile(...)
	return model

#  创建用于scikit-learn的分类器
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
# 在scikit-learn中用10折交叉验证评估模型
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
```

在 Sciki-Learn API 网页的 [Wrappers 上了解有关使用 Keras 深度学习模型和 scikit-learn 的更多信息。](http://keras.io/scikit-learn-api/)

## 第 07 课：绘制模型训练历史

您可以通过观察模型在训练期间随时间的表现来学习更多关于神经网络和深度学习模型的知识。

Keras 提供了训练深度学习模型时的注册回调功能。

训练深度学习模型时注册的默认回调之一是历史回调，它记录每个时期的训练指标。这包括损失和准确性（对于分类问题）以及验证数据集的损失和准确性（如果已设置）。

历史对象从调用于训练模型的`fit()`函数返回，度量标准存储在返回对象的历史成员中的字典中。

您本课程的目标是调查历史对象并在训练期间创建模型性能图。

例如，您可以打印历史记录对象收集的指标列表，如下所示：

```py
# list all data in history
history = model.fit(...)
print(history.history.keys())
```

您可以在 Keras 中了解有关 [History 对象和回调 API 的更多信息。](http://keras.io/callbacks/#history)

## 第 08 课：使用模型检查点在训练期间保存最佳模型

应用程序检查点是一种适用于长时间运行过程的容错技术。

Keras 库通过回调 API 提供检查点功能， ModelCheckpoint回调类允许您定义检查模型权重的位置，文件应该如何命名以及在什么情况下创建模型的检查点。

如果训练运行过早停止，则检查点可用于跟踪模型权重，并且其跟踪训练期间观察到的最佳模型也很有用。

在本课程中，您的目标是使用 Keras 中的 ModelCheckpoint 回调来跟踪训练期间观察到的最佳模型。

您可以定义 ModelCheckpoint，每次观察到改进时，都会将网络权重保存到同一文件中,例如：

```py
from keras.callbacks import ModelCheckpoint
...
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# 拟合网络
model.fit(..., callbacks=callbacks_list)
```

了解有关在 Keras 中使用 [ModelCheckpoint 回调的更多信息。](http://keras.io/callbacks/#modelcheckpoint)

## 第 09 课：通过Dropout正则化减少过拟合

神经网络过度学习训练数据集是一个大问题。

Dropout 是一种简单但非常有效的减少丢失的技术，并且已证明在大型深度学习模型中很有用。

dropout是一种在训练过程中忽略随机选择的神经元的技术，这些神经元是随机丢弃的 ，这意味着这些神经元对下游神经元激活行为在正向通过时被暂时消除，并且任何权重参数的更新都不会应用于反向通过的神经元。

您可以使用 Dropout 层类将Dropout层添加到你的训练神经网络训练模型中。

在本课程中，您的目标是尝试在神经网络的不同点添加dropout，并设置不同的dropout值概率。

例如，您可以创建一个概率为 20％的dropout层，并将其添加到您的模型中，如下所示：

```py
from keras.layers import Dropout
...
model.add(Dropout(0.2))
```

你可以在 Keras 中了解更多关于[Dropout的信息](http://keras.io/layers/core/#dropout)。

## 第 10 课：通过学习率表提高模型性能

您通常可以通过使用学习率表提高模型的性能。

这种训练模型时随机梯度下降使用的学习率发生改变的技术，通常称为自适应学习率或退火学习率。

Keras 具有基于时间的学习率表，该表内置于实现 SGD 类的随机梯度下降算法中。

在类的构造函数中，您可以指定衰减量，即您的学习率（也可以指定）每次迭代时减少的数量，当使用学习率衰减时，你应该增大学习率的初始值并考虑增加一个较大的动量项，如 0.8 或 0.9。

您在本课程中的目标是尝试 Keras 内置的基于时间的学习率计划。

例如，您可以指定从 0.1 开始，每次迭代下降0.0001的学习率表，如下所示：

```py
from keras.optimizers import SGD
...
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
model.compile(..., optimizer=sgd)
```

您可以在此处了解更多关于 Keras 的 [SGD 课程](http://keras.io/optimizers/#sgd)。

## 第 11 课：卷积神经网络中的速成课程

卷积神经网络是一种强大的人工神经网络技术。

卷积神经网络通过使用小方块的输入数据学习数据内部特征来表示并期望保持图像中像素之间的空间关系。

在整个图像中学习和使用的特征允许图像中的物体在场景中移动或平移，并且仍然可以被网络检测到，这就是这种类型的网络对于照片中以不同方向挑选数字，面孔，物体等识别非常有用的原因。

卷积神经网络中有三种类型的层：

1.  **卷积层**：由过滤器和特征图组成。
2.  **池化层**：从特征图中下采样激活。
3.  **全连接层**：搭建在插在网络模型的末端，可用于进行预测。

在本课中，您将熟悉描述卷积神经网络时使用的术语。

这可能需要你自己进行一点研究。

不要过分考虑它们如何工作，只需学习这种网络中使用的的术语和各种层的配置。

## 第 12 课：手写数字识别

手写数字识别的计算机视觉分类问题的难题。

[MNIST](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/) 数据集是用于评估手写数字识别问题的算法的标准问题,它包含可用于训练模型的 60,000 个数字图像，以及可用于评估其性能的 10,000 个图像。

![Example MNIST images](img/256dfb575d54b2eec4be14c906ce2c11.png)

MNIST 图像示例

使用卷积神经网络可以在 MNIST 问题上取得最先进的结果，Keras 能够使得加载 MNIST 数据集变得更加容易。

在本课程中，您的目标是为 MNIST 问题开发一个非常简单的卷积神经网络，该网络模型由一个卷积层，一个池化层和一个可以进行预测的全连接层组成（Dense是keras中最常用的全连接层）。

您可以在 Keras 中加载 MNIST 数据集，如下所示：

```py
from keras.datasets import mnist
...
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

将文件下载到您的计算机可能需要一些时间。

一个小提示：您会使用第一个隐藏层 Keras [Conv2D](http://keras.io/layers/convolutional/) 层将图像数据格式化为 x 宽 x 高格式的图像数据，其中 MNIST 数据具有 1 个通道，因为是灰度级并且宽度和高度为 28 像素的图像，您可以很轻松的改变 MNIST 数据集的形状，如下所示：

```py
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
```

您还需要对输出类值进行[独热编码](https://cloud.tencent.com/developer/article/1051795)，Keras 为此提供了一个方便的辅助函数来实现：



```py
from keras.utils import np_utils
...
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```

最后一个提示：以下是一个卷积神经网络的模型定义，您可以将其作为学习起点：

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

物体识别是模型必须能够准确识别出照片中的具体内容的问题。

深度学习的模型使用深度卷积神经网络在该问题方面达到了最先进的结果。

用于评估此类问题的模型使用最广泛的标准数据集称为 [CIFAR-10](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)。它包含 60,000 张小照片，每张照片包含10个物体中的一个物体，如猫，船或飞机等。

![Small Sample of CIFAR-10 Images](img/def90f3b9b58bf30829a905ae7da3e0d.png)

CIFAR-10 图像的小样本

与 MNIST 数据集一样，Keras 提供了一个方便的函数，您可以使用它来加载数据集，并在您第一次尝试加载数据集时将其下载到您的计算机,数据集大小为 163 MB，因此下载可能需要几分钟。

您在本课程中的目标是为 CIFAR-10 数据集开发一个深度卷积神经网络，考虑dropout试验和较长的训练时间，我建议您使用卷积层和池化层的重复模式。

您可以在 Keras 中加载 CIFAR-10 数据集并与卷积神经网络一起使用，如下所示：

```py
from keras.datasets import cifar10
from keras.utils import np_utils
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32') X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```

## 第 14 课：通过数据增强提高模型泛化能力

使用神经网络和深度学习模型时，需要进行数据预处理。

更复杂的物体识别任务也需要越来越多的[数据增强](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)，这意味着您使用随机翻转和移位修改数据集图像中物体的位置，这实质上会使您的训练数据集更大，并导致您的模型增加图像中物体的位置和方向。

Keras 提供了一个图像增强 API，可以及时在数据集中创建图像的修改版本。 [ImageDataGenerator](http://keras.io/preprocessing/image/) 类可用于定义要执行的图像增强操作，这些操作可以拟合到您的数据集中，然后在训练模型时用于代替数据集。

本课程的目标是使用您在上一课（如 MNIST 或 CIFAR-10）中熟悉的数据集实验 Keras 图像增强 API。

例如，下面的示例在 MNIST 数据集中创建最多 90 度图像的随机旋转。

```py
# 随机旋转
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# 加载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 重新改变数据形式为 [原始数据][像素][宽度][高度]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# 将整型转换为浮点型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 定义数据预处理
datagen = ImageDataGenerator(rotation_range=90)
# 利用数据拟合参数
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# 创建一个3×3图像的网格
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# 显示图像
	pyplot.show()
	break
```

您可以了解有关 [Keras 图像增强 API](http://keras.io/preprocessing/image/) 的更多信息。

## 深度学习迷你课程回顾

恭喜你，你做到了，做得好！

花点时间回顾一下你走了多远：

*   您在 python 中发现了深度学习库，包括强大的数值库 Theano 和 TensorFlow 以及用于应用深度学习且易于使用的 Keras 库。
*   您使用 Keras 构建了第一个神经网络，并学习了如何使用 scikit-learn 的深度学习模型以及如何检索和绘制模型的训练历史记录。
*   您了解了更多高级技术，例如dropout正则化和学习率表，以及如何在 Keras 中使用这些技术。
*   最后，您进行了下一步的学习，了解并开发了用于复杂计算机视觉任务的卷积神经网络，并了解了有关图像数据增强的相关知识。

不要轻视这一点，你在很短的时间内走了很长的路，这只是您在 Python 中深入学习的旅程的开始，请继续保持练习和拓展你的技能。

你喜欢这个迷你课吗？你有任何问题或疑点吗？请发表评论！