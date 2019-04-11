# Keras 深度学习库的多类分类教程

> 原文： [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)

Keras 是一个深度学习的 Python 库，它包含了高效的数值库 Theano 和 TensorFlow。

在本教程中，您将了解如何使用 Keras 开发和评估多类分类问题的神经网络模型。

完成本分步教程后，您将了解：

*   如何从 CSV 加载数据并使其可供 Keras 使用。
*   如何用神经网络建立多类分类数据进行建模。
*   如何用 scikit-learn 评估 Keras 神经网络模型。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **2017 年 6 月更新**：更新了在输出层使用 softmax 激活，更大隐藏层，默认权重初始化的示例。

![Multi-Class Classification Tutorial with the Keras Deep Learning Library](img/2dae7dc453b9ee9eecf2783612250927.png)

Keras 深度学习库的多类分类教程
[houroumono](https://www.flickr.com/photos/hourou/8922014724/) 的照片，保留一些权利。

## 1.问题描述

在本教程中，我们将使用称为[虹膜花数据集](http://archive.ics.uci.edu/ml/datasets/Iris)的标准机器学习问题。

这个数据集经过深入研究，是在神经网络上实践的一个很好的问题，因为所有 4 个输入变量都是数字的，并且具有相同的厘米尺度。每个实例描述观察到的花测量的属性，输出变量是特定的虹膜种类。

这是一个多类别的分类问题，意味着有两个以上的类需要预测，实际上有三种花种。这是用神经网络练习的一个重要问题类型，因为三个类值需要专门的处理。

虹膜花数据集是一个充分研究的问题，我们可以[期望在 95％至 97％的范围内实现模型精度](http://www.is.umk.pl/projects/rules.html#Iris)。这为开发我们的模型提供了一个很好的目标。

您可以[从 UCI 机器学习库下载虹膜花数据集](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)，并将其放在当前工作目录中，文件名为“ _iris.csv_ ”。

## 2.导入类和函数

我们可以从导入本教程中需要的所有类和函数开始。

这包括我们需要 Keras 的功能，还包括 [pandas](http://pandas.pydata.org/) 的数据加载以及 [scikit-learn](http://scikit-learn.org/) 的数据准备和模型评估。

```py
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
```

## 3.初始化随机数生成器

接下来，我们需要将随机数生成器初始化为常量值（7）。

这对于确保我们可以再次精确地实现从该模型获得的结果非常重要。它确保可以再现训练神经网络模型的随机过程。

```py
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

## 4.加载数据集

可以直接加载数据集。因为输出变量包含字符串，所以最简单的方法是使用 pandas 加载数据。然后我们可以将属性（列）拆分为输入变量（X）和输出变量（Y）。

```py
# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
```

## 5.编码输出变量

输出变量包含三个不同的字符串值。

在使用神经网络对多类分类问题进行建模时，最好将包含每个类值的值的向量的输出属性重新整形为一个矩阵，每个类值都有一个布尔值，以及给定的实例是否具有该值是否有类值。

这称为[一个热编码](https://en.wikipedia.org/wiki/One-hot)或从分类变量创建虚拟变量。

例如，在这个问题中，三个类值是 Iris-setosa，Iris-versicolor 和 Iris-virginica。如果我们有观察结果：

```py
Iris-setosa
Iris-versicolor
Iris-virginica
```

我们可以将其转换为每个数据实例的单热编码二进制矩阵，如下所示：

```py
Iris-setosa,	Iris-versicolor,	Iris-virginica
1,		0,			0
0,		1, 			0
0, 		0, 			1
```

我们可以通过首先使用 scikit-learn 类 LabelEncoder 将字符串一致地编码为整数来完成此操作。然后使用 Keras 函数 to_categorical（）将整数向量转换为一个热编码。

```py
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
```

## 6.定义神经网络模型

Keras 库提供了包装类，允许您在 scikit-learn 中使用 Keras 开发的神经网络模型。

Keras 中有一个 KerasClassifier 类，可以用作 scikit-learn 中的 Estimator，它是库中基本类型的模型。 KerasClassifier 将函数的名称作为参数。该函数必须返回构建的神经网络模型，为训练做好准备。

下面是一个函数，它将为虹膜分类问题创建一个基线神经网络。它创建了一个简单的完全连接的网络，其中一个隐藏层包含 8 个神经元。

隐藏层使用整流器激活功能，这是一种很好的做法。因为我们对虹膜数据集使用了单热编码，所以输出层必须创建 3 个输出值，每个类一个。具有最大值的输出值将被视为模型预测的类。

这个简单的单层神经网络的网络拓扑可以概括为：

```py
4 inputs -> [8 hidden nodes] -> 3 outputs
```

请注意，我们在输出层使用“ _softmax_ ”激活功能。这是为了确保输出值在 0 和 1 的范围内，并且可以用作预测概率。

最后，网络使用具有对数损失函数的高效 Adam 梯度下降优化算法，在 Keras 中称为“ _categorical_crossentropy_ ”。

```py
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

我们现在可以创建我们的 KerasClassifier 用于 scikit-learn。

我们还可以在构造 KerasClassifier 类中传递参数，该类将传递给内部用于训练神经网络的 fit（）函数。在这里，我们将时期数传递为 200，批量大小为 5，以便在训练模型时使用。通过将 verbose 设置为 0，在训练时也会关闭调试。

```py
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
```

## 7.使用 k-fold 交叉验证评估模型

我们现在可以在训练数据上评估神经网络模型。

scikit-learn 具有使用一套技术评估模型的出色能力。评估机器学习模型的黄金标准是 k 折交叉验证。

首先，我们可以定义模型评估程序。在这里，我们将折叠数设置为 10（一个很好的默认值）并在分区之前对数据进行洗牌。

```py
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
```

现在我们可以使用 10 倍交叉验证程序（kfold）在我们的数据集（X 和 dummy_y）上评估我们的模型（估计器）。

评估模型仅需要大约 10 秒钟，并返回一个对象，该对象描述了对数据集的每个分割的 10 个构建模型的评估。

```py
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

结果总结为数据集上模型精度的均值和标准差。这是对看不见的数据的模型表现的合理估计。对于这个问题，它也属于已知的最佳结果范围。

```py
Accuracy: 97.33% (4.42%)
```

## 摘要

在这篇文章中，您发现了如何使用 Keras Python 库开发和评估神经网络以进行深度学习。

通过完成本教程，您了解到：

*   如何加载数据并使其可用于 Keras。
*   如何使用一个热编码准备多类分类数据进行建模。
*   如何使用 Keras 神经网络模型与 scikit-learn。
*   如何使用 Keras 定义神经网络进行多类分类。
*   如何使用带有 k-fold 交叉验证的 scikit-learn 来评估 Keras 神经网络模型

您对 Keras 或此帖的深度学习有任何疑问吗？

在下面的评论中提出您的问题，我会尽力回答。