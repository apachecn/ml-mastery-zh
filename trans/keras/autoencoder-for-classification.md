# 自动编码器特征提取分类

> 原文:[https://machinelearning master . com/auto encoder-for-class/](https://machinelearningmastery.com/autoencoder-for-classification/)

**自动编码器**是一种神经网络，可用于学习原始数据的压缩表示。

自动编码器由编码器和解码器子模型组成。编码器压缩输入，解码器尝试根据编码器提供的压缩版本重新创建输入。训练后，编码器模型被保存，解码器被丢弃。

然后，编码器可用作数据准备技术，对原始数据执行特征提取，该原始数据可用于训练不同的机器学习模型。

在本教程中，您将了解如何开发和评估用于分类预测建模的自动编码器。

完成本教程后，您将知道:

*   自动编码器是一种神经网络模型，可用于学习原始数据的压缩表示。
*   如何在训练数据集上训练自动编码器模型，并只保存模型的编码器部分。
*   训练机器学习模型时如何将编码器作为数据准备步骤？

我们开始吧。

![How to Develop an Autoencoder for Classification](img/738aea1b3ed188d654bad74e8f245f5d.png)

如何开发分类自动编码器
图片由[贝恩德·泰勒](https://www.flickr.com/photos/bernd_thaller/23727171246/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  用于特征提取的自动编码器
2.  用于分类的自动编码器
3.  编码器作为预测模型的数据准备

## 用于特征提取的自动编码器

自动编码器[是一种神经网络模型，旨在学习输入的压缩表示。](https://en.wikipedia.org/wiki/Autoencoder)

> 自动编码器是一个神经网络，它被训练成试图将其输入复制到其输出。

—第 502 页，[深度学习](https://amzn.to/3kV7gdV)，2016。

它们是一种无监督的学习方法，尽管从技术上来说，它们是使用有监督的学习方法训练的，称为自监督。

自动编码器通常被训练为试图重建输入的更广泛模型的一部分。

例如:

*   X =模型.预测(X)

自动编码器模型的设计有目的地将架构限制在模型中点的瓶颈上，从而使这一点具有挑战性，输入数据的重建就是从这个瓶颈开始的。

自动编码器有许多类型，它们的用途各不相同，但可能更常见的用途是作为一个学习或自动特征提取模型。

在这种情况下，一旦模型被拟合，模型的重建方面可以被丢弃，并且直到瓶颈点的模型可以被使用。瓶颈处模型的输出是一个固定长度的向量，它提供了输入数据的压缩表示。

> 通常，它们受到限制，只允许近似复制，并且只复制类似于训练数据的输入。因为模型被迫优先考虑应该复制输入的哪些方面，所以它经常学习数据的有用属性。

—第 502 页，[深度学习](https://amzn.to/3kV7gdV)，2016。

然后可以将来自域的输入数据提供给模型，并且瓶颈处的模型输出可以用作监督学习模型中的特征向量，用于可视化，或者更一般地用于降维。

接下来，让我们探索如何针对分类预测建模问题开发用于特征提取的自动编码器。

## 用于分类的自动编码器

在本节中，我们将开发一个自动编码器来学习分类预测建模问题的输入特征的压缩表示。

首先，让我们定义一个分类预测建模问题。

我们将使用[make _ classification()sci kit-learn 函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个包含 100 个输入特征(列)和 1000 个示例(行)的合成二进制(2 类)分类任务。重要的是，我们将以这样的方式定义问题，即大多数输入变量是冗余的(100%或 90%中的 90%)，允许自动编码器稍后学习有用的压缩表示。

下面的示例定义了数据集并总结了它的形状。

```
# synthetic classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例定义数据集并打印数组的形状，确认行数和列数。

```
(1000, 100) (1000,)
```

接下来，我们将开发一个多层感知器(MLP)自动编码器模型。

该模型将采用所有输入列，然后输出相同的值。它将学会精确地重新创建输入模式。

自动编码器由两部分组成:编码器和解码器。编码器学习如何解释输入，并将其压缩为瓶颈层定义的内部表示。解码器获取编码器的输出(瓶颈层)，并尝试重新创建输入。

一旦自动编码器被训练，解码器被丢弃，我们只保留编码器，并使用它将输入的例子压缩成瓶颈层输出的向量。

在第一个自动编码器中，我们根本不会压缩输入，而是使用与输入大小相同的瓶颈层。这应该是一个简单的问题，模型将学习得近乎完美，并旨在确认我们的模型被正确实现。

我们将使用功能 API 定义模型；如果您不熟悉，我推荐本教程:

*   [如何使用 Keras 函数 API 进行深度学习](https://machinelearningmastery.com/keras-functional-api-deep-learning/)

在定义和拟合模型之前，我们将把数据分成训练集和测试集，并通过将值归一化到 0-1 的范围来缩放输入数据，这是 MLPs 的一个很好的实践。

```
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
```

我们将编码器定义为具有两个隐藏层，第一个具有两倍数量的输入(例如 200)，第二个具有相同数量的输入(100)，随后是具有与数据集相同数量的输入的瓶颈层(100)。

为了保证模型学习良好，我们将使用批处理规范化和泄漏 [ReLU 激活](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)。

```
...
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
```

解码器将以类似的结构来定义，尽管是相反的。

它将有两个隐藏层，第一个具有数据集中的输入数量(例如 100)，第二个具有两倍的输入数量(例如 200)。输出层的节点数将与输入数据中的列数相同，并将使用线性激活函数输出数值。

```
...
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
```

考虑到重建是一种多输出回归问题，模型将使用随机梯度下降的有效 Adam 版本进行拟合，并最小化均方误差。

```
...
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
```

我们可以在自动编码器模型中绘制图层，以了解数据如何在模型中流动。

```
...
# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
```

下图显示了自动编码器的曲线图。

![Plot of Autoencoder Model for Classification With No Compression](img/89172eec95c45fb8cabffc5340956d15.png)

无压缩分类自动编码器模型图

接下来，我们可以训练模型来重现输入，并在等待测试集上跟踪模型的性能。

```
...
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))
```

训练后，我们可以为训练集和测试集绘制学习曲线，以确认模型很好地学习了重建问题。

```
...
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

最后，如果需要，我们可以保存编码器模型供以后使用。

```
...
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')
```

作为保存编码器的一部分，我们还将绘制编码器模型，以获得瓶颈层输出的形状感觉，例如 100 元素向量。

下面提供了该图的示例。

![Plot of Encoder Model for Classification With No Compression](img/b28c7b667a29944a534d0a239dc35580.png)

无压缩分类编码器模型图

将所有这些结合在一起，下面列出了一个完整的自动编码器示例，用于在瓶颈层没有任何压缩的情况下重建分类数据集的输入数据。

```
# train autoencoder for classification with no compression in the bottleneck layer
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# number of input columns
n_inputs = X.shape[1]
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')
```

运行该示例符合模型，并报告沿途火车和测试集的损失。

**注意**:如果在创建模型的地块时遇到问题，可以注释掉导入，调用 *plot_model()* 函数。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到损耗变低了，但是在瓶颈层没有压缩的情况下，损耗没有达到零(正如我们可能预期的那样)。也许需要进一步调整模型架构或学习超参数。

```
...
42/42 - 0s - loss: 0.0032 - val_loss: 0.0016
Epoch 196/200
42/42 - 0s - loss: 0.0031 - val_loss: 0.0024
Epoch 197/200
42/42 - 0s - loss: 0.0032 - val_loss: 0.0015
Epoch 198/200
42/42 - 0s - loss: 0.0032 - val_loss: 0.0014
Epoch 199/200
42/42 - 0s - loss: 0.0031 - val_loss: 0.0020
Epoch 200/200
42/42 - 0s - loss: 0.0029 - val_loss: 0.0017
```

创建的学习曲线图表明，该模型在重构输入时获得了良好的拟合，在整个训练过程中保持稳定，而不是过度拟合。

![Learning Curves of Training the Autoencoder Model Without Compression](img/b470c287235ed252447631fa702b9b4f.png)

无压缩训练自动编码器模型的学习曲线

目前为止，一切顺利。我们知道如何开发一个没有压缩的自动编码器。

接下来，让我们更改模型的配置，以便瓶颈层有一半的节点数量(例如 50 个)。

```
...
# bottleneck
n_bottleneck = round(float(n_inputs) / 2.0)
bottleneck = Dense(n_bottleneck)(e)
```

将这些联系在一起，完整的示例如下所示。

```
# train autoencoder for classification with with compression in the bottleneck layer
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# number of input columns
n_inputs = X.shape[1]
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 2.0)
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')
```

运行该示例符合模型，并报告沿途火车和测试集的损失。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到，在没有压缩的情况下，损失变得与上面的示例类似低，这表明，在瓶颈只有一半大的情况下，模型的性能可能也一样好。

```
...
42/42 - 0s - loss: 0.0029 - val_loss: 0.0010
Epoch 196/200
42/42 - 0s - loss: 0.0029 - val_loss: 0.0013
Epoch 197/200
42/42 - 0s - loss: 0.0030 - val_loss: 9.4472e-04
Epoch 198/200
42/42 - 0s - loss: 0.0028 - val_loss: 0.0015
Epoch 199/200
42/42 - 0s - loss: 0.0033 - val_loss: 0.0021
Epoch 200/200
42/42 - 0s - loss: 0.0027 - val_loss: 8.7731e-04
```

创建了学习曲线的图，再次表明模型在重构输入时获得了良好的拟合，在整个训练过程中保持稳定，而不是过度拟合。

![Learning Curves of Training the Autoencoder Model With Compression](img/f5eddc1854392afed6eebbbb40402f49.png)

压缩训练自动编码器模型的学习曲线

训练好的编码器保存到文件“ *encoder.h5* ”中，我们以后可以加载使用。

接下来，让我们探索如何使用训练好的编码器模型。

## 编码器作为预测模型的数据准备

在本节中，我们将使用自动编码器中经过训练的编码器来压缩输入数据，并训练不同的预测模型。

首先，让我们在这个问题上建立一个性能基线。这一点很重要，因为如果压缩编码不能提高模型的性能，那么压缩编码就不能增加项目的价值，就不应该使用。

我们可以直接在训练数据集上训练逻辑回归模型，并在保持测试集上评估模型的性能。

下面列出了完整的示例。

```
# baseline in performance with logistic regression model
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
# define model
model = LogisticRegression()
# fit model on training set
model.fit(X_train, y_train)
# make prediction on test set
yhat = model.predict(X_test)
# calculate accuracy
acc = accuracy_score(y_test, yhat)
print(acc)
```

运行该示例适合训练数据集上的逻辑回归模型，并在测试集上对其进行评估。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到该模型实现了大约 89.3%的分类准确率。

我们希望并期望逻辑回归模型适合输入的编码版本，以获得更好的准确性，使编码被认为是有用的。

```
0.8939393939393939
```

我们可以更新示例，首先使用上一节中训练的编码器模型对数据进行编码。

首先，我们可以从文件中加载训练好的编码器模型。

```
...
# load the model from file
encoder = load_model('encoder.h5')
```

然后，我们可以使用编码器将原始输入数据(例如 100 列)转换为瓶颈向量(例如 50 个元素向量)。

这个过程可以应用于训练和测试数据集。

```
...
# encode the train data
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)
```

然后，我们可以像以前一样，使用这些编码数据来训练和评估逻辑回归模型。

```
...
# define the model
model = LogisticRegression()
# fit the model on the training set
model.fit(X_train_encode, y_train)
# make predictions on the test set
yhat = model.predict(X_test_encode)
```

将这些联系在一起，完整的示例如下所示。

```
# evaluate logistic regression on encoded input
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
# load the model from file
encoder = load_model('encoder.h5')
# encode the train data
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)
# define the model
model = LogisticRegression()
# fit the model on the training set
model.fit(X_train_encode, y_train)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(y_test, yhat)
print(acc)
```

运行该示例首先使用编码器对数据集进行编码，然后在训练数据集上拟合逻辑回归模型，并在测试集上对其进行评估。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到该模型实现了大约 93.9%的分类准确率。

这比在原始数据集上评估的相同模型具有更好的分类精度，表明编码对我们选择的模型和测试工具有帮助。

```
0.9393939393939394
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [LSTM 自动编码器简介](https://machinelearningmastery.com/lstm-autoencoders/)
*   [如何使用 Keras 函数 API 进行深度学习](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
*   [TensorFlow 2 教程:使用 tf.keras 开始深度学习](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)

### 书

*   [深度学习](https://amzn.to/3kV7gdV)，2016 年。

### 蜜蜂

*   [sklearn . datasets . make _ classification API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . model _ selection . train _ test _ split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。

### 文章

*   [自我编码，维基百科](https://en.wikipedia.org/wiki/Autoencoder)。

## 摘要

在本教程中，您发现了如何开发和评估用于分类预测建模的自动编码器。

具体来说，您了解到:

*   自动编码器是一种神经网络模型，可用于学习原始数据的压缩表示。
*   如何在训练数据集上训练自动编码器模型，并只保存模型的编码器部分。
*   训练机器学习模型时如何将编码器作为数据准备步骤？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。