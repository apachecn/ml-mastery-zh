# 如何在 Keras 中检查深度学习模型

> 原文： [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

深度学习模型可能需要数小时，数天甚至数周才能进行训练。

如果意外停止运行，则可能会丢失大量工作。

在这篇文章中，您将了解如何使用 Keras 库在 Python 训练期间检查您的深度学习模型。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Check-Point Deep Learning Models in Keras](img/299672cd2b9efa26d634b06c4df2e751.png)

如何在 Keras 检查深度学习模型
照片由 [saragoldsmith](https://www.flickr.com/photos/saragoldsmith/2353051153/) ，保留一些权利。

## 检验点神经网络模型

[应用程序检查点](https://en.wikipedia.org/wiki/Application_checkpointing)是一种容错技术，适用于长时间运行的进程。

这是一种在系统出现故障时采用系统状态快照的方法。如果出现问题，并非全部丢失。检查点可以直接使用，或者用作新运行的起点，从中断处开始。

在训练深度学习模型时，检查点是模型的权重。这些权重可用于按原样进行预测，或用作持续训练的基础。

Keras 库通过回调 API 提供[检查点功能。](http://keras.io/callbacks/#modelcheckpoint)

ModelCheckpoint 回调类允许您定义检查模型权重的位置，文件应如何命名以及在何种情况下创建模型的检查点。

API 允许您指定要监控的度量标准，例如训练或验证数据集的丢失或准确性。您可以指定是否在最大化或最小化分数时寻求改进。最后，用于存储权重的文件名可以包含诸如迭代号或度量的变量。

然后，在模型上调用 fit（）函数时，可以将 ModelCheckpoint 传递给训练过程。

注意，您可能需要安装 [h5py 库](http://www.h5py.org/)以输出 HDF5 格式的网络权重。

## 检查点神经网络模型改进

检查点的良好用途是每次在训练期间观察到改进时输出模型权重。

下面的例子为皮马印第安人糖尿病二元分类问题创建了一个小型神经网络。该示例假设 _pima-indians-diabetes.csv_ 文件位于您的工作目录中。

您可以从此处下载数据集：

*   [皮马印第安人糖尿病数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

该示例使用 33％的数据进行验证。

只有在验证数据集（monitor ='val_acc'和 mode ='max'）的分类准确性有所提高时，才会设置检验点以保存网络权重。权重存储在一个文件中，该文件包含文件名中的分数（权重改进 - {val_acc = .2f} .hdf5）。

```py
# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
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
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
```

运行该示例将生成以下输出（为简洁起见，将其截断）：

```py
...
Epoch 00134: val_acc did not improve
Epoch 00135: val_acc did not improve
Epoch 00136: val_acc did not improve
Epoch 00137: val_acc did not improve
Epoch 00138: val_acc did not improve
Epoch 00139: val_acc did not improve
Epoch 00140: val_acc improved from 0.83465 to 0.83858, saving model to weights-improvement-140-0.84.hdf5
Epoch 00141: val_acc did not improve
Epoch 00142: val_acc did not improve
Epoch 00143: val_acc did not improve
Epoch 00144: val_acc did not improve
Epoch 00145: val_acc did not improve
Epoch 00146: val_acc improved from 0.83858 to 0.84252, saving model to weights-improvement-146-0.84.hdf5
Epoch 00147: val_acc did not improve
Epoch 00148: val_acc improved from 0.84252 to 0.84252, saving model to weights-improvement-148-0.84.hdf5
Epoch 00149: val_acc did not improve
```

您将在工作目录中看到许多文件，其中包含 HDF5 格式的网络权重。例如：

```py
...
weights-improvement-53-0.76.hdf5
weights-improvement-71-0.76.hdf5
weights-improvement-77-0.78.hdf5
weights-improvement-99-0.78.hdf5
```

这是一个非常简单的检查点策略。如果验证准确度在训练时期上下移动，则可能会创建大量不必要的检查点文件。然而，它将确保您拥有在运行期间发现的最佳模型的快照。

## 仅限检查点最佳神经网络模型

更简单的检查点策略是将模型权重保存到同一文件中，当且仅当验证准确度提高时。

这可以使用上面相同的代码轻松完成，并将输出文件名更改为固定（不包括分数或迭代信息）。

在这种情况下，只有当验证数据集上模型的分类精度提高到目前为止最佳时，模型权重才会写入文件“weights.best.hdf5”。

```py
# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
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
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
```

运行此示例提供以下输出（为简洁起见，将其截断）：

```py
...
Epoch 00139: val_acc improved from 0.79134 to 0.79134, saving model to weights.best.hdf5
Epoch 00140: val_acc did not improve
Epoch 00141: val_acc did not improve
Epoch 00142: val_acc did not improve
Epoch 00143: val_acc did not improve
Epoch 00144: val_acc improved from 0.79134 to 0.79528, saving model to weights.best.hdf5
Epoch 00145: val_acc improved from 0.79528 to 0.79528, saving model to weights.best.hdf5
Epoch 00146: val_acc did not improve
Epoch 00147: val_acc did not improve
Epoch 00148: val_acc did not improve
Epoch 00149: val_acc did not improve
```

您应该在本地目录中看到权重文件。

```py
weights.best.hdf5
```

这是一个方便的检查点策略，在您的实验中始终使用。它将确保为运行保存最佳模型，以便您以后使用。它避免了您需要在训练时包含代码以手动跟踪和序列化最佳模型。

## 加载检查指向神经网络模型

现在您已经了解了如何在训练期间检查您的深度学习模型，您需要查看如何加载和使用检查点模型。

检查点仅包括模型权重。它假设您了解网络结构。这也可以序列化为 JSON 或 YAML 格式的文件。

在下面的示例中，模型结构是已知的，最佳权重从上一个实验加载，存储在 weights.best.hdf5 文件的工作目录中。

然后使用该模型对整个数据集进行预测。

```py
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

运行该示例将生成以下输出：

```py
Created model and loaded weights from file
acc: 77.73%
```

## 摘要

在这篇文章中，您已经发现了深度学习模型在长时间训练中的重要性。

您学习了两个检查点策略，您可以在下一个深度学习项目中使用它们：

1.  检查点模型改进。
2.  Checkpoint 最佳型号。

您还学习了如何加载检查点模型并进行预测。

您对深度学习模型或此帖的检查点有任何疑问吗？在评论中提出您的问题，我会尽力回答。