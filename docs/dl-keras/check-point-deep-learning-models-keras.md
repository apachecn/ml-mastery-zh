# 如何在 Keras 中检查深度学习模型

> 原文： [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

深度学习模型在训练时可能需要花费数小时，数天甚至数周。

如果意外停止运行，则可能会丢失大量成果。

在这篇文章中，您将了解如何使用Python中的keras库在模型训练期间检查您的深度学习模型。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Check-Point Deep Learning Models in Keras](img/299672cd2b9efa26d634b06c4df2e751.png)


照片由 [saragoldsmith](https://www.flickr.com/photos/saragoldsmith/2353051153/) 提供，并保留其所属权利。

## 检验点神经网络模型

[应用程序检查点](https://en.wikipedia.org/wiki/Application_checkpointing)是一种容错技术，适用于长时间运行的进程。

这是一种在系统出现故障时采用系统状态快照的方法，如果出现问题，任务并非全部丢失，检查点可以直接使用，或者从中断处开始，用作程序重新运行的起点。

在训练深度学习模型时，检查点是模型的权重参数，这些权重可用于按原样做出预测，或用作持续训练的基础。

Keras 库通过回调 API 提供[检查点功能。](http://keras.io/callbacks/#modelcheckpoint)

ModelCheckpoint 回调类允许您定义检查模型权重的位置，文件应如何命名以及在何种情况下创建模型的检查点。

API 允许您指定要监控的度量标准，例如训练或验证数据集的损失或准确率，您可以指定是否在最大化或最小化分数时寻求改进，最后，用于存储权重的文件名可以包含诸如迭代数量或度量的变量。

然后，在模型上调用`fit()`函数时，可以将 ModelCheckpoint 传递给训练过程。

注意，您可能需要安装 [h5py 库](http://www.h5py.org/)以输出 HDF5 格式的网络权重。

## 检查点神经网络模型改进

检查点的良好用途是每次在训练期间观察到表现提升时输出模型权重参数。

下面的例子为皮马印第安人糖尿病二分类问题创建了一个小型神经网络。该示例假设 _pima-indians-diabetes.csv_ 文件位于您的工作目录中。

您可以从此处下载数据集：

*   [皮马印第安人糖尿病数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

该示例使用 33％的数据作为验证集。

只有在验证数据集`（monitor ='val_acc'和 mode ='max'）`的分类准确率有所提高时，才会设置检验点以保存网络权重参数。权重参数存储在一个文件中，该.hdf5文件的文件名为当前精度值（格式化输出为：`权重改进 - {val_acc = .2f} .hdf5`）。

```py
# 当验证集的精度有所提高时，需要保存当前的权重参数
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# 固定随机种子再现性
seed = 7
numpy.random.seed(seed)
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入变量X和输出变量Y
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 检查点
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# 拟合模型
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
```

运行该示例将生成以下输出（为简洁表示，只显示其一部分结果）：

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

这是一个非常简单的检查点策略，如果验证准确度在训练时期上下移动，则可能会创建大量不必要的检查点文件，然而，它将确保您发现模型运行期间的最佳快照。

## 仅限检查点最佳神经网络模型

更简单的检查点策略是当且仅当验证准确度提高时将模型权重保存到同一文件中。

这可以使用上面相同的代码轻松完成，并将输出文件名更改为固定的字符串（不包括分数或迭代信息）。

在这种情况下，只有当验证数据集上模型的分类精度提高到当前最佳时，模型权重才会被写入文件`weights.best.hdf5`.

```py
#当验证模型准确率最高时，保存权重
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
#固定随机种子再现性
seed = 7
numpy.random.seed(seed)
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入变量和输出变量
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 检查点
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# 拟合模型
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
```

运行此示例提供以下输出（为简洁表示，只显示其一部分结果）：

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

您应该可以在本地目录中看到权重文件。

```py
weights.best.hdf5
```

这是在您的实验中能够始终使用的一个方便的检查点策略，它将确保为运行保存最佳模型，以便您以后使用，这个策略避免了您在训练时需要包含代码以手动跟踪和序列化最佳模型。

## 加载一个检查点的神经网络模型

现在您已经了解了如何在训练期间检查您的深度学习模型，您现在需要了解如何加载和使用检查点模型。

检查点仅包括模型权重，假设您了解网络结构，这些模型权重也可以序列化为 JSON 或 YAML 格式的文件。

在下面的示例中，模型结构是已知的，最佳权重从上一个实验加载，存储在 weights.best.hdf5 文件的工作目录中。

然后使用该模型对整个数据集做出预测。

```py
# 怎样从一个检查点加载和使用权重参数
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# 固定随机种子再现性
numpy.random.seed(seed)
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# 加载权重参数
model.load_weights("weights.best.hdf5")
# 编译模型（需要做出预测）
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入变量和输出变量
X = dataset[:,0:8]
Y = dataset[:,8]
# 在整个数据集上使用加载的权重参数评估模型表现
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

运行该示例将生成以下输出：

```py
Created model and loaded weights from file
acc: 77.73%
```

## 摘要

在这篇文章中，您已经了解了检查点在深度学习模型中长时间训练中的重要性。

您学习了两个检查点策略，您可以在下一个深度学习项目中使用它们：

1.  检查点模型改进。
2.  唯一的检查点最佳模型。

您还学习了如何加载检查点模型并做出预测。

如果您对深度学习模型检查点或此篇文章有任何疑问，在评论中提出您的问题，我会尽力回答。