# 评估 Keras 中深度学习模型的表现

> 原文： [https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)

Keras 是一个易于使用且功能强大的 Python 库，用于深度学习。

在设计和配置深度学习模型时，需要做出很多决定。大多数决策必须通过试验和错误凭经验解决，并根据实际数据进行评估。

因此，有一种强大的方法来评估神经网络和深度学习模型的表现至关重要。

在这篇文章中，您将发现使用 Keras 评估模型表现的几种方法。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Evaluate the Performance Of Deep Learning Models in Keras](img/e1c3b3955f7f82b7e5b64209a3825ab4.png)

评估 Keras 中深度学习模型的表现
照片由 [Thomas Leuthard](https://www.flickr.com/photos/thomasleuthard/7273077758/) 拍摄，保留一些权利。

## 根据经验评估网络配置

在设计和配置深度学习模型时，必须做出无数决定。

其中许多决策可以通过复制其他人的网络结构并使用启发式方法来解决。最终，最好的技术是实际设计小型实验并使用实际数据凭经验评估选项。

这包括高级决策，例如网络中层的数量，大小和类型。它还包括较低级别的决策，如损失函数的选择，激活函数，优化过程和时期数。

深度学习通常用于具有非常大的数据集的问题。那就是数万或数十万个实例。

因此，您需要拥有一个强大的测试工具，可以让您估计给定配置在看不见的数据上的表现，并可靠地将表现与其他配置进行比较。

## 数据拆分

大量数据和模型的复杂性需要非常长的训练时间。

因此，通常使用简单的数据分离到训练和测试数据集或训练和验证数据集中。

Keras 提供了两种方便的方式来评估您的深度学习算法：

1.  使用自动验证数据集。
2.  使用手动验证数据集。

### 使用自动验证数据集

Keras 可以将训练数据的一部分分离为验证数据集，并在每个时期评估模型在该验证数据集上的表现。

您可以通过将 **fit** （）函数上的 **validation_split** 参数设置为训练数据集大小的百分比来完成此操作。

例如，20％的合理值可能是 0.2 或 0.33，或者为了验证而保留的训练数据的 33％。

下面的示例演示了如何在小二元分类问题上使用自动验证数据集。本文中的所有实例均使用[皮马印第安人糖尿病数据集](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)。您可以[从 UCI 机器学习库下载](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)，并使用文件名 **pima-indians-diabetes.csv** 将数据文件保存到当前工作目录中（更新：[从这里](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

```py
# MLP with automatic validation set
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
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
```

运行该示例，您可以看到每个时期的详细输出显示了训练数据集和验证数据集的损失和准确性。

```py
...
Epoch 145/150
514/514 [==============================] - 0s - loss: 0.5252 - acc: 0.7335 - val_loss: 0.5489 - val_acc: 0.7244
Epoch 146/150
514/514 [==============================] - 0s - loss: 0.5198 - acc: 0.7296 - val_loss: 0.5918 - val_acc: 0.7244
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.5175 - acc: 0.7335 - val_loss: 0.5365 - val_acc: 0.7441
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.5219 - acc: 0.7354 - val_loss: 0.5414 - val_acc: 0.7520
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.5089 - acc: 0.7432 - val_loss: 0.5417 - val_acc: 0.7520
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.5148 - acc: 0.7490 - val_loss: 0.5549 - val_acc: 0.7520
```

### 使用手动验证数据集

Keras 还允许您手动指定在训练期间用于验证的数据集。

在这个例子中，我们使用 Python [scikit-learn](http://scikit-learn.org/stable/index.html) 机器学习库中的方便 [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) （）函数将我们的数据分成训练和测试数据集。我们使用 67％用于训练，剩余 33％用于验证。

可以通过 **validation_data** 参数将验证数据集指定给 Keras 中的 **fit** （）函数。它需要输入和输出数据集的元组。

```py
# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```

与之前一样，运行该示例提供了详细的训练输出，其中包括模型在每个时期的训练和验证数据集上的丢失和准确性。

```py
...
Epoch 145/150
514/514 [==============================] - 0s - loss: 0.4847 - acc: 0.7704 - val_loss: 0.5668 - val_acc: 0.7323
Epoch 146/150
514/514 [==============================] - 0s - loss: 0.4853 - acc: 0.7549 - val_loss: 0.5768 - val_acc: 0.7087
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.4864 - acc: 0.7743 - val_loss: 0.5604 - val_acc: 0.7244
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.4831 - acc: 0.7665 - val_loss: 0.5589 - val_acc: 0.7126
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.4961 - acc: 0.7782 - val_loss: 0.5663 - val_acc: 0.7126
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.4967 - acc: 0.7588 - val_loss: 0.5810 - val_acc: 0.6929
```

## 手动 k-fold 交叉验证

机器学习模型评估的黄金标准是 [k 倍交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics))。

它提供了对未见数据模型表现的可靠估计。它通过将训练数据集分成 k 个子集并在所有子集上轮流训练模型（除了一个被保持的外部）并在所保持的验证数据集上评估模型表现来实现这一点。重复该过程，直到所有子集都有机会成为保持的验证集。然后在所有创建的模型中对表现度量进行平均。

由于计算费用较高，交叉验证通常不用于评估深度学习模型。例如，k 倍交叉验证通常使用 5 或 10 倍。因此，必须构建和评估 5 或 10 个模型，这大大增加了模型的评估时间。

然而，当问题足够小或者你有足够的计算资源时，k-fold 交叉验证可以让你对模型的表现进行较少的偏差估计。

在下面的例子中，我们使用来自 [scikit-learn](http://scikit-learn.org/stable/index.html) Python 机器学习库的方便的 [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html) 类将训练数据集分成 10 倍。折叠是分层的，这意味着算法试图平衡每个折叠中每个类的实例数。

该示例使用 10 个数据分割创建和评估 10 个模型，并收集所有分数。通过将 verbose = 0 传递给模型上的 **fit（）**和 **evaluate（）**函数来关闭每个迭代的详细输出。

为每个型号打印表现并将其存储。然后在运行结束时打印模型表现的平均值和标准偏差，以提供模型精度的稳健估计。

```py
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
```

运行该示例将花费不到一分钟，并将产生以下输出：

```py
acc: 77.92%
acc: 68.83%
acc: 72.73%
acc: 64.94%
acc: 77.92%
acc: 35.06%
acc: 74.03%
acc: 68.83%
acc: 34.21%
acc: 72.37%
64.68% (+/- 15.50%)
```

## 摘要

在这篇文章中，您发现了使用一种强大的方法来估计深度学习模型在看不见的数据上的表现的重要性。

您发现了三种使用 Keras 库在 Python 中估计深度学习模型表现的方法：

*   使用自动验证数据集。
*   使用手动验证数据集。
*   使用手动 k-fold 交叉验证。

您对 Keras 或此帖的深度学习有任何疑问吗？在评论中提出您的问题，我会尽力回答。