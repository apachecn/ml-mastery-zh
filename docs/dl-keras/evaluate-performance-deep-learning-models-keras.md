# 评估 Keras 中深度学习模型的表现

> 原文： [https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)

Keras 是一个易于使用且功能强大被用于深度学习的 Python 库。

在设计和配置深度学习模型时，需要做出许多决策,这些决策大多必须通过反复试验和根据真实数据进行评估，以经验方式解决。

因此，有一种强大的方法来评估神经网络和深度学习模型的表现至关重要。

在这篇文章中，您将发现使用 Keras 评估模型表现的几种方法。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Evaluate the Performance Of Deep Learning Models in Keras](img/e1c3b3955f7f82b7e5b64209a3825ab4.png)


照片由 [Thomas Leuthard](https://www.flickr.com/photos/thomasleuthard/7273077758/) 拍摄，保留所属权利

## 根据经验评估网络配置

在设计和配置深度学习模型时，必须做出很多决策。

其中许多决策可以通过复制其他人的网络结构并使用启发式方法来解。最终，最好的技术是根据实际设计小型实验并使用实际数据凭经验评估相关选项。

这包括高级决策，例如网络中层的数量，大小和类型，它还包括较低级别的决策，如损失函数的选择，激活函数，优化过程和迭代次数。

深度学习通常用于具有非常大的数据集的问题，例如有数万或数十万个实例。

因此，您需要拥有一个强大的测试工具，可以让您评估给定配置在不可见的数据上的表现，并将较为可靠的表现表现与其他配置进行比较。

## 数据拆分

大量数据和复杂的模型需要非常长的训练时间。

因此，通常将数据分为测试数据集和验证数据集。

Keras 提供了两种方便的方式来评估您的深度学习算法：

1.  使用自动验证数据集。
2.  使用手动验证数据集。

### 使用自动验证数据集

Keras 可以将训练数据的一部分划分为为验证数据集，并在每个迭代中评估模型在该验证数据集上的表现。

您可以通过将`fit()`函数上的 **validation_split** 参数设置为训练数据集大小的百分比来完成此操作。

例如，20％的合理值可能是 0.2 或 0.33，或者为了验证而将验证数据集的数量选择的训练数据集的 33％。

下面的示例演示了如何在小二分类问题上使用自动验证数据集。本文中的所有实例均使用[皮马印第安人糖尿病数据集](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)。您可以[从 UCI 机器学习库下载](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)，并使用文件名 **pima-indians-diabetes.csv** 将数据文件保存到当前工作目录中（更新：[从这里](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

```py
# 使用自动验证集的MLP
from keras.models import Sequential
from keras.layers import Dense
import numpy
#固定随机种子再现性
numpy.random.seed(7)
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入变量和输出变量
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 拟合模型
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
```

运行该示例，您可以看到每个时期的详细输出显示了训练数据集和验证数据集的损失和准确率。

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

在这个例子中，我们使用 Python [scikit-learn](http://scikit-learn.org/stable/index.html) 机器学习库中方便的 [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html)()函数将我们的数据分成训练和测试数据集。我们使用 67％用于训练，剩余 33％用于验证。

可以通过 **validation_data** 参数将验证数据集指定给 Keras 中的`fit()`函数，该函数输出和输入的数据类型是数据集中的元组类型。

```py
# 使用手动验证集的MLP
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
#固定随机种子再现性
seed = 7
numpy.random.seed(seed)
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入变量X和输出变量Y
X = dataset[:,0:8]
Y = dataset[:,8]
# 将数据集划分为67%的训练集和33%的测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 拟合模型
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```

与之前一样，运行该示例提供了详细的训练输出数据，其中包括模型在每个迭代期间在训练集和验证数据集上的损失和精确度。

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

机器学习模型评估的黄金标准是 [k 折交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics)。

它提供了模型对不可见数据的表现的可靠估计，它通过将训练数据集拆分为 k 子集来实现此，并对所有子集（除保留的子集外）轮流训练模型，并评估已保留验证数据集上的模型表现，该过程将重复，直到所有子集都有机会成为已执行的验证集。然后，在创建的所有模型中对表现度量值进行平均。

交叉验证通常不用于评估深度学习模型，因为计算的花费会更高。例如，k 折交叉验证通常用于 5 或 10 折，因此，必须构造和评估 5 或 10 个模型，从而大大增加了模型的评估时间。

然而，当问题足够小或者你有足够的计算资源时，k-fold 交叉验证可以让你对模型的表现进行较少的偏差估计。

在下面的例子中，我们使用来自 [scikit-learn](http://scikit-learn.org/stable/index.html) Python 机器学习库的方便的 [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html) 类将训练数据集分成 10 折，折叠是分层的，这意味着算法试图平衡每个折叠中每个类的实例数。

该示例使用数据的 10 个拆分创建评估 10 个模型，并收集所有的表现分数，并通过将`verbose=0`传递给模型上的`fit()`函数和`evaluate()`函数，并将每个迭代期间的详细输出关闭。

为每个模型存储和打印相关的表现数据，然后，在运行结束时打印模型表现的平均差和标准差，以提供对模型准确率的可靠估计。

```py
# 为Pima Indians 数据集使用10折交叉验证的MLP
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# 固定随机种子再现性
seed = 7
numpy.random.seed(seed)
# 加载数据集
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 将数据集划分为输入数据X和输出数据Y
X = dataset[:,0:8]
Y = dataset[:,8]
# 定义10折交叉验证测试线束
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # 创建模型
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# 拟合模型
	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
	# 评估模型
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

在这篇文章中，您发现了使用一种强大的方法来估计深度学习模型在不可见数据集上的模型表现的重要性。

您发现了三种使用 Keras 库在 Python 中估计深度学习模型表现的方法：

*   使用自动验证数据集。
*   使用手动验证数据集。
*   使用手动 k-fold 交叉验证。

您对 Keras 深度学习或者此文章还有疑问吗？在评论中提出您的问题，我会尽力回答。