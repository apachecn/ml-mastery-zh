# 开发钞票鉴别的神经网络

> 原文：<https://machinelearningmastery.com/neural-network-for-banknote-authentication/>

最后更新于 2021 年 10 月 22 日

为新数据集开发神经网络预测模型可能具有挑战性。

一种方法是首先检查数据集，并为哪些模型可能起作用提出想法，然后探索数据集上简单模型的学习动态，最后利用强大的测试工具为数据集开发和调整模型。

该过程可用于开发用于分类和回归预测建模问题的有效神经网络模型。

在本教程中，您将发现如何为钞票二进制分类数据集开发多层感知器神经网络模型。

完成本教程后，您将知道:

*   如何加载和汇总钞票数据集，并使用结果建议数据准备和模型配置以供使用。
*   如何探索数据集上简单 MLP 模型的学习动态。
*   如何对模型性能进行稳健的估计，调整模型性能并对新数据进行预测。

我们开始吧。

*   **2021 年 10 月更新**:已弃用 predict _ classes()语法

![Develop a Neural Network for Banknote Authentication](img/891c0a3995aae255ce3cc56f8506f619.png)

开发用于钞票鉴别的神经网络
图片由[莱尼·K 摄影](https://flickr.com/photos/lennykphotography/21242956935/)拍摄，保留部分权利。

## 教程概述

本教程分为 4 个部分；它们是:

1.  钞票分类数据集
2.  神经网络学习动力学
3.  稳健模型评估
4.  最终模型和做出预测

## 钞票分类数据集

第一步是定义和探索数据集。

我们将使用“*钞票*”标准二进制分类数据集。

钞票数据集包括预测给定钞票是否是真实的，给定从照片中获得的多个度量。

数据集包含 1，372 行 5 个数值变量。这是一个有两类的分类问题(二元分类)。

下面提供了数据集中五个变量的列表。

*   小波变换图像的方差(连续的)。
*   小波变换图像的偏斜度(连续)。
*   小波变换图像的峰度(连续的)。
*   图像熵(连续的)。
*   类(整数)。

下面是数据集前 5 行的示例

```py
3.6216,8.6661,-2.8073,-0.44699,0
4.5459,8.1674,-2.4586,-1.4621,0
3.866,-2.6383,1.9242,0.10645,0
3.4566,9.5228,-4.0112,-3.5944,0
0.32924,-4.4552,4.5718,-0.9888,0
4.3684,9.6718,-3.9606,-3.1625,0
...
```

您可以在此了解有关数据集的更多信息:

*   [钞票数据集(钞票 _ 认证. csv)](https://github.com/jbrownlee/Datasets/blob/master/banknote_authentication.csv)
*   [钞票数据集详细信息(钞票 _ 认证.名称)](https://github.com/jbrownlee/Datasets/blob/master/banknote_authentication.names)

我们可以直接从网址将数据集加载为熊猫数据帧；例如:

```py
# load the banknote dataset and summarize the shape
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize shape
print(df.shape)
```

运行该示例直接从 URL 加载数据集，并报告数据集的形状。

在这种情况下，我们可以确认数据集有 5 个变量(4 个输入和 1 个输出)，并且数据集有 1，372 行数据。

对于神经网络来说，这不是很多行的数据，这表明一个小的网络，也许带有正则化，将是合适的。

它还建议使用 k 倍交叉验证将是一个好主意，因为它将给出比训练/测试分割更可靠的模型性能估计，并且因为单个模型将在几秒钟内适合最大数据集，而不是几小时或几天。

```py
(1372, 5)
```

接下来，我们可以通过查看汇总统计数据和数据图来了解更多关于数据集的信息。

```py
# show summary statistics and plots of the banknote dataset
from pandas import read_csv
from matplotlib import pyplot
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
# load the dataset
df = read_csv(url, header=None)
# show summary statistics
print(df.describe())
# plot histograms
df.hist()
pyplot.show()
```

运行该示例首先加载之前的数据，然后打印每个变量的汇总统计信息。

我们可以看到，数值随着不同的平均值和标准偏差而变化，也许在建模之前需要一些规范化或标准化。

```py
                 0            1            2            3            4
count  1372.000000  1372.000000  1372.000000  1372.000000  1372.000000
mean      0.433735     1.922353     1.397627    -1.191657     0.444606
std       2.842763     5.869047     4.310030     2.101013     0.497103
min      -7.042100   -13.773100    -5.286100    -8.548200     0.000000
25%      -1.773000    -1.708200    -1.574975    -2.413450     0.000000
50%       0.496180     2.319650     0.616630    -0.586650     0.000000
75%       2.821475     6.814625     3.179250     0.394810     1.000000
max       6.824800    12.951600    17.927400     2.449500     1.000000
```

然后为每个变量创建直方图。

我们可以看到，也许前两个变量具有类高斯分布，后两个输入变量可能具有偏斜高斯分布或指数分布。

我们在每个变量上使用幂变换可能会有一些好处，以便使概率分布不那么偏斜，这可能会提高模型性能。

![Histograms of the Banknote Classification Dataset](img/e9766a535d636478fbfc7e308ee5bd08.png)

钞票分类数据集的直方图

现在我们已经熟悉了数据集，让我们探索如何开发一个神经网络模型。

## 神经网络学习动力学

我们将使用张量流为数据集开发一个多层感知器(MLP)模型。

我们无法知道什么样的学习超参数的模型架构对这个数据集是好的或最好的，所以我们必须实验并发现什么是好的。

假设数据集很小，小批量可能是个好主意，例如 16 或 32 行。开始时使用亚当版本的随机梯度下降是一个好主意，因为它会自动调整学习速率，并且在大多数数据集上运行良好。

在我们认真评估模型之前，最好回顾学习动态，调整模型架构和学习配置，直到我们有稳定的学习动态，然后看看如何从模型中获得最大收益。

我们可以通过对数据进行简单的训练/测试分割并查看[学习曲线](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)的曲线来做到这一点。这将有助于我们了解自己是学习过度还是学习不足；然后我们可以相应地调整配置。

首先，我们必须确保所有输入变量都是浮点值，并将目标标签编码为整数值 0 和 1。

```py
...
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
```

接下来，我们可以将数据集分成输入和输出变量，然后分成 67/33 训练集和测试集。

```py
...
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

我们可以定义一个最小 MLP 模型。在这种情况下，我们将使用一个具有 10 个节点的隐藏层和一个输出层(任意选择)。我们将使用隐藏层中的 [ReLU 激活功能](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)和 *he_normal* 权重初始化，作为一个整体，它们是一个很好的实践。

模型的输出是用于二进制分类的 [sigmoid 激活](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)，我们将最小化二进制交叉熵损失。

```py
...
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
```

我们将使模型适合 50 个训练时期(任意选择)，批量大小为 32，因为它是一个小数据集。

我们正在原始数据上拟合模型，我们认为这可能是一个好主意，但这是一个重要的起点。

```py
...
# fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test,y_test))
```

在训练结束时，我们将评估模型在测试数据集上的性能，并将性能报告为分类精度。

```py
...
# predict test set and convert to class label
ypred = model.predict(X_test)
yhat = (ypred > 0.5).flatten().astype(int)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % score)
```

最后，我们将绘制训练和测试集上交叉熵损失的学习曲线。

```py
...
# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
```

综上所述，下面列出了在钞票数据集上评估我们的第一个 MLP 的完整示例。

```py
# fit a simple mlp model on the banknote and review learning curves
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
# fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test,y_test))
# predict test set and convert to class label
ypred = model.predict(X_test)
yhat = (ypred > 0.5).flatten().astype(int) 
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % score)
# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
```

运行该示例首先在训练数据集上拟合模型，然后在测试数据集上报告分类精度。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型达到了 100%的高精度或完美精度。这可能表明预测问题很容易，和/或神经网络很适合这个问题。

```py
Accuracy: 1.000
```

然后创建列车和测试集上的损耗线图。

我们可以看到，模型似乎收敛得很好，没有任何过度拟合或拟合不足的迹象。

![Learning Curves of Simple Multilayer Perceptron on Banknote Dataset](img/13f7aede3daf57f59afece1ebf28031b.png)

钞票数据集上简单多层感知器的学习曲线

我们第一次尝试就做得非常好。

现在，我们已经对数据集上的简单 MLP 模型的学习动态有了一些了解，我们可以考虑对数据集上的模型性能进行更稳健的评估。

## 稳健模型评估

k 倍交叉验证程序可以提供更可靠的 MLP 性能估计，尽管它可能非常慢。

这是因为 k 模型必须被拟合和评估。当数据集尺寸较小时，例如钞票数据集，这不是问题。

我们可以使用[stratifiedfold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)类手动枚举每个折叠，拟合模型，对其进行评估，然后在程序结束时报告评估分数的平均值。

```py
...
# prepare cross validation
kfold = KFold(10)
# enumerate splits
scores = list()
for train_ix, test_ix in kfold.split(X, y):
	# fit and evaluate the model...
	...
...
# summarize all scores
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

我们可以使用这个框架，利用我们的基本配置，甚至利用一系列不同的数据准备、模型架构和学习配置，来开发 MLP 模型性能的可靠估计。

重要的是，在使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)估计性能之前，我们首先了解了模型在前面部分的数据集上的学习动态。如果我们开始直接调整模型，我们可能会得到好的结果，但如果没有，我们可能不知道为什么，例如，模型过度或拟合不足。

如果我们再次对模型进行大的更改，最好返回并确认模型正在适当收敛。

下面列出了评估前一节中的基本 MLP 模型的框架的完整示例。

```py
# k-fold cross-validation of base model for the banknote dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(10)
# enumerate splits
scores = list()
for train_ix, test_ix in kfold.split(X, y):
	# split data
	X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
	# determine the number of input features
	n_features = X.shape[1]
	# define model
	model = Sequential()
	model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy')
	# fit the model
	model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
	# predict test set and convert to class label
        ypred = model.predict(X_test)
        yhat = (ypred > 0.5).flatten().astype(int) 
	# evaluate predictions
	score = accuracy_score(y_test, yhat)
	print('>%.3f' % score)
	scores.append(score)
# summarize all scores
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会报告评估程序每次迭代的模型性能，并在运行结束时报告分类精度的平均值和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 MLP 模型达到了大约 99.9%的平均精度。

这证实了我们的期望，即基本模型配置对于这个数据集非常有效，并且实际上模型非常适合这个问题，并且这个问题可能很难解决。

这(对我来说)令人惊讶，因为我本以为需要一些数据扩展，或许还需要一次电源转换。

```py
>1.000
>1.000
>1.000
>1.000
>0.993
>1.000
>1.000
>1.000
>1.000
>1.000
Mean Accuracy: 0.999 (0.002)
```

接下来，让我们看看如何拟合最终模型并使用它进行预测。

## 最终模型和做出预测

一旦我们选择了一个模型配置，我们就可以在所有可用的数据上训练一个最终模型，并使用它来对新数据进行预测。

在这种情况下，我们将使用具有脱落和小批量的模型作为最终模型。

我们可以像以前一样准备数据并拟合模型，尽管是在整个数据集上，而不是数据集的训练子集上。

```py
...
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
le = LabelEncoder()
y = le.fit_transform(y)
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
```

然后，我们可以使用这个模型对新数据进行预测。

首先，我们可以定义一行新数据。

```py
...
# define a row of new data
row = [3.6216,8.6661,-2.8073,-0.44699]
```

注意:我从数据集的第一行提取了这一行，预期的标签是“0”。

然后我们可以做一个预测。

```py
...
# make prediction and convert to class label
ypred = model.predict([row])
yhat = (ypred > 0.5).flatten().astype(int)
```

然后反转预测上的转换，这样我们就可以使用或解释正确标签中的结果(对于这个数据集，它只是一个整数)。

```py
...
# invert transform to get label for class
yhat = le.inverse_transform(yhat)
```

在这种情况下，我们将简单地报告预测。

```py
...
# report prediction
print('Predicted: %s' % (yhat[0]))
```

将所有这些结合起来，下面列出了为钞票数据集拟合最终模型并使用它对新数据进行预测的完整示例。

```py
# fit a final model and make predictions on new data for the banknote dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
le = LabelEncoder()
y = le.fit_transform(y)
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
# fit the model
model.fit(X, y, epochs=50, batch_size=32, verbose=0)
# define a row of new data
row = [3.6216,8.6661,-2.8073,-0.44699]
# make prediction and convert to class label
ypred = model.predict([row])
yhat = (ypred > 0.5).flatten().astype(int)
# invert transform to get label for class
yhat = le.inverse_transform(yhat)
# report prediction
print('Predicted: %s' % (yhat[0]))
```

运行该示例使模型适合整个数据集，并对单行新数据进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型为输入行预测了一个“0”标签。

```py
Predicted: 0.0
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何开发预测电离层扰动的神经网络](https://machinelearningmastery.com/predicting-disturbances-in-the-ionosphere/)
*   [标准机器学习数据集的最佳结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [TensorFlow 2 教程:使用 tf.keras 开始深度学习](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)
*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

## 摘要

在本教程中，您发现了如何为钞票二进制分类数据集开发多层感知器神经网络模型。

具体来说，您了解到:

*   如何加载和汇总钞票数据集，并使用结果建议数据准备和模型配置以供使用。
*   如何探索数据集上简单 MLP 模型的学习动态。
*   如何对模型性能进行稳健的估计，调整模型性能并对新数据进行预测。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。