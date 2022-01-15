# 为癌症存活数据集开发神经网络

> 原文：<https://machinelearningmastery.com/neural-network-for-cancer-survival-dataset/>

为新数据集开发神经网络预测模型可能具有挑战性。

一种方法是首先检查数据集，并为哪些模型可能起作用提出想法，然后探索数据集上简单模型的学习动态，最后利用强大的测试工具为数据集开发和调整模型。

该过程可用于开发用于分类和回归预测建模问题的有效神经网络模型。

在本教程中，您将发现如何为癌症存活二元分类数据集开发多层感知器神经网络模型。

完成本教程后，您将知道:

*   如何加载和总结癌症存活数据集，并使用结果建议数据准备和模型配置使用。
*   如何探索数据集上简单 MLP 模型的学习动态。
*   如何对模型表现进行稳健的估计，调整模型表现并对新数据进行预测。

我们开始吧。

![Develop a Neural Network for Cancer Survival Dataset](img/79ccd8569f154da92b3b3e8b26672a47.png)

为癌症存活数据集开发神经网络
图片由[贝恩德·泰勒](https://flickr.com/photos/bernd_thaller/47315883761/)提供，保留部分权利。

## 教程概述

本教程分为 4 个部分；它们是:

1.  哈贝曼乳腺癌存活数据集
2.  神经网络学习动力学
3.  稳健模型评估
4.  最终模型和做出预测

## 哈贝曼乳腺癌存活数据集

第一步是定义和探索数据集。

我们将使用“*哈贝曼*”标准二进制分类数据集。

数据集描述了乳腺癌患者数据，结果是患者存活率。具体来说，患者是否存活了 5 年或更长时间，或者患者是否没有存活。

这是一个用于不平衡分类研究的标准数据集。根据数据集描述，手术于 1958 年至 1970 年在芝加哥大学比林斯医院进行。

数据集中有 306 个例子，有 3 个输入变量；它们是:

*   手术时患者的年龄。
*   运营的两位数年份。
*   检测到的“*阳性腋窝淋巴结*的数量，这是衡量癌症是否已经扩散的指标。

因此，除了数据集中可用的情况之外，我们无法控制组成数据集的情况或在这些情况下使用的要素的选择。

尽管数据集描述了乳腺癌患者的存活率，但鉴于数据集规模较小，并且数据基于几十年前的乳腺癌诊断和手术，因此任何基于该数据集构建的模型都不可一概而论。

**注:说得再清楚不过**了，我们是不是“*解决乳腺癌*”。我们正在探索一个标准的分类数据集。

下面是数据集前 5 行的示例

```py
30,64,1,1
30,62,3,1
30,65,0,1
31,59,2,1
31,65,4,1
...
```

您可以在此了解有关数据集的更多信息:

*   [哈贝曼存活数据集(haberman.csv)](https://github.com/jbrownlee/Datasets/blob/master/haberman.csv)
*   [哈贝曼存活数据集详情(哈贝曼. name)](https://github.com/jbrownlee/Datasets/blob/master/haberman.names)

我们可以直接从网址将数据集加载为熊猫数据帧；例如:

```py
# load the haberman dataset and summarize the shape
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize shape
print(df.shape)
```

运行该示例直接从 URL 加载数据集，并报告数据集的形状。

在这种情况下，我们可以确认数据集有 4 个变量(3 个输入和 1 个输出)，并且数据集有 306 行数据。

对于神经网络来说，这不是很多行的数据，这表明一个小的网络，也许带有正则化，将是合适的。

它还建议使用 k 倍交叉验证将是一个好主意，因为它将给出比训练/测试分割更可靠的模型表现估计，并且因为单个模型将在几秒钟内适合最大数据集，而不是几小时或几天。

```py
(306, 4)
```

接下来，我们可以通过查看汇总统计数据和数据图来了解更多关于数据集的信息。

```py
# show summary statistics and plots of the haberman dataset
from pandas import read_csv
from matplotlib import pyplot
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
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
                0           1           2           3
count  306.000000  306.000000  306.000000  306.000000
mean    52.457516   62.852941    4.026144    1.264706
std     10.803452    3.249405    7.189654    0.441899
min     30.000000   58.000000    0.000000    1.000000
25%     44.000000   60.000000    0.000000    1.000000
50%     52.000000   63.000000    1.000000    1.000000
75%     60.750000   65.750000    4.000000    2.000000
max     83.000000   69.000000   52.000000    2.000000
```

然后为每个变量创建直方图。

我们可以看到，也许第一个变量具有类高斯分布，接下来的两个输入变量可能具有指数分布。

我们在每个变量上使用幂变换可能会有一些好处，以便使概率分布不那么偏斜，这可能会提高模型表现。

![Histograms of the Haberman Breast Cancer Survival Classification Dataset](img/eac089342cd334e5b9ecb1b6c9e94021.png)

哈贝曼乳腺癌存活分类数据集的直方图

我们可以看到两个类之间的例子分布有些偏斜，这意味着分类问题是不平衡的。这是不平衡的。

了解数据集实际上有多不平衡可能会有所帮助。

我们可以使用 Counter 对象来统计每个类中的示例数量，然后使用这些计数来总结分布。

下面列出了完整的示例。

```py
# summarize the class ratio of the haberman dataset
from pandas import read_csv
from collections import Counter
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
# define the dataset column names
columns = ['age', 'year', 'nodes', 'class']
# load the csv file as a data frame
dataframe = read_csv(url, header=None, names=columns)
# summarize the class distribution
target = dataframe['class'].values
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

运行该示例总结了数据集的类分布。

我们可以看到存活类 1 在 225 处有最多的例子，大约占数据集的 74%。我们可以看到非存活类 2 的例子更少，只有 81 个，约占数据集的 26%。

阶级分布是倾斜的，但并不严重不平衡。

```py
Class=1, Count=225, Percentage=73.529%
Class=2, Count=81, Percentage=26.471%
```

这是有帮助的，因为如果我们使用分类精确率，那么任何达到低于约 73.5%精确率的模型都不具备在这个数据集上的技能。

现在我们已经熟悉了数据集，让我们探索如何开发一个神经网络模型。

## 神经网络学习动力学

我们将使用[张量流](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)为数据集开发一个多层感知器(MLP)模型。

我们无法知道什么样的学习超参数的模型架构对这个数据集是好的或最好的，所以我们必须实验并发现什么是好的。

假设数据集很小，小批量可能是个好主意，例如 16 或 32 行。开始时使用亚当版本的随机梯度下降是一个好主意，因为它会自动调整学习速率，并且在大多数数据集上运行良好。

在我们认真评估模型之前，最好回顾学习动态，调整模型架构和学习配置，直到我们有稳定的学习动态，然后看看如何从模型中获得最大收益。

我们可以通过使用简单的数据训练/测试分割和学习曲线的回顾图来做到这一点。这将有助于我们了解自己是学习过度还是学习不足；然后我们可以相应地调整配置。

首先，我们必须确保所有输入变量都是浮点值，并将目标标签编码为整数值 0 和 1。

```py
...
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
```

接下来，我们可以将数据集分成输入和输出变量，然后分成 67/33 训练集和测试集。

我们必须确保按类对拆分进行分层，确保训练集和测试集具有与主数据集相同的类标签分布。

```py
...
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=3)
```

我们可以定义一个最小 MLP 模型。在这种情况下，我们将使用一个具有 10 个节点的隐藏层和一个输出层(任意选择)。我们将使用隐藏层中的 [ReLU 激活函数](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)和*he _ normal*[权重初始化](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)，作为一个整体，它们是一个很好的练习。

模型的输出是二进制分类的 sigmoid 激活，我们将最小化二进制[交叉熵损失](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)。

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

我们将使模型适合 200 个训练时期(任意选择)，批量大小为 16，因为它是一个小数据集。

我们正在原始数据上拟合模型，我们认为这可能是一个好主意，但这是一个重要的起点。

```py
...
# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_test,y_test))
```

在训练结束时，我们将评估模型在测试数据集上的表现，并将表现报告为分类精确率。

```py
...
# predict test set
yhat = model.predict_classes(X_test)
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

将所有这些结合起来，下面列出了在癌症存活数据集上评估我们的第一个 MLP 的完整示例。

```py
# fit a simple mlp model on the haberman and review learning curves
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=3)
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_test,y_test))
# predict test set
yhat = model.predict_classes(X_test)
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

运行该示例首先在训练数据集上拟合模型，然后在测试数据集上报告分类精确率。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

在这种情况下，我们可以看到该模型比无技能模型表现得更好，假设准确率在 73.5%以上。

```py
Accuracy: 0.765
```

然后创建列车和测试集上的损耗线图。

我们可以看到，模型很快在数据集上找到了一个很好的拟合，并且看起来没有过度拟合或拟合不足。

![Learning Curves of Simple Multilayer Perceptron on Cancer Survival Dataset](img/4520a0264ac7035bbb20444496ffe4b9.png)

癌症存活数据集上简单多层感知器的学习曲线

现在，我们已经对数据集上的简单 MLP 模型的学习动态有了一些了解，我们可以考虑对数据集上的模型表现进行更稳健的评估。

## 稳健模型评估

k 倍交叉验证程序可以提供更可靠的 MLP 表现估计，尽管它可能非常慢。

这是因为 k 模型必须被拟合和评估。当数据集规模较小时，例如癌症存活数据集，这不是问题。

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

我们可以使用这个框架，利用我们的基本配置，甚至利用一系列不同的数据准备、模型架构和学习配置，来开发 MLP 模型表现的可靠估计。

重要的是，在使用 k-fold 交叉验证来估计表现之前，我们首先了解了上一节中模型在数据集上的学习动态。如果我们开始直接调整模型，我们可能会得到好的结果，但如果没有，我们可能不知道为什么，例如，模型过度或拟合不足。

如果我们再次对模型进行大的更改，最好返回并确认模型正在适当收敛。

下面列出了评估前一节中的基本 MLP 模型的框架的完整示例。

```py
# k-fold cross-validation of base model for the haberman dataset
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
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(10, random_state=1)
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
	model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
	# predict test set
	yhat = model.predict_classes(X_test)
	# evaluate predictions
	score = accuracy_score(y_test, yhat)
	print('>%.3f' % score)
	scores.append(score)
# summarize all scores
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会报告评估程序每次迭代的模型表现，并在运行结束时报告分类精确率的平均值和标准偏差。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

在这种情况下，我们可以看到 MLP 模型获得了大约 75.2%的平均精确率，这与我们在前面部分中的粗略估计非常接近。

这证实了我们的预期，即对于这个数据集，基本模型配置可能比简单模型工作得更好

```py
>0.742
>0.774
>0.774
>0.806
>0.742
>0.710
>0.767
>0.800
>0.767
>0.633
Mean Accuracy: 0.752 (0.048)
```

这是个好结果吗？

事实上，这是一个具有挑战性的分类问题，达到 74.5%以上的分数是好的。

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
row = [30,64,1]
```

注意:我从数据集的第一行提取了这一行，预期的标签是“1”。

然后我们可以做一个预测。

```py
...
# make prediction
yhat = model.predict_classes([row])
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

将所有这些结合起来，下面列出了为 haberman 数据集拟合最终模型并使用它对新数据进行预测的完整示例。

```py
# fit a final model and make predictions on new data for the haberman dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
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
model.fit(X, y, epochs=200, batch_size=16, verbose=0)
# define a row of new data
row = [30,64,1]
# make prediction
yhat = model.predict_classes([row])
# invert transform to get label for class
yhat = le.inverse_transform(yhat)
# report prediction
print('Predicted: %s' % (yhat[0]))
```

运行该示例使模型适合整个数据集，并对单行新数据进行预测。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

在这种情况下，我们可以看到模型为输入行预测了一个“1”标签。

```py
Predicted: 1
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何建立乳腺癌患者存活概率模型](https://machinelearningmastery.com/how-to-develop-a-probabilistic-model-of-breast-cancer-patient-survival/)
*   [如何开发预测电离层扰动的神经网络](https://machinelearningmastery.com/predicting-disturbances-in-the-ionosphere/)
*   [标准机器学习数据集的最佳结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [TensorFlow 2 教程:使用 tf.keras 开始深度学习](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)
*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

## 摘要

在本教程中，您发现了如何为癌症存活二元分类数据集开发多层感知器神经网络模型。

具体来说，您了解到:

*   如何加载和总结癌症存活数据集，并使用结果建议数据准备和模型配置使用。
*   如何探索数据集上简单 MLP 模型的学习动态。
*   如何对模型表现进行稳健的估计，调整模型表现并对新数据进行预测。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。