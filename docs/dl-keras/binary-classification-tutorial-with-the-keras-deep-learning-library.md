# Keras 深度学习库的二分类教程

> 原文： [https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/](https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/)

Keras 是一个用于深度学习的 Python 库，它包含了高效的数值库 TensorFlow 和 Theano。

Keras 允许您快速简单地设计和训练神经网络和深度学习模型。

在这篇文章中，您将逐步完成二分类项目并了解如何在机器学习项目中高效使用 Keras 库。

完成本教程后，您将了解：

*   如何加载训练数据并将其提供给 Keras。
*   如何设计和训练表格数据形式的神经网络。
*   如何评估 Keras 神经网络模型在不可见的数据上的表现。
*   如何在使用神经网络时进行数据预处理以提升模型表现。
*   如何调整 Keras 中神经网络的拓扑和配置。

让我们现在开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![Binary Classification Worked Example with the Keras Deep Learning Library](img/813384c9d73f15abe34aa5f55bd5ddfa.png)


图片由[Mattia Merlo](https://www.flickr.com/photos/h_crimson/9405280189/) 提供，并保留所属权利。

## 1.数据集的描述

我们将在本教程中使用的数据集是 [Sonar 数据集](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))。

这是一个描述声纳啁啾返回弹跳不同服务的数据集，60 个输入变量是不同角度的回报强度。这是一个二分类问题，需要构建一个模型来区分岩石和金属。

您可以在 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))上了解有关此数据集的更多信息。您可以[免费下载数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)并将其放在工作目录中，文件名为 sonar.csv。

这是一个众所周知的数据集。所有变量都是连续的，通常在 0 到 1 的范围内。输出变量是代表矿石的字符“M”和代表岩石的字符“R”

使用此数据集的好处是它是一个标准的基准问题，这意味着我们对构建一个较好模型的预期技巧有所了解。使用交叉验证，神经网络[应该能够达到 84％左右的表现](http://www.is.umk.pl/projects/datasets.html#Sonar)，定制模型的精确度上限约为 88％。

## 2.基线神经网络模型表现

让我们为这个问题创建一个基线模型和结果。

我们将首先导入我们需要的所有类和函数。

```py
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

接下来，我们可以初始化随机数生成器，以确保在执行此代码时始终获得相同的结果。如果我们正在调试模型，这将对我们有所帮助。

```py
# 固定随机种子的在线性
seed = 7
numpy.random.seed(seed)
```

现在我们可以使用 [pandas](http://pandas.pydata.org/) 加载数据集，并将列拆分为 60 个输入变量（X）和 1 个输出变量（Y），我们使用 pandas 来加载数据，因为它可以轻松处理字符串（输出变量），如果尝试使用 NumPy 直接加载数据会比较困难。

```py
# 加载数据集
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# 将数据集分割为输入变量和输出变量
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

输出变量是字符串值，我们必须将它们转换为整数值 0 和 1。

我们可以使用 scikit-learn 中的 LabelEncoder 类来完成此操作。此类将通过 `fit()`函数使用整个数据集对所需的编码进行建模，然后应用编码以使用`transform()`函数创建新的输出变量。

```py
# 转换字符编码
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
```

我们现在准备使用 Keras 创建我们的神经网络模型。

我们将使用 scikit-learn 来使用分层 k 折交叉验证来评估模型，这是一种重采样技术，可以提供模型表现的估计。它通过将数据集分成 k 个部分来实现这一点，除了作为测试集的一部分用以评估模型的表现外，在其他所有训练数据上训练模型，该过程重复 k 次，并且所有构建的模型的平均分数被用作模型表现的稳健估计，因为它是分层的设计，这意味着它将查看输出值并尝试平衡属于数据 k 分裂中每个类的实例数。

要将 Keras 模型与 scikit-learn 一起使用，我们必须使用 KerasClassifier封装器。该类采用创建并返回神经网络模型的函数，将参数传递给 `fit()`调用，例如迭代数和批量大小。

让我们从定义创建基线模型的函数开始,我们的模型将具有单个完全连接的隐藏层，其具有与输入变量相同数量的神经元,这是创建神经网络时的一个很好的默认起点。

使用较小的高斯随机数初始化权重参数，使用整流器激活函数，输出层包含单个神经元以做出预测，它使用 `sigmod()`激活函数，以产生 0 到 1 范围内的概率输出，可以轻松地转换为清晰的类值（如0和1）。

最后，我们在训练期间使用对数损失函数（binary_crossentropy），这是二分类问题的首选损失函数。该模型还使用高效的 Adam 优化算法进行梯度下降，并在训练模型时收集精度度量指标。

```py
# 基线模型
def create_baseline():
	# 创建模型
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

现在可以在在 scikit-learn 框架中使用分层交叉验证来评估这个模型。

我们再次使用合理的默认值将训练时期的数量传递给 KerasClassifier，设定00模型将被创建 10 次以进行 10 次交叉验证，并且关闭详细输出选项（即参数`verbose = 0`）。

```py
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此代码将生成以下输出，显示模型在不可见数据上的估计精度的平均值和标准差。

```py
Baseline: 81.68% (7.26%)
```

这是一个很好的分数，没有做任何艰苦的工作。

## 3.使用数据预处理并重新运行基线模型

在建模之前进行数据预处理是一种很好的做法。

神经网络模型尤其适用于在数据规模和分布方面具有一致性的输入值。

建立神经网络模型时表格数据的有效数据chuli1方案是标准化，这是重新调整数据，使得数据每个属性的平均值为 0，标准偏差为 1.这种处理方法保留了高斯和高斯类分布，同时规范了数据每个属性的中心趋势。

我们可以使用 scikit-learn 使用 [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 类来执行 Sonar 数据集的标准化。

优化算法的方法是在交叉验证运行的过程中对训练数据进行标准化处理，并使用经过训练的标准化数据来预处理“不可见”的测试折叠，而不是对整个数据集执行标准化。这使得标准化成为交叉验证过程中模型预处理的一个必要步骤，并且能够阻止算法在评估期间具有从数据预处理过程中传递的“不可见”信息，如更清晰的分布。
我们可以使用 [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 在 scikit-learn 中实现这一点，此管道是一个包装器，它在交叉验证过程的传递中执行一个或多个模型。在这里，我们可以使用 StandardScaler 定义管道，然后将其应用于我们的神经网络模型。

```py
# 使用标准数据集评估基线模型
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此示例结果如下所示，我们确实看到精度值有一个较小的提升。

```py
Standardized: 84.56% (5.74%)
```

## 4.调整模型中的层和神经元数量

在神经网络上需要设置很多东西，例如权重初始化，激活函数，优化过程等。

具有超大的影响方面的的网络本身被称为网络拓扑，在本节中，我们将看一下关于网络结构的两个实验：将其缩小和将其放大。

这些实验能够在调节神经网络的问题上给予你帮助。

### 4.1. 评估较小的网络

我认为描述声纳数据集的输入变量有较大的冗余。

数据描述了来自不同角度的相同信号。也许其中一些角度比其他角度更有意义。我们可以通过限制第一个隐藏层中的表示空间来强制网络进行特征提取。

在这个实验中，我们将采用隐藏层中有 60 个神经元的基线模型，并将其减少一半到 30 个。这将在训练期间对网络施加压力，以挑选输入数据中最重要的结构进行建模。

我们还将在数据准备的前一个实验中对数据进行标准化，并尝试利用数据标准化后表现较小的提升。

```py
# 更小的模型
def create_smaller():
	# 创建模型
	model = Sequential()
	model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此示例结果如下所示，我们可以看到，我们对平均估计精度有一个非常小的提高，并且模型的精度分数的标准偏差（平均分布）显着降低。

这是一个很好的结果，因为我们在网络规模减半的情况下做得更好一些，而这仅需要一半的时间来训练。

```py
Smaller: 86.04% (4.00%)
```

### 4.2. 评估更大的网络

具有更多层的神经网络拓扑为网络提供了更多机会来提取关键特征并以更有用的非线性方式重新组合它们。

我们可以通过对用于创建模型的函数进行另一个小的调整来评估是否向网络添加更多层以轻松地改善表现。在这里，我们向网络添加一个新层（一行），即也就是在第一个隐藏层之后引入另一个隐藏层，其中包含 30 个神经元。

我们的网络现在具有拓扑结构：

```py
60 inputs -> [60 -> 30] -> 1 output
```

这里的方法是，网络有机会在瓶颈之前对所有输入变量进行建模，并被强制将表示能力减半，就像我们在上书的缩小网络拓扑的实验中所做的那样。

我们有一个额外的隐藏层来实现这个过程，而不是压缩输入层本身的表示。

```py
# 更大的模型
def create_larger():
	# 创建模型
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行此示例将生成以下结果。我们可以看到，我们在模型表现方面并没有得到提升，这可能是统计噪音或需要进一步训练的迹象。

```py
Larger: 83.14% (4.52%)
```

通过进一步调整诸如优化算法和训练时期的数量之类的方面，预期结果可以进一步提升。您可以在此数据集上获得的最佳分数是多少？

## 摘要

在这篇文章中，您了解了 Python 中的 Keras 深度学习库。

您了解了如何使用 Keras 逐步完成二分类问题，具体如下：

*   如何加载和准备在 Keras 中使用的数据。
*   如何创建基线神经网络模型。
*   如何使用 scikit-learn 和分层 k 折交叉验证来评估 Keras 模型。
*   数据预处理方案如何提升模型的表现。
*   调整网络拓扑的实验如何提升模型表现。

您对 Keras 的深度学习或此帖子有任何疑问吗？在评论中提出您的问题，我会尽力回答。