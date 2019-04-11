# Python 中的 Keras 深度学习库的回归教程

> 原文： [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)

Keras 是一个深度学习库，包含高效的数字库 Theano 和 TensorFlow。

在这篇文章中，您将了解如何使用 Keras 开发和评估神经网络模型以获得回归问题。

完成本分步教程后，您将了解：

*   如何加载 CSV 数据集并使其可供 Keras 使用。
*   如何使用 Keras 创建一个回归问题的神经网络模型。
*   如何使用 scras-learn 与 Keras 一起使用交叉验证来评估模型。
*   如何进行数据准备以提高 Keras 模型的技能。
*   如何使用 Keras 调整模型的网络拓扑。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。
*   **Update Apr / 2018** ：将 nb_epoch 参数更改为 epochs。

![Regression Tutorial with Keras Deep Learning Library in Python](img/4ae1e83ece36ea618e1ec2f6cbbbeb1f.png)

使用 Python 中的 Keras 深度学习库的回归教程
[Salim Fadhley](https://www.flickr.com/photos/salimfadhley/130295135/) 的照片，保留一些权利。

## 1.问题描述

我们将在本教程中看到的问题是[波士顿房价数据集](https://archive.ics.uci.edu/ml/datasets/Housing)。

您可以下载此数据集并将其保存到当前工作中，直接使用文件名 _housing.csv_ （更新：[从此处下载数据](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data)）。

该数据集描述了波士顿郊区房屋的 13 个数字属性，并涉及以数千美元模拟这些郊区房屋的价格。因此，这是回归预测建模问题。输入属性包括犯罪率，非经营业务占地比例，化学品浓度等。

这是机器学习中经过深入研究的问题。使用起来很方便，因为所有输入和输出属性都是数字的，并且有 506 个实例可供使用。

使用均方误差（MSE）评估的模型的合理表现约为 20 平方千美元（如果取平方根则为 4,500 美元）。这是一个很好的目标，旨在与我们的神经网络模型。

## 2.开发基线神经网络模型

在本节中，我们将为回归问题创建基线神经网络模型。

让我们从包含本教程所需的所有函数和对象开始。

```py
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

我们现在可以从本地目录中的文件加载数据集。

事实上，数据集在 UCI 机器学习库中不是 CSV 格式，而是用空格分隔属性。我们可以使用 pandas 库轻松加载它。然后我们可以分割输入（X）和输出（Y）属性，以便使用 Keras 和 scikit-learn 更容易建模。

```py
# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
```

我们可以使用 Keras 库提供的方便的包装器对象创建 Keras 模型并使用 scikit-learn 来评估它们。这是可取的，因为 scikit-learn 在评估模型方面表现优异，并且允许我们使用强大的数据准备和模型评估方案，只需很少的代码。

Keras 包装器需要一个函数作为参数。我们必须定义的这个函数负责创建要评估的神经网络模型。

下面我们定义用于创建要评估的基线模型的函数。它是一个简单的模型，它有一个完全连接的隐藏层，与输入属性具有相同数量的神经元（13）。网络使用良好的实践，例如隐藏层的整流器激活功能。没有激活函数用于输出层，因为它是回归问题，我们有兴趣直接预测数值而不进行变换。

使用有效的 ADAM 优化算法并且优化均方误差损失函数。这将与我们用于评估模型表现的指标相同。这是一个理想的指标，因为通过取平方根给出了一个误差值，我们可以在问题的背景下直接理解（数千美元）。

```py
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
```

用于 scikit-learn 作为回归估计器的 Keras 包装器对象称为 KerasRegressor。我们创建一个实例并将其传递给函数的名称以创建神经网络模型以及一些参数以便稍后传递给模型的 fit（）函数，例如时期数和批量大小。这两个都设置为合理的默认值。

我们还使用常量随机种子初始化随机数生成器，我们将为本教程中评估的每个模型重复该过程。这是为了确保我们一致地比较模型。

```py
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
```

最后一步是评估此基线模型。我们将使用 10 倍交叉验证来评估模型。

```py
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

运行此代码可以估算出模型在看不见的数据问题上的表现。结果报告了均方误差，包括交叉验证评估的所有 10 倍的平均值和标准偏差（平均方差）。

```py
Baseline: 31.64 (26.82) MSE
```

## 3.建模标准化数据集

波士顿房价数据集的一个重要问题是输入属性的尺度各不相同，因为它们测量的数量不同。

在使用神经网络模型对数据进行建模之前准备数据几乎总是好的做法。

继续上述基线模型，我们可以使用输入数据集的标准化版本重新评估相同的模型。

我们可以使用 scikit-learn 的 [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 框架在模型评估过程中，在交叉验证的每个折叠内执行标准化。这确保了每个测试集交叉验证折叠中没有数据泄漏到训练数据中。

下面的代码创建了一个 scikit-learn Pipeline，它首先标准化数据集，然后创建和评估基线神经网络模型。

```py
# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

运行该示例可提供比基线模型更高的表现，而无需标准化数据，从而丢弃错误。

```py
Standardized: 29.54 (27.87) MSE
```

此部分的进一步扩展将类似地对输出变量应用重新缩放，例如将其归一化到 0-1 的范围，并在输出层上使用 Sigmoid 或类似的激活函数，以将输出预测缩小到相同的范围。

## 4.调整神经网络拓扑

有许多问题可以针对神经网络模型进行优化。

也许最大的杠杆点是网络本身的结构，包括层数和每层神经元的数量。

在本节中，我们将评估另外两种网络拓扑，以进一步提高模型的表现。我们将研究更深入和更广泛的网络拓扑。

### 4.1。评估更深入的网络拓扑

提高神经网络表现的一种方法是添加更多层。这可能允许模型提取并重新组合数据中嵌入的高阶特征。

在本节中，我们将评估向模型添加一个隐藏层的效果。这就像定义一个新函数一样简单，这个函数将创建这个更深层次的模型，从上面的基线模型中复制出来。然后我们可以在第一个隐藏层之后插入一个新行。在这种情况下，神经元的数量约为一半。

```py
# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
```

我们的网络拓扑现在看起来像：

```py
13 inputs -> [13 -> 6] -> 1 output
```

我们可以采用与上述相同的方式评估此网络拓扑，同时还使用上面显示的数据集的标准化来提高表现。

```py
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

运行此模型确实表明表现从 28 降至 24,000 平方美元进一步改善。

```py
Larger: 22.83 (25.33) MSE
```

### 4.2。评估更广泛的网络拓扑

增加模型的表示能力的另一种方法是创建更广泛的网络。

在本节中，我们将评估保持浅层网络架构的效果，并使一个隐藏层中的神经元数量几乎翻倍。

同样，我们需要做的就是定义一个创建神经网络模型的新函数。在这里，与 13 到 20 的基线模型相比，我们增加了隐藏层中神经元的数量。

```py
# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
```

我们的网络拓扑现在看起来像：

```py
13 inputs -> [20] -> 1 output
```

我们可以使用与上面相同的方案评估更广泛的网络拓扑：

```py
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

建立模型的确看到误差进一步下降到大约 2.1 万平方美元。对于这个问题，这不是一个糟糕的结果。

```py
Wider: 21.64 (23.75) MSE
```

很难想象更广泛的网络在这个问题上会胜过更深层次的网络。结果证明了在开发神经网络模型时经验测试的重要性。

## 摘要

在这篇文章中，您发现了用于建模回归问题的 Keras 深度学习库。

通过本教程，您学习了如何开发和评估神经网络模型，包括：

*   如何加载数据和开发基线模型。
*   如何使用标准化等数据准备技术提升表现。
*   如何针对问题设计和评估具有不同变化拓扑的网络。

您对 Keras 深度学习库或这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。