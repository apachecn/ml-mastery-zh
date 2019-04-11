# 如何使用 Keras 在 Python 中网格搜索深度学习模型的超参数

> 原文： [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

超参数优化是深度学习的重要组成部分。

原因是神奇网络众所周知难以配置，并且需要设置许多参数。最重要的是，单个模型的训练速度可能非常慢。

在这篇文章中，您将了解如何使用 scikit-learn python 机器学习库中的网格搜索功能来调整 Keras 深度学习模型的超参数。

阅读这篇文章后你会知道：

*   如何包装 Keras 模型用于 scikit-learn 以及如何使用网格搜索。
*   如何网格搜索常见的神经网络参数，如学习率，dropout率，时期和神经元数量。
*   如何在自己的项目中定义自己的超参数调整实验。

让我们开始吧。

*   **2016 年 11 月更新**：修复了在代码示例中显示网格搜索结果的小问题。
*   **2016 年 10 月更新**：更新了 Keras 1.1.0，TensorFlow 0.10.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **2017 年 9 月更新**：更新了使用 Keras 2“epochs”代替 Keras 1“nb_epochs”的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](img/4ae90ba8e51222dd47048dad32a96128.png)

如何使用 Keras 网格搜索 Python 中深度学习模型的超参数
照片由 [3V Photo](https://www.flickr.com/photos/107439982@N02/10635372184/) ，保留一些权利。

## 概观

在这篇文章中，我想向您展示如何使用 scikit-learn 网格搜索功能，并为您提供一组示例，您可以将这些示例复制并粘贴到您自己的项目中作为起点。

以下是我们将要讨论的主题列表：

1.  如何在 scikit-learn 中使用 Keras 模型。
2.  如何在 scikit-learn 中使用网格搜索。
3.  如何调整批量大小和训练时期。
4.  如何调整优化算法。
5.  如何调整学习率和动力。
6.  如何调整网络权重初始化。
7.  如何调整激活功能。
8.  如何调整退出正则化。
9.  如何调整隐藏层中的神经元数量。

## 如何在 scikit-learn 中使用 Keras 模型

Keras 模型可以通过使用 **KerasClassifier** 或 **KerasRegressor** 类包装来用于 scikit-learn。

要使用这些包装器，您必须定义一个创建并返回 Keras 顺序模型的函数，然后在构造 **KerasClassifier** 类时将此函数传递给 **build_fn** 参数。

例如：

```py
def create_model():
	...
	return model

model = KerasClassifier(build_fn=create_model)
```

**KerasClassifier** 类的构造函数可以使用传递给 **model.fit（）**的调用的默认参数，例如迭代数和批量大小。

例如：

```py
def create_model():
	...
	return model

model = KerasClassifier(build_fn=create_model, epochs=10)
```

**KerasClassifier** 类的构造函数也可以获取可以传递给自定义 **create_model（）**函数的新参数。这些新参数也必须在 **create_model（）**函数的签名中使用默认参数进行定义。

例如：

```py
def create_model(dropout_rate=0.0):
	...
	return model

model = KerasClassifier(build_fn=create_model, dropout_rate=0.2)
```

您可以在 Keras API 文档中了解有关 [scikit-learn 包装器的更多信息。](http://keras.io/scikit-learn-api/)

## 如何在 scikit-learn 中使用网格搜索

网格搜索是一种模型超参数优化技术。

在 scikit-learn 中，这种技术在 **GridSearchCV** 类中提供。

构造此类时，必须提供一个超参数字典，以便在 **param_grid** 参数中进行评估。这是模型参数名称的映射和要尝试的值数组。

默认情况下，精度是优化的分数，但其他分数可以在 **GridSearchCV** 构造函数的**得分**参数中指定。

默认情况下，网格搜索仅使用一个线程。通过将 **GridSearchCV** 构造函数中的 **n_jobs** 参数设置为-1，该进程将使用计算机上的所有核心。根据您的 Keras 后端，这可能会干扰主要的神经网络训练过程。

然后， **GridSearchCV** 过程将为每个参数组合构建和评估一个模型。交叉验证用于评估每个单独的模型，并使用默认的 3 倍交叉验证，尽管可以通过指定 **GridSearchCV** 构造函数的 **cv** 参数来覆盖它。

下面是定义简单网格搜索的示例：

```py
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
```

完成后，您可以在 **grid.fit（）**返回的结果对象中访问网格搜索的结果。 **best_score_** 成员提供对优化过程中观察到的最佳分数的访问， **best_params_** 描述了获得最佳结果的参数组合。

您可以在 scikit-learn API 文档中了解有关 [GridSearchCV 类的更多信息。](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV)

## 问题描述

既然我们知道如何使用 scras 模型学习 keras 模型以及如何在 scikit-learn 中使用网格搜索，那么让我们看看一堆例子。

所有例子都将在一个名为 [Pima Indians 糖尿病分类数据集](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)的小型标准机器学习数据集上进行演示。这是一个包含所有数字属性的小型数据集，易于使用。

1.  [下载数据集](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)并将其直接放入您当前正在使用的名称 **pima-indians-diabetes.csv** （更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

在我们继续本文中的示例时，我们将汇总最佳参数。这不是网格搜索的最佳方式，因为参数可以交互，但它有利于演示目的。

### 并行化网格搜索的注意事项

所有示例都配置为使用并行性（ **n_jobs = -1** ）。

如果您收到如下错误：

```py
INFO (theano.gof.compilelock): Waiting for existing lock by process '55614' (I am process '55613')
INFO (theano.gof.compilelock): To manually release the lock, delete ...
```

终止进程并更改代码以不并行执行网格搜索，设置 **n_jobs = 1** 。

## 如何调整批量大小和时期数量

在第一个简单的例子中，我们考虑调整批量大小和适合网络时使用的时期数。

[迭代梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method)中的批量大小是在更新权重之前向网络显示的模式数。它也是网络训练的优化，定义了一次读取多少个模式并保留在内存中。

时期数是训练期间整个训练数据集显示给网络的次数。一些网络对批量大小敏感，例如 LSTM 递归神经网络和卷积神经网络。

在这里，我们将评估一套不同的迷你批量大小，从 10 到 100，步长为 20。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.686198 using {'epochs': 100, 'batch_size': 20}
0.348958 (0.024774) with: {'epochs': 10, 'batch_size': 10}
0.348958 (0.024774) with: {'epochs': 50, 'batch_size': 10}
0.466146 (0.149269) with: {'epochs': 100, 'batch_size': 10}
0.647135 (0.021236) with: {'epochs': 10, 'batch_size': 20}
0.660156 (0.014616) with: {'epochs': 50, 'batch_size': 20}
0.686198 (0.024774) with: {'epochs': 100, 'batch_size': 20}
0.489583 (0.075566) with: {'epochs': 10, 'batch_size': 40}
0.652344 (0.019918) with: {'epochs': 50, 'batch_size': 40}
0.654948 (0.027866) with: {'epochs': 100, 'batch_size': 40}
0.518229 (0.032264) with: {'epochs': 10, 'batch_size': 60}
0.605469 (0.052213) with: {'epochs': 50, 'batch_size': 60}
0.665365 (0.004872) with: {'epochs': 100, 'batch_size': 60}
0.537760 (0.143537) with: {'epochs': 10, 'batch_size': 80}
0.591146 (0.094954) with: {'epochs': 50, 'batch_size': 80}
0.658854 (0.054904) with: {'epochs': 100, 'batch_size': 80}
0.402344 (0.107735) with: {'epochs': 10, 'batch_size': 100}
0.652344 (0.033299) with: {'epochs': 50, 'batch_size': 100}
0.542969 (0.157934) with: {'epochs': 100, 'batch_size': 100}
```

我们可以看到，20 和 100 个时期的批量大小达到了 68％准确度的最佳结果。

## 如何调整训练优化算法

Keras 提供一套不同的最先进的优化算法。

在此示例中，我们调整用于训练网络的优化算法，每个算法都使用默认参数。

这是一个奇怪的例子，因为通常您会先选择一种方法，而是专注于调整问题的参数（例如，参见下一个例子）。

在这里，我们将评估 Keras API 支持的[优化算法套件。](http://keras.io/optimizers/)

完整的代码清单如下。

```py
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.704427 using {'optimizer': 'Adam'}
0.348958 (0.024774) with: {'optimizer': 'SGD'}
0.348958 (0.024774) with: {'optimizer': 'RMSprop'}
0.471354 (0.156586) with: {'optimizer': 'Adagrad'}
0.669271 (0.029635) with: {'optimizer': 'Adadelta'}
0.704427 (0.031466) with: {'optimizer': 'Adam'}
0.682292 (0.016367) with: {'optimizer': 'Adamax'}
0.703125 (0.003189) with: {'optimizer': 'Nadam'}
```

结果表明 ADAM 优化算法是最好的，准确度大约为 70％。

## 如何调整学习率和动量

通常预先选择优化算法来训练您的网络并调整其参数。

到目前为止，最常见的优化算法是普通的老式[随机梯度下降](http://keras.io/optimizers/#sgd)（SGD），因为它非常清楚。在这个例子中，我们将研究优化 SGD 学习率和动量参数。

学习率控制在每批结束时更新权重的程度，并且动量控制让先前更新影响当前重量更新的程度。

我们将尝试一套小的标准学习率和 0.2 到 0.8 的动量值，步长为 0.2，以及 0.9（因为它在实践中可能是一个受欢迎的价值）。

通常，在这样的优化中也包括时期的数量是个好主意，因为每批学习量（学习率），每个时期的更新数量（批量大小）和数量之间存在依赖关系。时代。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the learning rate and momentum
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	optimizer = SGD(lr=learn_rate, momentum=momentum)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.680990 using {'learn_rate': 0.01, 'momentum': 0.0}
0.348958 (0.024774) with: {'learn_rate': 0.001, 'momentum': 0.0}
0.348958 (0.024774) with: {'learn_rate': 0.001, 'momentum': 0.2}
0.467448 (0.151098) with: {'learn_rate': 0.001, 'momentum': 0.4}
0.662760 (0.012075) with: {'learn_rate': 0.001, 'momentum': 0.6}
0.669271 (0.030647) with: {'learn_rate': 0.001, 'momentum': 0.8}
0.666667 (0.035564) with: {'learn_rate': 0.001, 'momentum': 0.9}
0.680990 (0.024360) with: {'learn_rate': 0.01, 'momentum': 0.0}
0.677083 (0.026557) with: {'learn_rate': 0.01, 'momentum': 0.2}
0.427083 (0.134575) with: {'learn_rate': 0.01, 'momentum': 0.4}
0.427083 (0.134575) with: {'learn_rate': 0.01, 'momentum': 0.6}
0.544271 (0.146518) with: {'learn_rate': 0.01, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.01, 'momentum': 0.9}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.0}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.2}
0.572917 (0.134575) with: {'learn_rate': 0.1, 'momentum': 0.4}
0.572917 (0.134575) with: {'learn_rate': 0.1, 'momentum': 0.6}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.1, 'momentum': 0.9}
0.533854 (0.149269) with: {'learn_rate': 0.2, 'momentum': 0.0}
0.427083 (0.134575) with: {'learn_rate': 0.2, 'momentum': 0.2}
0.427083 (0.134575) with: {'learn_rate': 0.2, 'momentum': 0.4}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.6}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.8}
0.651042 (0.024774) with: {'learn_rate': 0.2, 'momentum': 0.9}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.0}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.2}
0.455729 (0.146518) with: {'learn_rate': 0.3, 'momentum': 0.4}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.6}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.8}
0.348958 (0.024774) with: {'learn_rate': 0.3, 'momentum': 0.9}
```

我们可以看到相对 SGD 在这个问题上不是很好，但是使用 0.01 的学习率和 0.0 的动量以及约 68％的准确度获得了最佳结果。

## 如何调整网络权重初始化

神经网络权重初始化过去很简单：使用小的随机值。

现在有一套不同的技术可供选择。 [Keras 提供清单](http://keras.io/initializations/)。

在此示例中，我们将通过评估所有可用技术来调整网络权重初始化的选择。

我们将在每一层使用相同的权重初始化方法。理想情况下，根据每层使用的激活函数，使用不同的权重初始化方案可能更好。在下面的示例中，我们使用整流器作为隐藏层。我们使用 sigmoid 作为输出层，因为预测是二进制的。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the weight initialization
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(init_mode='uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init_mode, activation='relu'))
	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.720052 using {'init_mode': 'uniform'}
0.720052 (0.024360) with: {'init_mode': 'uniform'}
0.348958 (0.024774) with: {'init_mode': 'lecun_uniform'}
0.712240 (0.012075) with: {'init_mode': 'normal'}
0.651042 (0.024774) with: {'init_mode': 'zero'}
0.700521 (0.010253) with: {'init_mode': 'glorot_normal'}
0.674479 (0.011201) with: {'init_mode': 'glorot_uniform'}
0.661458 (0.028940) with: {'init_mode': 'he_normal'}
0.678385 (0.004872) with: {'init_mode': 'he_uniform'}
```

我们可以看到，使用均匀重量初始化方案实现了最佳结果，实现了约 72％的表现。

## 如何调整神经元激活功能

激活功能控制各个神经元的非线性以及何时触发。

通常，整流器激活功能是最流行的，但它曾经是 sigmoid 和 tanh 功能，这些功能可能仍然更适合于不同的问题。

在这个例子中，我们将评估 Keras 中可用的[不同激活函数套件。我们将仅在隐藏层中使用这些函数，因为我们在输出中需要 sigmoid 激活函数以用于二元分类问题。](http://keras.io/activations/)

通常，将数据准备到不同传递函数的范围是一个好主意，在这种情况下我们不会这样做。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the activation function
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(activation='relu'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation=activation))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.722656 using {'activation': 'linear'}
0.649740 (0.009744) with: {'activation': 'softmax'}
0.720052 (0.032106) with: {'activation': 'softplus'}
0.688802 (0.019225) with: {'activation': 'softsign'}
0.720052 (0.018136) with: {'activation': 'relu'}
0.691406 (0.019401) with: {'activation': 'tanh'}
0.680990 (0.009207) with: {'activation': 'sigmoid'}
0.691406 (0.014616) with: {'activation': 'hard_sigmoid'}
0.722656 (0.003189) with: {'activation': 'linear'}
```

令人惊讶的是（至少对我而言），“线性”激活功能获得了最佳结果，准确度约为 72％。

## 如何调整dropout规范化

在这个例子中，我们将研究调整正则化的dropout率，以限制过度拟合并提高模型的推广能力。

为了获得良好的结果，dropout最好与权重约束相结合，例如最大范数约束。

有关在 Keras 深度学习模型中使用 dropout 的更多信息，请参阅帖子：

*   [具有 Keras 的深度学习模型中的丢失正则化](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

这涉及拟合dropout率和权重约束。我们将尝试 0.0 到 0.9 之间的丢失百分比（1.0 没有意义）和 0 到 5 之间的 maxnorm 权重约束值。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the dropout rate
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.723958 using {'dropout_rate': 0.2, 'weight_constraint': 4}
0.696615 (0.031948) with: {'dropout_rate': 0.0, 'weight_constraint': 1}
0.696615 (0.031948) with: {'dropout_rate': 0.0, 'weight_constraint': 2}
0.691406 (0.026107) with: {'dropout_rate': 0.0, 'weight_constraint': 3}
0.708333 (0.009744) with: {'dropout_rate': 0.0, 'weight_constraint': 4}
0.708333 (0.009744) with: {'dropout_rate': 0.0, 'weight_constraint': 5}
0.710937 (0.008438) with: {'dropout_rate': 0.1, 'weight_constraint': 1}
0.709635 (0.007366) with: {'dropout_rate': 0.1, 'weight_constraint': 2}
0.709635 (0.007366) with: {'dropout_rate': 0.1, 'weight_constraint': 3}
0.695312 (0.012758) with: {'dropout_rate': 0.1, 'weight_constraint': 4}
0.695312 (0.012758) with: {'dropout_rate': 0.1, 'weight_constraint': 5}
0.701823 (0.017566) with: {'dropout_rate': 0.2, 'weight_constraint': 1}
0.710938 (0.009568) with: {'dropout_rate': 0.2, 'weight_constraint': 2}
0.710938 (0.009568) with: {'dropout_rate': 0.2, 'weight_constraint': 3}
0.723958 (0.027126) with: {'dropout_rate': 0.2, 'weight_constraint': 4}
0.718750 (0.030425) with: {'dropout_rate': 0.2, 'weight_constraint': 5}
0.721354 (0.032734) with: {'dropout_rate': 0.3, 'weight_constraint': 1}
0.707031 (0.036782) with: {'dropout_rate': 0.3, 'weight_constraint': 2}
0.707031 (0.036782) with: {'dropout_rate': 0.3, 'weight_constraint': 3}
0.694010 (0.019225) with: {'dropout_rate': 0.3, 'weight_constraint': 4}
0.709635 (0.006639) with: {'dropout_rate': 0.3, 'weight_constraint': 5}
0.704427 (0.008027) with: {'dropout_rate': 0.4, 'weight_constraint': 1}
0.717448 (0.031304) with: {'dropout_rate': 0.4, 'weight_constraint': 2}
0.718750 (0.030425) with: {'dropout_rate': 0.4, 'weight_constraint': 3}
0.718750 (0.030425) with: {'dropout_rate': 0.4, 'weight_constraint': 4}
0.722656 (0.029232) with: {'dropout_rate': 0.4, 'weight_constraint': 5}
0.720052 (0.028940) with: {'dropout_rate': 0.5, 'weight_constraint': 1}
0.703125 (0.009568) with: {'dropout_rate': 0.5, 'weight_constraint': 2}
0.716146 (0.029635) with: {'dropout_rate': 0.5, 'weight_constraint': 3}
0.709635 (0.008027) with: {'dropout_rate': 0.5, 'weight_constraint': 4}
0.703125 (0.011500) with: {'dropout_rate': 0.5, 'weight_constraint': 5}
0.707031 (0.017758) with: {'dropout_rate': 0.6, 'weight_constraint': 1}
0.701823 (0.018688) with: {'dropout_rate': 0.6, 'weight_constraint': 2}
0.701823 (0.018688) with: {'dropout_rate': 0.6, 'weight_constraint': 3}
0.690104 (0.027498) with: {'dropout_rate': 0.6, 'weight_constraint': 4}
0.695313 (0.022326) with: {'dropout_rate': 0.6, 'weight_constraint': 5}
0.697917 (0.014382) with: {'dropout_rate': 0.7, 'weight_constraint': 1}
0.697917 (0.014382) with: {'dropout_rate': 0.7, 'weight_constraint': 2}
0.687500 (0.008438) with: {'dropout_rate': 0.7, 'weight_constraint': 3}
0.704427 (0.011201) with: {'dropout_rate': 0.7, 'weight_constraint': 4}
0.696615 (0.016367) with: {'dropout_rate': 0.7, 'weight_constraint': 5}
0.680990 (0.025780) with: {'dropout_rate': 0.8, 'weight_constraint': 1}
0.699219 (0.019401) with: {'dropout_rate': 0.8, 'weight_constraint': 2}
0.701823 (0.015733) with: {'dropout_rate': 0.8, 'weight_constraint': 3}
0.684896 (0.023510) with: {'dropout_rate': 0.8, 'weight_constraint': 4}
0.696615 (0.017566) with: {'dropout_rate': 0.8, 'weight_constraint': 5}
0.653646 (0.034104) with: {'dropout_rate': 0.9, 'weight_constraint': 1}
0.677083 (0.012075) with: {'dropout_rate': 0.9, 'weight_constraint': 2}
0.679688 (0.013902) with: {'dropout_rate': 0.9, 'weight_constraint': 3}
0.669271 (0.017566) with: {'dropout_rate': 0.9, 'weight_constraint': 4}
0.669271 (0.012075) with: {'dropout_rate': 0.9, 'weight_constraint': 5}
```

我们可以看到，20％的dropout率和 4 的最大权重约束导致最佳准确度约为 72％。

## 如何调整隐藏层中的神经元数量

层中神经元的数量是调整的重要参数。通常，层中的神经元的数量控制网络的表示能力，至少在拓扑中的那个点处。

此外，通常，足够大的单层网络可以近似于任何其他神经网络，[至少在理论上](https://en.wikipedia.org/wiki/Universal_approximation_theorem)。

在这个例子中，我们将研究调整单个隐藏层中的神经元数量。我们将以 5 的步长尝试 1 到 30 的值。

较大的网络需要更多的训练，并且至少批量大小和时期数应理想地用神经元的数量来优化。

完整的代码清单如下。

```py
# Use scikit-learn to grid search the number of neurons
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将生成以下输出。

```py
Best: 0.714844 using {'neurons': 5}
0.700521 (0.011201) with: {'neurons': 1}
0.714844 (0.011049) with: {'neurons': 5}
0.712240 (0.017566) with: {'neurons': 10}
0.705729 (0.003683) with: {'neurons': 15}
0.696615 (0.020752) with: {'neurons': 20}
0.713542 (0.025976) with: {'neurons': 25}
0.705729 (0.008027) with: {'neurons': 30}
```

我们可以看到，在隐藏层中具有 5 个神经元的网络实现了最佳结果，精度约为 71％。

## 超参数优化提示

本节列出了调整神经网络超参数时要考虑的一些方便提示。

*   **k 倍交叉验证**。您可以看到本文中示例的结果显示出一些差异。使用默认的交叉验证 3，但是 k = 5 或 k = 10 可能更稳定。仔细选择交叉验证配置以确保结果稳定。
*   **回顾整个网格**。不要只关注最佳结果，检查整个结果网格并寻找支持配置决策的趋势。
*   **并行化**。如果可以的话，使用你所有的核心，神经网络训练很慢，我们经常想尝试很多不同的参数。考虑搞砸很多 [AWS 实例](http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)。
*   **使用数据集样本**。因为网络训练很慢，所以尝试在训练数据集的较小样本上训练它们，只是为了了解参数的一般方向而不是最佳配置。
*   **从粗网格**开始。从粗粒度网格开始，一旦缩小范围，就可以缩放到更细粒度的网格。
*   **不转移结果**。结果通常是特定于问题的。尝试在您看到的每个新问题上避免喜欢的配置。您在一个问题上发现的最佳结果不太可能转移到您的下一个项目。而是寻找更广泛的趋势，例如层数或参数之间的关系。
*   **再现性是一个问题**。虽然我们在 NumPy 中为随机数生成器设置种子，但结果不是 100％可重复的。当网格搜索包裹 Keras 模型时，重复性要高于本文中提供的内容。

## 摘要

在这篇文章中，您了解了如何使用 Keras 和 scikit-learn 在 Python 中调整深度学习网络的超参数。

具体来说，你学到了：

*   如何包装 Keras 模型用于 scikit-learn 以及如何使用网格搜索。
*   如何为 Keras 模型网格搜索一套不同的标准神经网络参数。
*   如何设计自己的超参数优化实验。

你有调整大型神经网络超参数的经验吗？请在下面分享您的故事。

您对神经网络的超参数优化还是关于这篇文章有什么疑问？在评论中提出您的问题，我会尽力回答。