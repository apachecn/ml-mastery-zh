# 在 Python 中使用 Keras 深度学习模型和 Scikit-Learn

> 原文： [https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/](https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/)

Keras 是用于研究和开发的 Python 中最受欢迎的深度学习库之一，因为它简单易用。

[scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/) 库是 Python 中一般机器学习最受欢迎的库。

在这篇文章中，您将了解如何使用 Keras 中的深度学习模型和 Python 中的 scikit-learn 库。

这将允许您利用 scikit-learn 库的功能来完成模型评估和模型超参数优化等任务。

让我们开始吧。

*   **更新**：有关使用 Keras 调整超参数的更大示例，请参阅帖子：
    *   [如何使用 Keras 网格搜索 Python 中的深度学习模型的超参数](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
*   **2016 年 10 月更新**：更新了 Keras 1.1.0 和 scikit-learn v0.18 的示例。
*   **2017 年 1 月更新**：修正了打印网格搜索结果的错误。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Use Keras Deep Learning Models with Scikit-Learn in Python](img/7a15249820eaff950ac0e36a690b5b32.png)

使用 Keras 深度学习模型与 Scikit-Learn 在 Python
照片由 [Alan Levine](https://www.flickr.com/photos/cogdog/7519589420/) ，保留一些权利。

## 概观

Keras 是一个用于 Python 深度学习的流行库，但该库的重点是深度学习。事实上，它致力于极简主义，只关注您需要快速简单地定义和构建深度学习模型。

Python 中的 scikit-learn 库建立在 SciPy 堆栈之上，用于高效的数值计算。它是一个功能齐全的通用机器学习库，提供许多有助于开发深度学习模型的实用程序。不少于：

*   使用重新取样方法（如 k-fold 交叉验证）评估模型。
*   高效搜索和评估模型超参数。

Keras 库为深度学习模型提供了一个方便的包装器，可用作 scikit-learn 中的分类或回归估计器。

在接下来的部分中，我们将介绍使用 KerasClassifier 包装器的示例，该包装器用于在 Keras 中创建并用于 scikit-learn 库的分类神经网络。

测试问题是[皮马印第安人发病的糖尿病分类数据集](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)。这是一个包含所有数字属性的小型数据集，易于使用。 [下载数据集](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)并将其直接放在您当前正在使用的名称 **pima-indians-diabetes.csv** （更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

以下示例假设您已成功安装 Keras 和 scikit-learn。

## 使用交叉验证评估深度学习模型

Keras 中的 KerasClassifier 和 KerasRegressor 类接受一个参数 **build_fn** ，它是要调用以获取模型的函数的名称。

您必须定义一个名为您定义模型的函数，编译并返回它。

在下面的示例中，我们定义了一个函数 **create_model（）**，它为问题创建了一个简单的多层神经网络。

我们通过 **build_fn** 参数将此函数名称传递给 KerasClassifier 类。我们还传递了 **nb_epoch = 150** 和 **batch_size = 10** 的其他参数。它们会自动捆绑并传递给 **fit（）**函数，该函数由 KerasClassifier 类在内部调用。

在这个例子中，我们使用 scikit-learn [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html) 来执行 10 倍分层交叉验证。这是一种重采样技术，可以提供对机器学习模型在看不见的数据上的表现的可靠估计。

我们使用 scikit-learn 函数 **cross_val_score（）**来使用交叉验证方案评估我们的模型并打印结果。

```py
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

运行该示例显示每个迭代的模型技能。创建和评估总共 10 个模型，并显示最终的平均精度。

```py
0.646838691487
```

## 网格搜索深度学习模型参数

前面的例子展示了从 Keras 包装深度学习模型并将其用于 scikit-learn 库的函数是多么容易。

在这个例子中，我们更进一步。在创建 KerasClassifier 包装器时，我们为 **build_fn** 参数指定的函数可以使用参数。我们可以使用这些参数来进一步自定义模型的构造。另外，我们知道我们可以为 **fit（）**函数提供参数。

在此示例中，我们使用网格搜索来评估神经网络模型的不同配置，并报告提供最佳估计表现的组合。

**create_model（）**函数被定义为采用两个参数 optimizer 和 init，两者都必须具有默认值。这将允许我们评估为我们的网络使用不同的优化算法和权重初始化方案的效果。

创建模型后，我们为要搜索的参数定义值数组，具体如下：

*   用于搜索不同重量值的优化器。
*   用于使用不同方案准备网络权重的初始化器。
*   用于训练模型的时期，用于对训练数据集进行不同次数的曝光。
*   用于在重量更新之前改变样本数量的批次。

选项被指定到字典中并传递给 [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) scikit-learn 类的配置。该类将针对每个参数组合评估我们的神经网络模型的版本（对于优化器，初始化，时期和批次的组合，2 x 3 x 3 x 3）。然后使用默认的 3 倍分层交叉验证评估每种组合。

这是很多模型和大量的计算。这不是一个你想要轻松使用的方案，因为它需要时间。您可以使用较小的数据子集设计小型实验，这些实验将在合理的时间内完成。在这种情况下，这是合理的，因为网络较小且数据集较小（少于 1000 个实例和 9 个属性）。

最后，显示最佳模型的表现和配置组合，然后显示所有参数组合的表现。

```py
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
```

在 CPU（而不是 CPU）上执行的工作站上可能需要大约 5 分钟才能完成。运行该示例显示以下结果。

我们可以看到，网格搜索发现使用统一初始化方案，rmsprop 优化器，150 个迭代和 5 个批量大小在此问题上实现了大约 75％的最佳交叉验证分数。

```py
Best: 0.752604 using {'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 150}
0.707031 (0.025315) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 50}
0.589844 (0.147095) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 50}
0.701823 (0.006639) with: {'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 50}
0.714844 (0.019401) with: {'init': 'normal', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 50}
0.718750 (0.016573) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 50}
0.688802 (0.032578) with: {'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 50}
0.657552 (0.075566) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 100}
0.696615 (0.026557) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 100}
0.727865 (0.022402) with: {'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 100}
0.736979 (0.030647) with: {'init': 'normal', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 100}
0.739583 (0.029635) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 100}
0.717448 (0.012075) with: {'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 100}
0.692708 (0.036690) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 150}
0.697917 (0.028940) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 150}
0.727865 (0.030647) with: {'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 150}
0.747396 (0.016053) with: {'init': 'normal', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 150}
0.729167 (0.007366) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'batch_size': 5, 'epochs': 150}
0.752604 (0.017566) with: {'init': 'uniform', 'optimizer': 'adam', 'batch_size': 5, 'epochs': 150}
0.662760 (0.035132) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'batch_size': 10, 'epochs': 50}
...
```

## 摘要

在这篇文章中，您了解了如何包装 Keras 深度学习模型并在 scikit-learn 通用机器学习库中使用它们。

您可以看到，使用 scikit-learn 进行标准机器学习操作（如模型评估和模型超参数优化）可以节省大量时间来自行实施这些方案。

包装模型允许您利用 scikit 中的强大工具学习，使您的深度学习模型适合您的一般机器学习过程。

您是否有任何关于在 scikit-learn 或此帖子中使用 Keras 模型的问题？在评论中提出您的问题，我会尽力回答。