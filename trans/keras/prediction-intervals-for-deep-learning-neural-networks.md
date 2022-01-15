# 深度学习神经网络的预测区间

> 原文:[https://machinelearning master . com/forecast-intervals-for-deep-learning-neural-networks/](https://machinelearningmastery.com/prediction-intervals-for-deep-learning-neural-networks/)

**预测间隔**为回归问题的预测提供了不确定性的度量。

例如，95%的预测间隔表示 100 次中的 95 次，真实值将落在范围的下限和上限之间。这不同于可能代表不确定区间中心的简单点预测。

在回归预测建模问题上，没有用于计算深度学习神经网络的预测区间的标准技术。然而，可以使用一组模型来估计快速且肮脏的预测区间，这些模型依次提供点预测的分布，从中可以计算区间。

在本教程中，您将发现如何计算深度学习神经网络的预测区间。

完成本教程后，您将知道:

*   预测区间提供了回归预测建模问题的不确定性度量。
*   如何在标准回归问题上开发和评估简单的多层感知器神经网络。
*   如何使用一组神经网络模型计算和报告预测区间。

我们开始吧。

![Prediction Intervals for Deep Learning Neural Networks](img/c765e5d591679da6d853a3b890133c29.png)

深度学习神经网络的预测间隔
摄影:尤金 _o ，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  预测数的变化范围
2.  回归的神经网络
3.  神经网络预测区间

## 预测数的变化范围

通常，回归问题的预测模型(即预测数值)进行点预测。

这意味着他们预测一个单一的值，但没有给出任何关于预测的不确定性的指示。

根据定义，预测是一种估计或近似，包含一些不确定性。不确定性来自模型本身的误差和输入数据中的噪声。该模型是输入变量和输出变量之间关系的近似。

预测区间是对预测不确定性的量化。

它为结果变量的估计提供了一个概率上下限。

> 单个未来观测值的预测区间是一个在特定置信度下包含从分布中随机选择的未来观测值的区间。

—第 27 页，[统计区间:从业者和研究者指南](https://amzn.to/2G8w3IL)，2017。

使用回归模型进行预测时，最常用的是预测间隔，其中预测的是数量。

预测区间围绕着模型做出的预测，并有望涵盖真实结果的范围。

有关一般预测间隔的更多信息，请参见教程:

*   [机器学习的预测间隔](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)

现在我们已经熟悉了什么是预测区间，我们可以考虑如何计算神经网络的区间。让我们首先定义一个回归问题和一个神经网络模型来解决它。

## 回归的神经网络

在本节中，我们将定义一个回归预测建模问题和一个神经网络模型来解决它。

首先，让我们介绍一个标准回归数据集。我们将使用房屋数据集。

外壳数据集是一个标准的机器学习数据集，包括 506 行数据，有 13 个数字输入变量和一个数字目标变量。

使用三次重复的重复分层 10 倍交叉验证的测试工具，一个简单的模型可以获得大约 6.6 的平均绝对误差(MAE)。一个[性能最好的模型](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)可以在同一个测试线束上达到 1.9 左右的 MAE。这提供了此数据集的预期性能范围。

该数据集包括预测美国波士顿郊区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例将数据集下载并加载为熊猫数据框，并总结了数据集的形状和前五行数据。

```py
# load and summarize the housing dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())
```

运行该示例确认了 506 行数据、13 个输入变量和一个数字目标变量(总共 14 个)。我们还可以看到，所有的输入变量都是数字。

```py
(506, 14)
        0     1     2   3      4      5   ...  8      9     10      11    12    13
0  0.00632  18.0  2.31   0  0.538  6.575  ...   1  296.0  15.3  396.90  4.98  24.0
1  0.02731   0.0  7.07   0  0.469  6.421  ...   2  242.0  17.8  396.90  9.14  21.6
2  0.02729   0.0  7.07   0  0.469  7.185  ...   2  242.0  17.8  392.83  4.03  34.7
3  0.03237   0.0  2.18   0  0.458  6.998  ...   3  222.0  18.7  394.63  2.94  33.4
4  0.06905   0.0  2.18   0  0.458  7.147  ...   3  222.0  18.7  396.90  5.33  36.2

[5 rows x 14 columns]
```

接下来，我们可以为建模准备数据集。

首先将数据集拆分为输入和输出列，然后将行[拆分为训练和测试数据集](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)。

在这种情况下，我们将使用大约 67%的行来训练模型，剩下的 33%用于估计模型的性能。

```py
...
# split into input and output values
X, y = values[:,:-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
```

在本教程中，您可以了解有关列车测试分割的更多信息:

*   [用于评估机器学习算法的训练-测试分割](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)

然后，我们将缩放所有输入列(变量)，使其范围为 0-1，称为数据规范化，这在使用神经网络模型时是一种很好的做法。

```py
...
# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

在本教程中，您可以了解更多关于使用最小最大缩放器规范化输入数据的信息:

*   [如何在 Python 中使用标准缩放器和最小最大缩放器变换](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

下面列出了为建模准备数据的完整示例。

```py
# load and prepare the dataset for modeling
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
values = dataframe.values
# split into input and output values
X, y = values[:,:-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# summarize
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

运行该示例会像以前一样加载数据集，然后将列拆分为输入和输出元素，将行拆分为训练集和测试集，最后将所有输入变量缩放到范围[0，1]

打印了列车和测试集的形状，显示我们有 339 行来训练模型，167 行来评估模型。

```py
(339, 13) (167, 13) (339,) (167,)
```

接下来，我们可以在数据集上定义、训练和评估多层感知器(MLP)模型。

我们将定义一个简单的模型，它有两个隐藏层和一个预测数值的输出层。我们将使用 [ReLU 激活功能](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)和 *he* 权重初始化，这是一个很好的实践。

每个隐藏层中的节点数是经过一点点尝试和错误后选择的。

```py
...
# define neural network model
features = X_train.shape[1]
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=features))
model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1))
```

我们将使用具有接近默认学习率和动量值的随机梯度下降的有效亚当版本，并使用均方误差(MSE)损失函数拟合模型，这是回归预测建模问题的标准。

```py
...
# compile the model and specify loss and optimizer
opt = Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999)
model.compile(optimizer=opt, loss='mse')
```

您可以在本教程中了解关于 Adam 优化算法的更多信息:

*   [从头开始编码亚当梯度下降优化](https://machinelearningmastery.com/adam-optimization-from-scratch/)

该模型将适用于 300 个时代，批量为 16 个样本。这种配置是经过反复试验后选择的。

```py
...
# fit the model on the training dataset
model.fit(X_train, y_train, verbose=2, epochs=300, batch_size=16)
```

您可以在本教程中了解有关批次和时期的更多信息:

*   [神经网络中批次和时期之间的差异](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

最后，该模型可用于对测试数据集进行预测，我们可以通过将预测与测试集中的期望值进行比较来评估预测，并计算平均绝对误差(MAE)，这是模型性能的一个有用度量。

```py
...
# make predictions on the test set
yhat = model.predict(X_test, verbose=0)
# calculate the average error in the predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

将这些联系在一起，完整的示例如下所示。

```py
# train and evaluate a multilayer perceptron neural network on the housing regression dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
values = dataframe.values
# split into input and output values
X, y = values[:, :-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)
# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# define neural network model
features = X_train.shape[1]
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=features))
model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1))
# compile the model and specify loss and optimizer
opt = Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999)
model.compile(optimizer=opt, loss='mse')
# fit the model on the training dataset
model.fit(X_train, y_train, verbose=2, epochs=300, batch_size=16)
# make predictions on the test set
yhat = model.predict(X_test, verbose=0)
# calculate the average error in the predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例加载和准备数据集，在训练数据集上定义和拟合 MLP 模型，并在测试集上评估其性能。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，该模型实现了大约 2.3 的平均绝对误差，这优于天真模型，并接近最佳模型。

毫无疑问，通过进一步调整模型，我们可以获得接近最优的性能，但这对于我们研究预测区间来说已经足够好了。

```py
...
Epoch 296/300
22/22 - 0s - loss: 7.1741
Epoch 297/300
22/22 - 0s - loss: 6.8044
Epoch 298/300
22/22 - 0s - loss: 6.8623
Epoch 299/300
22/22 - 0s - loss: 7.7010
Epoch 300/300
22/22 - 0s - loss: 6.5374
MAE: 2.300
```

接下来，让我们看看如何使用我们的 MLP 模型在住房数据集上计算预测区间。

## 神经网络预测区间

在本节中，我们将使用上一节中开发的回归问题和模型来开发预测区间。

与线性回归等线性方法相比，计算神经网络等非线性回归算法的预测区间具有挑战性，因为线性回归的预测区间计算量很小。没有标准的技术。

有许多方法可以计算神经网络模型的有效预测区间。我推荐一些列在“*进一步阅读*”部分的论文来了解更多。

在本教程中，我们将使用一种非常简单的方法，它有很大的扩展空间。我称之为*快而脏*，因为它快而易算，但有局限性。

它包括拟合多个最终模型(例如 10 到 30 个)。然后，来自集合成员的点预测的分布被用于计算点预测和预测间隔。

例如，一个点预测可以作为来自集合成员的点预测的平均值，95%的预测间隔可以作为与平均值的 [1.96 标准偏差](https://en.wikipedia.org/wiki/1.96)。

这是一个简单的高斯预测区间，尽管也可以使用替代方案，例如点预测的最小值和最大值。或者，自举方法可用于在不同的自举样本上训练每个集合成员，点预测的 2.5 和 97.5 百分位数可用作预测区间。

有关引导方法的更多信息，请参见教程:

*   [引导法的简单介绍](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)

这些延伸作为练习留着；我们将坚持简单的高斯预测区间。

让我们假设上一节中定义的训练数据集是整个数据集，并且我们正在这个整个数据集上训练一个或多个最终模型。然后，我们可以用测试集上的预测区间进行预测，并评估该区间在未来的有效性。

我们可以通过将上一节中开发的元素分成函数来简化代码。

首先，让我们定义一个函数，用于加载和准备由 URL 定义的回归数据集。

```py
# load and prepare the dataset
def load_dataset(url):
	dataframe = read_csv(url, header=None)
	values = dataframe.values
	# split into input and output values
	X, y = values[:, :-1], values[:,-1]
	# split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)
	# scale input data
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, X_test, y_train, y_test
```

接下来，我们可以定义一个函数，该函数将在给定训练数据集的情况下定义和训练 MLP 模型，然后返回拟合模型，准备进行预测。

```py
# define and fit the model
def fit_model(X_train, y_train):
	# define neural network model
	features = X_train.shape[1]
	model = Sequential()
	model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=features))
	model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
	model.add(Dense(1))
	# compile the model and specify loss and optimizer
	opt = Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999)
	model.compile(optimizer=opt, loss='mse')
	# fit the model on the training dataset
	model.fit(X_train, y_train, verbose=0, epochs=300, batch_size=16)
	return model
```

我们需要多个模型来进行点预测，这将定义点预测的分布，从中我们可以估计区间。

因此，我们需要在训练数据集上拟合多个模型。每个模型必须不同，这样才能做出不同的预测。这可以通过给定训练 MLP 模型的随机性质、给定随机初始权重以及给定随机梯度下降优化算法的使用来实现。

模型越多，点预测对模型性能的估计就越好。我会推荐至少 10 个型号，也许 30 个型号之外没什么好处。

下面的函数适合一组模型，并将它们存储在返回的列表中。

出于兴趣，每个拟合模型也在测试集上进行评估，测试集在每个模型拟合后报告。我们预计，每个模型在搁置测试集上的估计性能会略有不同，报告的分数将有助于我们确认这一预期。

```py
# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
	ensemble = list()
	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train)
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=0)
		mae = mean_absolute_error(y_test, yhat)
		print('>%d, MAE: %.3f' % (i+1, mae))
		# store the model
		ensemble.append(model)
	return ensemble
```

最后，我们可以使用训练好的模型集合进行点预测，这些点预测可以总结为一个预测区间。

下面的函数实现了这一点。首先，每个模型对输入数据进行点预测，然后计算 95%的预测区间，并返回区间的下、中、上值。

该函数被设计为以一行作为输入，但可以很容易地适应多行。

```py
# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X):
	# make predictions
	yhat = [model.predict(X, verbose=0) for model in ensemble]
	yhat = asarray(yhat)
	# calculate 95% gaussian prediction interval
	interval = 1.96 * yhat.std()
	lower, upper = yhat.mean() - interval, yhat.mean() + interval
	return lower, yhat.mean(), upper
```

最后，我们可以调用这些函数。

首先，加载和准备数据集，然后定义和拟合集合。

```py
...
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
X_train, X_test, y_train, y_test = load_dataset(url)
# fit ensemble
n_members = 30
ensemble = fit_ensemble(n_members, X_train, X_test, y_train, y_test)
```

然后，我们可以使用测试集中的单行数据，用预测间隔进行预测，然后报告结果。

我们还报告了预期值，我们预计该值将被预测区间覆盖(可能接近 95%的时间；这并不完全准确，只是一个粗略的近似值)。

```py
...
# make predictions with prediction interval
newX = asarray([X_test[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX)
print('Point prediction: %.3f' % mean)
print('95%% prediction interval: [%.3f, %.3f]' % (lower, upper))
print('True value: %.3f' % y_test[0])
```

将这些联系在一起，下面列出了使用多层感知器神经网络以预测间隔进行预测的完整示例。

```py
# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# load and prepare the dataset
def load_dataset(url):
	dataframe = read_csv(url, header=None)
	values = dataframe.values
	# split into input and output values
	X, y = values[:, :-1], values[:,-1]
	# split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)
	# scale input data
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, X_test, y_train, y_test

# define and fit the model
def fit_model(X_train, y_train):
	# define neural network model
	features = X_train.shape[1]
	model = Sequential()
	model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=features))
	model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
	model.add(Dense(1))
	# compile the model and specify loss and optimizer
	opt = Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999)
	model.compile(optimizer=opt, loss='mse')
	# fit the model on the training dataset
	model.fit(X_train, y_train, verbose=0, epochs=300, batch_size=16)
	return model

# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
	ensemble = list()
	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train)
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=0)
		mae = mean_absolute_error(y_test, yhat)
		print('>%d, MAE: %.3f' % (i+1, mae))
		# store the model
		ensemble.append(model)
	return ensemble

# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X):
	# make predictions
	yhat = [model.predict(X, verbose=0) for model in ensemble]
	yhat = asarray(yhat)
	# calculate 95% gaussian prediction interval
	interval = 1.96 * yhat.std()
	lower, upper = yhat.mean() - interval, yhat.mean() + interval
	return lower, yhat.mean(), upper

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
X_train, X_test, y_train, y_test = load_dataset(url)
# fit ensemble
n_members = 30
ensemble = fit_ensemble(n_members, X_train, X_test, y_train, y_test)
# make predictions with prediction interval
newX = asarray([X_test[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX)
print('Point prediction: %.3f' % mean)
print('95%% prediction interval: [%.3f, %.3f]' % (lower, upper))
print('True value: %.3f' % y_test[0])
```

运行该示例依次适合每个集成成员，并在等待测试集上报告其估计性能；最后，作出并报告具有预测区间的单一预测。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到每个模型的性能略有不同，这证实了我们对模型确实不同的预期。

最后，我们可以看到，集合做出了大约 30.5 的点预测，95%的预测间隔为[26.287，34.822]。我们还可以看到真实值是 28.2，间隔确实捕捉到了这个值，这很棒。

```py
>1, MAE: 2.259
>2, MAE: 2.144
>3, MAE: 2.732
>4, MAE: 2.628
>5, MAE: 2.483
>6, MAE: 2.551
>7, MAE: 2.505
>8, MAE: 2.299
>9, MAE: 2.706
>10, MAE: 2.145
>11, MAE: 2.765
>12, MAE: 3.244
>13, MAE: 2.385
>14, MAE: 2.592
>15, MAE: 2.418
>16, MAE: 2.493
>17, MAE: 2.367
>18, MAE: 2.569
>19, MAE: 2.664
>20, MAE: 2.233
>21, MAE: 2.228
>22, MAE: 2.646
>23, MAE: 2.641
>24, MAE: 2.492
>25, MAE: 2.558
>26, MAE: 2.416
>27, MAE: 2.328
>28, MAE: 2.383
>29, MAE: 2.215
>30, MAE: 2.408
Point prediction: 30.555
95% prediction interval: [26.287, 34.822]
True value: 28.200
```

如上所述，这是一种快速而肮脏的技术，用于利用神经网络的预测区间进行预测。

有一些简单的扩展，例如使用自举方法来进行可能更可靠的点预测，以及我建议您探索的下面列出的一些论文中描述的更高级的技术。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习的预测间隔](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)
*   [如何在 Python 中使用标准缩放器和最小最大缩放器变换](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)
*   [标准机器学习数据集的最佳结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [神经网络中批次和时期之间的差异](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
*   [从头开始编码亚当梯度下降优化](https://machinelearningmastery.com/adam-optimization-from-scratch/)

### 报纸

*   [深度学习的高质量预测区间:一种无分布、集合的方法](https://arxiv.org/abs/1802.07167)，2018。
*   [实际置信区间和预测区间](https://papers.nips.cc/paper/1306-practicalconfidence-and-prediction-intervals)，1994。

### 文章

*   1.96，维基百科。

## 摘要

在本教程中，您发现了如何计算深度学习神经网络的预测间隔。

具体来说，您了解到:

*   预测区间提供了回归预测建模问题的不确定性度量。
*   如何在标准回归问题上开发和评估简单的多层感知器神经网络。
*   如何使用一组神经网络模型计算和报告预测区间。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。