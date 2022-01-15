# 如何为人类活动识别时间序列分类开发 RNN 模型

> 原文： [https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)

人类活动识别是将由专用线束或智能电话记录的加速度计数据序列分类为已知的明确定义的运动的问题。

该问题的经典方法涉及基于固定大小的窗口和训练机器学习模型（例如决策树的集合）的时间序列数据中的手工制作特征。困难在于此功能工程需要该领域的强大专业知识。

最近，诸如 LSTM 之类的循环神经网络和利用一维卷积神经网络或 CNN 的变化等深度学习方法已经被证明可以在很少或没有数据的情况下提供具有挑战性的活动识别任务的最新结果特征工程，而不是使用原始数据的特征学习。

在本教程中，您将发现三种循环神经网络架构，用于对活动识别时间序列分类问题进行建模。

完成本教程后，您将了解：

*   如何开发一种用于人类活动识别的长短期记忆循环神经网络。
*   如何开发一维卷积神经网络 LSTM 或 CNN-LSTM 模型。
*   如何针对同一问题开发一维卷积 LSTM 或 ConvLSTM 模型。

让我们开始吧。

![How to Develop RNN Models for Human Activity Recognition Time Series Classification](img/d330ca8c16c51dde05533f60b321bc56.jpg)

如何开发用于人类活动识别的 RNN 模型时间序列分类
照片由 [Bonnie Moreland](https://www.flickr.com/photos/icetsarina/25033478158/) ，保留一些权利。

## 教程概述

本教程分为四个部分;他们是：

1.  使用智能手机数据集进行活动识别
2.  开发 LSTM 网络模型
3.  开发 CNN-LSTM 网络模型
4.  开发 ConvLSTM 网络模型

## 使用智能手机数据集进行活动识别

[人类活动识别](https://en.wikipedia.org/wiki/Activity_recognition)，或简称为 HAR，是基于使用传感器的移动痕迹来预测人正在做什么的问题。

标准的人类活动识别数据集是 2012 年推出的“使用智能手机数据集的活动识别”。

它由 Davide Anguita 等人准备并提供。来自意大利热那亚大学的 2013 年论文“[使用智能手机进行人类活动识别的公共领域数据集](https://upcommons.upc.edu/handle/2117/20897)”中对该数据集进行了全面描述。该数据集在他们的 2012 年论文中用机器学习算法建模，标题为“[使用多类硬件友好支持向量机](https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30)在智能手机上进行人类活动识别。“

数据集可用，可以从 UCI 机器学习库免费下载：

*   [使用智能手机数据集进行人类活动识别，UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

该数据来自 30 名年龄在 19 至 48 岁之间的受试者，其执行六项标准活动中的一项，同时佩戴记录运动数据的腰部智能手机。记录执行活动的每个受试者的视频，并从这些视频手动标记移动数据。

以下是在记录其移动数据的同时执行活动的主体的示例视频。

&lt;iframe allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" frameborder="0" height="375" src="https://www.youtube.com/embed/XOEN9W05_4A?feature=oembed" width="500"&gt;&lt;/iframe&gt;

进行的六项活动如下：

1.  步行
2.  走上楼
3.  走楼下
4.  坐在
5.  常设
6.  铺设

记录的运动数据是来自智能手机的 x，y 和 z 加速度计数据（线性加速度）和陀螺仪数据（角速度），特别是三星 Galaxy S II。以 50Hz（即每秒 50 个数据点）记录观察结果。每个受试者进行两次活动;一旦设备在左侧，一次设备在右侧。

原始数据不可用。相反，可以使用预处理版本的数据集。预处理步骤包括：

*   使用噪声滤波器预处理加速度计和陀螺仪。
*   将数据拆分为 2.56 秒（128 个数据点）的固定窗口，重叠率为 50％。将加速度计数据分割为重力（总）和身体运动分量。

特征工程应用于窗口数据，并且提供具有这些工程特征的数据的副本。

从每个窗口提取在人类活动识别领域中常用的许多时间和频率特征。结果是 561 元素的特征向量。

根据受试者的数据，将数据集分成训练（70％）和测试（30％）组。训练 21 个，测试 9 个。

使用旨在用于智能手机的支持向量机（例如定点算术）的实验结果导致测试数据集的预测准确度为 89％，实现与未修改的 SVM 实现类似的结果。

该数据集是免费提供的，可以从 UCI 机器学习库下载。

数据以单个 zip 文件的形式提供，大小约为 58 兆字节。此下载的直接链接如下：

*   [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

下载数据集并将所有文件解压缩到当前工作目录中名为“HARDataset”的新目录中。

## 开发 LSTM 网络模型

在本节中，我们将为人类活动识别数据集开发长期短期记忆网络模型（LSTM）。

LSTM 网络模型是一种循环神经网络，能够学习和记忆长输入数据序列。它们适用于由长序列数据组成的数据，最多 200 到 400 个时间步长。它们可能非常适合这个问题。

该模型可以支持多个并行的输入数据序列，例如加速度计的每个轴和陀螺仪数据。该模型学习从观察序列中提取特征以及如何将内部特征映射到不同的活动类型。

使用 LSTM 进行序列分类的好处是，他们可以直接从原始时间序列数据中学习，反过来不需要领域专业知识来手动设计输入功能。该模型可以学习时间序列数据的内部表示，并且理想地实现与适合具有工程特征的数据集版本的模型相当的表现。

本节分为四个部分;他们是：

1.  加载数据
2.  拟合和评估模型
3.  总结结果
4.  完整的例子

## 加载数据

第一步是将原始数据集加载到内存中。

原始数据中有三种主要信号类型：总加速度，车身加速度和车身陀螺仪。每个都有 3 个数据轴。这意味着每个时间步长总共有九个变量。

此外，每个数据系列已被划分为 2.56 秒数据或 128 个时间步长的重叠窗口。这些数据窗口对应于上一节中工程特征（行）的窗口。

这意味着一行数据具有（128 * 9）或 1,152 个元素。这比前一节中 561 个元素向量的大小小一倍，并且可能存在一些冗余数据。

信号存储在 train 和 test 子目录下的/ _Inertial Signals_ /目录中。每个信号的每个轴都存储在一个单独的文件中，这意味着每个训练和测试数据集都有九个要加载的输入文件和一个要加载的输出文件。在给定一致的目录结构和文件命名约定的情况下，我们可以批量加载这些文件。

输入数据采用 CSV 格式，其中列由空格分隔。这些文件中的每一个都可以作为 NumPy 数组加载。下面的`load_file()`函数在给定文件填充路径的情况下加载数据集，并将加载的数据作为 NumPy 数组返回。

```py
# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

然后，我们可以将给定组（训练或测试）的所有数据加载到单个三维 NumPy 数组中，其中数组的尺寸为[_ 样本，时间步长，特征 _]。

为了更清楚，有 128 个时间步和 9 个特征，其中样本数是任何给定原始信号数据文件中的行数。

下面的`load_group()`函数实现了这种行为。 [dstack（）NumPy 函数](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.dstack.html)允许我们将每个加载的 3D 数组堆叠成单个 3D 数组，其中变量在第三维（特征）上分开。

```py
# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
```

我们可以使用此功能加载给定组的所有输入信号数据，例如训练或测试。

下面的`load_dataset_group()`函数使用目录之间的一致命名约定加载单个组的所有输入信号数据和输出数据。

```py
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

最后，我们可以加载每个训练和测试数据集。

输出数据定义为类号的整数。我们必须对这些类整数进行热编码，以使数据适合于拟合神经网络多分类模型。我们可以通过调用 [to_categorical（）Keras 函数](https://keras.io/utils/#to_categorical)来实现。

下面的`load_dataset()`函数实现了这种行为，并返回训练并测试 X 和 y 元素，以便拟合和评估定义的模型。

```py
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy
```

### 拟合和评估模型

现在我们已将数据加载到内存中以便进行建模，我们可以定义，拟合和评估 LSTM 模型。

我们可以定义一个名为`evaluate_model()`的函数，它接受训练和测试数据集，拟合训练数据集上的模型，在测试数据集上对其进行评估，并返回模型表现的估计值。

首先，我们必须使用 Keras 深度学习库来定义 LSTM 模型。该模型需要使用[_ 样本，时间步长，特征 _]进行三维输入。

这正是我们加载数据的方式，其中一个样本是时间序列数据的一个窗口，每个窗口有 128 个时间步长，时间步长有九个变量或特征。

模型的输出将是一个六元素向量，包含属于六种活动类型中每种活动类型的给定窗口的概率。

在拟合模型时需要输入和输出尺寸，我们可以从提供的训练数据集中提取它们。

```py
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
```

为简单起见，该模型被定义为顺序 Keras 模型。

我们将模型定义为具有单个 LSTM 隐藏层。接下来是一个脱落层，旨在减少模型过拟合到训练数据。最后，在使用最终输出层做出预测之前，使用密集的完全连接层来解释由 LSTM 隐藏层提取的特征。

随机梯度下降的有效 [Adam 版本将用于优化网络，并且鉴于我们正在学习多类别分类问题，将使用分类交叉熵损失函数。](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

下面列出了该模型的定义。

```py
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

该模型适用于固定数量的时期，在这种情况下为 15，并且将使用 64 个样本的批量大小，其中在更新模型的权重之前将 64 个数据窗口暴露给模型。

模型拟合后，将在测试数据集上进行评估，并返回测试数据集上拟合模型的精度。

注意，在拟合 LSTM 时，通常不对值序列数据进行混洗。这里我们在训练期间对输入数据的窗口进行随机播放（默认）。在这个问题中，我们感兴趣的是利用 LSTM 的能力来学习和提取窗口中时间步长的功能，而不是跨窗口。

下面列出了完整的`evaluate_model()`函数。

```py
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

网络结构或选择的超参数没有什么特别之处，它们只是这个问题的起点。

### 总结结果

我们无法从单一评估中判断模型的技能。

其原因是神经网络是随机的，这意味着当在相同数据上训练相同的模型配置时将产生不同的特定模型。

这是网络的一个特征，它为模型提供了自适应能力，但需要对模型进行稍微复杂的评估。

我们将多次重复对模型的评估，然后在每次运行中总结模型的表现。例如，我们可以调用`evaluate_model()`共 10 次。这将导致必须总结的模型评估分数。

```py
# repeat experiment
scores = list()
for r in range(repeats):
	score = evaluate_model(trainX, trainy, testX, testy)
	score = score * 100.0
	print('>#%d: %.3f' % (r+1, score))
	scores.append(score)
```

我们可以通过计算和报告绩效的均值和标准差来总结得分样本。均值给出了数据集上模型的平均精度，而标准差给出了精度与平均值的平均方差。

下面的函数`summarize_results()`总结了运行的结果。

```py
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

我们可以将重复评估，结果收集和结果汇总捆绑到实验的主要功能中，称为 _run_experiment（）_，如下所示。

默认情况下，在报告模型表现之前，会对模型进行 10 次评估。

```py
# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
```

### 完整的例子

现在我们已经拥有了所有的部分，我们可以将它们组合成一个有效的例子。

完整的代码清单如下。

```py
# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

运行该示例首先打印已加载数据集的形状，然后打印训练和测试集的形状以及输入和输出元素。这确认了样本数，时间步长和变量，以及类的数量。

接下来，创建和评估模型，并为每个模型打印调试消息。

最后，打印分数样本，然后是平均值和标准差。我们可以看到该模型表现良好，在原始数据集上实现了约 89.7％的分类准确度，标准偏差约为 1.3。

这是一个很好的结果，考虑到原始论文发表了 89％的结果，在具有重域特定特征工程的数据集上进行了训练，而不是原始数据集。

注意：鉴于算法的随机性，您的具体结果可能会有所不同。如果是这样，请尝试运行几次代码。

```py
(7352, 128, 9) (7352, 1)
(2947, 128, 9) (2947, 1)
(7352, 128, 9) (7352, 6) (2947, 128, 9) (2947, 6)

>#1: 90.058
>#2: 85.918
>#3: 90.974
>#4: 89.515
>#5: 90.159
>#6: 91.110
>#7: 89.718
>#8: 90.295
>#9: 89.447
>#10: 90.024

[90.05768578215134, 85.91788259246692, 90.97387173396675, 89.51476077366813, 90.15948422124194, 91.10960298608755, 89.71835765184933, 90.29521547336275, 89.44689514760775, 90.02375296912113]

Accuracy: 89.722% (+/-1.371)
```

现在我们已经了解了如何开发用于时间序列分类的 LSTM 模型，让我们看看如何开发更复杂的 CNN LSTM 模型。

## 开发 CNN-LSTM 网络模型

CNN LSTM 架构涉及使用卷积神经网络（CNN）层对输入数据进行特征提取以及 LSTM 以支持序列预测。

CNN LSTM 是针对视觉时间序列预测问题以及从图像序列（例如视频）生成文本描述的应用而开发的。具体来说，问题是：

*   **活动识别**：生成在一系列图像中演示的活动的文本描述。
*   **图像说明**：生成单个图像的文本描述。
*   **视频说明**：生成图像序列的文本描述。

您可以在帖子中了解有关 CNN LSTM 架构的更多信息：

*   [CNN 长短期记忆网络](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)

要了解有关组合这些模型的后果的更多信息，请参阅论文：

*   [卷积，长短期记忆，完全连接的深度神经网络](https://ieeexplore.ieee.org/document/7178838/)，2015。

CNN LSTM 模型将以块为单位读取主序列的子序列，从每个块中提取特征，然后允许 LSTM 解释从每个块提取的特征。

实现此模型的一种方法是将 128 个时间步的每个窗口拆分为 CNN 模型要处理的子序列。例如，每个窗口中的 128 个时间步长可以分成 32 个时间步长的四个子序列。

```py
# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
```

然后我们可以定义一个 CNN 模型，该模型期望以 32 个时间步长和 9 个特征的长度读取序列。

整个 CNN 模型可以包裹在 [TimeDistributed](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) 层中，以允许相同的 CNN 模型在窗口的四个子序列中的每一个中读取。然后将提取的特征展平并提供给 LSTM 模型以进行读取，在最终映射到活动之前提取其自身的特征。

```py
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
```

通常使用两个连续的 CNN 层，然后是丢失和最大池层，这是 CNN LSTM 模型中使用的简单结构。

下面列出了更新的 _evaluate_model（）_。

```py
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

我们可以像上一节中的直线 LSTM 模型一样评估此模型。

完整的代码清单如下。

```py
# cnn lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

运行该示例总结了 10 个运行中每个运行的模型表现，然后报告了测试集上模型表现的最终摘要。

我们可以看到该模型的表现约为 90.6％，标准偏差约为 1％。

注意：鉴于算法的随机性，您的具体结果可能会有所不同。如果是这样，请尝试运行几次代码。

```py
>#1: 91.517
>#2: 91.042
>#3: 90.804
>#4: 92.263
>#5: 89.684
>#6: 88.666
>#7: 91.381
>#8: 90.804
>#9: 89.379
>#10: 91.347

[91.51679674244994, 91.04173736002714, 90.80420766881574, 92.26331862911435, 89.68442483881914, 88.66644044791313, 91.38106549032915, 90.80420766881574, 89.37902952154734, 91.34713267729894]

Accuracy: 90.689% (+/-1.051)
```

## 开发 ConvLSTM 网络模型

CNN LSTM 想法的进一步扩展是执行 CNN 的卷积（例如 CNN 如何读取输入序列数据）作为 LSTM 的一部分。

这种组合称为卷积 LSTM，简称 ConvLSTM，与 CNN LSTM 一样，也用于时空数据。

与直接读取数据以计算内部状态和状态转换的 LSTM 不同，并且与解释 CNN 模型的输出的 CNN LSTM 不同，ConvLSTM 直接使用卷积作为读取 LSTM 单元本身的输入的一部分。

有关如何在 LSTM 单元内计算 ConvLSTM 方程的更多信息，请参阅文章：

*   [卷积 LSTM 网络：用于降水预报的机器学习方法](https://arxiv.org/abs/1506.04214v1)，2015。

Keras 库提供 [ConvLSTM2D 类](https://keras.io/layers/recurrent/#convlstm2d)，支持用于 2D 数据的 ConvLSTM 模型。它可以配置为 1D 多变量时间序列分类。

默认情况下，ConvLSTM2D 类要求输入数据具有以下形状：

```py
(samples, time, rows, cols, channels)
```

其中每个时间步数据被定义为（行*列）数据点的图像。

在上一节中，我们将给定的数据窗口（128 个时间步长）划分为 32 个时间步长的四个子序列。我们可以在定义 ConvLSTM2D 输入时使用相同的子序列方法，其中时间步数是窗口中子序列的数量，当我们处理一维数据时行数是 1，列数代表子序列中的时间步长数，在本例中为 32。

对于这个选择的问题框架，ConvLSTM2D 的输入因此是：

*   **样本**：n，表示数据集中的窗口数。
*   **时间**：4，对于我们将 128 个时间步长的窗口分成四个子序列。
*   **行**：1，用于每个子序列的一维形状。
*   **列**：32，表示输入子序列中的 32 个时间步长。
*   **频道**：9，为九个输入变量。

我们现在可以为 ConvLSTM2D 模型准备数据。

```py
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
```

ConvLSTM2D 类需要根据 CNN 和 LSTM 进行配置。这包括指定滤波器的数量（例如 64），二维内核大小，在这种情况下（子序列时间步长的 1 行和 3 列），以及激活函数，在这种情况下是整流的线性。

与 CNN 或 LSTM 模型一样，输出必须展平为一个长向量，然后才能通过密集层进行解释。

```py
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
```

然后我们可以在之前对 LSTM 和 CNN LSTM 模型进行评估。

下面列出了完整的示例。

```py
# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

与之前的实验一样，运行模型会在每次拟合和评估时打印模型的表现。最终模型表现的摘要在运行结束时给出。

我们可以看到，该模型在实现约 90％的准确度的问题上始终表现良好，可能比较大的 CNN LSTM 模型具有更少的资源。

注意：鉴于算法的随机性，您的具体结果可能会有所不同。如果是这样，请尝试运行几次代码。

```py
>#1: 90.092
>#2: 91.619
>#3: 92.128
>#4: 90.533
>#5: 89.243
>#6: 90.940
>#7: 92.026
>#8: 91.008
>#9: 90.499
>#10: 89.922

[90.09161859518154, 91.61859518154056, 92.12758737699356, 90.53274516457415, 89.24329826942655, 90.93993892093654, 92.02578893790296, 91.00780454699695, 90.49881235154395, 89.92195453003053]

Accuracy: 90.801% (+/-0.886)
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **数据准备**。考虑探索简单的数据扩展方案是否可以进一步提升模型表现，例如标准化，标准化和电源转换。
*   **LSTM 变化**。 LSTM 架构的变体可以在此问题上实现更好的表现，例如栈式 LSTM 和双向 LSTM。
*   **超参数调整**。考虑探索模型超参数的调整，例如单位数，训练时期，批量大小等。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [使用智能手机进行人类活动识别的公共领域数据集](https://upcommons.upc.edu/handle/2117/20897)，2013 年。
*   [智能手机上的人类活动识别使用多类硬件友好支持向量机](https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30)，2012。
*   [卷积，长短期记忆，完全连接的深度神经网络](https://ieeexplore.ieee.org/document/7178838/)，2015。
*   [卷积 LSTM 网络：用于降水预报的机器学习方法](https://arxiv.org/abs/1506.04214v1)，2015。

### 用品

*   [使用智能手机数据集进行人类活动识别，UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
*   [活动识别，维基百科](https://en.wikipedia.org/wiki/Activity_recognition)
*   [使用智能手机传感器的活动识别实验，视频](https://www.youtube.com/watch?v=XOEN9W05_4A)。

## 摘要

在本教程中，您发现了三种循环神经网络架构，用于对活动识别时间序列分类问题进行建模。

具体来说，你学到了：

*   如何开发一种用于人类活动识别的长短期记忆循环神经网络。
*   如何开发一维卷积神经网络 LSTM 或 CNN LSTM 模型。
*   如何针对同一问题开发一维卷积 LSTM 或 ConvLSTM 模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。