# 如何评估用于人类活动识别的机器学习算法

> 原文： [https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/](https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/)

人类活动识别是将由专用线束或智能电话记录的加速度计数据序列分类为已知的明确定义的运动的问题。

该问题的经典方法涉及基于固定大小的窗口和训练机器学习模型（例如决策树的集合）的时间序列数据中的手工制作特征。困难在于此功能工程需要该领域的深厚专业知识。

最近，已经证明，诸如循环神经网络和一维卷积神经网络（CNN）之类的深度学习方法可以在很少或没有数据特征工程的情况下提供具有挑战性的活动识别任务的最新结果，而不是使用特征学习原始数据。

在本教程中，您将了解如何在“_ 使用智能手机的活动识别 _”数据集上评估各种机器学习算法。

完成本教程后，您将了解：

*   如何在特征设计版本的活动识别数据集上加载和评估非线性和集成机器学习算法。
*   如何在活动识别数据集的原始信号数据上加载和评估机器学习算法。
*   如何定义能够进行特征学习的更复杂算法的预期表现的合理上下界，例如深度学习方法。

让我们开始吧。

![How to Evaluate Machine Learning Algorithms for Human Activity Recognition](img/9f387fe9e2be7d88dcf3a61fb4de878c.jpg)

如何评估用于人类活动识别的机器学习算法
照片由 [Murray Foubister](https://www.flickr.com/photos/mfoubister/41564699865/) ，保留一些权利。

## 教程概述

本教程分为三个部分;他们是：

1.  使用智能手机数据集进行活动识别
2.  建模特征工程数据
3.  建模原始数据

## 使用智能手机数据集进行活动识别

[人类活动识别](https://en.wikipedia.org/wiki/Activity_recognition)，或简称为 HAR，是基于使用传感器的移动痕迹来预测人正在做什么的问题。

标准人类活动识别数据集是 2012 年提供的“使用智能手机进行活动识别”数据集。

它由 Davide Anguita 等人准备并提供。来自意大利热那亚大学的 2013 年论文“[使用智能手机进行人类活动识别的公共领域数据集](https://upcommons.upc.edu/handle/2117/20897)”中对该数据集进行了全面描述。该数据集在他们的 2012 年论文中用机器学习算法建模，标题为“[使用多类硬件友好支持向量机](https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30)在智能手机上进行人类活动识别。“

数据集可用，可以从 UCI 机器学习库免费下载：

*   [使用智能手机数据集进行人类活动识别，UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

该数据来自 30 名年龄在 19 至 48 岁之间的受试者，其执行六项标准活动中的一项，同时佩戴记录运动数据的腰部智能手机。记录执行活动的每个受试者的视频，并从这些视频手动标记移动数据。

以下是在记录其移动数据的同时执行活动的主体的示例视频。

&lt;iframe allow="autoplay; encrypted-media" allowfullscreen="" frameborder="0" height="375" src="https://www.youtube.com/embed/XOEN9W05_4A?feature=oembed" width="500"&gt;&lt;/iframe&gt;

进行的六项活动如下：

1.  步行
2.  走上楼
3.  走楼下
4.  坐在
5.  常设
6.  铺设

记录的运动数据是来自智能手机的 x，y 和 z 加速度计数据（线性加速度）和陀螺仪数据（角速度），特别是三星 Galaxy S II。以 50Hz（即每秒 50 个数据点）记录观察结果。每个受试者进行两次活动，一次是左侧设备，另一次是右侧设备。

原始数据不可用。相反，可以使用预处理版本的数据集。预处理步骤包括：

*   使用噪声滤波器预处理加速度计和陀螺仪。
*   将数据拆分为 2.56 秒（128 个数据点）的固定窗口，重叠率为 50％。
*   将加速度计数据分割为重力（总）和身体运动分量。

特征工程应用于窗口数据，并且提供具有这些工程特征的数据的副本。

从每个窗口提取在人类活动识别领域中常用的许多时间和频率特征。结果是 561 元素的特征向量。

根据受试者的数据，将数据集分成训练（70％）和测试（30％）组。训练 21 个，测试 9 个。

使用旨在用于智能手机的支持向量机（例如定点算术）的实验结果导致测试数据集的预测准确度为 89％，实现与未修改的 SVM 实现类似的结果。

该数据集是免费提供的，可以从 UCI 机器学习库下载。

数据以单个 zip 文件的形式提供，大小约为 58 兆字节。此下载的直接链接如下：

*   [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

下载数据集并将所有文件解压缩到当前工作目录中名为“HARDataset”的新目录中。

## 建模特征工程数据

在本节中，我们将开发代码来加载数据集的特征工程版本并评估一套非线性机器学习算法，包括原始论文中使用的 SVM。

目标是在测试数据集上实现至少 89％的准确率。

使用特征工程版数据集的方法的结果为为原始数据版本开发的任何方法提供了基线。

本节分为五个部分;他们是：

*   加载数据集
*   定义模型
*   评估模型
*   总结结果
*   完整的例子

### 加载数据集

第一步是加载训练并测试输入（X）和输出（y）数据。

具体来说，以下文件：

*   _HARDataset / train / X_train.txt_
*   _HARDataset / train / y_train.txt_
*   _HARDataset / test / X_test.txt_
*   _HARDataset / test / y_test.txt_

输入数据采用 CSV 格式，其中列通过空格分隔。这些文件中的每一个都可以作为 NumPy 数组加载。

下面的`load_file()`函数在给定文件的文件路径的情况下加载数据集，并将加载的数据作为 NumPy 数组返回。

```py
# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

考虑到目录布局和文件名的相似性，我们可以调用此函数来加载给定训练或测试集组的`X`和`y`文件。下面的`load_dataset_group()`函数将为一个组加载这两个文件，并将 X 和 y 元素作为 NumPy 数组返回。然后，此函数可用于加载训练和测试组的 X 和 y 元素。

```py
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	# load input data
	X = load_file(prefix + group + '/X_'+group+'.txt')
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

最后，我们可以加载 train 和 test 数据集，并将它们作为 NumPy 数组返回，以便为拟合和评估机器学习模型做好准备。

```py
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy
```

我们可以调用这个函数来加载所有需要的数据;例如：

```py
# load dataset
trainX, trainy, testX, testy = load_dataset()
```

### 定义模型

接下来，我们可以定义一个机器学习模型列表来评估这个问题。

我们将使用默认配置评估模型。我们目前不是在寻找这些模型的最佳配置，只是对具有默认配置的复杂模型在这个问题上表现如何的一般概念。

我们将评估一组不同的非线性和集成机器学习算法，具体来说：

非线性算法：

*   k-最近邻居
*   分类和回归树
*   支持向量机
*   朴素贝叶斯

集合算法：

*   袋装决策树
*   随机森林
*   额外的树木
*   梯度增压机

我们将定义模型并将它们存储在字典中，该字典将模型对象映射到有助于分析结果的简短名称。

下面的`define_models()`函数定义了我们将评估的八个模型。

```py
# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models
```

此功能非常易于扩展，您可以轻松更新以定义您希望的任何机器学习模型或模型配置。

### 评估模型

下一步是评估加载的数据集中定义的模型。

该步骤分为单个模型的评估和所有模型的评估。

我们将通过首先将其拟合到训练数据集上，对测试数据集做出预测，然后使用度量来评估预测来评估单个模型。在这种情况下，我们将使用分类精度来捕获模型的表现（或误差），给出六个活动（或类）的平衡观察。

下面的`evaluate_model()`函数实现了此行为，评估给定模型并将分类精度返回为百分比。

```py
# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0
```

我们现在可以为每个定义的模型重复调用`evaluate_model()`函数。

下面的`evaluate_models()`函数实现此行为，获取已定义模型的字典，并返回映射到其分类精度的模型名称字典。

因为模型的评估可能需要几分钟，所以该函数在评估每个模型作为一些详细的反馈之后打印它们的表现。

```py
# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results
```

### 总结结果

最后一步是总结研究结果。

我们可以按降序排列分类准确度对所有结果进行排序，因为我们对最大化准确率感兴趣。

然后可以打印评估模型的结果，清楚地显示每个评估模型的相对等级。

下面的`summarize_results()`函数实现了这种行为。

```py
# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))
```

### 完整的例子

我们知道我们已经完成了所有工作。

下面列出了在数据集的特征工程版本上评估一套八个机器学习模型的完整示例。

```py
# spot check on engineered-features
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	# load input data
	X = load_file(prefix + group + '/X_'+group+'.txt')
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
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)
```

运行该示例首先加载训练和测试数据集，显示每个输入和输出组件的形状。

然后依次评估八个模型，打印每个模型的表现。

最后，显示模型在测试集上的表现等级。

我们可以看到，ExtraTrees 集合方法和支持向量机非线性方法在测试集上实现了大约 94％的准确率。

这是一个很好的结果，超过原始论文中 SVM 报告的 89％。

考虑到算法的随机性，每次运行代码时，具体结果可能会有所不同。然而，考虑到数据集的大小，算法表现之间的相对关系应该相当稳定。

```py
(7352, 561) (7352, 1)
(2947, 561) (2947, 1)
(7352, 561) (7352,) (2947, 561) (2947,)
Defined 8 models
>knn: 90.329
>cart: 86.020
>svm: 94.028
>bayes: 77.027
>bag: 89.820
>rf: 92.772
>et: 94.028
>gbm: 93.756

Name=et, Score=94.028
Name=svm, Score=94.028
Name=gbm, Score=93.756
Name=rf, Score=92.772
Name=knn, Score=90.329
Name=bag, Score=89.820
Name=cart, Score=86.020
Name=bayes, Score=77.027
```

这些结果显示了在准备数据和领域特定功能的工程中给定的领域专业知识的可能性。因此，这些结果可以作为通过更先进的方法可以追求的表现的上限，这些方法可以自动学习特征作为拟合模型的一部分，例如深度学习方法。

任何这样的先进方法都适合并评估从中得到工程特征的原始数据。因此，直接评估该数据的机器学习算法的表现可以提供任何更高级方法的表现的预期下限。

我们将在下一节中探讨这一点。

## 建模原始数据

我们可以使用相同的框架来评估原始数据上的机器学习模型。

原始数据确实需要更多工作才能加载。

原始数据中有三种主要信号类型：总加速度，车身加速度和车身陀螺仪。每个都有三个数据轴。这意味着每个时间步长总共有九个变量。

此外，每个数据系列已被划分为 2.65 秒数据的重叠窗口，或 128 个时间步长。这些数据窗口对应于上一节中工程特征（行）的窗口。

这意味着一行数据具有 128 * 9 或 1,152 个元素。这比前一节中 561 个元素向量的大小小一倍，并且可能存在一些冗余数据。

信号存储在训练和测试子目录下的 _/ Inertial Signals /_ 目录中。每个信号的每个轴都存储在一个单独的文件中，这意味着每个训练和测试数据集都有九个要加载的输入文件和一个要加载的输出文件。在给定一致的目录结构和文件命名约定的情况下，我们可以批量加载这些文件。

首先，我们可以将给定组的所有数据加载到单个三维 NumPy 数组中，其中数组的维数为[样本，时间步长，特征]。为了更清楚，有 128 个时间步和 9 个特征，其中样本数是任何给定原始信号数据文件中的行数。

下面的`load_group()`函数实现了这种行为。`dstack()`NumPy 函数允许我们将每个加载的 3D 数组堆叠成单个 3D 数组，其中变量在第三维（特征）上分开。

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

作为准备加载数据的一部分，我们必须将窗口和特征展平为一个长向量。

我们可以使用 NumPy 重塑功能执行此操作，并将[_ 样本，时间步长，特征 _]的三个维度转换为[_ 样本，时间步长*特征 _]的两个维度。

下面的`load_dataset()`函数实现了这种行为，并返回训练并测试`X`和`y`元素，以便拟合和评估定义的模型。

```py
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# flatten X
	trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
	testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy
```

综合这些，下面列出了完整的例子。

```py
# spot check on raw data
from numpy import dstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
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
	# flatten X
	trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
	testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)
```

首先运行该示例加载数据集。

我们可以看到原始序列和测试集具有与工程特征（分别为 7352 和 2947）相同数量的样本，并且正确加载了三维数据。我们还可以看到平面数据和将提供给模型的 1152 输入向量。

接下来依次评估八个定义的模型。

最终结果表明，决策树的集合在原始数据上表现最佳。梯度增强和额外树木以最高 87％和 86％的准确度表现最佳，比数据集的特征工程版本中表现最佳的模型低约 7 个点。

令人鼓舞的是，Extra Trees 集合方法在两个数据集上都表现良好;它表明它和类似的树集合方法可能适合这个问题，至少在这个简化的框架中。

我们还可以看到 SVM 下降到约 72％的准确度。

决策树集合的良好表现可能表明需要特征选择和集合方法能够选择与预测相关活动最相关的那些特征。

```py
(7352, 128, 9) (7352, 1)
(2947, 128, 9) (2947, 1)
(7352, 1152) (7352,) (2947, 1152) (2947,)
Defined 8 models
>knn: 61.893
>cart: 72.141
>svm: 76.960
>bayes: 72.480
>bag: 84.527
>rf: 84.662
>et: 86.902
>gbm: 87.615

Name=gbm, Score=87.615
Name=et, Score=86.902
Name=rf, Score=84.662
Name=bag, Score=84.527
Name=svm, Score=76.960
Name=bayes, Score=72.480
Name=cart, Score=72.141
Name=knn, Score=61.893
```

如前一节所述，这些结果为可能尝试从原始数据自动学习更高阶特征（例如，通过深度学习方法中的特征学习）的任何更复杂的方法提供了精确度的下限。

总之，此类方法的界限在原始数据上从 GBM 的约 87％准确度扩展到高度处理的数据集上的额外树和 SVM 的约 94％，[87％至 94％]。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **更多算法**。在这个问题上只评估了八种机器学习算法;尝试一些线性方法，也许还有一些非线性和集合方法。
*   **算法调整**。没有调整机器学习算法;主要使用默认配置。选择一种方法，例如 SVM，ExtraTrees 或 Gradient Boosting，并搜索一组不同的超参数配置，以查看是否可以进一步提升问题的表现。
*   **数据缩放**。数据已经按比例缩放到[-1,1]。探索额外的扩展（例如标准化）是否可以带来更好的表现，可能是对这种扩展敏感的方法（如 kNN）。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [使用智能手机进行人类活动识别的公共领域数据集](https://upcommons.upc.edu/handle/2117/20897)，2013 年。
*   [智能手机上的人类活动识别使用多类硬件友好支持向量机](https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30)，2012。

### 用品

*   [使用智能手机数据集进行人类活动识别，UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
*   [活动识别，维基百科](https://en.wikipedia.org/wiki/Activity_recognition)
*   [使用智能手机传感器的活动识别实验，视频](https://www.youtube.com/watch?v=XOEN9W05_4A)。

## 摘要

在本教程中，您了解了如何在'_ 使用智能手机的活动识别 _'数据集上评估各种机器学习算法。

具体来说，你学到了：

*   如何在特征设计版本的活动识别数据集上加载和评估非线性和集成机器学习算法。
*   如何在活动识别数据集的原始信号数据上加载和评估机器学习算法。
*   如何定义能够进行特征学习的更复杂算法的预期表现的合理上下界，例如深度学习方法。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。