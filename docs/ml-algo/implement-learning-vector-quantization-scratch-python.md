# 如何用Python从零开始实现学习向量量化

> 原文： [https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/](https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/)

[k-最近邻居](http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)的限制是您必须保留一个大型训练样例数据库才能做出预测。

[学习向量量化](http://machinelearningmastery.com/learning-vector-quantization-for-machine-learning/)算法通过学习最能代表训练数据的更小的模式子集来解决这个问题。

在本教程中，您将了解如何使用Python从零开始实现学习向量量化算法。

完成本教程后，您将了解：

*   如何从训练数据集中学习一组码本向量。
*   如何使用学习的码本向量做出预测。
*   如何将学习向量量化应用于实际预测建模问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement Learning Vector Quantization From Scratch With Python](img/9116ae5fdf9659e09d1603c386bd07bd.jpg)

如何用Python从零开始实现学习向量量化
照片由 [Tony Faiola](https://www.flickr.com/photos/tonyfaiola/10303914233/) ，保留一些权利。

## 描述

本节简要介绍了学习向量量化算法和我们将在本教程中使用的电离层分类问题

### 学习向量量化

学习向量量化（LVQ）算法很像k-Nearest Neighbors。

通过在模式库中找到最佳匹配来做出预测。不同之处在于，模式库是从训练数据中学习的，而不是使用训练模式本身。

模式库称为码本向量，每个模式称为码本。将码本向量初始化为来自训练数据集的随机选择的值。然后，在许多时期，它们适于使用学习算法最佳地总结训练数据。

学习算法一次显示一个训练记录，在码本向量中找到最佳匹配单元，如果它们具有相同的类，则将其移动到更接近训练记录，或者如果它们具有不同的类，则更远离训练记录。

一旦准备好，码本向量用于使用k-Nearest Neighbors算法做出预测，其中k = 1。

该算法是为分类预测建模问题而开发的，但可以适用于回归问题。

### 电离层数据集

电离层数据集根据雷达返回数据预测电离层的结构。

每个实例都描述了大气层雷达回波的特性，任务是预测电离层中是否存在结构。

共有351个实例和34个数字输入变量，每对雷达脉冲有17对2，通常具有0-1的相同比例。类值是一个字符串，其值为“g”表示良好返回，“b”表示不良返回。

使用零规则算法预测具有最多观测值的类，可以实现64.286％的基线准确度。

您可以从 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Ionosphere)了解更多信息并下载数据集。

下载数据集并将其放在当前工作目录中，名称为 **ionosphere.csv** 。

## 教程

本教程分为4个部分：

1.  欧几里德距离。
2.  最佳匹配单位。
3.  训练码本向量。
4.  电离层案例研究。

这些步骤将为实现LVQ算法并将其应用于您自己的预测建模问题奠定基础。

### 欧几里德距离

需要的第一步是计算数据集中两行之间的距离。

数据行主要由数字组成，计算两行或数字向量之间的距离的简单方法是绘制一条直线。这在2D或3D中是有意义的，并且可以很好地扩展到更高的尺寸。

我们可以使用欧几里德距离测量来计算两个向量之间的直线距离。它被计算为两个向量之间的平方差之和的平方根。

```py
distance = sqrt( sum( (x1_i - x2_i)^2 )
```

其中 **x1** 是第一行数据， **x2** 是第二行数据， **i** 是特定列的索引，因为我们对所有列求和。

对于欧几里德距离，值越小，两个记录就越相似。值为0表示两个记录之间没有差异。

下面是一个名为 **euclidean_distance（）**的函数，它在Python中实现了这一功能。

```py
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
```

您可以看到该函数假定每行中的最后一列是从距离计算中忽略的输出值。

我们可以用一个小的人为分类数据集测试这个距离函数。当我们构造LVQ算法所需的元素时，我们将使用该数据集几次。

```py
X1			X2			Y
2.7810836		2.550537003		0
1.465489372		2.362125076		0
3.396561688		4.400293529		0
1.38807019		1.850220317		0
3.06407232		3.005305973		0
7.627531214		2.759262235		1
5.332441248		2.088626775		1
6.922596716		1.77106367		1
8.675418651		-0.242068655		1
7.673756466		3.508563011		1
```

综上所述，我们可以编写一个小例子，通过打印第一行和所有其他行之间的距离来测试我们的距离函数。我们希望第一行和它自己之间的距离为0，这是一个值得注意的好事。

下面列出了完整的示例。

```py
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
	print(distance)
```

运行此示例将打印数据集中第一行和每一行之间的距离，包括其自身。

```py
0.0
1.32901739153
1.94946466557
1.55914393855
0.535628072194
4.85094018699
2.59283375995
4.21422704263
6.52240998823
4.98558538245
```

现在是时候使用距离计算来定位数据集中的最佳匹配单位。

### 2.最佳匹配单位

最佳匹配单元或BMU是与新数据最相似的码本向量。

要在数据集中找到BMU以获取新的数据，我们必须首先计算每个码本与新数据之间的距离。我们可以使用上面的距离函数来做到这一点。

计算距离后，我们必须按照与新数据的距离对所有码本进行排序。然后我们可以返回第一个或最相似的码本向量。

我们可以通过跟踪数据集中每个记录的距离作为元组来进行此操作，按距离（按降序排序）对元组列表进行排序，然后检索BMU。

下面是一个名为 **get_best_matching_unit（）**的函数，它实现了这个功能。

```py
# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]
```

您可以看到上一步中开发的 **euclidean_distance（）**函数用于计算每个码本与新 **test_row** 之间的距离。

在使用自定义键的情况下对码本和距离元组的列表进行排序，以确保在排序操作中使用元组中的第二项（ **tup [1]** ）。

最后，返回顶部或最相似的码本向量作为BMU。

我们可以使用上一节中准备的小型人为数据集来测试此功能。

下面列出了完整的示例。

```py
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Test best matching unit function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
test_row = dataset[0]
bmu = get_best_matching_unit(dataset, test_row)
print(bmu)
```

运行此示例将数据集中的BMU打印到第一个记录。正如预期的那样，第一条记录与自身最相似，位于列表的顶部。

```py
[2.7810836, 2.550537003, 0]
```

使用一组码本向量做出预测是一回事。

我们使用1最近邻居算法。也就是说，对于我们希望做出预测的每个新模式，我们在集合中找到最相似的码本向量并返回其关联的类值。

现在我们知道如何从一组码本向量中获得最佳匹配单元，我们需要学习如何训练它们。

### 3.训练码本向量

训练一组码本向量的第一步是初始化该集合。

我们可以使用训练数据集中随机特征构建的模式对其进行初始化。

下面是一个名为 **random_codebook（）**的函数，它实现了这个功能。从训练数据中选择随机输入和输出特征。

```py
# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook
```

在将码本向量初始化为随机集之后，必须调整它们以最好地总结训练数据。

这是迭代完成的。

1.  **时期**：在顶层，对于固定数量的时期或训练数据的曝光重复该过程。
2.  **训练数据集**：在一个时期内，每次使用一个训练模式来更新该码本向量集。
3.  **模式特征**：对于给定的训练模式，更新最佳匹配码本向量的每个特征以使其移近或远离。

为每个训练模式找到最佳匹配单元，并且仅更新该最佳匹配单元。训练模式和BMU之间的差异被计算为误差。比较类值（假定为列表中的最后一个值）。如果它们匹配，则将错误添加到BMU以使其更接近训练模式，否则，将其减去以将其推得更远。

调整BMU的量由学习率控制。这是对所有BMU所做更改量的加权。例如，学习率为0.3意味着BMU仅移动了训练模式和BMU之间的误差或差异的30％。

此外，调整学习率以使其在第一时期具有最大效果并且随着训练继续进行直到其在最后时期中具有最小效果的效果较小。这称为线性衰减学习率计划，也可用于人工神经网络。

我们可以按时期总结学习率的衰减如下：

```py
rate = learning_rate * (1.0 - (epoch/total_epochs))
```

我们可以通过假设学习率为0.3和10个时期来测试这个等式。每个时期的学习率如下：

```py
Epoch		Effective Learning Rate
0		0.3
1		0.27
2		0.24
3		0.21
4		0.18
5		0.15
6		0.12
7		0.09
8		0.06
9		0.03
```

我们可以把所有这些放在一起。下面是一个名为 **train_codebooks（）**的函数，它实现了在给定训练数据集的情况下训练一组码本向量的过程。

该函数对训练数据集，创建和训练的码本向量的数量，初始学习率和训练码本向量的时期数量采用3个附加参数。

您还可以看到该函数记录每个时期的总和平方误差，并打印一条消息，显示时期编号，有效学习率和总和平方误差分数。在调试训练函数或给定预测问题的特定配置时，这很有用。

您可以看到使用 **random_codebook（）**初始化码本向量和 **get_best_matching_unit（）**函数来查找一个迭代内每个训练模式的BMU。

```py
# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += error**2
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
	return codebooks
```

我们可以将它与上面的示例结合起来，为我们设计的数据集学习一组代码簿向量。

以下是完整的示例。

```py
from math import sqrt
from random import randrange
from random import seed

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += error**2
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
	return codebooks

# Test the training function
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
learn_rate = 0.3
n_epochs = 10
n_codebooks = 2
codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs)
print('Codebooks: %s' % codebooks)
```

运行该示例训练一组2个码本向量用于10个时期，初始学习率为0.3。每个时期打印细节，并显示从训练数据中学习的一组2​​个码本向量。

我们可以看到学习率的变化符合我们上面针对每个时期探讨的期望。我们还可以看到，每个时期的总和平方误差在训练结束时继续下降，并且可能有机会进一步调整示例以实现更少的错误。

```py
>epoch=0, lrate=0.300, error=43.270
>epoch=1, lrate=0.270, error=30.403
>epoch=2, lrate=0.240, error=27.146
>epoch=3, lrate=0.210, error=26.301
>epoch=4, lrate=0.180, error=25.537
>epoch=5, lrate=0.150, error=24.789
>epoch=6, lrate=0.120, error=24.058
>epoch=7, lrate=0.090, error=23.346
>epoch=8, lrate=0.060, error=22.654
>epoch=9, lrate=0.030, error=21.982
Codebooks: [[2.432316086217663, 2.839821664184211, 0], [7.319592257892681, 1.97013382654341, 1]]
```

现在我们知道如何训练一组码本向量，让我们看看如何在真实数据集上使用这个算法。

### 4.电离层案例研究

在本节中，我们将学习向量量化算法应用于电离层数据集。

第一步是加载数据集并将加载的数据转换为我们可以与欧氏距离计算一起使用的数字。为此我们将使用辅助函数 **load_csv（）**来加载文件， **str_column_to_float（）**将字符串数转换为浮点数， **str_column_to_int（）**转换​​为class列到整数值。

我们将使用5倍折叠交叉验证来评估算法。这意味着每个折叠中将有351/5 = 70.2或仅超过70个记录。我们将使用辅助函数 **evaluate_algorithm（）**来评估具有交叉验证的算法和 **accuracy_metric（）**来计算预测的准确率。

The complete example is listed below.

```py
# LVQ for the Ionosphere Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks

# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)

# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
filename = 'ionosphere.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
learn_rate = 0.3
n_epochs = 50
n_codebooks = 20
scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

运行此示例将打印每个折叠的分类准确度以及所有折叠的平均分类精度。

我们可以看出，87.143％的准确率优于64.286％的基线。我们还可以看到，我们的20个码本向量库远远少于保存整个训练数据集。

```py
Scores: [90.0, 88.57142857142857, 84.28571428571429, 87.14285714285714, 85.71428571428571]
Mean Accuracy: 87.143%
```

## 扩展

本节列出了您可能希望探索的教程的扩展。

*   **调谐参数**。上述示例中的参数未进行调整，请尝试使用不同的值来提高分类准确度。
*   **不同的距离测量**。尝试不同的距离测量，如曼哈顿距离和闵可夫斯基距离。
*   **多次通过LVQ** 。可以通过多次训练运行来更新码本向量。通过大学习率的训练进行实验，接着是大量具有较小学习率的时期来微调码本。
*   **更新更多BMU** 。尝试在训练时选择多个BMU，并将其从训练数据中拉出。
*   **更多问题**。将LVQ应用于UCI机器学习存储库中的更多分类问题。

**你有没有探索过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何在Python中从零开始实现学习向量量化算法。

具体来说，你学到了：

*   如何计算模式之间的距离并找到最佳匹配单元。
*   如何训练一组码本向量以最好地总结训练数据集。
*   如何将学习向量量化算法应用于实际预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。