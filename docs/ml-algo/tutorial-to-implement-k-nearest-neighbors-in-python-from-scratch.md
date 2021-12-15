# 从零开始在Python中实现 K 最近邻

> 原文： [https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

K 最近邻算法（或简称kNN）是一种易于理解和实现的算法，是您可以随意使用的强大工具。

在本教程中，您将从零开始在Python（2.7）中实现K 最近邻算法。实现将特定于分类问题，并将使用虹膜花分类问题进行演示。

本教程适合您，如果您是Python程序员，或者可以快速获取python的程序员，并且您对如何从零开始实现K 最近邻算法感兴趣。

[![K 最近邻 algorithm](img/db0046e6a5d2174b405ded8fd90d8edb.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/09/k-Nearest-Neighbors-algorithm.png)

K 最近邻 algorithm
图片来自 [Wikipedia](http://en.wikipedia.org/wiki/File:Map1NN.png) ，保留所有权利

## 什么是K 最近邻

kNN的模型是整个训练数据集。当对于看不见的数据实例需要预测时，kNN算法将在训练数据集中搜索k个最相似的实例。总结了最相似实例的预测属性，并将其作为未见实例的预测返回。

相似性度量取决于数据类型。对于实值数据，可以使用欧几里德距离。可以使用其他类型的数据，例如分类或二进制数据，汉明距离。

在回归问题的情况下，可以返回预测属性的平均值。在分类的情况下，可以返回最普遍的类别。

## K 最近邻是如何工作的

kNN算法属于基于实例的竞争学习和懒惰学习算法的家族。

基于实例的算法是使用数据实例（或行）对问题建模以便做出预测决策的算法。 kNN算法是基于实例的方法的一种极端形式，因为所有训练观察都被保留作为模型的一部分。

它是一种竞争性学习算法，因为它在内部使用模型元素（数据实例）之间的竞争来做出预测决策。数据实例之间的客观相似性度量使得每个数据实例竞争“赢”或与给定的未见数据实例最相似并且有助于预测。

延迟学习是指算法在需要预测之前不构建模型的事实。这是懒惰的，因为它只在最后一秒工作。这样做的好处是只包括与看不见的数据相关的数据，称为本地化模型。缺点是在较大的训练数据集上重复相同或类似的搜索可能在计算上是昂贵的。

最后，kNN非常强大，因为它不会假设任何有关数据的内容，除了可以在任何两个实例之间一致地计算距离度量。因此，它被称为非参数或非线性，因为它不假设功能形式。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 使用测量对花进行分类

我们将在本教程中使用的测试问题是虹膜分类。

该问题包括150个来自三个不同物种的鸢尾花的观察结果。给定花有4种尺寸：萼片长度，萼片宽度，花瓣长度和花瓣宽度，均以厘米为单位。预测的属性是物种，它是setosa，versicolor或virginica之一。

它是一个标准数据集，其中物种对于所有实例都是已知的。因此，我们可以将数据拆分为训练和测试数据集，并使用结果来评估我们的算法实现。该问题的良好分类准确度高于90％正确，通常为96％或更高。

*   [下载鸢尾花数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

使用文件名“`iris.data`”将文件保存在当前工作目录中。

## 如何在Python中实现K 最近邻

本教程分为以下几个步骤：

1.  **句柄**数据：从CSV打开数据集并拆分为测试/训练数据集。
2.  **相似度**：计算两个数据实例之间的距离。
3.  **邻居**：找到k个最相似的数据实例。
4.  **响应**：从一组数据实例生成响应。
5.  **准确度**：总结预测的准确率。
6.  **主**：将它们捆绑在一起。

### 1.处理数据

我们需要做的第一件事是加载我们的数据文件。数据为CSV格式，没有标题行或任何引号。我们可以使用open函数打开文件，并使用 [csv](https://docs.python.org/2/library/csv.html) 模块中的reader函数读取数据行。

Read a CSV File in Python Python

```py
import csv
with open('iris.data', 'rb') as csvfile:
	lines = csv.reader(csvfile)
	for row in lines:
		print ', '.join(row)
```

接下来，我们需要将数据拆分为kNN可用于做出预测的训练数据集和我们可用于评估模型准确率的测试数据集。

我们首先需要将作为字符串加载的花卉度量转换为我们可以使用的数字。接下来，我们需要将数据集随机分成训练和数据集。训练/测试的比率为67/33是使用的标准比率。

将它们全部拉到一起，我们可以定义一个名为 **loadDataset** 的函数，该函数使用提供的文件名加载CSV，并使用提供的分割比率将其随机分成训练和测试数据集。

Load a dataset and spit into train and test sets in Python Python

```py
import csv
import random
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
```

将iris flowers数据集CSV文件下载到本地目录。我们可以使用我们的iris数据集测试此函数，如下所示：

Test the loadDataset function in Python Python

```py
trainingSet=[]
testSet=[]
loadDataset('iris.data', 0.66, trainingSet, testSet)
print 'Train: ' + repr(len(trainingSet))
print 'Test: ' + repr(len(testSet))
```

### 2.相似性

为了做出预测，我们需要计算任何两个给定数据实例之间的相似性。这是必需的，以便我们可以在训练数据集中为测试数据集的给定成员定位k个最相似的数据实例，然后做出预测。

鉴于所有四个花测量都是数字并具有相同的单位，我们可以直接使用欧几里德距离测量。这被定义为两个数字数组之间的平方差之和的平方根（再读几次并让它沉入其中）。

另外，我们想要控制在距离计算中包括哪些字段。具体来说，我们只想包含前4个属性。一种方法是将欧氏距离限制为固定长度，忽略最终尺寸。

将所有这些放在一起我们可以定义 **euclideanDistance** 函数如下：

Calculate Euclidean distance in Python Python

```py
import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
```

我们可以使用一些示例数据测试此函数，如下所示：

Test the euclideanDistance function in Python Python

```py
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print 'Distance: ' + repr(distance)
```

### 3.邻居

现在我们有一个相似性度量，我们可以使用它为给定的看不见的实例收集k个最相似的实例。

这是计算所有实例的距离并选择具有最小距离值的子集的直接过程。

下面是 **getNeighbors** 函数，该函数从给定测试实例的训练集中返回k个最相似的邻居（使用已定义的 **euclideanDistance** 函数）

Locate most similar neighbours in Python Python

```py
import operator 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
```

我们可以测试这个函数如下：

Test the getNeighbors function in Python Python

```py
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)
```

### 4.回应

一旦我们为测试实例找到了最相似的邻居，下一个任务就是根据这些邻居设计预测响应。

我们可以通过允许每个邻居投票选择他们的类属性来做到这一点，并将多数投票作为预测。

下面提供了从多个邻居获得多数投票回复的功能。它假定该类是每个邻居的最后一个属性。

Summarize a prediction from neighbours in Python Python

```py
import operator
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
```

我们可以用一些测试邻居测试这个函数，如下所示：

Test the getResponse function in Python Python

```py
neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbors)
print(response)
```

在绘制的情况下，此方法返回一个响应，但您可以以特定方式处理此类情况，例如不返回响应或选择无偏的随机响应。

### 5.准确率

我们已经完成了kNN算法的所有部分。一个重要的问题是如何评估预测的准确率。

评估模型准确率的一种简单方法是计算所有预测中的总正确预测的比率，称为分类准确度。

下面是 **getAccuracy** 函数，该函数对总正确预测进行求和，并以正确分类的百分比形式返回准确度。

Calculate accuracy of predictions in Python Python

```py
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
```

我们可以使用测试数据集和预测来测试此函数，如下所示：

Test the getAccuracy function in Python Python

```py
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)
```

### 6.主要

我们现在拥有算法的所有元素，我们可以将它们与主函数绑定在一起。

下面是在Python中从零开始实现kNN算法的完整示例。

Example of kNN implemented from Scratch in Python Python

```py
# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')

main()
```

运行该示例，您将看到每个预测的结果与测试集中的实际类值进行比较。在运行结束时，您将看到模型的准确率。在这种情况下，略高于98％。

Example output

```py
...
> predicted='Iris-virginica', actual='Iris-virginica'
> predicted='Iris-virginica', actual='Iris-virginica'
> predicted='Iris-virginica', actual='Iris-virginica'
> predicted='Iris-virginica', actual='Iris-virginica'
> predicted='Iris-virginica', actual='Iris-virginica'
Accuracy: 98.0392156862745%
```

## 扩展的想法

本节为您提供了可以应用的扩展的概念，并使用您在本教程中实现的Python代码进行调查。

*   **回归**：您可以调整实现以适应回归问题（预测实值属性）。最接近的实例的摘要可涉及取预测属性的均值或中值。
*   **标准化**：当属性之间的度量单位不同时，属性可能在它们对距离度量的贡献中占主导地位。对于这些类型的问题，您需要在计算相似度之前将所有数据属性重新缩放到0-1范围内（称为规范化）。更新模型以支持数据规范化。
*   **替代距离测量**：有许多可用的距离测量，如果您愿意，您甚至可以开发自己的特定于域的距离测量。实现替代距离测量，例如曼哈顿距离或向量点积。

您可能希望探索此算法的更多扩展。另外两个想法包括支持对预测的k-最相似实例的距离加权贡献以及用于搜索类似实例的更高级的基于数据树的结构。

## 资源了解更多

本节将提供一些资源，您可以使用这些资源来了解K 最近邻算法的更多信息，包括它的工作原理和原理以及在代码中实现它的实际问题。

### 问题

*   [维基百科上的鸢尾花数据集](http://en.wikipedia.org/wiki/Iris_flower_data_set)
*   [UCI机器学习库Iris数据集](https://archive.ics.uci.edu/ml/datasets/Iris)

### 码

本节链接到流行的机器学习库中的kNN的开源实现。如果您正在考虑实现自己的方法版本以供操作使用，请查看这些内容。

*   [在scikit-learn中实现kNN](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors)
*   [在Weka中实现kNN](https://github.com/baron/weka/blob/master/weka/src/main/java/weka/classifiers/lazy/IBk.java) （非正式）

### 图书

您可能有一本或多本关于应用机器学习的书籍。本节重点介绍有关机器学习的常见应用书籍中涉及K 最近邻的章节或章节。

*   [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，第159和350页。
*   [数据挖掘：实用机器学习工具和技术，第三版（数据管理系统中的Morgan Kaufmann系列）](http://www.amazon.com/dp/0123748569?tag=inspiredalgor-20)，第76,128和235页。
*   [黑客机器学习](http://www.amazon.com/dp/1449303714?tag=inspiredalgor-20)，第10章。
*   [机器学习在行动](http://www.amazon.com/dp/1617290181?tag=inspiredalgor-20)，第2章。
*   [编程集体智慧：构建智能Web 2.0应用程序](http://www.amazon.com/dp/0596529325?tag=inspiredalgor-20)，第2章和第8章以及第293页。

## 教程摘要

在本教程中，您了解了k-Nearest Neighbor算法，它的工作原理以及可用于思考算法并将其与其他算法相关联的一些隐喻。您从零开始在Python中实现kNN算法，以便您了解每行代码并调整实现以探索扩展并满足您自己的项目需求。

以下是本教程的5个主要学习内容：

*   **k-最近邻**：一种理解和实现的简单算法，以及强大的非参数方法。
*   **基于实例的方法**：使用数据实例（观察）对问题建模。
*   **竞争学习**：学习和预测决策是由模型元素之间的内部竞争决定的。
*   **懒惰学习**：为了做出预测，只有在需要时才构建模型。
*   **相似性度量**：计算数据实例之间的客观距离度量是算法的关键特征。

您是否使用本教程实现了kNN？你是怎么去的？你学到了什么？