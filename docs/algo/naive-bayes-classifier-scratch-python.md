# 如何在 Python 中从零开始实现朴素贝叶斯

> 原文： [https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)

朴素贝叶斯算法简单有效，应该是您尝试分类问题的第一种方法之一。

在本教程中，您将学习 Naive Bayes 算法，包括它的工作原理以及如何在 Python 中从零开始实现它。

*   **更新**：查看关于使用朴素贝叶斯算法的提示的后续内容：“ [Better Naive Bayes：从 Naive Bayes 算法中获取最多的 12 个技巧](http://machinelearningmastery.com/better-naive-bayes/ "Better Naive Bayes: 12 Tips To Get The Most From The Naive Bayes Algorithm")”。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

[![naive bayes classifier](img/e6a92a6bcab0d5c51968019190f71f21.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/12/naive-bayes-classifier.jpg)

朴素贝叶斯分类器
摄影： [Matt Buck](https://www.flickr.com/photos/mattbuck007/3676624894) ，保留一些权利

## 关于朴素贝叶斯

朴素贝叶斯算法是一种直观的方法，它使用属于每个类的每个属性的概率来做出预测。如果您想要概率性地建模预测性建模问题，那么您将提出监督学习方法。

朴素贝叶斯通过假设属于给定类值的每个属性的概率独立于所有其他属性来简化概率的计算。这是一个强有力的假设，但会产生一种快速有效的方法。

给定属性值的类值的概率称为条件概率。通过将条件概率乘以给定类值的每个属性，我们得到属于该类的数据实例的概率。

为了做出预测，我们可以计算属于每个类的实例的概率，并选择具有最高概率的类值。

朴素碱基通常使用分类数据来描述，因为它易于使用比率进行描述和计算。用于我们目的的更有用的算法版本支持数字属性并假设每个数字属性的值是正态分布的（落在钟形曲线上的某处）。同样，这是一个强有力的假设，但仍然提供了可靠的结果。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的 60 多种算法的方便思维导图。

下载，打印并使用它。

## 预测糖尿病的发病

我们将在本教程中使用的测试问题是[皮马印第安人糖尿病问题](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)。

这个问题包括对 Pima 印第安人专利的医疗细节的 768 次观察。记录描述了从患者身上获取的瞬时测量值，例如他们的年龄，怀孕次数和血液检查次数。所有患者均为 21 岁或以上的女性。所有属性都是数字，其单位因属性而异。

每个记录具有类别值，该类别值指示患者在进行测量（1）或不进行测量（0）的 5 年内是否患有糖尿病。

这是一个标准的数据集，已在机器学习文献中进行了大量研究。良好的预测准确率为 70％-76％。

下面是来自 _pima-indians.data.csv_ 文件的示例，以了解我们将要使用的数据（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

Sample from the pima-indians.data.csv file

```py
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
```

## 朴素贝叶斯算法教程

本教程分为以下几个步骤：

1.  **句柄数据**：从 CSV 文件加载数据并将其拆分为训练和测试数据集。
2.  **汇总数据**：总结训练数据集中的属性，以便我们可以计算概率并做出预测。
3.  **做出预测**：使用数据集的摘要生成单个预测。
4.  **制作预测**：根据测试数据集和汇总的训练数据集生成预测。
5.  **评估准确度**：评估为测试数据集做出的预测的准确率，作为所有预测中的正确百分比。
6.  **将它绑在一起**：使用所有代码元素来呈现 Naive Bayes 算法的完整且独立的实现。

### 1.处理数据

我们需要做的第一件事是加载我们的数据文件。数据为 CSV 格式，没有标题行或任何引号。我们可以使用 open 函数打开文件，并使用 csv 模块中的 reader 函数读取数据行。

我们还需要将作为字符串加载的属性转换为可以使用它们的数字。下面是用于加载 Pima indians 数据集的 **loadCsv（）**函数。

Load a CSV file of scalars into memory Python

```py
import csv
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
```

我们可以通过加载 pima indians 数据集并打印已加载的数据实例的数量来测试此函数。

Test the loadCsv() function Python

```py
filename = 'pima-indians-diabetes.data.csv'
dataset = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset))
```

运行此测试，您应该看到类似的内容：

Example output of testing the loadCsv() function

```py
Loaded data file pima-indians-diabetes.data.csv rows
```

接下来，我们需要将数据拆分为 Naive Bayes 可用于做出预测的训练数据集和我们可用于评估模型准确率的测试数据集。我们需要将数据集随机分成训练和数据集，比率为 67％训练和 33％测试（这是在数据集上测试算法的常用比率）。

下面是 **splitDataset（）**函数，它将给定数据集拆分为给定的分割比率。

Split a loaded dataset into a train and test datasets Python

```py
import random
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
```

我们可以通过定义一个包含 5 个实例的模拟数据集来测试它，将其拆分为训练和测试数据集并打印出来以查看哪些数据实例最终到达哪里。

Test the splitDataset() function Python

```py
dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset, splitRatio)
print('Split {0} rows into train with {1} and test with {2}').format(len(dataset), train, test)
```

Running this test, you should see something like:

Example output from testing the splitDataset() function

```py
Split 5 rows into train with [[4], [3], [5]] and test with [[1], [2]]
```

### 2.总结数据

朴素贝叶斯模型由训练数据集中的数据摘要组成。然后在做出预测时使用此摘要。

收集的训练数据摘要涉及每个属性的平均值和标准偏差，按类别值。例如，如果有两个类值和 7 个数值属性，那么我们需要每个属性（7）和类值（2）组合的均值和标准差，即 14 个属性摘要。

在做出预测以计算属于每个类值的特定属性值的概率时，这些是必需的。

我们可以将此摘要数据的准备工作分解为以下子任务：

1.  按类别分开数据
2.  计算平均值
3.  计算标准差
4.  总结数据集
5.  按类别汇总属性

#### 按类别分开数据

第一个任务是按类值分隔训练数据集实例，以便我们可以计算每个类的统计量。我们可以通过创建每个类值的映射到属于该类的实例列表并将实例的整个数据集排序到适当的列表中来实现。

下面的 **separateByClass（）**函数就是这样做的。

The separateByClass() function

```py
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
```

您可以看到该函数假定最后一个属性（-1）是类值。该函数将类值映射返回到数据实例列表。

我们可以使用一些示例数据测试此函数，如下所示：

Testing the separateByClass() function

```py
dataset = [[1,20,1], [2,21,0], [3,22,1]]
separated = separateByClass(dataset)
print('Separated instances: {0}').format(separated)
```

Running this test, you should see something like:

Output when testing the separateByClass() function

```py
Separated instances: {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
```

#### 计算平均值

我们需要计算类值的每个属性的平均值。均值是数据的中心中心或中心趋势，我们将在计算概率时将其用作高斯分布的中间。

我们还需要计算类值的每个属性的标准偏差。标准偏差描述了数据传播的变化，我们将用它来表征计算概率时高斯分布中每个属性的预期传播。

标准偏差计算为方差的平方根。方差计算为每个属性值与平均值的平方差的平均值。注意我们使用的是 N-1 方法，它在计算方差时从属性值的数量中减去 1。

Functions to calculate the mean and standard deviations of attributes

```py
import math
def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
```

我们可以通过取 1 到 5 的数字的平均值来测试这个。

Code to test the mean() and stdev() functions

```py
numbers = [1,2,3,4,5]
print('Summary of {0}: mean={1}, stdev={2}').format(numbers, mean(numbers), stdev(numbers))
```

Running this test, you should see something like:

Output of testing the mean() and stdev() functions

```py
Summary of [1, 2, 3, 4, 5]: mean=3.0, stdev=1.58113883008
```

#### 总结数据集

现在我们有了汇总数据集的工具。对于给定的实例列表（对于类值），我们可以计算每个属性的均值和标准差。

zip 函数将数据实例中每个属性的值分组到它们自己的列表中，以便我们可以计算属性的均值和标准差值。

The summarize() function

```py
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
```

我们可以用一些测试数据来测试这个 **summarize（）**函数，该数据显示第一和第二数据属性的平均值和标准偏差值明显不同。

Code to test the summarize() function

```py
dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset)
print('Attribute summaries: {0}').format(summary)
```

Running this test, you should see something like:

Output of testing the summarize() function

```py
Attribute summaries: [(2.0, 1.0), (21.0, 1.0)]
```

#### 按类别汇总属性

我们可以通过首先将训练数据集分成按类分组的实例来将它们整合在一起。然后计算每个属性的摘要。

The summarizeByClass() function

```py
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
```

我们可以用一个小的测试数据集测试这个 **summarizeByClass（）**函数。

Code to test the summarizeByClass() function

```py
dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset)
print('Summary by class value: {0}').format(summary)
```

Running this test, you should see something like:

Output from testing the summarizeByClass() function

```py
Summary by class value: 
{0: [(3.0, 1.4142135623730951), (21.5, 0.7071067811865476)], 
1: [(2.0, 1.4142135623730951), (21.0, 1.4142135623730951)]}
```

### 3.做出预测

我们现在准备使用从我们的训练数据准备的摘要做出预测。做出预测涉及计算给定数据实例属于每个类的概率，然后选择具有最大概率的类作为预测。

我们可以将这部分分为以下任务：

1.  计算高斯概率密度函数
2.  计算类概率
3.  做一个预测
4.  估计准确度

#### 计算高斯概率密度函数

在给定从训练数据估计的属性的已知平均值和标准偏差的情况下，我们可以使用高斯函数来估计给定属性值的概率。

假定为每个属性和类值准备的属性汇总，结果是给定类值的给定属性值的条件概率。

有关高斯概率密度函数的详细信息，请参阅参考资料。总之，我们将已知细节插入高斯（属性值，平均值和标准偏差）并读取属性值属于类的可能性。

在 **calculateProbability（）**函数中，我们首先计算指数，然后计算主要除法。这让我们可以在两条线上很好地拟合方程。

The calculateProbability() function

```py
import math
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
```

我们可以使用一些示例数据对此进行测试，如下所示。

Code to test the calculateProbability() function

```py
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x, mean, stdev)
print('Probability of belonging to this class: {0}').format(probability)
```

Running this test, you should see something like:

Output from testing the calculateProbability() function

```py
Probability of belonging to this class: 0.0624896575937
```

#### 计算类概率

现在我们可以计算出属于某个类的属性的概率，我们可以组合数据实例的所有属性值的概率，并得出整个数据实例属于该类的概率。

我们将概率乘以它们，将概率结合在一起。在下面的 **calculateClassProbabilities（）**中，通过将每个类的属性概率相乘来计算给定数据实例的概率。结果是类值与概率的映射。

Code for the calculateClassProbabilities() function

```py
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
```

我们可以测试 **calculateClassProbabilities（）**函数。

Code to test the calculateClassProbabilities() function

```py
summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print('Probabilities for each class: {0}').format(probabilities)
```

Running this test, you should see something like:

Output from testing the calculateClassProbabilities() function

```py
Probabilities for each class: {0: 0.7820853879509118, 1: 6.298736258150442e-05}
```

#### 做一个预测

现在，可以计算属于每个类值的数据实例的概率，我们可以查找最大概率并返回关联类。

**predict（）**函数属于那个。

Implementation of the predict() function

```py
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
```

我们可以测试 **predict（）**函数如下：

Code to test the predict() function

```py
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print('Prediction: {0}').format(result)
```

Running this test, you should see something like:

Output of testing the predict() function

```py
Prediction: A
```

### 4.做出预测

最后，我们可以通过对测试数据集中的每个数据实例做出预测来估计模型的准确率。 **getPredictions（）**将执行此操作并返回每个测试实例的预测列表。

Code for the getPredictions() function

```py
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
```

我们可以测试 **getPredictions（）**函数。

Code to test the getPredictions() function

```py
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
testSet = [[1.1, '?'], [19.1, '?']]
predictions = getPredictions(summaries, testSet)
print('Predictions: {0}').format(predictions)
```

Running this test, you should see something like:

Output from testing the getPredictions() function

```py
Predictions: ['A', 'B']
```

### 5.获得准确率

可以将预测与测试数据集中的类值进行比较，并且可以将分类精度计算为 0 和 0 之间的准确度比率。和 100％。 **getAccuracy（）**将计算此准确率。

Code for the getAccuracy() function

```py
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
```

我们可以使用下面的示例代码测试 **getAccuracy（）**函数。

Code to test the getAccuracy() function

```py
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}').format(accuracy)
```

Running this test, you should see something like:

Output from testing the getAccuracy() function

```py
Accuracy: 66.6666666667
```

### 6.把它绑在一起

最后，我们需要将它们结合在一起。

下面提供了从零开始在 Python 中实现的 Naive Bayes 的完整代码清单。

Complete code for implementing Naive Bayes from scratch in Python Python

```py
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()
```

运行该示例提供如下输出：

Output from running the final code

```py
Split 768 rows into train=514 and test=254 rows
Accuracy: 76.3779527559%
```

## 实现扩展

本节为您提供了可以应用的扩展的概念，并使用您在本教程中实现的 Python 代码进行调查。

您已经从零开始在 python 中实现了自己的 Gaussian Naive Bayes 版本。

您可以进一步扩展实现。

*   **计算类概率**：更新示例以概括属于每个类的数据实例的概率作为比率。这可以被计算为属于一个类的数据实例的概率除以属于每个类的数据实例的概率之和。例如，A 类的概率为 0.02，B 类的概率为 0.001，属于 A 类的实例的可能性为（0.02 /(0.02 + 0.001））* 100，约为 95.23％。
*   **对数概率**：给定属性值的每个类的条件概率很小。当它们相乘时会产生非常小的值，这可能导致浮点下溢（数字太小而无法在 Python 中表示）。对此的常见修复是将概率的对数组合在一起。研究并实现这一改进。
*   **标称属性**：更新实现以支持名义属性。这非常相似，您可以为每个属性收集的摘要信息是每个类的类别值的比率。深入了解参考资料以获取更多信息。
*   **不同的密度函数**（`bernoulli`或 _ 多项式 _）：我们已经看过高斯朴素贝叶斯，但你也可以看看其他分布。实现不同的分布，例如多项式，bernoulli 或内核朴素贝叶斯，它们对属性值的分布和/或它们与类值的关系做出不同的假设。

## 资源和进一步阅读

本节将提供一些资源，您可以使用这些资源来了解 Naive Bayes 算法的更多信息，包括它的工作原理和原理以及在代码中实现它的实际问题。

### 问题

有关预测糖尿病发病问题的更多资源。

*   [Pima Indians 糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)：此页面提供对数据集文件的访问，描述属性并列出使用该数据集的论文。
*   [数据集文件](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)：数据集文件。
*   [数据集摘要](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names)：数据集属性的描述。
*   [糖尿病数据集结果](http://www.is.umk.pl/projects/datasets.html#Diabetes)：该数据集上许多标准算法的准确率。

### 码

本节链接到流行的机器学习库中朴素贝叶斯的开源实现。如果您正在考虑实现自己的方法版本以供操作使用，请查看这些内容。

*   [Scikit-Learn 中的朴素贝叶斯](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py)：在 scikit-learn 库中实现朴素的贝叶斯。
*   [朴素贝叶斯文档](http://scikit-learn.org/stable/modules/naive_bayes.html)：朴素贝叶斯的 Scikit-Learn 文档和示例代码

### 图书

您可能有一本或多本关于应用机器学习的书籍。本节重点介绍有关机器学习的常见应用书籍中涉及朴素贝叶斯的部分或章节。

*   [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，第 353 页
*   [数据挖掘：实用机器学习工具和技术](http://www.amazon.com/dp/0123748569?tag=inspiredalgor-20)，第 94 页
*   [黑客机器学习](http://www.amazon.com/dp/1449303714?tag=inspiredalgor-20)，第 78 页
*   [统计学习简介：在 R](http://www.amazon.com/dp/1461471370?tag=inspiredalgor-20) 中的应用，第 138 页
*   [机器学习：算法视角](http://www.amazon.com/dp/1420067184?tag=inspiredalgor-20)，第 171 页
*   [机器学习在行动](http://www.amazon.com/dp/1617290181?tag=inspiredalgor-20)，第 61 页（第 4 章）
*   [机器学习](http://www.amazon.com/dp/0070428077?tag=inspiredalgor-20)，第 177 页（第 6 章）

## 下一步

采取行动。

按照教程从零开始实现 Naive Bayes。使示例适应另一个问题。遵循扩展并改进实现。

发表评论并分享您的经验。

**更新**：查看关于使用朴素贝叶斯算法的提示的后续内容：“ [Better Naive Bayes：从 Naive Bayes 算法中获取最多的 12 个技巧](http://machinelearningmastery.com/better-naive-bayes/ "Better Naive Bayes: 12 Tips To Get The Most From The Naive Bayes Algorithm")”