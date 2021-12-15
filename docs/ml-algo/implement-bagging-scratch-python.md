# 如何用Python从头开始实现Bagging

> 原文： [https://machinelearningmastery.com/implement-bagging-scratch-python/](https://machinelearningmastery.com/implement-bagging-scratch-python/)

决策树是一种简单而强大的预测建模技术，但它们存在高度差异。

这意味着在给定不同的训练数据的情况下，树可以得到非常不同

使决策树更加健壮并实现更好表现的技术称为引导程序聚合或简称包装。

在本教程中，您将了解如何使用Python从头开始使用决策树实现装袋过程。

完成本教程后，您将了解：

*   如何创建数据集的引导样本。
*   如何使用自举模型做出预测。
*   如何将装袋应用于您自己的预测建模问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **2017年2月更新**：修复了build_tree中的错误。
*   **2017年8月更新**：修正了基尼计算中的一个错误，根据组大小添加了组基尼评分缺失的权重（感谢迈克尔！）。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Implement Bagging From Scratch With Python](img/45674f0c1bf7e879f1bfecec292ebded.jpg)

如何用Python实现套装
照片由 [Michael Cory](https://www.flickr.com/photos/khouri/5457862281/) 拍摄，保留一些权利。

## 说明

本节简要介绍Bootstrap Aggregation和将在本教程中使用的Sonar数据集。

### Bootstrap聚合算法

引导程序是具有替换的数据集的样本。

这意味着从现有数据集的随机样本创建新数据集，其中可以选择给定行并将其多次添加到样本中。

当您只有可用的有限数据集时，在估算诸如更广泛数据集的均值等值时使用它是一种有用的方法。通过创建数据集的样本并估算这些样本的均值，您可以获取这些估计的平均值，并更好地了解潜在问题的真实均值。

这种方法可以与具有高方差的机器学习算法一起使用，例如决策树。针对每个数据引导样本以及用于做出预测的那些模型的平均输出，训练单独的模型。这种技术简称为bootstrap聚合或装袋。

方差意味着算法的表现对训练数据敏感，高方差表明训练数据的变化越多，算法的表现就越差。

通过训练许多树并取其预测的平均值，可以改善诸如未修剪的决策树之类的高方差机器学习算法的表现。结果通常优于单个决策树。

除了提高表现之外，装袋的另一个好处是袋装决策树不能过度配合问题。可以继续添加树木，直到达到最大表现。

### 声纳数据集

我们将在本教程中使用的数据集是Sonar数据集。

这是一个描述声纳啁啾返回从不同表面反弹的数据集。 60个输入变量是不同角度的回报强度。这是一个二分类问题，需要一个模型来区分岩石和金属圆柱。共有208个观测结果。

这是一个众所周知的数据集。所有变量都是连续的，通常在0到1的范围内。输出变量是我的字符串“M”和摇滚的“R”，需要将其转换为整数1和0。

通过预测数据集（M或矿）中具有最多观测值的类，零规则算法可以实现53％的准确度。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))中了解有关此数据集的更多信息。

免费下载数据集并将其放在工作目录中，文件名为 **sonar.all-data.csv** 。

## 教程

本教程分为两部分：

1.  Bootstrap Resample。
2.  声纳数据集案例研究。

这些步骤提供了实现和将决策树的引导聚合应用于您自己的预测建模问题所需的基础。

### 1\. Bootstrap Resample

让我们首先深入了解bootstrap方法的工作原理。

我们可以通过从数据集中随机选择行并将它们添加到新列表来创建数据集的新样本。我们可以针对固定数量的行重复此操作，或者直到新数据集的大小与原始数据集的大小的比率匹配为止。

我们可以通过不删除已选择的行来允许替换采样，以便将来可以选择。

下面是一个名为 **subsample（）**的函数，它实现了这个过程。来自随机模块的 **randrange（）**函数用于选择随机行索引以在循环的每次迭代中添加到样本。样本的默认大小是原始数据集的大小。

```py
# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample
```

我们可以使用此函数来估计人为数据集的平均值。

首先，我们可以创建一个包含20行和0到9之间的单列随机数的数据集，并计算平均值。

然后，我们可以制作原始数据集的引导样本，计算平均值，并重复此过程，直到我们有一个均值列表。取这些样本均值的平均值可以给出我们对整个数据集平均值的可靠估计。

下面列出了完整的示例。

每个bootstrap样本创建为原始20个观察数据集的10％样本（或2个观察值）。然后，我们通过创建原始数据集的1,10,100个引导样本，计算它们的平均值，然后平均所有这些估计的平均值来进行实验。

```py
from random import seed
from random import random
from random import randrange

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers) / float(len(numbers))

seed(1)
# True mean
dataset = [[randrange(10)] for i in range(20)]
print('True Mean: %.3f' % mean([row[0] for row in dataset]))
# Estimated means
ratio = 0.10
for size in [1, 10, 100]:
	sample_means = list()
	for i in range(size):
		sample = subsample(dataset, ratio)
		sample_mean = mean([row[0] for row in sample])
		sample_means.append(sample_mean)
	print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))
```

运行该示例将打印我们要估计的原始平均值。

然后我们可以从各种不同数量的自举样本中看到估计的平均值。我们可以看到，通过100个样本，我们可以很好地估计平均值。

```py
True Mean: 4.450
Samples=1, Estimated Mean: 4.500
Samples=10, Estimated Mean: 3.300
Samples=100, Estimated Mean: 4.480
```

我们可以从每个子样本创建一个模型，而不是计算平均值。

接下来，让我们看看如何组合多个bootstrap模型的预测。

### 2.声纳数据集案例研究

在本节中，我们将随机森林算法应用于Sonar数据集。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **sonar.all-data.csv** 。

首先加载数据集，将字符串值转换为数字，并将输出列从字符串转换为0到1的整数值。这可以通过辅助函数 **load_csv（）**， **str_column_to_float（ ）**和 **str_column_to_int（）**加载和准备数据集。

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型误差。分类精度将用于评估每个模型。这些行为在 **cross_validation_split（）**， **accuracy_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们还将使用适用于装袋的分类和回归树（CART）算法的实现，包括辅助函数 **test_split（）**将数据集分成组， **gini_index（）**来评估分裂点， **get_split（）**找到最佳分裂点， **to_terminal（）**， **split（）**和 **build_tree（）**使用创建单个决策树，**预测（）**用决策树和上一步骤中描述的**子样本（）**函数做出预测，以制作训练数据集的子样本

开发了一个名为 **bagging_predict（）**的新函数，负责使用每个决策树做出预测并将预测组合成单个返回值。这是通过从袋装树所做的预测列表中选择最常见的预测来实现的。

最后，开发了一个名为 **bagging（）**的新功能，负责创建训练数据集的样本，在每个样本上训练决策树，然后使用袋装树列表对测试数据集做出预测。

The complete example is listed below.

```py
# Bagging Algorithm on the Sonar dataset
from random import seed
from random import randrange
from csv import reader

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

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
		# for i in range(len(dataset)):
		# 	row = dataset[randrange(len(dataset))]
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

# Test bagging on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 6
min_size = 2
sample_size = 0.50
for n_trees in [1, 5, 10, 50]:
	scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

k值为5用于交叉验证，每次迭代时评估每个折叠208/5 = 41.6或仅超过40个记录。

构建深度树，最大深度为6，每个节点为2的最小训练行数。训练数据集的样本创建为原始数据集大小的50％。这是为了强制用于训练每棵树的数据集子样本中的某些变体。装袋的默认设置是使样本数据集的大小与原始训练数据集的大小相匹配。

评估了一系列4种不同数量的树以显示算法的行为。

打印每个折叠的精度和每个配置的平均精度。随着树木数量的增加，我们可以看到表现略有提升的趋势。

```py
Trees: 1
Scores: [87.8048780487805, 65.85365853658537, 65.85365853658537, 65.85365853658537, 73.17073170731707]
Mean Accuracy: 71.707%

Trees: 5
Scores: [60.97560975609756, 80.48780487804879, 78.04878048780488, 82.92682926829268, 63.41463414634146]
Mean Accuracy: 73.171%

Trees: 10
Scores: [60.97560975609756, 73.17073170731707, 82.92682926829268, 80.48780487804879, 68.29268292682927]
Mean Accuracy: 73.171%

Trees: 50
Scores: [63.41463414634146, 75.60975609756098, 80.48780487804879, 75.60975609756098, 85.36585365853658]
Mean Accuracy: 76.098%
```

这种方法的一个难点是，即使构建了深树，创建的袋装树也非常相似。反过来，这些树的预测也是相似的，并且我们希望在训练数据集的不同样本上训练的树之间的高方差减小。

这是因为在构造选择相同或相似分裂点的树时使用的贪婪算法。

本教程试图通过约束用于训练每棵树的样本大小来重新注入此方差。更强大的技术是约束在创建每个分割点时可以评估的特征。这是随机森林算法中使用的方法。

## 扩展

*   **调整示例**。探索树木数量甚至单个树配置的不同配置，以了解您是否可以进一步改善结果。
*   **Bag另一种算法**。其他算法可与套袋一起使用。例如，具有低k值的k-最近邻算法将具有高方差并且是用于装袋的良好候选者。
*   **回归问题**。套袋可以与回归树一起使用。您可以从袋装树中返回预测的平均值，而不是从预测集中预测最常见的类值。回归问题的实验。

**你有没有试过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何使用Python从头开始实现引导程序聚合。

具体来说，你学到了：

*   如何创建子样本并估计引导数量。
*   如何创建决策树集合并使用它们做出预测。
*   如何将装袋应用于现实世界的预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。