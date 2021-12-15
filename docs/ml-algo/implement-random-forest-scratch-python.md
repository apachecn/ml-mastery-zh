# 如何在Python中从零开始实现随机森林

> 原文： [https://machinelearningmastery.com/implement-random-forest-scratch-python/](https://machinelearningmastery.com/implement-random-forest-scratch-python/)

决策树可能遭受高度变化，这使得它们的结果对于所使用的特定训练数据而言是脆弱的。

从训练数据样本中构建多个模型（称为装袋）可以减少这种差异，但树木具有高度相关性。

随机森林是套袋的扩展，除了根据训练数据的多个样本构建树木之外，它还限制了可用于构建树木的特征，迫使树木变得不同。反过来，这可以提升表现。

在本教程中，您将了解如何在Python中从零开始实现随机森林算法。

完成本教程后，您将了解：

*   袋装决策树与随机森林算法的区别。
*   如何构造具有更多方差的袋装决策树。
*   如何将随机森林算法应用于预测建模问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **2017年2月更新**：修复了build_tree中的错误。
*   **2017年8月更新**：修正了基尼计算中的一个错误，根据组大小添加了组基尼评分缺失的权重（感谢迈克尔！）。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Implement Random Forest From Scratch in Python](img/bd883c81857661285726387e7733a170.jpg)

如何在Python中从零开始实现随机森林
照片由 [InspireFate Photography](https://www.flickr.com/photos/inspirefatephotography/7569736320/) ，保留一些权利。

## 描述

本节简要介绍随机森林算法和本教程中使用的Sonar数据集。

### 随机森林算法

决策树涉及在每个步骤中从数据集中贪婪地选择最佳分割点。

如果没有修剪，该算法使决策树易受高方差影响。通过使用训练数据集的不同样本（问题的不同视图）创建多个树并组合它们的预测，可以利用和减少这种高方差。这种方法简称为bootstrap聚合或装袋。

套袋的限制是使用相同的贪婪算法来创建每个树，这意味着可能在每个树中选择相同或非常相似的分裂点，使得不同的树非常相似（树将相关）。反过来，这使他们的预测相似，减轻了最初寻求的差异。

我们可以通过限制贪婪算法在创建树时在每个分割点处可以评估的特征（行）来强制决策树不同。这称为随机森林算法。

与装袋一样，采集训练数据集的多个样本，并对每个样本进行不同的训练。不同之处在于，在每个点处对数据进行拆分并添加到树中，只能考虑固定的属性子集。

对于分类问题，我们将在本教程中看到的问题类型，要考虑拆分的属性数量限制为输入要素数量的平方根。

```py
num_features_for_split = sqrt(total_input_features)
```

这一个小变化的结果是彼此更加不同的树（不相关）导致更多样化的预测和组合预测，其通常具有单个树或单独装袋的更好表现。

### 声纳数据集

我们将在本教程中使用的数据集是Sonar数据集。

这是一个描述声纳啁啾返回从不同表面反弹的数据集。 60个输入变量是不同角度的回报强度。这是一个二分类问题，需要一个模型来区分岩石和金属圆柱。共有208个观测结果。

这是一个众所周知的数据集。所有变量都是连续的，通常在0到1的范围内。输出变量是我的字符串“M”和摇滚的“R”，需要将其转换为整数1和0。

通过预测数据集（M或矿）中具有最多观测值的类，零规则算法可以实现53％的准确度。

您可以在 [UCI机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))中了解有关此数据集的更多信息。

免费下载数据集并将其放在工作目录中，文件名为 **sonar.all-data.csv** 。

## 教程

本教程分为两个步骤。

1.  计算拆分。
2.  声纳数据集案例研究。

这些步骤提供了实现和应用随机森林算法到您自己的预测建模问题所需的基础。

### 1.计算拆分

在决策树中，通过查找导致成本最低的属性和该属性的值来选择拆分点。

对于分类问题，此成本函数通常是Gini索引，用于计算分割点创建的数据组的纯度。基尼系数为0是完美纯度，在两类分类问题的情况下，类值完全分为两组。

在决策树中查找最佳分割点涉及评估每个输入变量的训练数据集中每个值的成本。

对于装袋和随机森林，此过程在训练数据集的样本上执行，由替换完成。对替换进行采样意味着可以选择相同的行并将其多次添加到样本中。

我们可以更新随机森林的这个程序。如果具有最低成本的拆分，我们可以创建要考虑的输入属性的样本，而不是在搜索中枚举输入属性的所有值。

此输入属性样本可以随机选择而无需替换，这意味着在查找成本最低的分割点时，每个输入属性只需要考虑一次。

下面是一个函数名 **get_split（）**，它实现了这个过程。它将数据集和固定数量的输入要素作为输入参数进行评估，其中数据集可以是实际训练数据集的样本。

辅助函数 **test_split（）**用于通过候选分割点分割数据集， **gini_index（）**用于评估创建的行组的给定分割的成本。

我们可以看到通过随机选择特征索引并将它们添加到列表（称为**特征**）来创建特征列表，然后枚举该特征列表并将训练数据集中的特定值评估为分割点。

```py
# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
```

现在我们知道如何修改决策树算法以与随机森林算法一起使用，我们可以将其与包装的实现结合起来并将其应用于真实世界的数据集。

### 2.声纳数据集案例研究

在本节中，我们将随机森林算法应用于Sonar数据集。

该示例假定数据集的CSV副本位于当前工作目录中，文件名为 **sonar.all-data.csv** 。

首先加载数据集，将字符串值转换为数字，然后将输出列从字符串转换为0和1的整数值。这可以通过辅助函数 **load_csv（）**， **str_column_to_float（ ）**和 **str_column_to_int（）**加载和准备数据集。

我们将使用k-fold交叉验证来估计学习模型在看不见的数据上的表现。这意味着我们将构建和评估k模型并将表现估计为平均模型误差。分类精度将用于评估每个模型。这些行为在 **cross_validation_split（）**， **accuracy_metric（）**和 **evaluate_algorithm（）**辅助函数中提供。

我们还将使用适用于装袋的分类和回归树（CART）算法的实现，包括辅助函数 **test_split（）**将数据集分成组， **gini_index（）**来评估分裂点，我们在上一步中讨论的修改后的 **get_split（）**函数， **to_terminal（）**， **split（）**和 **build_tree（）[HTG11用于创建单个决策树，**预测（）**使用决策树做出预测， **subsample（）**制作训练数据集的子样本和 **bagging_predict（ ）**使用决策树列表做出预测。**

开发了一个新的函数名 **random_forest（）**，它首先从训练数据集的子样本创建决策树列表，然后使用它们做出预测。

如上所述，随机森林和袋装决策树之间的关键区别是树木创建方式的一个小变化，这里是 **get_split（）**函数。

下面列出了完整的示例。

```py
# Random Forest Algorithm on Sonar Dataset
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
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
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
def split(node, max_depth, min_size, n_features, depth):
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
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
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

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
	scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

k值为5用于交叉验证，每次迭代时评估每个折叠208/5 = 41.6或仅超过40个记录。

构建深度树，最大深度为10，每个节点为1的最小训练行数。训练数据集的样本创建的大小与原始数据集相同，这是随机森林算法的默认期望值。

在每个分割点处考虑的要素数设置为sqrt（num_features）或sqrt（60）= 7.74四舍五入为7个要素。

评估了一组3种不同数量的树木进行比较，显示随着更多树木的增加，技能越来越高。

运行该示例打印每个折叠的分数和每个配置的平均分数。

```py
Trees: 1
Scores: [56.09756097560976, 63.41463414634146, 60.97560975609756, 58.536585365853654, 73.17073170731707]
Mean Accuracy: 62.439%

Trees: 5
Scores: [70.73170731707317, 58.536585365853654, 85.36585365853658, 75.60975609756098, 63.41463414634146]
Mean Accuracy: 70.732%

Trees: 10
Scores: [75.60975609756098, 80.48780487804879, 92.6829268292683, 73.17073170731707, 70.73170731707317]
Mean Accuracy: 78.537%
```

## 扩展

本节列出了您可能有兴趣探索的本教程的扩展。

*   **算法调整**。发现本教程中使用的配置有一些试验和错误，但没有进行优化。尝试使用更多树，不同数量的功能甚至不同的树配置来提高表现。
*   **更多问题**。将该技术应用于其他分类问题，甚至使用新的成本函数和用于组合树木预测的新方法使其适应回归。

**你有没有试过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何从零开始实现随机森林算法。

具体来说，你学到了：

*   随机森林与袋装决策树的区别。
*   如何更新决策树的创建以适应随机森林过程。
*   如何将随机森林算法应用于现实世界的预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。