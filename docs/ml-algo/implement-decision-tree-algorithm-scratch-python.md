# 如何在Python中从头开始实现决策树算法

> 原文： [https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)

决策树是一种强大的预测方法，非常受欢迎。

它们很受欢迎，因为最终模型很容易被从业者和领域专家所理解。最终决策树可以准确解释为什么进行特定预测，使其对操作使用非常有吸引力。

决策树还为更先进的集合方法提供了基础，例如装袋，随机森林和梯度增强。

在本教程中，您将了解如何使用Python从头开始实现[分类和回归树算法](http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)。

完成本教程后，您将了解：

*   如何计算和评估数据中的候选分裂点。
*   如何安排拆分为决策树结构。
*   如何将分类和回归树算法应用于实际问题。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **2017年2月更新**：修复了build_tree中的错误。
*   **2017年8月更新**：修正了基尼计算中的一个错误，根据组大小添加了组基尼评分缺失的权重（感谢迈克尔！）。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How To Implement The Decision Tree Algorithm From Scratch In Python](img/6ae5db916a10295dfba3cc0aaba924b8.jpg)

如何在Python中从头开始实施决策树算法
[Martin Cathrae](https://www.flickr.com/photos/suckamc/4325870801/) 的照片，保留一些权利。

## 说明

本节简要介绍了本教程中使用的分类和回归树算法以及Banknote数据集。

### 分类和回归树

分类和回归树或简称CART是Leo Breiman引用的首字母缩略词，用于指代可用于分类或回归预测建模问题的决策树算法。

我们将在本教程中专注于使用CART进行分类。

CART模型的表示是二叉树。这是来自算法和数据结构的相同二叉树，没什么太花哨的（每个节点可以有零个，一个或两个子节点）。

假设变量是数字，节点表示单个输入变量（X）和该变量上的分割点。树的叶节点（也称为终端节点）包含用于做出预测的输出变量（y）。

一旦创建，就可以使用拆分在每个分支之后使用新的数据行来导航树，直到进行最终预测。

创建二元决策树实际上是划分输入空间的过程。贪婪的方法用于划分称为递归二进制分裂的空间。这是一个数值程序，其中所有值都排成一行，并使用成本函数尝试和测试不同的分裂点。

选择具有最佳成本（最低成本，因为我们最小化成本）的分割。基于成本函数，以贪婪的方式评估和选择所有输入变量和所有可能的分裂点。

*   **回归**：为选择分割点而最小化的成本函数是落在矩形内的所有训练样本的总和平方误差。
*   **分类**：使用基尼成本函数，其表示节点的纯度，其中节点纯度是指分配给每个节点的训练数据的混合程度。

拆分继续，直到节点包含最少数量的训练示例或达到最大树深度。

### 钞票数据集

钞票数据集涉及根据从照片中采取的若干措施来预测给定钞票是否是真实的。

数据集包含1,372行，包含5个数字变量。这是两个类的分类问题（二分类）。

下面提供了数据集中五个变量的列表。

1.  小波变换图像的方差（连续）。
2.  小波变换图像的偏度（连续）。
3.  小波峰度变换图像（连续）。
4.  图像熵（连续）。
5.  class（整数）。

下面是数据集的前5行的示例

```py
3.6216,8.6661,-2.8073,-0.44699,0
4.5459,8.1674,-2.4586,-1.4621,0
3.866,-2.6383,1.9242,0.10645,0
3.4566,9.5228,-4.0112,-3.5944,0
0.32924,-4.4552,4.5718,-0.9888,0
4.3684,9.6718,-3.9606,-3.1625,0
```

使用零规则算法预测最常见的类值，问题的基线准确率约为50％。

您可以从 [UCI机器学习库](http://archive.ics.uci.edu/ml/datasets/banknote+authentication)了解更多信息并下载数据集。

下载数据集并将其放在当前工作目录中，文件名为 **data_banknote_authentication.csv** 。

## 教程

本教程分为5个部分：

1.  基尼指数。
2.  创建拆分。
3.  建树。
4.  做一个预测。
5.  钞票案例研究。

这些步骤将为您提供从头开始实施CART算法所需的基础，并将其应用于您自己的预测建模问题。

### 基尼系数

Gini索引是用于评估数据集中拆分的成本函数的名称。

数据集中的拆分涉及一个输入属性和该属性的一个值。它可用于将训练模式划分为两组行。

基尼分数通过分割创建的两个组中的类的混合程度，可以了解分割的好坏程度。完美分离导致基尼评分为0，而最差情况分裂导致每组50/50分类导致基尼评分为0.5（对于2类问题）。

通过示例可以最好地演示计算基尼系数。

我们有两组数据，每组有2行。第一组中的行都属于类0，第二组中的行属于类1，因此它是完美的分割。

我们首先需要计算每组中班级的比例。

```py
proportion = count(class_value) / count(rows)
```

这个例子的比例是：

```py
group_1_class_0 = 2 / 2 = 1
group_1_class_1 = 0 / 2 = 0
group_2_class_0 = 0 / 2 = 0
group_2_class_1 = 2 / 2 = 1
```

然后为每个子节点计算Gini，如下所示：

```py
gini_index = sum(proportion * (1.0 - proportion))
gini_index = 1.0 - sum(proportion * proportion)
```

然后，必须根据组的大小，相对于父组中的所有样本，对每组的基尼系数进行加权。当前正在分组的所有样本。我们可以将此权重添加到组的Gini计算中，如下所示：

```py
gini_index = (1.0 - sum(proportion * proportion)) * (group_size/total_samples)
```

在此示例中，每组的基尼评分计算如下：

```py
Gini(group_1) = (1 - (1*1 + 0*0)) * 2/4
Gini(group_1) = 0.0 * 0.5 
Gini(group_1) = 0.0 
Gini(group_2) = (1 - (0*0 + 1*1)) * 2/4
Gini(group_2) = 0.0 * 0.5 
Gini(group_2) = 0.0
```

然后在分割点处的每个子节点上添加分数，以给出可以与其他候选分割点进行比较的分割点的最终基尼分数。

然后，此分裂点的基尼计算为0.0 + 0.0或完美基尼得分为0.0。

下面是一个名为 **gini_index（）**的函数，它计算组列表的Gini索引和已知类值的列表。

你可以看到那里有一些安全检查，以避免空组的除以零。

```py
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
```

我们可以使用上面的工作示例测试此函数。我们还可以测试每组中50/50分裂的最坏情况。下面列出了完整的示例。

```py
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

# test Gini values
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
```

运行该示例打印两个Gini分数，首先是最差情况的分数为0.5，然后是最佳情况的分数为0.0。

```py
0.5
0.0
```

现在我们知道如何评估拆分的结果，让我们看一下创建拆分。

### 2.创建拆分

拆分由数据集中的属性和值组成。

我们可以将其概括为要拆分的属性的索引以及在该属性上拆分行的值。这只是索引数据行的有用简写。

创建拆分涉及三个部分，第一部分我们已经看过计算基尼评分。剩下的两部分是：

1.  拆分数据集。
2.  评估所有拆分。

我们来看看每一个。

#### 2.1。拆分数据集

拆分数据集意味着在给定属性索引和该属性的拆分值的情况下将数据集分成两个行列表。

一旦我们拥有这两个组，我们就可以使用上面的Gini评分来评估拆分的成本。

拆分数据集涉及迭代每一行，检查属性值是低于还是高于拆分值，并分别将其分配给左侧或右侧组。

下面是一个名为 **test_split（）**的函数，它实现了这个过程。

```py
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
```

不是很多。

请注意，右侧组包含索引值大于或等于拆分值的所有行。

#### 2.2。评估所有拆分

通过上面的Gini函数和测试分割函数，我们现在拥有评估分割所需的一切。

给定一个数据集，我们必须检查每个属性的每个值作为候选分割，评估分割的成本并找到我们可以做出的最佳分割。

找到最佳拆分后，我们可以将其用作决策树中的节点。

这是一个详尽而贪婪的算法。

我们将使用字典来表示决策树中的节点，因为我们可以按名称存储数据。当选择最佳分割并将其用作树的新节点时，我们将存储所选属性的索引，要分割的属性的值以及由所选分割点分割的两组数据。

每组数据都是其自己的小数据集，只有那些通过拆分过程分配给左或右组的行。您可以想象我们如何在构建决策树时递归地再次拆分每个组。

下面是一个名为 **get_split（）**的函数，它实现了这个过程。您可以看到它迭代每个属性（类值除外），然后遍历该属性的每个值，分割和评估拆分。

记录最佳分割，然后在所有检查完成后返回。

```py
# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
```

我们可以设计一个小数据集来测试这个函数和我们的整个数据集拆分过程。

```py
X1			X2			Y
2.771244718		1.784783929		0
1.728571309		1.169761413		0
3.678319846		2.81281357		0
3.961043357		2.61995032		0
2.999208922		2.209014212		0
7.497545867		3.162953546		1
9.00220326		3.339047188		1
7.444542326		0.476683375		1
10.12493903		3.234550982		1
6.642287351		3.319983761		1
```

我们可以为每个类使用单独的颜色绘制此数据集。您可以看到，手动选择X1的值（图中的x轴）来分割此数据集并不困难。

![CART Contrived Dataset](img/dd963418a62770e69fbbf7d35b673dda.jpg)

CART Contrived Dataset

下面的例子将所有这些放在一起。

```py
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
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
```

**get_split（）**功能被修改为打印出每个分割点，并且在评估时它是基尼指数。

运行该示例打印所有Gini分数，然后在X1的数据集中打印最佳分割的分数＆lt; 6.642，基尼指数为0.0或完美分裂。

```py
X1 < 2.771 Gini=0.444
X1 < 1.729 Gini=0.500
X1 < 3.678 Gini=0.286
X1 < 3.961 Gini=0.167
X1 < 2.999 Gini=0.375
X1 < 7.498 Gini=0.286
X1 < 9.002 Gini=0.375
X1 < 7.445 Gini=0.167
X1 < 10.125 Gini=0.444
X1 < 6.642 Gini=0.000
X2 < 1.785 Gini=0.500
X2 < 1.170 Gini=0.444
X2 < 2.813 Gini=0.320
X2 < 2.620 Gini=0.417
X2 < 2.209 Gini=0.476
X2 < 3.163 Gini=0.167
X2 < 3.339 Gini=0.444
X2 < 0.477 Gini=0.500
X2 < 3.235 Gini=0.286
X2 < 3.320 Gini=0.375
Split: [X1 < 6.642]
```

现在我们知道如何在数据集或行列表中找到最佳分割点，让我们看看如何使用它来构建决策树。

### 3.建造一棵树

创建树的根节点很简单。

我们使用整个数据集调用上面的 **get_split（）**函数。

向树中添加更多节点更有趣。

构建树可以分为3个主要部分：

1.  终端节点。
2.  递归拆分。
3.  建造一棵树。

#### 3.1。终端节点

我们需要决定何时停止种树。

我们可以使用节点在训练数据集中负责的行数和行数来实现。

*   **最大树深**。这是树的根节点的最大节点数。一旦满足树的最大深度，我们必须停止拆分添加新节点。更深的树木更复杂，更有可能过拟合训练数据。
*   **最小节点记录**。这是给定节点负责的最小训练模式数。一旦达到或低于此最小值，我们必须停止拆分和添加新节点。预计训练模式太少的节点过于具体，可能会过度训练训练数据。

这两种方法将是用户指定的树构建过程参数。

还有一个条件。可以选择所有行属于一个组的拆分。在这种情况下，我们将无法继续拆分和添加子节点，因为我们将无法在一侧或另一侧拆分记录。

现在我们有一些关于何时停止种植树木的想法。当我们在给定点停止增长时，该节点被称为终端节点并用于进行最终预测。

这是通过获取分配给该节点的行组并选择组中最常见的类值来完成的。这将用于做出预测。

下面是一个名为 **to_terminal（）**的函数，它将为一组行选择一个类值。它返回行列表中最常见的输出值。

```py
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
```

#### 3.2。递归拆分

我们知道如何以及何时创建终端节点，现在我们可以构建我们的树。

构建决策树涉及在为每个节点创建的组上反复调用上面开发的 **get_split（）**函数。

添加到现有节点的新节点称为子节点。节点可以具有零个子节点（终端节点），一个子节点（一侧直接做出预测）或两个子节点。我们将在给定节点的字典表示中将子节点称为左和右。

创建节点后，我们可以通过再次调用相同的函数，对拆分中的每组数据递归创建子节点。

下面是一个实现此递归过程的函数。它将节点作为参数以及节点中的最大深度，最小模式数和节点的当前深度。

您可以想象这可能首先如何在根节点中传递，并且深度为1.此函数最好用以下步骤解释：

1.  首先，提取节点分割的两组数据以供使用并从节点中删除。当我们处理这些组时，节点不再需要访问这些数据。
2.  接下来，我们检查左侧或右侧行组是否为空，如果是，我们使用我们拥有的记录创建终端节点。
3.  然后我们检查是否已达到最大深度，如果是，我们创建一个终端节点。
4.  然后我们处理左子节点，如果行组太小则创建终端节点，否则以深度优先方式创建和添加左节点，直到在该分支上到达树的底部。
5.  然后以相同的方式处理右侧，因为我们将构造的树恢复到根。

```py
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
```

#### 3.3。建造一棵树

我们现在可以将所有部分组合在一起。

构建树包括创建根节点并调用 **split（）**函数，然后递归调用自身以构建整个树。

下面是实现此过程的小 **build_tree（）**函数。

```py
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
```

我们可以使用上面设计的小数据集测试整个过程。

以下是完整的示例。

还包括一个小的 **print_tree（）**函数，它递归地打印出决策树的节点，每个节点一行。虽然没有真正的决策树图那么引人注目，但它给出了树结构和决策的概念。

```py
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

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)
```

我们可以在运行此示例时更改最大深度参数，并查看对打印树的影响。

最大深度为1（调用 **build_tree（）**函数时的第二个参数），我们可以看到树使用了我们在上一节中发现的完美分割。这是一个具有一个节点的树，也称为决策树桩。

```py
[X1 < 6.642]
 [0]
 [1]
```

将最大深度增加到2，即使不需要，我们也会强制树进行拆分。然后，根节点的左右子节点再次使用 **X1** 属性来拆分已经完美的类混合。

```py
[X1 < 6.642]
 [X1 < 2.771]
  [0]
  [0]
 [X1 < 7.498]
  [1]
  [1]
```

最后，反过来说，我们可以强制一个更高级别的分裂，最大深度为3。

```py
[X1 < 6.642]
 [X1 < 2.771]
  [0]
  [X1 < 2.771]
   [0]
   [0]
 [X1 < 7.498]
  [X1 < 7.445]
   [1]
   [1]
  [X1 < 7.498]
   [1]
   [1]
```

这些测试表明，很有可能优化实现以避免不必要的拆分。这是一个扩展。

现在我们可以创建一个决策树，让我们看看如何使用它来对新数据做出预测。

### 4.做出预测

使用决策树做出预测涉及使用专门提供的数据行导航树。

同样，我们可以使用递归函数实现此功能，其中使用左子节点或右子节点再次调用相同的预测例程，具体取决于拆分如何影响提供的数据。

我们必须检查子节点是否是要作为预测返回的终端值，或者它是否是包含要考虑的另一级树的字典节点。

下面是实现此过程的 **predict（）**函数。您可以看到给定节点中的索引和值

您可以看到给定节点中的索引和值如何用于评估提供的数据行是否位于拆分的左侧或右侧。

```py
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
```

我们可以使用我们设计的数据集来测试这个功能。下面是一个使用硬编码决策树的示例，该决策树具有最佳分割数据的单个节点（决策树桩）。

该示例对数据集中的每一行做出预测。

```py
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

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]

#  predict with a stump
stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for row in dataset:
	prediction = predict(stump, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
```

运行该示例将按预期为每行打印正确的预测。

```py
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
```

我们现在知道如何创建决策树并使用它来做出预测。现在，让我们将它应用于真实的数据集。

### 5.钞票案例研究

本节将CART算法应用于Bank Note数据集。

第一步是加载数据集并将加载的数据转换为可用于计算分割点的数字。为此，我们将使用辅助函数 **load_csv（）**来加载文件，使用 **str_column_to_float（）**将字符串数转换为浮点数。

我们将使用5倍折叠交叉验证来评估算法。这意味着每个折叠中将使用1372/5 = 274.4或仅超过270个记录。我们将使用辅助函数 **evaluate_algorithm（）**来评估具有交叉验证的算法和 **accuracy_metric（）**来计算预测的准确率。

开发了一个名为 **decision_tree（）**的新函数来管理CART算法的应用，首先从训练数据集创建树，然后使用树对测试数据集做出预测。

下面列出了完整的示例。

```py
# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

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

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

该示例使用5层的最大树深度和每个节点的最小行数为10.这些CART参数通过一些实验选择，但绝不是最佳的。

运行该示例打印每个折叠的平均分类准确度以及所有折叠的平均表现。

您可以看到CART和所选配置的平均分类精度达到了约97％，这明显优于达到50％精度的零规则算法。

```py
Scores: [96.35036496350365, 97.08029197080292, 97.44525547445255, 98.17518248175182, 97.44525547445255]
Mean Accuracy: 97.299%
```

## 扩展

本节列出了您可能希望探索的本教程的扩展。

*   **算法调整**。未调整CART在Bank Note数据集中的应用。尝试使用不同的参数值，看看是否可以获得更好的表现。
*   **交叉熵**。用于评估分裂的另一个成本函数是交叉熵（logloss）。您可以实施和试验此替代成本函数。
*   **树修剪**。减少训练数据集过拟合的一项重要技术是修剪树木。调查并实施树修剪方法。
*   **分类数据集**。该示例设计用于具有数字或序数输入属性的输入数据，尝试分类输入数据和可能使用相等而不是排名的拆分。
*   **回归**。使用不同的成本函数和方法调整树以进行回归以创建终端节点。
*   **更多数据集**。将算法应用于UCI机器学习库中的更多数据集。

**你有没有探索过这些扩展？**
在下面的评论中分享您的经验。

## 评论

在本教程中，您了解了如何使用Python从头开始实现决策树算法。

具体来说，你学到了：

*   如何选择和评估训练数据集中的分割点。
*   如何从多个拆分中递归构建决策树。
*   如何将CART算法应用于现实世界的分类预测建模问题。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。