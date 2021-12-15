# 如何在Python中从零开始实现重采样方法

> 原文： [https://machinelearningmastery.com/implement-resampling-methods-scratch-python/](https://machinelearningmastery.com/implement-resampling-methods-scratch-python/)

预测建模的目标是创建能够对新数据进行良好预测的模型。

我们在训练时无法访问这些新数据，因此我们必须使用统计方法来估计模型在新数据上的表现。

这类方法称为重采样方法，因为它们重新采样您可用的训练数据。

在本教程中，您将了解如何在Python中从零开始实现重采样方法。

完成本教程后，您将了解：

*   如何实施训练并测试您的数据分割。
*   如何实现数据的k折交叉验证拆分。

让我们开始吧。

*   **2017年1月更新**：将cross_validation_split（）中的fold_size计算更改为始终为整数。修复了Python 3的问题。
*   **更新May / 2018** ：修正了错误的LOOCV。
*   **更新Aug / 2018** ：经过测试和更新，可与Python 3.6配合使用。

![How to Implement Resampling Methods From Scratch In Python](img/a65fa70aa5d4e102c4493609da25fa36.jpg)

如何在Python中实现重新取样方法
照片由 [Andrew Lynch](https://www.flickr.com/photos/newandrew/8478102656/) ，保留一些权利。

## 描述

重新采样方法的目标是充分利用您的训练数据，以便准确地估计模型在新的未见数据上的表现。

然后可以使用准确的表现估计来帮助您选择要使用的模型参数集或要选择的模型。

选择模型后，您可以在整个训练数据集上训练最终模型，并开始使用它来做出预测。

您可以使用两种常见的重采样方法：

*   训练和测试分割您的数据。
*   k折交叉验证。

在本教程中，我们将介绍使用each和when使用一种方法而不是另一种方法。

## 教程

本教程分为3个部分：

1.  训练和测试分裂。
2.  k-fold交叉验证拆分。
3.  如何选择重采样方法。

这些步骤将为您处理重新采样数据集以估计新数据的算法表现提供所需的基础。

### 1.训练和测试分裂

训练和测试分割是最简单的重采样方法。

因此，它是最广泛使用的。

训练和测试拆分涉及将数据集分成两部分：

*   训练数据集。
*   测试数据集。

训练数据集由机器学习算法用于训练模型。保留测试数据集并用于评估模型的表现。

分配给每个数据集的行是随机选择的。这是为了确保模型的训练和评估是客观的。

如果比较多个算法或比较相同算法的多个配置，则应使用相同的训练和数据集的测试分割。这是为了确保表现的比较是一致的或是苹果对苹果。

我们可以通过在分割数据之前以相同的方式为随机数生成器播种，或者通过保持数据集的相同分割以供多个算法使用来实现此目的。

我们可以在单个函数中实现数据集的训练和测试分割。

下面是一个名为 **train_test_split（）**的函数，用于将数据集拆分为训练并进行测试拆分。它接受两个参数，即要作为列表列表拆分的数据集和可选的拆分百分比。

使用默认分割百分比0.6或60％。这将为训练数据集分配60％的数据集，并将剩余的40％留给测试数据集。训练/测试的60/40是数据的良好默认分割。

该函数首先根据提供的数据集计算训练集所需的行数。制作原始数据集的副本。从复制的数据集中选择并删除随机行，并将其添加到训练数据集，直到训练数据集包含目标行数。

然后，将保留在数据集副本中的行作为测试数据集返回。

随机模型中的 **randrange（）**函数用于生成0到列表大小范围内的随机整数。

```py
from random import randrange

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
```

我们可以使用10行的人为数据集来测试这个函数，每个行都有一个列。

下面列出了完整的示例。

```py
from random import seed
from random import randrange

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# test train/test split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = train_test_split(dataset)
print(train)
print(test)
```

该示例在拆分训练数据集之前修复随机种子。这是为了确保每次执行代码时都进行完全相同的数据分割。如果我们想多次使用相同的拆分来评估和比较不同算法的表现，这很方便。

运行该示例将生成以下输出。

打印训练和测试集中的数据，显示6/10或60％的记录分配给训练数据集，4/10或40％的记录分配给测试集。

```py
[[3], [2], [7], [1], [8], [9]]
[[4], [5], [6], [10]]
```

### 2\. k-fold交叉验证拆分

使用训练和测试分割方法的局限性在于您获得了算法表现的噪声估计。

k折交叉验证方法（也称为交叉验证）是一种重采样方法，可提供更准确的算法表现估计。

它通过首先将数据分成k组来完成此操作。然后训练该算法并评估k次，并通过取平均表现得分来总结表现。每组数据称为折叠，因此名称为k-fold交叉验证。

它的工作原理是首先在k-1组数据上训练算法，然后在第k个保持组上作为测试集进行评估。重复这一过程，使得k组中的每一组都有机会被伸出并用作测试装置。

因此，k的值应该可以被训练数据集中的行数整除，以确保每个k组具有相同的行数。

您应该为k选择一个值，该值将数据拆分为具有足够行的组，每个组仍然代表原始数据集。对于较小的数据集，使用的良好默认值是k = 3，对于较大的数据集，k = 10。检查折叠尺寸是否具有代表性的快速方法是计算汇总统计量，例如平均值和标准差，并查看值与整个数据集的相同统计量的差异。

我们可以重复我们在上一节中学习的内容，在实现k-fold交叉验证时创建一个列和测试分割。

我们必须返回k-folds或k组数据，而不是两组。

下面是一个名为 **cross_validation_split（）**的函数，它实现了数据的交叉验证拆分。

和以前一样，我们创建了一个数据集的副本，从中可以绘制随机选择的行。

我们计算每个折叠的大小，作为数据集的大小除以所需的折叠数。

```py
fold size = total rows / total folds
```

如果数据集没有干净地除以折叠数，则可能会有一些剩余行，并且它们不会在拆分中使用。

然后，我们创建一个具有所需大小的行列表，并将它们添加到折叠列表中，然后在最后返回。

```py
from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
```

我们可以在与上面相同的小型人工数据集上测试这种重采样方法。每行只有一个列值，但我们可以想象这可能如何扩展到标准机器学习数据集。

The complete example is listed below.

和以前一样，我们为随机数生成器修复种子，以确保每次执行代码时在相同的折叠中使用相同的行。

k值为4用于演示目的。我们可以预期，将10行划分为4行将导致每行2行，剩余2将不会用于拆分。

```py
from random import seed
from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# test cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)
```

运行该示例将生成以下输出。打印折叠列表，显示确实如预期的那样每个折叠有两行。

```py
[[[3], [2]], [[7], [1]], [[8], [9]], [[10], [6]]]
```

### 3.如何选择重采样方法

用于估计机器学习算法在新数据上的表现的黄金标准是k倍交叉验证。

当配置良好时，与其他方法（如训练和测试分割）相比，k折交叉验证可提供稳健的表现估计。

交叉验证的缺点是运行起来可能非常耗时，需要训练和评估k个不同的模型。如果您有一个非常大的数据集，或者您正在评估需要很长时间训练的模型，则会出现问题。

训练和测试分割重采样方法是最广泛使用的。这是因为它易于理解和实现，并且因为它可以快速估算算法表现。

只构建和评估单个模型。

尽管训练和测试分割方法可以对新数据的模型表现进行噪声或不可靠估计，但如果您拥有非常大的数据集，则这不会成为问题。

大型数据集是数十万或数百万条记录中的数据集，大到足以将其分成两半，导致两个数据集具有几乎相同的统计属性。

在这种情况下，可能几乎不需要使用k折交叉验证作为算法的评估，并且训练和测试分裂可能同样可靠。

## 扩展

在本教程中，我们研究了两种最常见的重采样方法。

您可能需要调查和实现其他方法作为本教程的扩展。

例如：

*   **重复训练和测试**。这是使用训练和测试分割的地方，但过程重复多次。
*   **LOOCV或Leave One Out Cross Validation** 。这是k折交叉验证的一种形式，其中k的值固定为n（训练样本的数量）。
*   **分层**。在分类问题中，这是每组中类值的平衡被迫与原始数据集匹配的地方。

你实施了扩展吗？
在下面的评论中分享您的经历。

## 评论

在本教程中，您了解了如何从零开始在Python中实现重采样方法。

具体来说，你学到了：

*   如何实施训练和测试分割方法。
*   如何实现k-fold交叉验证方法。
*   何时使用每种方法。

您对重新采样方法或此帖有任何疑问吗？
在评论中提出您的问题，我会尽力回答。