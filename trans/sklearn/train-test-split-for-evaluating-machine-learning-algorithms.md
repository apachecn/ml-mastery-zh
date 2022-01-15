# 用于评估机器学习算法的训练-测试分割

> 原文：<https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/>

最后更新于 2020 年 8 月 26 日

当机器学习算法用于对未用于训练模型的数据进行预测时，训练-测试分割过程用于估计机器学习算法的表现。

这是一个快速且容易执行的过程，其结果允许您针对您的预测建模问题比较机器学习算法的表现。虽然使用和解释起来很简单，但有时不应该使用该过程，例如当您有一个小数据集时，以及需要额外配置的情况下，例如当它用于分类并且数据集不平衡时。

在本教程中，您将发现如何使用训练-测试分割来评估机器学习模型。

完成本教程后，您将知道:

*   当你有一个非常大的数据集，一个昂贵的模型需要训练，或者需要一个好的模型表现的快速估计时，训练-测试分割过程是合适的。
*   如何使用 scikit-learn 机器学习库执行训练-测试分割过程？
*   如何使用训练-测试分割评估用于分类和回归的机器学习算法？

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Train-Test Split for Evaluating Machine Learning Algorithms](img/ea03895d2b88f766cc3b5453209ca0a2.png)

用于评估机器学习算法的训练-测试分割
图片由[保罗·范德瓦尔夫](https://flickr.com/photos/pavdw/27885794643/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  列车测试分割评估
    1.  何时使用列车测试分割
    2.  如何配置列车测试分割
2.  Scikit-Learn 中的训练-测试分割程序
    1.  可重复的列车测试分割
    2.  分层列车试验分割
3.  评估机器学习模型的训练-测试分割
    1.  用于分类的列车测试分割
    2.  回归的训练-测试分割

## 列车测试分割评估

训练-测试分割是一种评估机器学习算法表现的技术。

它可以用于分类或回归问题，也可以用于任何监督学习算法。

该过程包括获取一个数据集并将其分成两个子集。第一个子集用于拟合模型，称为训练数据集。第二子集不用于训练模型；相反，数据集的输入元素被提供给模型，然后进行预测并与期望值进行比较。第二个数据集被称为测试数据集。

*   **训练数据集**:用于拟合机器学习模型。
*   **测试数据集**:用于评估拟合机器学习模型。

目标是估计机器学习模型在新数据上的表现:不用于训练模型的数据。

这就是我们期望在实践中使用该模型的方式。也就是说，用已知的输入和输出来拟合可用的数据，然后对未来我们没有预期输出或目标值的新例子进行预测。

当有足够大的可用数据集时，列车测试程序是合适的。

### 何时使用列车测试分割

“足够大”的概念是特定于每个预测建模问题的。这意味着有足够的数据将数据集分成训练数据集和测试数据集，并且每个训练数据集和测试数据集都是问题域的合适表示。这要求原始数据集也是问题域的合适表示。

问题域的适当表示意味着有足够的记录来覆盖域中所有常见的情况和最不常见的情况。这可能意味着实际观察到的输入变量的组合。这可能需要数千、数十万或数百万个例子。

相反，当可用的数据集很小时，训练测试过程是不合适的。原因是，当数据集被拆分为训练集和测试集时，训练数据集中没有足够的数据供模型学习输入到输出的有效映射。测试集中也没有足够的数据来有效地评估模型表现。估计的表现可能过于乐观(好)或过于悲观(坏)。

如果你没有足够的数据，那么一个合适的替代模型评估程序将是 k 倍交叉验证程序。

除了数据集大小，使用训练-测试分割评估程序的另一个原因是计算效率。

一些模型的训练成本非常高，在这种情况下，在其他程序中使用的重复评估是难以处理的。一个例子可能是深度神经网络模型。在这种情况下，通常使用列车测试程序。

或者，一个项目可能有一个有效的模型和一个庞大的数据集，尽管可能需要快速估计模型表现。同样，在这种情况下，将采用列车测试分割程序。

使用随机选择将来自原始训练数据集的样本分成两个子集。这是为了确保训练和测试数据集代表原始数据集。

### 如何配置列车测试分割

该程序有一个主要配置参数，即列车和测试集的大小。对于训练数据集或测试数据集，这通常表示为 0 到 1 之间的百分比。例如，大小为 0.67(67%)的训练集意味着剩余百分比 0.33(33%)被分配给测试集。

没有最佳分割百分比。

您必须选择符合项目目标的拆分百分比，考虑因素包括:

*   训练模型的计算成本。
*   评估模型的计算成本。
*   训练集代表性。
*   测试集代表性。

然而，常见的分割百分比包括:

*   培训:80%，测试:20%
*   培训:67%，测试:33%
*   培训:50%，测试:50%

现在我们已经熟悉了训练-测试分割模型评估过程，让我们看看如何在 Python 中使用这个过程。

## Scikit-Learn 中的训练-测试分割程序

scikit-learn Python 机器学习库通过 [train_test_split()函数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)提供了 train-test split 评估程序的实现。

该函数将加载的数据集作为输入，并返回分割成两个子集的数据集。

```py
...
# split into train test sets
train, test = train_test_split(dataset, ...)
```

理想情况下，您可以将原始数据集拆分为输入( *X* )和输出( *y* )列，然后调用传递这两个数组的函数，并让它们适当地拆分为训练和测试子集。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
```

分割的大小可以通过“ *test_size* ”参数来指定，该参数取数据集大小在 0 到 1 之间的行数(整数)或百分比(浮点数)。

后者是最常见的，使用的值如 0.33，其中 33%的数据集将分配给测试集，67%的数据集将分配给训练集。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

我们可以使用包含 1000 个示例的合成分类数据集来演示这一点。

下面列出了完整的示例。

```py
# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_blobs(n_samples=1000)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

运行该示例将数据集分为训练集和测试集，然后打印新数据集的大小。

我们可以看到，670 个示例(67%)被分配给了训练集，330 个示例(33%)被分配给了测试集，正如我们所指定的。

```py
(670, 2) (330, 2) (670,) (330,)
```

或者，可以通过指定“ *train_size* ”参数来拆分数据集，该参数可以是行数(整数)或原始数据集在 0 到 1 之间的百分比，例如 0.67 代表 67%。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)
```

### 可重复的列车测试分割

另一个重要的考虑是，行被随机分配给训练集和测试集。

这样做是为了确保数据集是原始数据集的代表性样本(例如随机样本)，而原始数据集又应该是来自问题域的观察的代表性样本。

当比较机器学习算法时，希望(也许是必需的)在数据集的相同子集上对它们进行拟合和评估。

这可以通过固定分割数据集时使用的伪随机数生成器的种子来实现。如果您不熟悉伪随机数生成器，请参阅教程:

*   [Python 机器学习随机数生成器介绍](https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/)

这可以通过将“ *random_state* ”设置为整数值来实现。任何价值都行；它不是可调超参数。

```py
...
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

下面的示例演示了这一点，并显示了数据的两个独立拆分会产生相同的结果。

```py
# demonstrate that the train-test split procedure is repeatable
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_blobs(n_samples=100)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
```

运行该示例会拆分数据集并打印训练数据集的前五行。

再次分割数据集，打印训练数据集的前五行，显示相同的值，确认当我们固定伪随机数发生器的种子时，我们得到原始数据集的相同分割。

```py
[[-2.54341511  4.98947608]
 [ 5.65996724 -8.50997751]
 [-2.5072835  10.06155749]
 [ 6.92679558 -5.91095498]
 [ 6.01313957 -7.7749444 ]]

[[-2.54341511  4.98947608]
 [ 5.65996724 -8.50997751]
 [-2.5072835  10.06155749]
 [ 6.92679558 -5.91095498]
 [ 6.01313957 -7.7749444 ]]
```

### 分层列车试验分割

最后一个考虑是分类问题。

一些分类问题对于每个类别标签没有均衡数量的例子。因此，希望将数据集拆分为训练集和测试集，以保持每个类中示例的比例与原始数据集中观察到的比例相同。

这被称为分层列车测试分割。

我们可以通过将“*分层*”参数设置为原始数据集的 y 分量来实现这一点。这将由 *train_test_split()* 功能使用，以确保在提供的“ *y* ”数组中，列车和测试集在每个类别中都有一定比例的示例。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
```

我们可以用一个分类数据集的例子来证明这一点，其中一个类有 94 个例子，第二个类有 6 个例子。

首先，我们可以将数据集分割成训练集和测试集，而不需要“*分层*”参数。下面列出了完整的示例。

```py
# split imbalanced dataset into train and test sets without stratification
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_classification(n_samples=100, weights=[0.94], flip_y=0, random_state=1)
print(Counter(y))
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
print(Counter(y_train))
print(Counter(y_test))
```

运行该示例首先按类标签报告数据集的组成，显示预期的 94%对 6%。

然后分割数据集，并报告训练集和测试集的组成。我们可以看到训练集有 45/5 个例子，测试集中有 49/1 个例子。列车和测试集的组成不同，这是不可取的。

```py
Counter({0: 94, 1: 6})
Counter({0: 45, 1: 5})
Counter({0: 49, 1: 1})
```

接下来，我们可以对列车测试分割进行分层，并比较结果。

```py
# split imbalanced dataset into train and test sets with stratification
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_classification(n_samples=100, weights=[0.94], flip_y=0, random_state=1)
print(Counter(y))
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
print(Counter(y_train))
print(Counter(y_test))
```

假设我们对训练集和测试集使用了 50%的分割，我们预计训练集和测试集在训练集/测试集中分别有 47/3 个示例。

运行该示例，我们可以看到在这种情况下，分层版本的 train-test split 已经创建了 train 和 test 数据集，在 train/test 集中有 47/3 个示例，正如我们所期望的那样。

```py
Counter({0: 94, 1: 6})
Counter({0: 47, 1: 3})
Counter({0: 47, 1: 3})
```

现在我们已经熟悉了 *train_test_split()* 函数，让我们看看如何使用它来评估一个机器学习模型。

## 评估机器学习模型的训练-测试分割

在本节中，我们将探索使用训练-测试分割过程来评估标准分类和回归预测建模数据集上的机器学习模型。

### 用于分类的列车测试分割

我们将演示如何使用训练测试分割来评估声纳数据集上的随机森林算法。

声纳数据集是一个标准的机器学习数据集，由 208 行数据组成，包含 60 个数字输入变量和一个具有两个类值的目标变量，例如二进制分类。

该数据集包括预测声纳回波显示的是岩石还是模拟地雷。

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集描述(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

下面的示例下载数据集并总结其形状。

```py
# summarize the sonar dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

运行该示例会下载数据集，并将其拆分为输入和输出元素。不出所料，我们可以看到有 208 行数据，60 个输入变量。

```py
(208, 60) (208,)
```

我们现在可以使用列车测试分割来评估模型。

首先，加载的数据集必须分成输入和输出组件。

```py
...
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

接下来，我们可以分割数据集，以便 67%用于训练模型，33%用于评估模型。这种分裂是任意选择的。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

然后，我们可以在训练数据集上定义和拟合模型。

```py
...
# fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
```

然后使用拟合模型进行预测，并使用分类精确率表现度量来评估预测。

```py
...
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)
```

将这些联系在一起，完整的示例如下所示。

```py
# train-test split evaluation random forest on the sonar dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)
```

运行该示例首先加载数据集，并确认输入和输出元素中的行数。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

数据集被分成训练集和测试集，我们可以看到有 139 行用于训练，69 行用于测试集。

最后，在测试集上对模型进行评估，当对新数据进行预测时，模型的表现具有大约 78.3%的准确性。

```py
(208, 60) (208,)
(139, 60) (69, 60) (139,) (69,)
Accuracy: 0.783
```

### 回归的训练-测试分割

我们将演示如何使用训练测试分割来评估房屋数据集上的随机森林算法。

外壳数据集是一个标准的机器学习数据集，由 506 行数据组成，有 13 个数字输入变量和一个数字目标变量。

该数据集包括预测美国波士顿郊区的房价。

*   [房屋数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房屋描述(房屋名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集；我们将自动下载它作为我们工作示例的一部分。

以下示例将数据集下载并加载为熊猫数据框，并总结了数据集的形状。

```py
# load and summarize the housing dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
```

运行该示例确认了 506 行数据、13 个输入变量和单个数值目标变量(总共 14 个)。

```py
(506, 14)
```

我们现在可以使用列车测试分割来评估模型。

首先，加载的数据集必须分成输入和输出组件。

```py
...
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
```

接下来，我们可以分割数据集，以便 67%用于训练模型，33%用于评估模型。这种分裂是任意选择的。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

然后，我们可以在训练数据集上定义和拟合模型。

```py
...
# fit the model
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
```

然后使用拟合模型进行预测，并使用平均绝对误差(MAE)表现指标评估预测。

```py
...
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

将这些联系在一起，完整的示例如下所示。

```py
# train-test split evaluation random forest on the housing dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fit the model
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例首先加载数据集，并确认输入和输出元素中的行数。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

数据集被分成训练集和测试集，我们可以看到有 339 行用于训练，167 行用于测试集。

最后，在测试集上对模型进行评估，当对新数据进行预测时，模型的表现平均绝对误差约为 2.211(千美元)。

```py
(506, 13) (506,)
(339, 13) (167, 13) (339,) (167,)
MAE: 2.157
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [sklearn . model _ selection . train _ test _ split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
*   [sklearn . datasets . make _ classification API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . dataset . make _ blobs API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)。

## 摘要

在本教程中，您发现了如何使用训练-测试分割来评估机器学习模型。

具体来说，您了解到:

*   当你有一个非常大的数据集，一个昂贵的模型需要训练，或者需要一个好的模型表现的快速估计时，训练-测试分割过程是合适的。
*   如何使用 scikit-learn 机器学习库执行训练-测试分割过程？
*   如何使用训练-测试分割评估用于分类和回归的机器学习算法？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。