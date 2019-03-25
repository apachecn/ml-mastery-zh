# 如何在Python中最好地调整XGBoost的多线程支持

> 原文： [https://machinelearningmastery.com/best-tune-multithreading-support-xgboost-python/](https://machinelearningmastery.com/best-tune-multithreading-support-xgboost-python/)

用于梯度增强用途的XGBoost库专为高效的多核并行处理而设计。

这使它可以在训练时有效地使用系统中的所有CPU核心。

在这篇文章中，您将发现Python中XGBoost的并行处理功能。

阅读这篇文章后你会知道：

*   如何确认XGBoost多线程支持在您的系统上正在运行。
*   如何评估增加XGBoost上的线程数的效果。
*   如何在使用交叉验证和网格搜索时充分利用多线程XGBoost。

让我们开始吧。

*   **2017年1月更新**：已更新，以反映scikit-learn API版本0.18.1中的更改​​。

![How to Best Tune Multithreading Support for XGBoost in Python](img/ed8730017a4b756792937527e1a8af75.jpg)

如何最好地调整Python中XGBoost的多线程支持
照片由 [Nicholas A. Tonelli](https://www.flickr.com/photos/nicholas_t/14946860658/) ，保留一些权利。

## 问题描述：Otto Dataset

在本教程中，我们将使用 [Otto Group产品分类挑战](https://www.kaggle.com/c/otto-group-product-classification-challenge)数据集。

此数据集可从Kaggle获得（您需要注册Kaggle才能下载此数据集）。您可以从[数据页面](https://www.kaggle.com/c/otto-group-product-classification-challenge/data)下载训练数据集 **train.zip** ，并将解压缩的 **trian.csv** 文件放入您的工作目录。

该数据集描述了超过61,000种产品的93个模糊细节，这些产品分为10个产品类别（例如时尚，电子等）。输入属性是某种不同事件的计数。

目标是对新产品进行预测，因为10个类别中的每个类别都有一组概率，并且使用多类对数损失（也称为交叉熵）来评估模型。

这个竞赛在2015年5月完成，这个数据集对XGBoost来说是一个很好的挑战，因为有大量的例子和问题的难度以及需要很少的数据准备这一事实（除了将字符串类变量编码为整数）。

## 线程数的影响

XGBoost在C ++中实现，以明确地使用 [OpenMP API](https://en.wikipedia.org/wiki/OpenMP) 进行并行处理。

梯度增强中的并行性可以在单个树的构造中实现，而不是像随机森林那样并行创建树。这是因为在增强中，树木被顺序添加到模型中。 XGBoost的速度既可以在单个树木的构造中增加平行度，也可以有效地准备输入数据，以帮助加快树木的构建。

根据您的平台，您可能需要专门编译XGBoost以支持多线程。有关详细信息，请参阅 [XGBoost安装说明](https://github.com/dmlc/xgboost/blob/master/doc/build.md)。

用于scikit-learn的XGBoost的 **XGBClassifier** 和 **XGBRegressor** 包装类提供了 **nthread** 参数，用于指定XGBoost在训练期间可以使用的线程数。

默认情况下，此参数设置为-1以使用系统中的所有核心。

```
model = XGBClassifier(nthread=-1)
```

通常，您应该为XGBoost安装获得多线程支持，而无需任何额外的工作。

根据您的Python环境（例如Python 3），您可能需要显式启用XGBoost的多线程支持。如果您需要帮助， [XGBoost库提供了一个示例](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_parallel.py)。

您可以通过构建许多不同的XGBoost模型来确认XGBoost多线程支持是否正常工作，指定线程数并计算构建每个模型所需的时间。这一趋势将向您展示启用了多线程支持，并指出了构建模型时的效果。

例如，如果您的系统有4个核心，您可以训练8个不同的模型，并计算创建每个模型所需的时间（以秒为单位），然后比较时间。

```
# evaluate the effect of the number of threads
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
	start = time.time()
	model = XGBClassifier(nthread=n)
	model.fit(X_train, y_train)
	elapsed = time.time() - start
	print(n, elapsed)
	results.append(elapsed)
```

我们可以在Otto数据集上使用这种方法。下面提供完整示例以确保完整性。

您可以更改 **num_threads** 阵列以满足系统上的核心数。

```
# Otto, tune number of threads
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
from matplotlib import pyplot
# load data
data = read_csv('train.csv')
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# evaluate the effect of the number of threads
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
	start = time.time()
	model = XGBClassifier(nthread=n)
	model.fit(X, label_encoded_y)
	elapsed = time.time() - start
	print(n, elapsed)
	results.append(elapsed)
# plot results
pyplot.plot(num_threads, results)
pyplot.ylabel('Speed (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show()
```

运行此示例总结了每个配置的执行时间（以秒为单位），例如：

```
(1, 115.51652717590332)
(2, 62.7727689743042)
(3, 46.042901039123535)
(4, 40.55334496498108)
```

下面提供了这些时间的图表。

![XGBoost Tune Number of Threads for Single Model](img/146726c5dc2eed3eec56e9a56373187a.jpg)

XGBoost Tune单个模型的线程数

随着线程数量的增加，我们可以看到执行时间减少的好趋势。

如果您没有看到每个新线程的运行时间有所改善，您可能需要研究如何在安装过程中或运行时在XGBoost中启用多线程支持。

我们可以在具有更多内核的机器上运行相同的代码。据报道，大型Amazon Web Services EC2实例具有32个核心。我们可以调整上面的代码来计算训练具有1到32个内核的模型所需的时间。结果如下。

![XGBoost Time to Train Model on 1 to 32 Cores](img/146f19ae6f7ee6886994a2b084b410b3.jpg)

XGBoost在1到32个核心上训练模型的时间

值得注意的是，除了16个线程（大约7秒）之外，我们没有看到太多改进。我希望其原因是亚马逊实例仅在硬件中提供16个内核，并且超线程可以提供额外的16个内核。结果表明，如果您的计算机具有超线程，则可能需要将 **num_threads** 设置为等于计算机中物理CPU核心的数量。

使用OpenMP进行XGBoost的低级优化实现会挤出像这样的大型机器的每个最后一个周期。

## 交叉验证XGBoost模型时的并行性

scikit-learn中的k-fold交叉验证支持也支持多线程。

例如， **cross_val_score（）**函数上的 **n_jobs** 参数用于使用k-fold交叉验证评估数据集上的模型，允许您指定要运行的并行作业数。

默认情况下，此值设置为1，但可以设置为-1以使用系统上的所有CPU核心，这是一种很好的做法。例如：

```
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='log_loss', n_jobs=-1, verbose=1)
```

这就提出了如何配置交叉验证的问题：

*   禁用XGBoost中的多线程支持，并允许交叉验证在所有核心上运行。
*   在交叉验证中禁用多线程支持，并允许XGBoost在所有核心上运行。
*   为XGBoost和Cross验证启用多线程支持。

我们可以通过简单计算在每种情况下评估模型所需的时间来得到这个问题的答案。

在下面的示例中，我们使用10倍交叉验证来评估Otto训练数据集上的默认XGBoost模型。评估上述每个场景，并报告评估模型所花费的时间。

完整的代码示例如下所示。

```
# Otto, parallel cross validation
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import time
# load data
data = read_csv('train.csv')
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# Single Thread XGBoost, Parallel Thread CV
start = time.time()
model = XGBClassifier(nthread=1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
print("Single Thread XGBoost, Parallel Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost, Single Thread CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
elapsed = time.time() - start
print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost and CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
print("Parallel Thread XGBoost and CV: %f" % (elapsed))
```

运行该示例将打印以下结果：

```
Single Thread XGBoost, Parallel Thread CV: 359.854589
Parallel Thread XGBoost, Single Thread CV: 330.498101
Parallel Thread XGBoost and CV: 313.382301
```

我们可以看到，通过交叉验证折叠并行化XGBoost会带来好处。这是有道理的，因为10个连续快速任务比（10除以num_cores）慢任务更好。

有趣的是，我们可以看到通过在XGBoost和交叉验证中启用多线程来实现最佳结果。这是令人惊讶的，因为这意味着num_cores数量的并行XGBoost模型在其模型构造中竞争相同的num_cores。然而，这实现了最快的结果，并且建议使用XGBoost进行交叉验证。

因为网格搜索使用相同的基础方法来实现并行性，所以我们期望同样的发现可用于优化XGBoost的超参数。

## 摘要

在这篇文章中，您发现了XGBoost的多线程功能。

你了解到：

*   如何检查系统上是否启用了XGBoost中的多线程支持。
*   如何增加线程数会影响训练XGBoost模型的性能。
*   如何在Python中最佳地配置XGBoost和交叉验证，以最短的运行时间。

您对XGBoost或此帖子的多线程支持有任何疑问吗？在评论中提出您的问题，我会尽力回答。