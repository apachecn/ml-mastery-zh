# 使用 Python 和 Scikit-Learn 重缩放机器学习数据

> 原文： [https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/](https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/)

在构建模型之前，必须准备好数据。 [数据准备](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/ "How to Prepare Data For Machine Learning")过程可以包括三个步骤：数据选择，数据预处理和数据转换。

在这篇文章中，您将发现两种简单的数据转换方法，您可以使用 [scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library") 将这些方法应用于 Python 中的数据。

**更新**：[有关更新的示例集](http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)，请参阅此帖子。

[![Data Rescaling](img/fc8259088110f3e7b9ccb32457b8dc37.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/Data-Rescaling.jpg)

数据重缩放
照片由 [Quinn Dombrowski](https://www.flickr.com/photos/quinnanya/4508825094) 拍摄，保留一些权利。

## 数据重新缩放

您的预处理数据可能包含各种数量的比例混合的属性，如美元，千克和销售量。

如果数据属性具有相同的比例，许多机器学习方法期望或更有效。两种流行的[数据缩放](http://en.wikipedia.org/wiki/Feature_scaling)方法是[标准化](http://en.wikipedia.org/wiki/Normalization_(statistics))和标准化。

## 数据规范化

规范化是指将实值数字属性重新缩放到 0 和 1 范围内。

缩放依赖于值的大小的模型的输入属性是有用的，例如 k-最近邻居中使用的距离度量以及回归中的系数的准备。

以下示例演示了 Iris 花数据集的数据标准化。

Normalize the data attributes for the Iris dataset Python

```
# Normalize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
y = iris.target
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
```

有关更多信息，请参阅 API 文档中的[规范化函数](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html)。

## 数据标准化

标准化是指将每个属性的分布转换为平均值为零，标准差为 1（单位方差）。

标准化依赖于诸如高斯过程之类的属性分布的模型的属性是有用的。

以下示例演示了 Iris 花数据集的数据标准化。

Standardize the data attributes for the Iris dataset Python

```
# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the Iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data and target attributes
X = iris.data
y = iris.target
# standardize the data attributes
standardized_X = preprocessing.scale(X)
```

有关更多信息，请参阅 API 文档中的[比例功能](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)。

## 提示：使用哪种方法

在应用数据之前，很难知道重缩放数据是否会提高算法的表现。如果经常可以，但并非总是如此。

一个很好的建议是创建数据集的重新缩放副本，并使用您的测试工具和一些您想要检查的算法将它们相互竞争。这可以快速突出显示使用给定模型重新缩放数据的好处（或缺少），以及哪种重新缩放方法可能值得进一步调查。

## 摘要

在应用机器学习算法之前，数据重新缩放是数据准备的重要部分。

在这篇文章中，您发现数据重新缩放适用于应用机器学习的过程和两种方法：规范化和标准化，您可以使用 scikit-learn 库在 Python 中重新缩放数据。

**Update**: [See this post for a more up to date set of examples](http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/).