# 如何使用scikit-learn在Python中开发您的第一个XGBoost模型

> 原文： [https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)

XGBoost是梯度提升决策树的一种实现，旨在提高竞争机器学习速度和性能。

在这篇文章中，您将了解如何在Python中安装和创建第一个XGBoost模型。

阅读这篇文章后你会知道：

*   如何在您的系统上安装XGBoost以便在Python中使用。
*   如何准备数据并训练您的第一个XGBoost模型。
*   如何使用XGBoost模型进行预测。

让我们开始吧。

*   **2017年1月更新**：已更新，以反映scikit-learn API版本0.18.1中的更改​​。
*   **2017年3月更新**：添加缺失导入，使导入更清晰。
*   **更新March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Develop Your First XGBoost Model in Python with scikit-learn](img/c4698227a6e8a3bd125ec3366c9ee135.jpg)

如何用scikit-learn开发你的第一个XGBoost模型
照片由 [Justin Henry](https://www.flickr.com/photos/zappowbang/524307651/) 开发，保留一些权利。

## 教程概述

本教程分为以下6个部分：

1.  安装XGBoost以与Python一起使用。
2.  问题定义和下载数据集。
3.  加载并准备数据。
4.  训练XGBoost模型。
5.  进行预测并评估模型。
6.  将它们结合在一起并运行示例。

## 1.安装XGBoost以便在Python中使用

假设您有一个可用的SciPy环境，可以使用pip轻松安装XGBoost。

例如：

```
sudo pip install xgboost
```

要更新XGBoost的安装，您可以键入：

```
sudo pip install --upgrade xgboost
```

如果您不能使用pip或者想要从GitHub运行最新代码，则另一种安装XGBoost的方法要求您复制XGBoost项目并执行手动构建和安装。

例如，要在Mac OS X上没有多线程构建XGBoost（已经通过macports或homebrew安装了GCC），您可以键入：

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
cp make/minimum.mk ./config.mk
make -j4
cd python-package
sudo python setup.py install
```

您可以在 [XGBoost安装指南](http://xgboost.readthedocs.io/en/latest/build.html)上了解有关如何为不同平台安装XGBoost的更多信息。有关安装XGBoost for Python的最新说明，请参阅 [XGBoost Python包](https://github.com/dmlc/xgboost/tree/master/python-package)。

作为参考，您可以查看 [XGBoost Python API参考](http://xgboost.readthedocs.io/en/latest/python/python_api.html)。

## 2.问题描述：预测糖尿病的发病

在本教程中，我们将使用皮马印第安人糖尿病数据集。

该数据集由描述患者医疗细节的8个输入变量和一个输出变量组成，以指示患者是否在5年内患有糖尿病。

您可以在UCI机器学习存储库网站上了解有关此数据集的更多信息。

这是第一个XGBoost模型的一个很好的数据集，因为所有输入变量都是数字的，问题是一个简单的二进制分类问题。对于XGBoost算法来说，它不一定是一个好问题，因为它是一个相对较小的数据集，并且很容易建模。

下载此数据集并将其放入当前工作目录，文件名为“ **pima-indians-diabetes.csv** ”（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

## 3.加载和准备数据

在本节中，我们将从文件加载数据并准备用于培训和评估XGBoost模型。

我们将从导入我们打算在本教程中使用的类和函数开始。

```
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

接下来，我们可以使用NumPy函数 **loadtext（）**将CSV文件作为NumPy数组加载。

```
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
```

我们必须将数据集的列（属性或特征）分成输入模式（X）和输出模式（Y）。我们可以通过以NumPy数组格式指定列索引来轻松完成此操作。

```
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
```

最后，我们必须将X和Y数据拆分为训练和测试数据集。训练集将用于准备XGBoost模型，测试集将用于进行新的预测，我们可以从中评估模型的性能。

为此，我们将使用scikit-learn库中的 **train_test_split（）**函数。我们还为随机数生成器指定种子，以便每次执行此示例时始终获得相同的数据分割。

```
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
```

我们现在准备训练我们的模型。

## 4.训练XGBoost模型

XGBoost提供了一个包装类，允许在scikit-learn框架中将模型视为分类器或回归器。

这意味着我们可以使用带有XGBoost模型的完整scikit-learn库。

用于分类的XGBoost模型称为 **XGBClassifier** 。我们可以创建并使其适合我们的训练数据集。使用scikit-learn API和 **model.fit（）**函数拟合模型。

训练模型的参数可以传递给构造函数中的模型。在这里，我们使用合理的默认值。

```
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
```

您可以通过打印模型来查看训练模型中使用的参数，例如：

```
print(model)
```

您可以在 [XGBoost Python scikit-learn API](http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) 中了解有关 **XGBClassifier** 和 **XGBRegressor** 类的默认值的更多信息。

您可以在 [XGBoost参数页面](http://xgboost.readthedocs.io/en/latest//parameter.html)上了解有关每个参数含义以及如何配置它们的更多信息。

我们现在准备使用训练有素的模型进行预测。

## 5.使用XGBoost模型进行预测

我们可以使用测试数据集上的拟合模型进行预测。

为了进行预测，我们使用scikit-learn函数 **model.predict（）**。

默认情况下，XGBoost进行的预测是概率。因为这是二元分类问题，所以每个预测是输入模式属于第一类的概率。我们可以通过将它们四舍五入为0或1来轻松地将它们转换为二进制类值。

```
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
```

现在我们已经使用拟合模型对新数据进行预测，我们可以通过将预测值与预期值进行比较来评估预测的性能。为此，我们将在scikit-learn中使用内置的 **accuracy_score（）**函数。

```
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

## 6.将它们捆绑在一起

我们可以将所有这些部分组合在一起，下面是完整的代码清单。

```
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例将生成以下输出。

```
Accuracy: 77.95%
```

对于这个问题，这是一个[良好的准确度得分，我们可以期待，考虑到模型的能力和问题的适度复杂性。](http://www.is.umk.pl/projects/datasets.html#Diabetes)

## 摘要

在这篇文章中，您了解了如何在Python中开发第一个XGBoost模型。

具体来说，你学到了：

*   如何在您的系统上安装XGBoost以备Python使用。
*   如何在标准机器学习数据集上准备数据并训练您的第一个XGBoost模型。
*   如何使用scikit-learn进行预测并评估训练有素的XGBoost模型的性能。

您对XGBoost或该帖子有任何疑问吗？在评论中提出您的问题，我会尽力回答。