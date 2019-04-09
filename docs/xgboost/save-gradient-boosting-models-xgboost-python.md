# 如何在 Python 中使用 XGBoost 保存梯度提升模型

> 原文： [https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/](https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/)

XGBoost 可用于使用梯度提升算法为表格数据创建一些表现最佳的模型。

经过训练，将模型保存到文件中以便以后用于预测新的测试和验证数据集以及全新数据通常是一种很好的做法。

在本文中，您将了解如何使用标准 Python pickle API 将 XGBoost 模型保存到文件中。

完成本教程后，您将了解：

*   如何使用 pickle 保存并稍后加载训练有素的 XGBoost 模型。
*   如何使用 joblib 保存并稍后加载训练有素的 XGBoost 模型。

让我们开始吧。

*   **2017 年 1 月更新**：已更新，以反映 scikit-learn API 版本 0.18.1 中的更改​​。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![How to Save Gradient Boosting Models with XGBoost in Python](img/5a3953dc573c491c8f0f4131ffbd4ec7.jpg)

如何在 Python 中使用 XGBoost 保存梯度提升模型
照片来自 [Keoni Cabral](https://www.flickr.com/photos/keoni101/5334841889/) ，保留一些权利。

## 使用 Pickle 序列化您的 XGBoost 模型

Pickle 是在 Python 中序列化对象的标准方法。

您可以使用 [Python pickle API](https://docs.python.org/2/library/pickle.html) 序列化您的机器学习算法并将序列化格式保存到文件中，例如：

```py
# save model to file
pickle.dump(model, open("pima.pickle.dat", "wb"))
```

稍后您可以加载此文件以反序列化模型并使用它来进行新的预测，例如：

```py
# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
```

以下示例演示了如何在 [Pima 印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)上训练 XGBoost 模型，将模型保存到文件中，然后加载它以进行预测（更新：[从此处下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) ）。

完整性代码清单如下所示。

```py
# Train XGBoost model, save to file using pickle, load and make predictions
from numpy import loadtxt
import xgboost
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
# save model to file
pickle.dump(model, open("pima.pickle.dat", "wb"))

# some time later...

# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例将训练有素的 XGBoost 模型保存到当前工作目录中的 **pima.pickle.dat** pickle 文件中。

```py
pima.pickle.dat
```

加载模型并对训练数据集进行预测后，将打印模型的准确性。

```py
Accuracy: 77.95%
```

## 使用 joblib 序列化 XGBoost 模型

Joblib 是 SciPy 生态系统的一部分，并提供用于管道化 Python 作业的实用程序。

[Joblib API](https://pypi.python.org/pypi/joblib) 提供了用于保存和加载有效利用 NumPy 数据结构的 Python 对象的实用程序。对于非常大的模型，使用它可能是一种更快捷的方法。

API 看起来很像 pickle API，例如，您可以保存训练有素的模型，如下所示：

```py
# save model to file
joblib.dump(model, "pima.joblib.dat")
```

您可以稍后从文件加载模型并使用它来进行如下预测：

```py
# load model from file
loaded_model = joblib.load("pima.joblib.dat")
```

下面的示例演示了如何训练 XGBoost 模型在 Pima Indians 糖尿病数据集开始时进行分类，使用 Joblib 将模型保存到文件中，并在以后加载它以进行预测。

```py
# Train XGBoost model, save to file using joblib, load and make predictions
from numpy import loadtxt
import xgboost
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
# save model to file
joblib.dump(model, "pima.joblib.dat")

# some time later...

# load model from file
loaded_model = joblib.load("pima.joblib.dat")
# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行该示例将模型保存为当前工作目录中的 **pima.joblib.dat** 文件，并为模型中的每个 NumPy 数组创建一个文件（在本例中为两个附加文件）。

```py
pima.joblib.dat
pima.joblib.dat_01.npy
pima.joblib.dat_02.npy
```

加载模型后，将在训练数据集上对其进行评估，并打印预测的准确性。

```py
Accuracy: 77.95%
```

## 摘要

在这篇文章中，您了解了如何序列化经过训练的 XGBoost 模型，然后加载它们以进行预测。

具体来说，你学到了：

*   如何使用 pickle API 序列化并稍后加载训练有素的 XGBoost 模型。
*   如何使用 joblib API 序列化并稍后加载训练有素的 XGBoost 模型。

您对序列化 XGBoost 模型或此帖子有任何疑问吗？在评论中提出您的问题，我会尽力回答。