# 在 Python 中使用 XGBoost 进行梯度提升的数据准备

> 原文： [https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/](https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/)

XGBoost 因其速度和表现而成为 Gradient Boosting 的流行实现。

在内部，XGBoost 模型将所有问题表示为回归预测建模问题，仅将数值作为输入。如果您的数据格式不同，则必须将其准备为预期格式。

在这篇文章中，您将了解如何准备数据，以便在 Python 中使用 XGBoost 库进行梯度提升。

阅读这篇文章后你会知道：

*   如何编码字符串输出变量进行分类。
*   如何使用一个热编码准备分类输入变量。
*   如何使用 XGBoost 自动处理丢失的数据。

让我们开始吧。

*   **2016 年 9 月更新**：我在 impute 示例中更新了一些小错字。
*   **2017 年 1 月更新**：已更新，以反映 scikit-learn API 版本 0.18.1 中的更改​​。
*   **2017 年 1 月更新**：更新了将输入数据转换为字符串的乳腺癌示例。

![Data Preparation for Gradient Boosting with XGBoost in Python](img/93adf08afc03afb270f7ec43a70644cd.jpg)

使用 Python 中的 XGBoost 进行梯度提升的数据准备
照片由 [Ed Dunens](https://www.flickr.com/photos/blachswan/14990404869/) 拍摄，保留一些权利。

## 标签编码字符串类值

虹膜花分类问题是具有字符串类值的问题的示例。

这是一个预测问题，其中以厘米为单位给出鸢尾花的测量值，任务是预测给定花属于哪个物种。

下面是原始数据集的示例。您可以从 [UCI 机器学习库](http://archive.ics.uci.edu/ml/datasets/Iris)中了解有关此数据集的更多信息并以 CSV 格式下载原始数据。

```py
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
```

XGBoost 无法按原样对此问题进行建模，因为它要求输出变量为数字。

我们可以使用 [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 轻松地将字符串值转换为整数值。三个类值（Iris-setosa，Iris-versicolor，Iris-virginica）被映射到整数值（0,1,2）。

```py
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
```

我们将标签编码器保存为单独的对象，以便我们可以使用相同的编码方案转换训练以及稍后的测试和验证数据集。

下面是一个演示如何加载虹膜数据集的完整示例。请注意，Pandas 用于加载数据以处理字符串类值。

```py
# multiclass classification
import pandas
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# load data
data = pandas.read_csv('iris.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:4]
Y = dataset[:,4]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行该示例将生成以下输出：

```py
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
Accuracy: 92.00%
```

请注意 XGBoost 模型如何配置为使用 **multi：softprob** 目标自动建模多类分类问题，该目标是 softmax loss 函数的一种变体，用于模拟类概率。这表明在内部，输出类自动转换为一种热类型编码。

## 一个热编码分类数据

一些数据集仅包含分类数据，例如乳腺癌数据集。

该数据集描述了乳腺癌活组织检查的技术细节，预测任务是预测患者是否复发癌症。

下面是原始数据集的示例。您可以在 [UCI 机器学习库](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer)中了解有关此数据集的更多信息，并从 [mldata.org](http://mldata.org/repository/data/viewslug/datasets-uci-breast-cancer/) 以 CSV 格式下载。

```py
'40-49','premeno','15-19','0-2','yes','3','right','left_up','no','recurrence-events'
'50-59','ge40','15-19','0-2','no','1','right','central','no','no-recurrence-events'
'50-59','ge40','35-39','0-2','no','2','left','left_low','no','recurrence-events'
'40-49','premeno','35-39','0-2','yes','3','right','left_low','yes','no-recurrence-events'
'40-49','premeno','30-34','3-5','yes','2','left','right_up','no','recurrence-events'
```

我们可以看到所有 9 个输入变量都是分类的，并以字符串格式描述。问题是二进制分类预测问题，输出类值也以字符串格式描述。

我们可以重用上一节中的相同方法，并将字符串类值转换为整数值，以使用 LabelEncoder 对预测进行建模。例如：

```py
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
```

我们可以在 X 中的每个输入要素上使用相同的方法，但这只是一个起点。

```py
# encode string input values as integers
features = []
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	features.append(feature)
encoded_x = numpy.array(features)
encoded_x = encoded_x.reshape(X.shape[0], X.shape[1])
```

XGBoost 可以假设每个输入变量的编码整数值具有序数关系。例如，对于 breast-quad 变量，“left-up”编码为 0 并且“left-low”编码为 1 具有作为整数的有意义的关系。在这种情况下，这种假设是不真实的。

相反，我们必须将这些整数值映射到新的二进制变量，每个分类值都有一个新变量。

例如，breast-quad 变量具有以下值：

```py
left-up
left-low
right-up
right-low
central
```

我们可以将其建模为 5 个二进制变量，如下所示：

```py
left-up, left-low, right-up, right-low, central
1,0,0,0,0
0,1,0,0,0
0,0,1,0,0
0,0,0,1,0
0,0,0,0,1
```

这称为[一个热编码](https://en.wikipedia.org/wiki/One-hot)。我们可以使用 scikit-learn 中的 [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 类对所有分类输入变量进行热编码。

在我们对其进行标签编码后，我们可以对每个功能进行热编码。首先，我们必须将要素数组转换为 2 维 NumPy 数组，其中每个整数值是长度为 1 的要素向量。

```py
feature = feature.reshape(X.shape[0], 1)
```

然后我们可以创建 OneHotEncoder 并对特征数组进行编码。

```py
onehot_encoder = OneHotEncoder(sparse=False)
feature = onehot_encoder.fit_transform(feature)
```

最后，我们可以通过逐个连接一个热编码特征来建立输入数据集，将它们作为新列添加（轴= 2）。我们最终得到一个由 43 个二进制输入变量组成的输入向量。

```py
# encode string input values as integers
encoded_x = None
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = numpy.concatenate((encoded_x, feature), axis=1)
print("X shape: : ", encoded_x.shape)
```

理想情况下，我们可以尝试不使用一个热编码输入属性，因为我们可以使用显式序数关系对它们进行编码，例如第一个列的年龄值为'40 -49'和'50 -59'。如果您有兴趣扩展此示例，则将其留作练习。

下面是带有标签和一个热编码输入变量和标签编码输出变量的完整示例。

```py
# binary classification, breast cancer dataset, label and one hot encoded
import numpy
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# load data
data = read_csv('datasets-uci-breast-cancer.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:9]
X = X.astype(str)
Y = dataset[:,9]
# encode string input values as integers
encoded_x = None
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = numpy.concatenate((encoded_x, feature), axis=1)
print("X shape: : ", encoded_x.shape)
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例，我们得到以下输出：

```py
('X shape: : ', (285, 43))
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
Accuracy: 71.58%
```

我们再次看到 XGBoost 框架自动选择' **binary：logistic** '目标，这是二进制分类问题的正确目标。

## 支持缺失数据

XGBoost 可以自动学习如何最好地处理丢失的数据。

事实上，XGBoost 被设计为处理稀疏数据，如前一节中的一个热编码数据，并且通过最小化损失函数来处理丢失数据的方式与处理稀疏或零值的方式相同。

有关如何在 XGBoost 中处理缺失值的技术细节的更多信息，请参见文章 [XGBoost：可伸缩树升压系统](https://arxiv.org/abs/1603.02754)中的第 3.4 节“_ 稀疏感知拆分查找 _”。

Horse Colic 数据集是演示此功能的一个很好的示例，因为它包含大部分缺失数据，大约 30％。

您可以了解有关 Horse Colic 数据集的更多信息，并从 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/Horse+Colic)下载原始数据文件。

这些值由空格分隔，我们可以使用 Pandas 函数 [read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) 轻松加载它。

```py
dataframe = read_csv("horse-colic.csv", delim_whitespace=True, header=None)
```

加载后，我们可以看到缺少的数据标有问号字符（'？'）。我们可以将这些缺失值更改为 XGBoost 预期的稀疏值，即值零（0）。

```py
# set missing values to 0
X[X == '?'] = 0
```

由于缺少的数据被标记为字符串，因此缺少数据的那些列都作为字符串数据类型加载。我们现在可以将整个输入数据集转换为数值。

```py
# convert to numeric
X = X.astype('float32')
```

最后，这是一个二元分类问题，尽管类值用整数 1 和 2 标记。我们将 XGBoost 中的二进制分类问题建模为逻辑 0 和 1 值。我们可以使用 LabelEncoder 轻松地将 Y 数据集转换为 0 和 1 整数，就像我们在虹膜花示例中所做的那样。

```py
# encode Y class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
```

完整性代码清单如下所示。

```py
# binary classification, missing data
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# load data
dataframe = read_csv("horse-colic.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split data into X and y
X = dataset[:,0:27]
Y = dataset[:,27]
# set missing values to 0
X[X == '?'] = 0
# convert to numeric
X = X.astype('float32')
# encode Y class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例将生成以下输出。

```py
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
Accuracy: 83.84%
```

我们可以通过使用非零值（例如 1）标记缺失值来梳理 XGBoost 自动处理缺失值的效果。

```py
X[X == '?'] = 1
```

重新运行该示例表明模型的准确性下降。

```py
Accuracy: 79.80%
```

我们还可以使用特定值来估算缺失的数据。

通常使用列的平均值或中值。我们可以使用 scikit-learn [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) 类轻松地估算缺失的数据。

```py
# impute missing values as the mean
imputer = Imputer()
imputed_x = imputer.fit_transform(X)
```

下面是完整的示例，其中缺少的数据与每列的平均值估算。

```py
# binary classification, missing data, impute with mean
import numpy
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
# load data
dataframe = read_csv("horse-colic.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split data into X and y
X = dataset[:,0:27]
Y = dataset[:,27]
# set missing values to 0
X[X == '?'] = numpy.nan
# convert to numeric
X = X.astype('float32')
# impute missing values as the mean
imputer = Imputer()
imputed_x = imputer.fit_transform(X)
# encode Y class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(imputed_x, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

运行此示例，我们看到的结果等同于将值固定为一（1）。这表明至少在这种情况下，我们最好用不同的零（0）值而不是有效值（1）或估算值来标记缺失值。

```py
Accuracy: 79.80%
```

当您缺少值时，尝试这两种方法（自动处理和输入）是一个很好的教训。

## 摘要

在这篇文章中，您发现了如何使用 Python 中的 XGBoost 为梯度提升准备机器学习数据。

具体来说，你学到了：

*   如何使用标签编码为二进制分类准备字符串类值。
*   如何使用一个热编码准备分类输入变量以将它们建模为二进制变量。
*   XGBoost 如何自动处理丢失的数据以及如何标记和估算缺失值。

您对如何为 XGBoost 或此帖子准备数据有任何疑问吗？在评论中提出您的问题，我会尽力回答。