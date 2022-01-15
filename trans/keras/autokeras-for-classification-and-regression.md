# 如何使用 AutoKeras 进行分类和回归

> 原文:[https://machinelearning master . com/auto keras-for-classing-and-revolution/](https://machinelearningmastery.com/autokeras-for-classification-and-regression/)

最后更新于 2020 年 9 月 6 日

AutoML 指的是自动发现给定数据集的最佳模型的技术。

当应用于神经网络时，这涉及发现模型架构和用于训练模型的超参数，通常称为**神经架构搜索**。

AutoKeras 是一个开源库，用于为深度学习模型执行 AutoML。通过 TensorFlow tf.keras API 使用所谓的 Keras 模型执行搜索。

它提供了一种简单有效的方法，可以自动为各种预测建模任务找到性能最佳的模型，包括表格或所谓的结构化分类和回归数据集。

在本教程中，您将发现如何使用 AutoKeras 为分类和回归任务找到良好的神经网络模型。

完成本教程后，您将知道:

*   AutoKeras 是 AutoML 的一个实现，用于使用神经架构搜索的深度学习。
*   如何使用 AutoKeras 为二进制分类数据集找到性能最佳的模型。
*   如何使用 AutoKeras 为回归数据集查找性能最佳的模型。

我们开始吧。

*   **更新 2020 年 9 月**:更新 AutoKeras 版本和安装说明。

![How to Use AutoKeras for Classification and Regression](img/6b6912c8c8c336786e733dc05cbd65bf.png)

如何使用 AutoKeras 进行分类和回归
图片由 [kanu101](https://flickr.com/photos/kateure1309/24972028303/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  深度学习自动课程
2.  用于分类的自动分类器
3.  回归的自动 Keras

## 深度学习自动课程

[自动机器学习](https://en.wikipedia.org/wiki/Automated_machine_learning)，简称 AutoML，是指为预测建模问题自动寻找数据准备、模型和模型超参数的最佳组合。

AutoML 的好处是允许机器学习从业者以很少的输入快速有效地处理预测建模任务，例如“开火并忘记”。

> 随着机器学习技术的广泛应用，自动机器学习已经成为一个非常重要的研究课题。AutoML 的目标是让机器学习背景知识有限的人能够轻松使用机器学习模型。

——[Auto-keras:一个高效的神经架构搜索系统](https://www.kdd.org/kdd2019/accepted-papers/view/auto-keras-an-efficient-neural-architecture-search-system)，2019。

AutoKeras 是 AutoML 的一个实现，用于使用 Keras API 的深度学习模型，特别是 TensorFlow 2 提供的 [tf.keras API。](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)

它使用搜索神经网络架构的过程来最好地解决建模任务，更一般地称为[神经架构搜索](https://en.wikipedia.org/wiki/Neural_architecture_search)，简称 NAS。

> ……基于我们提出的方法，我们开发了一个被广泛采用的开源 AutoML 系统，即 Auto-Keras。它是一个开源的 AutoML 系统，可以在本地下载和安装。

——[Auto-keras:一个高效的神经架构搜索系统](https://www.kdd.org/kdd2019/accepted-papers/view/auto-keras-an-efficient-neural-architecture-search-system)，2019。

本着 Keras 的精神，AutoKeras 为不同的任务提供了一个易于使用的界面，例如图像分类、结构化数据分类或回归等等。用户只需要指定数据的位置和要尝试的模型数量，并返回一个在该数据集上获得最佳性能(在配置的约束下)的模型。

**注意** : AutoKeras 提供的是 TensorFlow 2 Keras 模型(例如 tf.keras)，而不是 Standalone Keras 模型。因此，该库假设您安装了 Python 3 和 TensorFlow 2.3.0 或更高版本。

在编写时，您需要手动安装名为 [keras-tuner](https://keras-team.github.io/keras-tuner/) 的必备库。您可以按如下方式安装此库:

```py
sudo pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
```

如果情况再次发生变化，就像快速移动的开源项目经常发生的那样，请参见这里的官方安装说明:

*   [AutoKeras 安装说明](https://autokeras.com/install/)

现在我们可以安装 AutoKeras 了。

要安装 AutoKeras，可以使用 Pip，如下所示:

```py
sudo pip install autokeras
```

您可以确认安装成功，并按如下方式检查版本号:

```py
sudo pip show autokeras
```

您应该会看到如下输出:

```py
Name: autokeras
Version: 1.0.8
Summary: AutoML for deep learning
Home-page: http://autokeras.com
Author: Data Analytics at Texas A&M (DATA) Lab, Keras Team
Author-email: jhfjhfj1@gmail.com
License: MIT
Location: ...
Requires: tensorflow, packaging, pandas, scikit-learn
Required-by: 
```

一旦安装完成，您就可以应用 AutoKeras 为您的预测建模任务找到一个好的或很棒的神经网络模型。

我们将看两个常见的例子，在这些例子中，您可能希望对表格数据(所谓的结构化数据)使用 AutoKeras、分类和回归。

## 用于分类的自动分类器

AutoKeras 可用于发现表格数据分类任务的良好或伟大模型。

回想一下，表格数据是由行和列组成的数据集，如表格或您在电子表格中看到的数据。

在本节中，我们将为声纳分类数据集开发一个模型，用于将声纳回波分类为岩石或地雷。该数据集由 208 行数据组成，包含 60 个输入要素，目标类别标签为 0(岩石)或 1(矿山)。

一个简单的模型可以通过重复的 10 倍交叉验证达到大约 53.4%的分类准确率，这提供了一个下限。一个好的模型可以达到 88.2%左右的准确率，提供一个上限。

您可以在此了解有关数据集的更多信息:

*   [声纳数据集(声纳. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   [声纳数据集描述(声纳.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

不需要下载数据集；作为示例的一部分，我们将自动下载它。

首先，我们可以下载数据集，并将其分成随机选择的训练集和测试集，其中 33%用于测试，67%用于训练。

下面列出了完整的示例。

```py
# load the sonar dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
print(dataframe.shape)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# basic data preparation
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

运行该示例首先下载数据集并汇总形状，显示预期的行数和列数。

然后数据集被分成输入和输出元素，然后这些元素被进一步分成训练和测试数据集。

```py
(208, 61)
(208, 60) (208,)
(139, 60) (69, 60) (139,) (69,)
```

我们可以使用 AutoKeras 为这个数据集自动发现一个有效的神经网络模型。

这可以通过使用 [StructuredDataClassifier](https://autokeras.com/structured_data_classifier/) 类并指定要搜索的模型数量来实现。这定义了要执行的搜索。

```py
...
# define the search
search = StructuredDataClassifier(max_trials=15)
```

然后，我们可以使用加载的数据集执行搜索。

```py
...
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
```

这可能需要几分钟时间，并将报告搜索进度。

接下来，我们可以在测试数据集上评估模型，看看它在新数据上的表现。

```py
...
# evaluate the model
loss, acc = search.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)
```

然后，我们使用该模型对新的数据行进行预测。

```py
...
# use the model to make a prediction
row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
X_new = asarray([row]).astype('float32')
yhat = search.predict(X_new)
print('Predicted: %.3f' % yhat[0])
```

我们可以检索最终的模型，它是 TensorFlow Keras 模型的一个实例。

```py
...
# get the best performing model
model = search.export_model()
```

然后我们可以总结模型的结构，看看选择了什么。

```py
...
# summarize the loaded model
model.summary()
```

最后，我们可以将模型保存到文件中以备后用，可以使用 TensorFlow [load_model()函数](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)进行加载。

```py
...
# save the best performing model to file
model.save('model_sonar.h5')
```

将这些联系在一起，下面列出了应用 AutoKeras 为声纳数据集找到有效神经网络模型的完整示例。

```py
# use autokeras to find a model for the sonar dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
print(dataframe.shape)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# basic data preparation
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define the search
search = StructuredDataClassifier(max_trials=15)
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
# evaluate the model
loss, acc = search.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)
# use the model to make a prediction
row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
X_new = asarray([row]).astype('float32')
yhat = search.predict(X_new)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model_sonar.h5')
```

运行该示例将报告大量关于搜索进度的调试信息。

模型和结果都保存在当前工作目录中名为“*结构化数据分类器*的文件夹中。

```py
...
[Trial complete]
[Trial summary]
 |-Trial ID: e8265ad768619fc3b69a85b026f70db6
 |-Score: 0.9259259104728699
 |-Best step: 0
 > Hyperparameters:
 |-classification_head_1/dropout_rate: 0
 |-optimizer: adam
 |-structured_data_block_1/dense_block_1/dropout_rate: 0.0
 |-structured_data_block_1/dense_block_1/num_layers: 2
 |-structured_data_block_1/dense_block_1/units_0: 32
 |-structured_data_block_1/dense_block_1/units_1: 16
 |-structured_data_block_1/dense_block_1/units_2: 512
 |-structured_data_block_1/dense_block_1/use_batchnorm: False
 |-structured_data_block_1/dense_block_2/dropout_rate: 0.25
 |-structured_data_block_1/dense_block_2/num_layers: 3
 |-structured_data_block_1/dense_block_2/units_0: 32
 |-structured_data_block_1/dense_block_2/units_1: 16
 |-structured_data_block_1/dense_block_2/units_2: 16
 |-structured_data_block_1/dense_block_2/use_batchnorm: False
```

然后在搁置测试数据集上评估性能最佳的模型。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到该模型实现了大约 82.6%的分类准确率。

```py
Accuracy: 0.826
```

接下来，报告性能最佳的模型的体系结构。

我们可以看到一个模型有两个隐藏层，分别是 drop 和 ReLU 激活。

```py
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 60)]              0
_________________________________________________________________
categorical_encoding (Catego (None, 60)                0
_________________________________________________________________
dense (Dense)                (None, 256)               15616
_________________________________________________________________
re_lu (ReLU)                 (None, 256)               0
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               131584
_________________________________________________________________
re_lu_1 (ReLU)               (None, 512)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513
_________________________________________________________________
classification_head_1 (Sigmo (None, 1)                 0
=================================================================
Total params: 147,713
Trainable params: 147,713
Non-trainable params: 0
_________________________________________________________________
```

## 回归的自动 Keras

AutoKeras 也可以用于回归任务，即预测数值的预测建模问题。

我们将使用汽车保险数据集，该数据集包括根据索赔总数预测总付款。数据集有 63 行，一个输入变量和一个输出变量。

使用重复的 10 倍交叉验证，一个简单的模型可以获得大约 66 的平均绝对误差(MAE)，提供了预期性能的下限。一个好的模型可以达到 28 左右的 MAE，提供一个性能上限。

您可以在此了解有关此数据集的更多信息:

*   [车险数据集(auto-insurance.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv)
*   [汽车保险数据集(汽车保险.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.names)

我们可以加载数据集并将其分成输入和输出元素，然后训练和测试数据集。

下面列出了完整的示例。

```py
# load the sonar dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
print(dataframe.shape)
# split into input and output elements
data = dataframe.values
data = data.astype('float32')
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

运行该示例加载数据集，确认行数和列数，然后将数据集拆分为训练集和测试集。

```py
(63, 2)
(63, 1) (63,)
(42, 1) (21, 1) (42,) (21,)
```

可以使用 [StructuredDataRegressor](https://autokeras.com/structured_data_regressor/) 类将自动 Keras 应用于回归任务，并为要试用的模型数量进行配置。

```py
...
# define the search
search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
```

然后可以运行搜索并保存最佳模型，就像在分类案例中一样。

```py
...
# define the search
search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
```

然后，我们可以使用性能最好的模型，并在等待数据集上对其进行评估，对新数据进行预测，并总结其结构。

```py
...
# evaluate the model
mae, _ = search.evaluate(X_test, y_test, verbose=0)
print('MAE: %.3f' % mae)
# use the model to make a prediction
X_new = asarray([[108]]).astype('float32')
yhat = search.predict(X_new)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model_insurance.h5')
```

将这些联系在一起，下面列出了使用 AutoKeras 为汽车保险数据集发现有效神经网络模型的完整示例。

```py
# use autokeras to find a model for the insurance dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
print(dataframe.shape)
# split into input and output elements
data = dataframe.values
data = data.astype('float32')
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define the search
search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
# evaluate the model
mae, _ = search.evaluate(X_test, y_test, verbose=0)
print('MAE: %.3f' % mae)
# use the model to make a prediction
X_new = asarray([[108]]).astype('float32')
yhat = search.predict(X_new)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model_insurance.h5')
```

运行该示例将报告大量关于搜索进度的调试信息。

模型和结果都保存在当前工作目录中名为“*structured _ data _ reversor*”的文件夹中。

```py
...
[Trial summary]
|-Trial ID: ea28b767d13e958c3ace7e54e7cb5a14
|-Score: 108.62509155273438
|-Best step: 0
> Hyperparameters:
|-optimizer: adam
|-regression_head_1/dropout_rate: 0
|-structured_data_block_1/dense_block_1/dropout_rate: 0.0
|-structured_data_block_1/dense_block_1/num_layers: 2
|-structured_data_block_1/dense_block_1/units_0: 16
|-structured_data_block_1/dense_block_1/units_1: 1024
|-structured_data_block_1/dense_block_1/units_2: 128
|-structured_data_block_1/dense_block_1/use_batchnorm: True
|-structured_data_block_1/dense_block_2/dropout_rate: 0.5
|-structured_data_block_1/dense_block_2/num_layers: 2
|-structured_data_block_1/dense_block_2/units_0: 256
|-structured_data_block_1/dense_block_2/units_1: 64
|-structured_data_block_1/dense_block_2/units_2: 1024
|-structured_data_block_1/dense_block_2/use_batchnorm: True
```

然后在搁置测试数据集上评估性能最佳的模型。

**注**:考虑到算法或评估程序的随机性，或数值精度的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了大约 24 的 MAE。

```py
MAE: 24.916
```

接下来，报告性能最佳的模型的体系结构。

我们可以看到一个带有两个隐藏层的模型，带有 ReLU 激活。

```py
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
categorical_encoding (Catego (None, 1)                 0
_________________________________________________________________
dense (Dense)                (None, 64)                128
_________________________________________________________________
re_lu (ReLU)                 (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               33280
_________________________________________________________________
re_lu_1 (ReLU)               (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               65664
_________________________________________________________________
re_lu_2 (ReLU)               (None, 128)               0
_________________________________________________________________
regression_head_1 (Dense)    (None, 1)                 129
=================================================================
Total params: 99,201
Trainable params: 99,201
Non-trainable params: 0
_________________________________________________________________
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [自动机器学习，维基百科](https://en.wikipedia.org/wiki/Automated_machine_learning)。
*   [神经架构搜索，维基百科](https://en.wikipedia.org/wiki/Neural_architecture_search)。
*   [自动贩卖机主页](https://autokeras.com/)。
*   [AutoKeras GitHub 项目](https://github.com/keras-team/autokeras)。
*   [Auto-keras:一个高效的神经架构搜索系统](https://www.kdd.org/kdd2019/accepted-papers/view/auto-keras-an-efficient-neural-architecture-search-system)，2019。
*   [标准分类和回归机器学习数据集的结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)

## 摘要

在本教程中，您发现了如何使用 AutoKeras 为分类和回归任务找到良好的神经网络模型。

具体来说，您了解到:

*   AutoKeras 是 AutoML 的一个实现，用于使用神经架构搜索的深度学习。
*   如何使用 AutoKeras 为二进制分类数据集找到性能最佳的模型。
*   如何使用 AutoKeras 为回归数据集查找性能最佳的模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。