# 基于 Keras 的深度学习模型中的dropout正则化

> 原文： [https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

神经网络和深度学习模型的简单而强大的正则化技术是dropout。

在这篇文章中，您将发现 dropout 正则化技术以及如何将其应用于使用 Keras 的 Python 模型。

阅读这篇文章后你会知道：

*   dropout正则化技术如何工作。
*   如何在输入层上使用 dropout。
*   如何在隐藏层上使用 dropout。
*   如何调整问题的dropout水平。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0，TensorFlow 0.10.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![Dropout Regularization in Deep Learning Models With Keras](img/79c8df17f76c38e054a8760b935e6ed9.png)

使用 Keras 的深度学习模型中的dropout正规化
照片由 [Trekking Rinjani](https://www.flickr.com/photos/trekkingrinjani/4930552641/) ，保留一些权利。

## 神经网络的丢失正则化

Dropout 是 Srivastava 等人提出的神经网络模型的正则化技术。在他们的 2014 年论文[dropout：一种防止神经网络过度拟合的简单方法](http://jmlr.org/papers/v15/srivastava14a.html)（[下载 PDF](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) ）。

dropout是一种在训练过程中忽略随机选择的神经元的技术。他们随机“dropout”。这意味着它们对下游神经元激活的贡献在正向通过时暂时消除，并且任何重量更新都不会应用于向后通过的神经元。

当神经网络学习时，神经元权重在网络中的上下文中进行。针对特定特征调整神经元的权重，从而提供一些特化。相邻神经元变得依赖于这种专业化，如果采取太多可能导致脆弱的模型太专门于训练数据。这在训练期间依赖于神经元的背景被称为复杂的共同适应。

你可以想象，如果神经元在训练过程中随机掉出网络，那么其他神经元将不得不介入并处理对缺失神经元进行预测所需的表示。这被认为导致网络学习多个独立的内部表示。

其结果是网络对神经元的特定权重变得不那么敏感。这反过来导致网络能够更好地概括并且不太可能过度拟合训练数据。

## 科拉斯的dropout规范化

通过以每个权重更新周期的给定概率（例如 20％）随机选择要丢弃的节点，可以容易地实现丢失。这就是在卡拉斯实施 Dropout 的方式。 Dropout 仅在模型训练期间使用，在评估模型的技能时不使用。

接下来，我们将探讨在 Keras 中使用 Dropout 的几种不同方法。

这些示例将使用 [Sonar 数据集](http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))。这是一个二元分类问题，其目标是从声纳啁啾返回中正确识别岩石和模拟地雷。它是神经网络的一个很好的测试数据集，因为所有输入值都是数字的并且具有相同的比例。

数据集可以是从 UCI 机器学习库下载的[。您可以将声纳数据集放在当前工作目录中，文件名为 sonar.csv。](http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)

我们将使用带有 10 倍交叉验证的 scikit-learn 来评估开发的模型，以便更好地梳理结果中的差异。

有 60 个输入值和一个输出值，输入值在用于网络之前已标准化。基线神经网络模型具有两个隐藏层，第一个具有 60 个单元，第二个具有 30 个。随机梯度下降用于训练具有相对低的学习率和动量的模型。

下面列出了完整的基线模型。

```py
# Baseline Model on the Sonar Dataset
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行该示例可生成 86％的估计分类准确度。

```py
Baseline: 86.04% (4.58%)
```

## 在可见层上使用 Dropout

Dropout 可以应用于称为可见层的输入神经元。

在下面的示例中，我们在输入（或可见层）和第一个隐藏层之间添加一个新的 Dropout 层。dropout率设置为 20％，这意味着每个更新周期中将随机排除五分之一输入。

此外，正如 Dropout 原始论文中所建议的那样，对每个隐藏层的权重施加约束，确保权重的最大范数不超过值 3.这可以通过在密集上设置 kernel_constraint 参数来完成。构造层时的类。

学习率提高了一个数量级，动量增加到 0.9。原始 Dropout 论文中也推荐了这些学习率的提高。

继续上面的基线示例，下面的代码使用输入丢失来运行相同的网络。

```py
# dropout in the input layer with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(60,)))
	model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行该示例至少在单次测试运行中提供了分类精度的小幅下降。

```py
Visible: 83.52% (7.68%)
```

## 在隐藏层上使用 Dropout

Dropout 可以应用于网络模型体内的隐藏神经元。

在下面的示例中，Dropout 应用于两个隐藏层之间以及最后一个隐藏层和输出层之间。再次使用 20％的dropout率，以及对这些层的权重约束。

```py
# dropout in hidden layers with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

我们可以看到，针对此问题以及所选网络配置，在隐藏层中使用丢失并未提升表现。事实上，表现比基线差。

可能需要额外的训练时期或者需要进一步调整学习率。

```py
Hidden: 83.59% (7.31%)
```

## 使用 Dropout 的提示

关于 Dropout 的原始论文提供了一套标准机器学习问题的实验结果。因此，在实践中使用dropout时，他们提供了许多有用的启发式方法。

*   通常，使用 20％-50％神经元的小dropout值，20％提供良好的起点。概率太低具有最小的影响而且值太高会导致网络的学习不足。
*   使用更大的网络。当在较大的网络上使用 dropout 时，您可能会获得更好的表现，从而为模型提供更多学习独立表示的机会。
*   在传入（可见）和隐藏单位上使用 dropout。在网络的每一层应用丢失已经显示出良好的结果。
*   使用具有衰减和大动量的大学习率。将学习率提高 10 到 100 倍，并使用 0.9 或 0.99 的高动量值。
*   限制网络权重的大小。较大的学习率可能导致非常大的网络权重。对网络权重的大小施加约束，例如大小为 4 或 5 的最大范数正则化已被证明可以改善结果。

## 关于dropout的更多资源

以下是一些资源，您可以用它们来了解有关神经网络和深度学习模型中的丢失的更多信息。

*   [dropout：一种防止神经网络过度拟合的简单方法](http://jmlr.org/papers/v15/srivastava14a.html)（原始论文）。
*   [通过阻止特征检测器的共同适应来改善神经网络](http://arxiv.org/abs/1207.0580)。
*   [dropout方法如何在深度学习中发挥作用？ Quora 上的](https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning)。

## 摘要

在这篇文章中，您发现了深度学习模型的丢失正则化技术。你了解到：

*   dropout是什么以及如何运作。
*   如何在自己的深度学习模型中使用 dropout。
*   在您自己的模型上从dropout中获得最佳结果的提示。

您对dropout或这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。