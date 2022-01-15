# Keras 深度学习模型中的丢弃正则化

> 原文： [https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

神经网络和深度学习模型的简单而强大的正则化技术是 dropout。

在这篇文章中，您将了解 dropout 正则化技术以及如何将其应用于使用 Keras 用 Python 编写的模型中。

阅读这篇文章后你会知道：

*   dropout 正则化技术原理。
*   如何在输入层上使用 dropout。
*   如何在隐藏层上使用 dropout。
*   如何针对具体问题对 dropout 调优

让我们开始吧。

*   **2016 年 10 月更新**：更新了 Keras 1.1.0，TensorFlow 0.10.0 和 scikit-learn v0.18 的示例。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![Dropout Regularization in Deep Learning Models With Keras](img/79c8df17f76c38e054a8760b935e6ed9.png)

使用 Keras 的深度学习模型中的 dropout 正规化
照片由 [Trekking Rinjani](https://www.flickr.com/photos/trekkingrinjani/4930552641/) ，保留一些权利。

## 神经网络的 dropout 正则化

Dropout 是 Srivastava 等人提出的神经网络模型的正则化技术。在他们的 2014 年论文[dropout：一种防止神经网络过拟合的简单方法](http://jmlr.org/papers/v15/srivastava14a.html)（[下载 PDF](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) ）。

dropout 是一种在训练过程中忽略随机选择的神经元的技术。这些神经元被随机“dropout“，这意味着它们对激活下一层神经元的贡献在正向传递时暂时消除，并且在反向传递时任何权重更新也不会应用于这些神经元。

当神经网络学习时，网络中的神经元的权重将进行调整重置。神经元的权重针对某些特征进行调优，具有一些特殊化。周围的神经元则会依赖于这种特殊化，如果过于特殊化，模型会因为对训练数据过拟合而变得脆弱不堪。神经元在训练过程中的这种依赖于上下文的现象被称为复杂的协同适应（complex co-adaptations）。

你可以想象到如果神经元在训练过程中被随机丢弃，那么其他神经元因缺失神经元不得不介入并替代缺失神经元的那部分表征，为预测结果提供信息。人们认为这样网络模型可以学到多种相互独立的内部表征。

其结果是网络对神经元的特定权重变得不那么敏感。这反过来使得网络能够更好地泛化，减少了过拟合训练数据的可能性。

## Keras 的 dropout 规范化

通过以每轮权重更新时的给定概率（例如 20％）随机选择要丢弃的节点、。这就是在 Keras 实现 Dropout 的方式。 Dropout 仅在模型训练期间使用，在评估模型的表现时不使用。

接下来，我们将探讨在 Keras 中使用 Dropout 的几种不同方法。

这些示例将使用 [Sonar 数据集](http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))。这是一个二分类问题，其目标是利用声纳打印正确识别岩石和模拟地雷。它是神经网络的一个很好的测试数据集，因为所有输入值都是数字的并且具有相同的量纲。

数据集可以是从 UCI 机器学习库下载的[。您可以将声纳数据集放在当前工作目录中，文件名为 sonar.csv。](http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)

我们将使用带有 10 折交叉验证的 scikit-learn 来评估模型的质量，以便更好地梳理结果中的差异。

有 60 个输入值和一个输出值，输入值在用于网络之前已归一化。基准神经网络模型具有两个隐藏层，第一个具有 60 个单元，第二个具有 30 个。随机梯度下降用于训练具有相对低的学习率和冲量的模型。

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

运行该示例可生成 分类准确度约为 86%。

```py
Baseline: 86.04% (4.58%)
```

## 在可见层上使用 Dropout

Dropout 可以应用于称为可见层的输入神经元。

在下面的示例中，我们在输入（或可见层）和第一个隐藏层之间添加一个新的 Dropout 层。dropout 率设置为 20％，这意味着每个更新周期中将随机丢弃五分之一输入。

此外，正如 Dropout 那篇论文中所建议的那样，对每个隐藏层的权重施加约束，确保权重的最大范数不超过值 3.这可以通过在构造模型层时设置 kernel_constraint 参数来完成。

学习率提高了一个数量级，冲量增加到 0.9。 这也是 Dropout 论文中推荐的做法。

继续上面的基准示例，下面的代码使用输入层 dropout 的网络模型。

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

运行该示例在单次测试运行中分类精度小幅下降。

```py
Visible: 83.52% (7.68%)
```

## 在隐藏层上使用 Dropout

Dropout 可以应用于网络模型内的隐藏层节点。

在下面的示例中，Dropout 应用于两个隐藏层之间以及最后一个隐藏层和输出层之间。再次使用 20％的 dropout 率，并且对这些层进行权重约束。

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

我们可以看到，针对此问题以及所选模型配置参数，在隐藏层中使用 dropout 并未提升模型效果。事实上，表现比基准差。

可能需要更多的训练迭代次数或者需要进一步调整学习率。

```py
Hidden: 83.59% (7.31%)
```

## 使用 Dropout 的技巧

关于 Dropout 的原始论文提供了一套标准机器学习问题的实践性结论。因此在运用 dropout 时，会带来很多帮助。

*   通常，使用 20％-50％神经元的小 dropout 值，20％可作为良好的起点。比例太低具有最小的影响比列太高会导致模型的欠学习。
*   使用更大的网络。当在较大的网络上使用 dropout 时，模型可能会获得更好的表现，模型有更多的机会学习到多种独立的表征。
*   在输入层（可见层）和隐藏层都使用 dropout。在网络的每一层应用 dropout 已被证明具有良好的结果。
*   增加学习率和冲量。将学习率提高 10 到 100 倍，并使用 0.9 或 0.99 的高冲量值。
*   限制网络模型权重的大小。较大的学习率可能导致非常大的权重值。对网络的权重值做最大范数正则化等方法，例如大小为 4 或 5 的最大范数正则化已被证明可以改善结果。

## 关于 dropout 的更多资源

以下是一些资源，您可以用它们来了解有关神经网络和深度学习模型中的 dropout 的更多信息。

*   [dropout：一种防止神经网络过拟合的简单方法](http://jmlr.org/papers/v15/srivastava14a.html)（原始论文）。
*   [通过阻止特征检测器的共同适应来改善神经网络](http://arxiv.org/abs/1207.0580)。
*   [dropout 方法如何在深度学习中发挥作用？ Quora 上的](https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning)。

## 总结

在这篇文章中，您了解了深度学习模型的 dropout 正则化技术。你了解到：

*   dropout 含义和原理。
*   如何在自己的深度学习模型中使用 dropout。
*   使用 dropout 的技巧。

您对 dropout 或这篇文章有任何疑问吗？在评论中提出您的问题，我会尽力回答。
