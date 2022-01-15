# 如何用更多数据更新神经网络模型

> 原文：<https://machinelearningmastery.com/update-neural-network-models-with-more-data/>

用于预测建模的深度学习神经网络模型可能需要更新。

这可能是因为自模型开发和部署以来数据发生了变化，也可能是因为自模型开发以来额外的标记数据变得可用，并且预计额外的数据将提高模型的表现。

在为新数据更新神经网络模型时，用一系列不同的方法进行实验和评估是很重要的，尤其是如果模型更新将是自动的，例如定期进行。

**更新神经网络模型**的方法有很多，虽然两种主要的方法涉及要么使用现有模型作为起点并对其进行重新训练，要么保持现有模型不变，并将现有模型的预测与新模型相结合。

在本教程中，您将发现如何更新深度学习神经网络模型以响应新数据。

完成本教程后，您将知道:

*   当底层数据发生变化或有新的标记数据可用时，可能需要更新神经网络模型。
*   如何仅用新数据或新旧数据的组合来更新训练好的神经网络模型。
*   如何创建现有模型和新模型的集合，这些模型仅基于新数据或新旧数据的组合进行训练。

我们开始吧。

![How to Update Neural Network Models With More Data](img/782508ca44166b7d3808dc1d4f3af357.png)

如何用更多数据更新神经网络模型
朱迪·加拉格尔摄，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  更新神经网络模型
2.  再培训更新策略
    1.  仅在新数据上更新模型
    2.  新旧数据更新模型
3.  集合更新策略
    1.  仅基于新数据的集成模型
    2.  集成模型与新旧数据模型

## 更新神经网络模型

为预测建模项目选择并最终确定深度学习神经网络模型只是一个开始。

然后，您可以开始使用该模型对新数据进行预测。

您可能会遇到的一个可能的问题是，预测问题的性质可能会随着时间的推移而改变。

你可能会注意到这一点，因为预测的有效性可能会随着时间的推移而开始下降。这可能是因为在模型中做出和捕捉的假设正在改变或不再成立。

通常，这被称为“*概念漂移*”的问题，其中变量的潜在概率分布和变量之间的关系随着时间而变化，这可能对根据数据构建的模型产生负面影响。

有关概念漂移的更多信息，请参见教程:

*   [机器学习中概念漂移的温和介绍](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/)

概念漂移可能会在不同的时间影响您的模型，具体取决于您正在解决的预测问题以及为解决该问题而选择的模型。

随着时间的推移监视模型的表现，并使用模型表现的明显下降作为触发器来对模型进行更改，例如在新数据上对其进行重新训练，可能会有所帮助。

或者，您可能知道您的域中的数据变化足够频繁，以至于需要定期对模型进行更改，例如每周、每月或每年。

最后，您可能会运行一段时间您的模型，并积累具有已知结果的额外数据，您希望使用这些数据来更新您的模型，以提高预测表现。

重要的是，在响应问题的变化或新数据的可用性时，您有很大的灵活性。

例如，您可以采用经过训练的神经网络模型，并使用新数据更新模型权重。或者，我们可能希望保持现有模型不变，并将其预测与新模型相结合，以适应新获得的数据。

这些方法可能代表更新神经网络模型以响应新数据的两个一般主题，它们是:

*   重新培训更新策略。
*   集合更新策略。

让我们依次仔细看看每一个。

## 再培训更新策略

神经网络模型的一个好处是，随着不断的训练，它们的权重可以随时更新。

当响应底层数据的变化或新数据的可用性时，在更新神经网络模型时有几种不同的策略可供选择，例如:

*   仅在新数据上继续训练模型。
*   继续在新旧数据上训练模型。

我们还可以想象上述策略的变化，例如使用新数据的样本或新老数据的样本来代替所有可用的数据，以及对采样数据可能的基于实例的加权。

我们还可以考虑模型的扩展，冻结现有模型的层(例如，这样模型权重在训练期间就不能改变)，然后添加新的层，模型权重可以改变，移植到模型的扩展上来处理数据中的任何变化。也许这是下一节中的再培训和合奏方法的一种变体，我们现在就不说了。

然而，这是需要考虑的两个主要策略。

让我们用一个工作实例来具体说明这些方法。

### 仅在新数据上更新模型

我们只能根据新数据更新模型。

这种方法的一个极端版本是不使用任何新数据，而只是在旧数据上重新训练模型。这可能与*响应新数据不做任何事情*相同。在另一个极端，一个模型可能只适合新数据，抛弃旧数据和旧模型。

*   忽略新数据，什么都不做。
*   用新数据更新现有模型。
*   在新数据上安装新模型，丢弃旧模型和数据。

在这个例子中，我们将关注中间立场，但是在您的问题上测试所有三种方法并看看哪种方法最有效可能会很有趣。

首先，我们可以定义一个合成的二进制分类数据集，并将其分成两半，然后使用一部分作为“*旧数据*，另一部分作为“*新数据*

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
```

然后，我们可以定义一个多层感知机模型(MLP)，并将其仅适用于旧数据。

```py
...
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)
```

然后我们可以想象保存模型并使用一段时间。

随着时间的推移，我们希望根据已有的新数据对其进行更新。

这将涉及使用比正常情况小得多的学习速率，以便我们不会洗掉在旧数据上学习的权重。

**注**:你需要发现一个适合你的模型和数据集的学习率，这个学习率要比简单地从头拟合一个新模型获得更好的表现。

```py
...
# update model on new data only with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
```

然后，我们可以用这个较小的学习率将模型拟合到新数据上。

```py
...
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on new data
model.fit(X_new, y_new, epochs=100, batch_size=32, verbose=0)
```

将这些联系在一起，下面列出了仅在新数据上更新神经网络模型的完整示例。

```py
# update neural network with new data only
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# update model on new data only with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on new data
model.fit(X_new, y_new, epochs=100, batch_size=32, verbose=0)
```

接下来，让我们看看在新的和旧的数据上更新模型。

### 新旧数据更新模型

我们可以结合新旧数据更新模型。

这种方法的一个极端版本是丢弃模型，并简单地在所有可用的数据(新的和旧的)上拟合一个新的模型。一个不太极端的版本是使用现有模型作为起点，并基于组合数据集对其进行更新。

同样，测试这两种策略并看看什么适合您的数据集是个好主意。

在这种情况下，我们将重点关注不太极端的更新策略。

合成数据集和模型可以像以前一样适合旧数据集。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)
```

新数据可用，我们希望结合新旧数据更新模型。

首先，我们必须使用小得多的学习率，试图使用当前的权重作为搜索的起点。

**注**:你需要发现一个适合你的模型和数据集的学习率，这个学习率要比简单地从头拟合一个新模型获得更好的表现。

```py
...
# update model with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
```

然后，我们可以创建一个由新旧数据组成的复合数据集。

```py
...
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
```

最后，我们可以在这个复合数据集上更新模型。

```py
...
# fit the model on new data
model.fit(X_both, y_both, epochs=100, batch_size=32, verbose=0)
```

将这些联系在一起，下面列出了根据新旧数据更新神经网络模型的完整示例。

```py
# update neural network with both old and new data
from numpy import vstack
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# update model with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
# fit the model on new data
model.fit(X_both, y_both, epochs=100, batch_size=32, verbose=0)
```

接下来，让我们看看如何使用集成模型来响应新数据。

## 集合更新策略

集成是由多个其他模型组成的预测模型。

有许多不同类型的集合模型，尽管最简单的方法可能是平均来自多个不同模型的预测。

有关深度学习神经网络的集成算法的更多信息，请参见教程:

*   [深度学习神经网络的集成学习方法](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)

当响应底层数据的变化或新数据的可用性时，我们可以使用集成模型作为策略。

与上一节中的方法类似，我们可以考虑两种集成学习算法的方法，作为响应新数据的策略；它们是:

*   现有模型和新模型的集合仅适用于新数据。
*   现有模型和新模型的集成适用于新旧数据。

同样，我们可能会考虑这些方法的变化，例如新旧数据的样本，以及集成中包含的一个以上的现有模型或附加模型。

然而，这是需要考虑的两个主要策略。

让我们用一个工作实例来具体说明这些方法。

### 仅基于新数据的集成模型

我们可以创建现有模型的集合，而新模型只适合新数据。

期望集合预测比单独使用旧模型或新模型表现得更好或更稳定(方差更低)。在采用集成之前，应该对数据集进行检查。

首先，我们可以准备数据集并拟合旧模型，就像我们在前面几节中所做的那样。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)
```

随着时间的推移，新数据变得可用。

然后，我们可以在新数据上拟合新模型，自然地发现仅在新数据集上运行良好或最佳的模型和配置。

在这种情况下，我们将简单地使用与旧模型相同的模型架构和配置。

```py
...
# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
```

然后，我们可以只在新数据上拟合这个新模型。

```py
...
# fit the model on old data
new_model.fit(X_new, y_new, epochs=150, batch_size=32, verbose=0)
```

现在我们有了这两个模型，我们可以用每个模型进行预测，并将预测的平均值计算为“*集合预测*”

```py
...
# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```

将这些联系在一起，下面列出了使用现有模型和仅适用于新数据的新模型的集合进行更新的完整示例。

```py
# ensemble old neural network with new model fit on new data only
from numpy import hstack
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
new_model.fit(X_new, y_new, epochs=150, batch_size=32, verbose=0)

# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```

### 集成模型与新旧数据模型

我们可以创建一个现有模型和一个新模型的集合，以适应旧数据和新数据。

期望集合预测比单独使用旧模型或新模型表现得更好或更稳定(方差更低)。在采用集成之前，应该对数据集进行检查。

首先，我们可以准备数据集并拟合旧模型，就像我们在前面几节中所做的那样。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)
```

随着时间的推移，新数据变得可用。

然后，我们可以在新旧数据的组合上拟合新模型，自然地发现仅在新数据集上运行良好或最佳的模型和配置。

在这种情况下，我们将简单地使用与旧模型相同的模型架构和配置。

```py
...
# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
```

我们可以从旧数据和新数据创建一个复合数据集，然后在这个数据集上拟合新模型。

```py
...
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
# fit the model on old data
new_model.fit(X_both, y_both, epochs=150, batch_size=32, verbose=0)
```

最后，我们可以一起使用这两个模型来进行集合预测。

```py
...
# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```

将这些联系在一起，下面列出了使用现有模型和适合新旧数据的新模型的集合进行更新的完整示例。

```py
# ensemble old neural network with new model fit on old and new data
from numpy import hstack
from numpy import vstack
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
# fit the model on old data
new_model.fit(X_both, y_both, epochs=150, batch_size=32, verbose=0)

# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习中概念漂移的温和介绍](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/)
*   [深度学习神经网络的集成学习方法](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)

## 摘要

在本教程中，您发现了如何更新深度学习神经网络模型以响应新数据。

具体来说，您了解到:

*   当底层数据发生变化或有新的标记数据可用时，可能需要更新神经网络模型。
*   如何仅用新数据或新旧数据的组合来更新训练好的神经网络模型。
*   如何创建现有模型和新模型的集合，这些模型仅基于新数据或新旧数据的组合进行训练。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。