# 如何用Keras中的长短期记忆模型做出预测

> 原文： [https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/](https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/)

开发LSTM模型的目标是可以用于序列预测问题的最终模型。

在这篇文章中，您将了解如何最终确定模型并使用它来预测新数据。

完成这篇文章后，你会知道：

*   如何训练最终的LSTM模型。
*   如何保存最终的LSTM模型，然后再次加载它。
*   如何预测新数据。

让我们开始吧。

![How to Make Predictions with Long Short-Term Memory Models with Keras](img/cd1edf00ee74f318b8e52c73721422c7.jpg)

如何使用Keras
用长期短期记忆模型做出预测 [Damon jah](https://www.flickr.com/photos/damonjah/15490904985/) ，保留一些权利。

## 步骤1.训练最终模型

### 什么是最终的LSTM模型？

最终的LSTM模型是用于对新数据做出预测的模型。

也就是说，给定输入数据的新示例，您希望使用该模型来预测预期输出。这可以是分类（分配标签）或回归（实际值）。

序列预测项目的目标是获得最佳的最终模型，其中“最佳”定义为：

*   **数据**：您提供的历史数据。
*   **时间**：你必须在项目上花费的时间。
*   **程序**：数据准备步骤，算法或算法，以及所选的算法配置。

在项目中，您可以收集数据，花费时间，发现数据准备过程，要使用的算法以及如何配置它。

最终的模型是这个过程的顶峰，你寻求的目的是为了开始实际做出预测。

没有完美的模型这样的东西。只有你能发现的最好的模型。

### 如何敲定LSTM模型？

您可以通过在所有数据上应用所选的LSTM架构和配置来最终确定模型。

没有训练和测试拆分，也没有交叉验证折叠。将所有数据重新组合到一个大型训练数据集中，以适合您的模型。

而已。

使用最终模型，您可以：

*   保存模型以供以后或操作使用。
*   加载模型并对新数据做出预测。

有关训练最终模型的更多信息，请参阅帖子：

*   [如何训练最终机器学习模型](http://machinelearningmastery.com/train-final-machine-learning-model/)

## 第2步。保存最终模型

Keras提供了一个API，允许您将模型保存到文件中。

该模型以HDF5文件格式保存，可有效地在磁盘上存储大量数字。您需要确认已安装h5py Python库。它可以安装如下：

```py
sudo pip install h5py
```

您可以使用模型上的save（）函数将适合的Keras模型保存到文件中。

例如：

```py
# define model
model = Sequential()
model.add(LSTM(...))
# compile model
model.compile(...)
# fit model
model.fit(...)
# save model to single file
model.save('lstm_model.h5')
```

此单个文件将包含模型架构和权重。它还包括所选损失和优化算法的规范，以便您可以恢复训练。

可以使用load_model（）函数再次加载模型（来自不同Python会话中的不同脚本）。

```py
from keras.models import load_model
# load model from single file
model = load_model('lstm_model.h5')
# make predictions
yhat = model.predict(X, verbose=0)
print(yhat)
```

下面是一个完整的LSTM模型拟合示例，将其保存到单个文件中，然后再次加载。尽管模型的加载位于同一脚本中，但此部分可以从另一个Python会话中的另一个脚本运行。运行该示例将模型保存到文件lstm_model.h5。

```py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model

# return training data
def get_train():
seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
seq = array(seq)
X, y = seq[:, 0], seq[:, 1]
X = X.reshape((len(X), 1, 1))
return X, y

# define model
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1, activation='linear'))
# compile model
model.compile(loss='mse', optimizer='adam')
# fit model
X,y = get_train()
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# save model to single file
model.save('lstm_model.h5')

# snip...
# later, perhaps run from another script

# load model from single file
model = load_model('lstm_model.h5')
# make predictions
yhat = model.predict(X, verbose=0)
print(yhat)
```

有关保存和加载Keras模型的更多信息，请参阅帖子：

*   [保存并加载您的Keras深度学习模型](http://machinelearningmastery.com/save-load-keras-deep-learning-models/)

## 第3步。对新数据做出预测

在完成模型并将其保存到文件后，您可以加载它并使用它来做出预测。

For example:

*   在序列回归问题上，这可能是下一时间步的实际值的预测。
*   在序列分类问题上，这可能是给定输入序列的类结果。

或者它可能是基于序列预测问题细节的任何其他变化。您希望给出输入序列（X）的模型（yhat）的结果，其中序列（y）的真实结果当前是未知的。

您可能有兴趣在生产环境中做出预测，作为接口的后端或手动做出预测。这实际上取决于项目的目标。

在拟合最终模型之前对训练数据执行的任何数据准备也必须在做出预测之前应用于任何新数据。

预测是容易的部分。

它涉及获取准备好的输入数据（X）并在加载的模型上调用Keras预测方法之一。

请记住，做出预测（X）的输入仅包括做出预测所需的输入序列数据，而不是所有先前的训练数据。在预测一个序列中的下一个值的情况下，输入序列将是1个样本，具有固定数量的时间步长和在定义和拟合模型时使用的特征。

例如，可以通过调用模型上的predict（）函数来对输出层的激活函数的形状和比例进行原始预测：

```py
X = ...
model = ...
yhat = model.predict(X)
```

可以通过调用模型上的predict_classes（）函数来预测类索引。

```py
X = ...
model = ...
yhat = model.predict_classes(X)
```

可以通过调用模型上的predict_proba（）函数来预测概率。

```py
X = ...
model = ...
yhat = model.predict_proba(X)
```

有关Keras模型生命周期的更多信息，请参阅帖子：

*   [Keras中长期短期记忆模型的5步生命周期](http://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 帖子

*   [如何训练最终机器学习模型](http://machinelearningmastery.com/train-final-machine-learning-model/)
*   [保存并加载您的Keras深度学习模型](http://machinelearningmastery.com/save-load-keras-deep-learning-models/)
*   [Keras中长期短期记忆模型的5步生命周期](http://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)

### API

*   [如何保存Keras型号？在Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) 中。
*   [保存并加载Keras API](https://keras.io/models/about-keras-models/) 。

## 摘要

在这篇文章中，您了解了如何最终确定模型并使用它来预测新数据。

具体来说，你学到了：

*   如何训练最终的LSTM模型。
*   如何保存最终的LSTM模型，然后再次加载它。
*   如何预测新数据。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。