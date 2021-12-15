# Keras中长短期记忆模型的5步生命周期

> 原文： [https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)

使用Keras在Python中创建和评估深度学习神经网络非常容易，但您必须遵循严格的模型生命周期。

在这篇文章中，您将发现在Keras中创建，训练和评估长期短期记忆（LSTM）循环神经网络的分步生命周期，以及如何使用训练有素的模型做出预测。

阅读这篇文章后，你会知道：

*   如何在Keras中定义，编译，拟合和评估LSTM。
*   如何为回归和分类序列预测问题选择标准默认值。
*   如何将它们联系起来，在Keras开发和运行您的第一个LSTM循环神经网络。

让我们开始吧。

*   **2017年6月更新**：修复输入大小调整示例中的拼写错误。

![The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](img/af98dc2c534e319ebc72e7ee514fcfa4.jpg)

Keras长期短期记忆模型的5步生命周期
照片由 [docmonstereyes](https://www.flickr.com/photos/docmonstereyes/2755918484/) 拍摄，保留一些权利。

## 概观

下面概述了我们将要研究的Keras LSTM模型生命周期中的5个步骤。

1.  定义网络
2.  编译网络
3.  适合网络
4.  评估网络
5.  作出预测

### 环境

本教程假定您已安装Python SciPy环境。您可以在此示例中使用Python 2或3。

本教程假设您安装了TensorFlow或Theano后端的Keras v2.0或更高版本。

本教程还假设您安装了scikit-learn，Pandas，NumPy和Matplotlib。

接下来，让我们看看标准时间序列预测问题，我们可以将其用作此实验的上下文。

如果您在设置Python环境时需要帮助，请参阅以下帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 步骤1.定义网络

第一步是定义您的网络。

神经网络在Keras中定义为层序列。这些层的容器是Sequential类。

第一步是创建Sequential类的实例。然后，您可以创建层并按照它们应连接的顺序添加它们。由存储器单元组成的LSTM复现层称为LSTM（）。通常跟随LSTM层并用于输出预测的完全连接层称为Dense（）。

例如，我们可以分两步完成：

```py
model = Sequential()
model.add(LSTM(2))
model.add(Dense(1))
```

但是我们也可以通过创建一个层数组并将其传递给Sequential的构造函数来一步完成。

```py
layers = [LSTM(2), Dense(1)]
model = Sequential(layers)
```

网络中的第一层必须定义预期的输入数量。输入必须是三维的，包括样本，时间步和特征。

*   **样品**。这些是数据中的行。
*   **时间步**。这些是过去对特征的观察，例如滞后变量。
*   **功能**。这些是数据中的列。

假设您的数据作为NumPy数组加载，您可以使用NumPy中的reshape（）函数将2D数据集转换为3D数据集。如果您希望列成为一个功能的时间步长，您可以使用：

```py
data = data.reshape((data.shape[0], data.shape[1], 1))
```

如果您希望2D数据中的列成为具有一个时间步长的要素，则可以使用：

```py
data = data.reshape((data.shape[0], 1, data.shape[1]))
```

您可以指定input_shape参数，该参数需要包含时间步数和要素数的元组。例如，如果我们有两个时间步长和一个特征用于单变量时间序列，每行有两个滞后观察值，则将指定如下：

```py
model = Sequential()
model.add(LSTM(5, input_shape=(2,1)))
model.add(Dense(1))
```

可以通过将LSTM层添加到Sequential模型来堆叠LSTM层。重要的是，在堆叠LSTM层时，我们必须为每个输入输出一个序列而不是一个值，以便后续的LSTM层可以具有所需的3D输入。我们可以通过将return_sequences参数设置为True来完成此操作。例如：

```py
model = Sequential()
model.add(LSTM(5, input_shape=(2,1), return_sequences=True))
model.add(LSTM(5))
model.add(Dense(1))
```

将Sequential模型视为一个管道，将原始数据输入到最后，预测从另一个输出。

这在Keras中是一个有用的容器，因为传统上与层相关的关注点也可以拆分并作为单独的层添加，清楚地显示它们在从输入到预测的数据转换中的作用。

例如，可以提取转换来自层中每个神经元的求和信号的激活函数，并将其作为称为激活的层状对象添加到Sequential中。

```py
model = Sequential()
model.add(LSTM(5, input_shape=(2,1)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```

激活函数的选择对于输出层是最重要的，因为它将定义预测将采用的格式。

例如，下面是一些常见的预测建模问题类型以及可以在输出层中使用的结构和标准激活函数：

*   **回归**：线性激活函数，或“线性”，以及与输出数量匹配的神经元数量。
*   **二元分类（2类）**：Logistic激活函数，或'sigmoid'，以及一个神经元输出层。
*   **多类分类（＆gt; 2类）**：假设单热编码输出模式，Softmax激活函数或'softmax'，以及每类值一个输出神经元。

## 第2步。编译网络

一旦我们定义了网络，我们就必须编译它。

编译是一个效率步骤。它将我们定义的简单层序列转换为高效的矩阵变换系列，其格式应在GPU或CPU上执行，具体取决于Keras的配置方式。

将编译视为网络的预计算步骤。定义模型后始终需要它。

编译需要指定许多参数，专门用于训练您的网络。具体地，用于训练网络的优化算法和用于评估由优化算法最小化的网络的损失函数。

例如，下面是编译定义模型并指定随机梯度下降（sgd）优化算法和均值误差（mean_squared_error）损失函数的情况，用于回归类型问题。

```py
model.compile(optimizer='sgd', loss='mean_squared_error')
```

或者，可以在作为编译步骤的参数提供之前创建和配置优化程序。

```py
algorithm = SGD(lr=0.1, momentum=0.3)
model.compile(optimizer=algorithm, loss='mean_squared_error')
```

预测建模问题的类型对可以使用的损失函数的类型施加约束。

例如，下面是不同预测模型类型的一些标准损失函数：

*   **回归**：均值平方误差或'mean_squared_error'。
*   **二元分类（2类）**：对数损失，也称为交叉熵或“binary_crossentropy”。
*   **多类分类（＆gt; 2类）**：多类对数损失或'categorical_crossentropy'。

最常见的优化算法是随机梯度下降，但Keras还支持一套其他最先进的优化算法，这些算法在很少或没有配置的情况下都能很好地工作。

也许最常用的优化算法因为它们通常具有更好的表现：

*   **随机梯度下降**或'sgd'，需要调整学习速度和动量。
*   **ADAM** 或'adam'，需要调整学习率。
*   **RMSprop** 或'rmsprop'，需要调整学习率。

最后，除了损失函数之外，您还可以指定在拟合模型时收集的度量标准。通常，要收集的最有用的附加度量标准是分类问题的准确率。要收集的度量标准由数组中的名称指定。

例如：

```py
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
```

## 步骤3.适合网络

一旦网络被编译，它就可以适合，这意味着在训练数据集上调整权重。

安装网络需要指定训练数据，包括输入模式矩阵X和匹配输出模式数组y。

使用反向传播算法训练网络，并根据编译模型时指定的优化算法和损失函数进行优化。

反向传播算法要求网络训练指定数量的时期或暴露于训练数据集。

每个迭代可以被划分为称为批次的输入 - 输出模式对的组。这定义了在一个迭代内更新权重之前网络所接触的模式数。它也是一种效率优化，确保一次不会将太多输入模式加载到内存中。

拟合网络的最小例子如下：

```py
history = model.fit(X, y, batch_size=10, epochs=100)
```

适合后，将返回历史对象，该对象提供训练期间模型表现的摘要。这包括损失和编译模型时指定的任何其他指标，记录每个迭代。

训练可能需要很长时间，从几秒到几小时到几天，具体取决于网络的大小和训练数据的大小。

默认情况下，每个迭代的命令行上都会显示一个进度条。这可能会给您带来太多噪音，或者可能会给您的环境带来问题，例如您使用的是交互式笔记本电脑或IDE。

通过将详细参数设置为2，可以减少每个时期显示的信息量。您可以通过将详细设置为1来关闭所有输出。例如：

```py
history = model.fit(X, y, batch_size=10, epochs=100, verbose=0)
```

## 第4步。评估网络

一旦网络被训练，就可以对其进行评估。

可以在训练数据上评估网络，但是这不会提供作为预测模型的网络表现的有用指示，因为它之前已经看到了所有这些数据。

我们可以在测试期间看不到的单独数据集上评估网络的表现。这将提供对网络表现的估计，以便对未来看不见的数据做出预测。

该模型评估所有测试模式的损失，以及编译模型时指定的任何其他指标，如分类准确率。返回评估指标列表。

例如，对于使用精度度量编制的模型，我们可以在新数据集上对其进行评估，如下所示：

```py
loss, accuracy = model.evaluate(X, y)
```

与拟合网络一样，提供详细输出以了解评估模型的进度。我们可以通过将verbose参数设置为0来关闭它。

```py
loss, accuracy = model.evaluate(X, y, verbose=0)
```

## 第5步。做出预测

一旦我们对拟合模型的表现感到满意，我们就可以使用它来预测新数据。

这就像使用新输入模式数组调用模型上的predict（）函数一样简单。

For example:

```py
predictions = model.predict(X)
```

预测将以网络输出层提供的格式返回。

在回归问题的情况下，这些预测可以是直接问题的格式，由线性激活函数提供。

对于二元分类问题，预测可以是第一类的概率数组，其可以通过舍入转换为1或0。

对于多类分类问题，结果可以是概率数组的形式（假设一个热编码输出变量），可能需要使用argmax（）NumPy函数将其转换为单个类输出预测。

或者，对于分类问题，我们可以使用predict_classes（）函数，该函数会自动将uncrisp预测转换为清晰的整数类值。

```py
predictions = model.predict_classes(X)
```

与拟合和评估网络一样，提供详细输出以给出模型预测的进度的概念。我们可以通过将verbose参数设置为0来关闭它。

```py
predictions = model.predict(X, verbose=0)
```

## 端到端工作示例

让我们将所有这些与一个小例子结合起来。

这个例子将使用一个学习10个数字序列的简单问题。我们将向网络显示一个数字，例如0.0，并期望它预测为0.1。然后显示它0.1并期望它预测0.2，依此类推到0.9。

1.  **定义网络**：我们将构建一个LSTM神经网络，在可见层有1个输入时间步长和1个输入特征，LSTM隐藏层有10个存储单元，在完全连接的输出层有1个神经元线性（默认）激活功能。
2.  **编译网络**：我们将使用具有默认配置和均方误差丢失函数的高效ADAM优化算法，因为它是一个回归问题。
3.  **适合网络**：我们将使网络适合1,000个时期，并使用等于训练集中模式数量的批量大小。我们还将关闭所有详细输出。
4.  **评估网络**。我们将在训练数据集上评估网络。通常，我们会在测试或验证集上评估模型。
5.  **制作预测**。我们将对训练输入数据做出预测。同样，通常我们会对我们不知道正确答案的数据做出预测。

完整的代码清单如下。

```py
# Example of LSTM to learn a sequence
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# 1\. define network
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1))
# 2\. compile network
model.compile(optimizer='adam', loss='mean_squared_error')
# 3\. fit network
history = model.fit(X, y, epochs=1000, batch_size=len(X), verbose=0)
# 4\. evaluate network
loss = model.evaluate(X, y, verbose=0)
print(loss)
# 5\. make predictions
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])
```

运行此示例将生成以下输出，显示10个数字的原始输入序列，对整个序列做出预测时网络的均方误差损失以及每个输入模式的预测。

输出间隔开以便于阅读。

我们可以看到序列被很好地学习，特别是如果我们将预测舍入到第一个小数位。

```py
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

4.54527471447e-05

[ 0.11612834 0.20493418 0.29793766 0.39445466 0.49376178 0.59512401
0.69782174 0.80117452 0.90455914]
```

## 进一步阅读

*   [顺序模型](https://keras.io/models/sequential/)的Keras文档。
*   [LSTM层](https://keras.io/layers/recurrent/#lstm)的Keras文档。
*   [Keras优化算法文档](https://keras.io/optimizers/)。
*   [Keras损失函数文档](https://keras.io/losses/)。

## 摘要

在这篇文章中，您使用Keras库发现了LSTM循环神经网络的5步生命周期。

具体来说，你学到了：

*   如何在Keras中为LSTM网络定义，编译，拟合，评估和预测。
*   如何为分类和回归问题选择激活函数和输出层配置。
*   如何在Keras开发和运行您的第一个LSTM模型。

您对Keras的LSTM型号有任何疑问，或者关于这篇文章？
在评论中提出您的问题，我会尽力回答。