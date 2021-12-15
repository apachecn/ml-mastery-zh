# 如何在 Keras 中重塑长短期存储网络的输入数据

> 原文： [https://machinelearningmastery.com/reshape-in​​put-data-long-short-term-memory-networks-keras/](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)

可能很难理解如何准备序列数据以输入 LSTM 模型。

通常，如何为 LSTM 模型定义输入层存在混淆。

关于如何将可能是 1D 或 2D 数字矩阵的序列数据转换为 LSTM 输入层所需的 3D 格式也存在混淆。

在本教程中，您将了解如何为 LSTM 模型定义输入层以及如何为 LSTM 模型重新加载已加载的输入数据。

完成本教程后，您将了解：

*   如何定义 LSTM 输入层。
*   如何重塑 LSTM 模型的一维序列数据并定义输入层。
*   如何重塑 LSTM 模型的多个并行系列数据并定义输入层。

让我们开始吧。

![How to Reshape Input for Long Short-Term Memory Networks in Keras](img/cf2834ae15e6a15b3b12cdbc14b08330.jpg)

如何重塑 Keras 长期短期记忆网络的输入
图片来自 [Global Landscapes Forum](https://www.flickr.com/photos/152793655@N07/33495968401/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  LSTM 输入层
2.  具有单输入样本的 LSTM 示例
3.  具有多输入功能的 LSTM 示例
4.  LSTM 输入提示

### LSTM 输入层

LSTM 输入层由网络的第一个隐藏层上的“`input_shape`”参数指定。

这会让初学者感到困惑。

例如，下面是具有一个隐藏的 LSTM 层和一个密集输出层的网络的示例。

```py
model = Sequential()
model.add(LSTM(32))
model.add(Dense(1))
```

在此示例中，LSTM（）层必须指定输入的形状。

每个 LSTM 层的输入必须是三维的。

此输入的三个维度是：

*   **样品**。一个序列是一个样本。批次由一个或多个样品组成。
*   **时间步**。一个步骤是样品中的一个观察点。
*   **功能**。一个特征是在一个时间步骤的一个观察。

这意味着输入层在拟合模型和做出预测时需要 3D 数据数组，即使数组的特定尺寸包含单个值，例如，一个样本或一个特征。

定义 LSTM 网络的输入层时，网络假定您有一个或多个样本，并要求您指定时间步数和要素数。您可以通过指定“`input_shape`”参数的元组来完成此操作。

例如，下面的模型定义了一个输入层，该输入层需要 1 个或更多样本，50 个时间步长和 2 个特征。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(50, 2)))
model.add(Dense(1))
```

现在我们知道了如何定义 LSTM 输入层以及 3D 输入的期望，让我们看看如何为 LSTM 准备数据的一些示例。

## 单输入样本的 LSTM 示例

考虑具有多个时间步长和一个特征的一个序列的情况。

例如，这可能是 10 个值的序列：

```py
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

我们可以将这个数字序列定义为 NumPy 数组。

```py
from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
```

然后我们可以使用 NumPy 数组上的 _reshape（）_ 函数将这个一维数组重新整形为一个三维数组，每个时间步长有 1 个样本，10 个时间步长和 1 个特征。

在数组上调用时， _reshape（）_ 函数接受一个参数，该参数是定义数组新形状的元组。我们不能传递任何数字元组;重塑必须均匀地重新组织数组中的数据。

```py
data = data.reshape((1, 10, 1))
```

重新成形后，我们可以打印数组的新形状。

```py
print(data.shape)
```

将所有这些放在一起，下面列出了完整的示例。

```py
from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data = data.reshape((1, 10, 1))
print(data.shape)
```

运行该示例将打印单个样本的新 3D 形状。

```py
(1, 10, 1)
```

此数据现在可以用作输入（`X`）到 LSTM，其 input_shape 为（10,1）。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dense(1))
```

## 具有多输入功能的 LSTM 示例

考虑您有多个并行系列作为模型输入的情况。

例如，这可能是两个并行的 10 个值系列：

```py
series 1: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
series 2: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
```

我们可以将这些数据定义为包含 10 行的 2 列矩阵：

```py
from numpy import array
data = array([
	[0.1, 1.0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])
```

该数据可以被构造为 1 个样本，具有 10 个时间步长和 2 个特征。

它可以重新整形为 3D 数组，如下所示：

```py
data = data.reshape(1, 10, 2)
```

Putting all of this together, the complete example is listed below.

```py
from numpy import array
data = array([
	[0.1, 1.0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])
data = data.reshape(1, 10, 2)
print(data.shape)
```

Running the example prints the new 3D shape of the single sample.

```py
(1, 10, 2)
```

此数据现在可以用作输入（`X`）到 LSTM，其 input_shape 为（10,2）。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(10, 2)))
model.add(Dense(1))
```

## 更长的工作示例

有关准备数据的完整端到端工作示例，请参阅此帖子：

*   [如何为长期短期记忆网络准备单变量时间序列数据](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

## LSTM 输入提示

本节列出了在准备 LSTM 输入数据时可以帮助您的一些提示。

*   LSTM 输入层必须为 3D。
*   3 个输入维度的含义是：样本，时间步长和功能。
*   LSTM 输入层由第一个隐藏层上的`input_shape`参数定义。
*  `input_shape`参数采用两个值的元组来定义时间步长和特征的数量。
*   假设样本数为 1 或更多。
*   NumPy 数组上的 _reshape（）_ 功能可用于将您的 1D 或 2D 数据重塑为 3D。
*   _reshape（）_ 函数将元组作为定义新形状的参数。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [Recurrent Layers Keras API](https://keras.io/layers/recurrent/)
*   [Numpy reshape（）函数 API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.reshape.html)
*   [如何将时间序列转换为 Python 中的监督学习问题](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [时间序列预测作为监督学习](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

## 摘要

在本教程中，您了解了如何为 LSTM 定义输入层以及如何重新整形序列数据以输入 LSTM。

具体来说，你学到了：

*   如何定义 LSTM 输入层。
*   如何重塑 LSTM 模型的一维序列数据并定义输入层。
*   如何重塑 LSTM 模型的多个并行系列数据并定义输入层。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。