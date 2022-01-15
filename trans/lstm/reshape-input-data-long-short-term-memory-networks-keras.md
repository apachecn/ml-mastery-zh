# 如何重塑喀拉斯长短期记忆网络的输入数据

> 原文:[https://machinelearning master . com/resform-input-data-long-short-memory-networks-keras/](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)

最后更新于 2019 年 8 月 14 日

很难理解如何准备序列数据输入 LSTM 模型。

通常，对于如何定义 LSTM 模型的输入层存在困惑。

对于如何将可能是 1D 或 2D 数字矩阵的序列数据转换为 LSTM 输入图层所需的 3D 格式，也存在困惑。

在本教程中，您将了解如何定义 LSTM 模型的输入图层，以及如何为 LSTM 模型重塑加载的输入数据。

完成本教程后，您将知道:

*   如何定义 LSTM 输入层。
*   如何为 LSTM 模型重塑一维序列数据并定义输入图层。
*   如何为 LSTM 模型重塑多个并行系列数据并定义输入图层。

**用我的新书[Python 的长短期记忆网络](https://machinelearningmastery.com/lstms-with-python/)启动你的项目**，包括*循序渐进教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Reshape Input for Long Short-Term Memory Networks in Keras](img/8fc6bb4ace397f56877135f7cedfc102.png)

如何在喀拉斯重塑长短期记忆网络的输入
图片由[全球景观论坛](https://www.flickr.com/photos/152793655@N07/33495968401/)提供，保留部分权利。

## 教程概述

本教程分为 4 个部分；它们是:

1.  LSTM 输入层
2.  具有单输入样本的 LSTM 示例
3.  具有多种输入功能的 LSTM 示例
4.  LSTM 输入提示

### LSTM 输入层

LSTM 输入层由网络第一个隐藏层上的“ *input_shape* ”参数指定。

这可能会让初学者感到困惑。

例如，下面是具有一个隐藏 LSTM 层和一个密集输出层的网络示例。

```py
model = Sequential()
model.add(LSTM(32))
model.add(Dense(1))
```

在本例中，LSTM()图层必须指定输入的形状。

每个 LSTM 层的输入必须是三维的。

这种输入的三个维度是:

*   **样品**。一个序列就是一个样本。一批由一个或多个样品组成。
*   **时间步长**。一个时间步长是样本中的一个观察点。
*   **功能**。一个特点是一步一个观察。

这意味着，在拟合模型和进行预测时，输入图层需要一个三维数据阵列，即使阵列的特定维度包含单个值，例如一个样本或一个特征。

定义 LSTM 网络的输入图层时，网络假设您有 1 个或更多样本，并要求您指定时间步长数和要素数。您可以通过为“ *input_shape* ”参数指定一个元组来实现这一点。

例如，下面的模型定义了一个需要 1 个或更多样本、50 个时间步长和 2 个特征的输入层。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(50, 2)))
model.add(Dense(1))
```

既然我们已经知道了如何定义 LSTM 输入图层和对 3D 输入的期望，那么让我们来看一些如何为 LSTM 准备数据的示例。

## 单输入样本的 LSTM 示例

考虑这样的情况:您有一个由多个时间步骤组成的序列和一个特性。

例如，这可能是 10 个值的序列:

```py
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

我们可以将这个数字序列定义为一个 NumPy 数组。

```py
from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
```

然后，我们可以在 NumPy 数组上使用*重塑()*函数，将这个一维数组重塑为一个三维数组，每个时间步长有 1 个样本、10 个时间步长和 1 个特征。

当在数组上调用时，*重塑()*函数接受一个参数，该参数是定义数组新形状的元组。我们不能传入任何一组数字；整形必须均匀地重新组织数组中的数据。

```py
data = data.reshape((1, 10, 1))
```

一旦重塑，我们就可以打印新的阵列形状。

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

运行该示例将打印单个样本的新三维形状。

```py
(1, 10, 1)
```

该数据现在可以用作输入( *X* )到输入形状为(10，1)的 LSTM。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dense(1))
```

## 具有多种输入功能的 LSTM 示例

考虑有多个并行系列作为模型输入的情况。

例如，这可能是由 10 个值组成的两个并行系列:

```py
series 1: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
series 2: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
```

我们可以将这些数据定义为 2 列 10 行的矩阵:

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

该数据可以被构建为具有 10 个时间步长和 2 个特征的 1 个样本。

它可以按如下方式重新造型为三维阵列:

```py
data = data.reshape(1, 10, 2)
```

将所有这些放在一起，下面列出了完整的示例。

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

运行该示例将打印单个样本的新三维形状。

```py
(1, 10, 2)
```

该数据现在可以用作输入( *X* )到输入形状为(10，2)的 LSTM。

```py
model = Sequential()
model.add(LSTM(32, input_shape=(10, 2)))
model.add(Dense(1))
```

## 工作时间更长的示例

有关准备数据的完整端到端工作示例，请参见本文:

*   [如何为长短期记忆网络准备单变量时间序列数据](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

## LSTM 输入提示

本节列出了一些提示，可以帮助您准备 LSTMs 的输入数据。

*   LSTM 输入图层必须是 3D 的。
*   3 个输入维度的含义是:样本、时间步长和特征。
*   LSTM 输入图层由第一个隐藏图层上的*输入形状*参数定义。
*   *input_shape* 参数采用两个值的元组来定义时间步长和特征的数量。
*   假设样本数为 1 或更多。
*   NumPy 阵列上的*重塑()*功能可用于将 1D 或 2D 数据重塑为 3D。
*   *重塑()*函数将元组作为定义新形状的参数。

## 进一步阅读

如果您想了解更多信息，本节将提供更多相关资源。

*   [复发性蛋鸡角叉菜胶](https://keras.io/layers/recurrent/)
*   [Numpy 重塑()函数 API](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.reshape.html)
*   [如何在 Python 中将时间序列转换为监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [作为监督学习的时间序列预测](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

## 摘要

在本教程中，您发现了如何为 LSTMs 定义输入图层，以及如何重塑序列数据以输入到 LSTMs。

具体来说，您了解到:

*   如何定义 LSTM 输入层。
*   如何为 LSTM 模型重塑一维序列数据并定义输入图层。
*   如何为 LSTM 模型重塑多个并行系列数据并定义输入图层。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。