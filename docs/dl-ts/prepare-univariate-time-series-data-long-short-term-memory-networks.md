# 如何为长短期记忆网络准备单变量时间序列数据

> 原文： [https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

当您刚刚开始深度学习时，很难准备数据。

长期短期记忆，或 LSTM，循环神经网络期望在 Keras Python 深度学习库中进行三维输入。

如果您的时间序列数据中包含数千个长序列，则必须将时间序列拆分为样本，然后将其重新整形为 LSTM 模型。

在本教程中，您将了解如何使用 Keras 在 Python 中为 LSTM 模型准备单变量时间序列数据。

让我们开始吧。

![How to Prepare Univariate Time Series Data for Long Short-Term Memory Networks](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/11/How-to-Prepare-Univariate-Time-Series-Data-for-Long-Short-Term-Memory-Networks.jpg)

如何为长期短期记忆网络准备单变量时间序列数据
照片来自 [Miguel Mendez](https://www.flickr.com/photos/flynn_nrg/8487128120/) ，保留一些权利。

## 如何准备时间序列数据

也许我得到的最常见问题是如何为监督学习准备时间序列数据。

我写过一些关于这个主题的帖子，例如：

*   [如何将时间序列转换为 Python 中的监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [时间序列预测作为监督学习](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

但是，这些帖子对每个人都没有帮助。

我最近收到了这封邮件：

> 我的数据文件中有两列有 5000 行，第 1 列是时间（间隔 1 小时），第 2 列是比特/秒，我试图预测比特/秒。在这种情况下，您可以帮我设置[LSTM]的样本，时间步长和功能吗？

这里几乎没有问题：

*   LSTM 期望 3D 输入，并且第一次了解这个问题可能具有挑战性。
*   LSTM 不喜欢超过 200-400 个时间步长的序列，因此需要将数据拆分为样本。

在本教程中，我们将使用此问题作为显示为 Keras 中的 LSTM 网络专门准备数据的一种方法的基础。

## 1.加载数据

我假设你知道如何将数据加载为 Pandas Series 或 DataFrame。

如果没有，请参阅以下帖子：

*   [如何在 Python 中加载和探索时间序列数据](https://machinelearningmastery.com/load-explore-time-series-data-python/)
*   [如何在 Python 中加载机器学习数据](https://machinelearningmastery.com/load-machine-learning-data-python/)

在这里，我们将通过在 5,000 个时间步长内存中定义新数据集来模拟加载。

```py
from numpy import array

# load...
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = array(data)
print(data[:5, :])
print(data.shape)
```

运行此片段既会打印前 5 行数据，也会打印已加载数据的形状。

我们可以看到我们有 5,000 行和 2 列：标准的单变量时间序列数据集。

```py
[[ 1 10]
 [ 2 20]
 [ 3 30]
 [ 4 40]
 [ 5 50]]
(5000, 2)
```

## 2.停机时间

如果您的时间序列数据随着时间的推移是一致的并且没有缺失值，我们可以删除时间列。

如果没有，您可能希望查看插入缺失值，将数据重新采样到新的时间刻度，或者开发可以处理缺失值的模型。看帖子如：

*   [如何使用 Python 处理序列预测问题中的缺失时间步长](https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)
*   [如何使用 Python 处理缺失数据](https://machinelearningmastery.com/handle-missing-data-python/)
*   [如何使用 Python 重新取样和插值您的时间序列数据](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)

在这里，我们只删除第一列：

```py
# drop time
data = data[:, 1]
print(data.shape)
```

现在我们有一个 5,000 个值的数组。

```py
(5000,)
```

## 3.拆分成样品

LSTM 需要处理样品，其中每个样品是单个时间序列。

在这种情况下，5,000 个时间步长太长;根据我读过的一些论文，LSTM 可以更好地完成 200 到 400 个步骤。因此，我们需要将 5,000 个时间步骤分成多个较短的子序列。

我在这里写了更多关于拆分长序列的文章：

*   [如何处理具有长短期记忆循环神经网络的超长序列](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
*   [如何准备 Keras 中截断反向传播的序列预测](https://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)

有很多方法可以做到这一点，你可能想根据你的问题探索一些。

例如，您可能需要重叠序列，也许非重叠是好的，但您的模型需要跨子序列的状态等等。

在这里，我们将 5,000 个时间步骤分成 25 个子序列，每个子序列有 200 个时间步长。我们将采用老式方式，而不是使用 NumPy 或 Python 技巧，以便您可以看到正在发生的事情。

```py
# split into samples (e.g. 5000/200 = 25)
samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
	# grab from i to i + 200
	sample = data[i:i+length]
	samples.append(sample)
print(len(samples))
```

我们现在有 25 个子序列，每个子序列有 200 个时间步长。

```py
25
```

如果您更喜欢在一个班轮中这样做，那就去吧。我很想知道你能想出什么。
在下面的评论中发布您的方法。

## 4.重塑子序列

LSTM 需要具有[样本，时间步长和特征]格式的数据。

在这里，我们有 25 个样本，每个样本 200 个时间步长和 1 个特征。

首先，我们需要将我们的数组列表转换为 25 x 200 的 2D NumPy 数组。

```py
# convert list of arrays into 2d array
data = array(samples)
print(data.shape)
```

运行这件作品，你应该看到：

```py
(25, 200)
```

接下来，我们可以使用`reshape()`函数为我们的单个特征添加一个额外的维度。

```py
# reshape into [samples, timesteps, features]
# expect [25, 200, 1]
data = data.reshape((len(samples), length, 1))
print(data.shape)
```

就是这样。

现在，数据可用作 LSTM 模型的输入（X）。

```py
(25, 200, 1)
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 相关文章

*   [如何将时间序列转换为 Python 中的监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [时间序列预测作为监督学习](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
*   [如何在 Python 中加载和探索时间序列数据](https://machinelearningmastery.com/load-explore-time-series-data-python/)
*   [如何在 Python 中加载机器学习数据](https://machinelearningmastery.com/load-machine-learning-data-python/)
*   [如何使用 Python 处理序列预测问题中的缺失时间步长](https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)
*   [如何使用 Python 处理缺失数据](https://machinelearningmastery.com/handle-missing-data-python/)
*   [如何使用 Python 重新取样和插值您的时间序列数据](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)
*   [如何处理具有长短期记忆循环神经网络的超长序列](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
*   [如何准备 Keras 中截断反向传播的序列预测](https://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)

### API

*   [Keras 的 LSTM API](https://keras.io/layers/recurrent/#lstm)
*   [numpy.reshape API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)

## 摘要

在本教程中，您了解了如何将长的单变量时间序列数据转换为可用于在 Python 中训练 LSTM 模型的表单。

这篇文章有帮助吗？你有任何问题吗？
请在下面的评论中告诉我。