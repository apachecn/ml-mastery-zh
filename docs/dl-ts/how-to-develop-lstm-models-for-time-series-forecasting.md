# 如何开发 LSTM 模型进行时间序列预测

> 原文： [https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

长短期内存网络（简称 LSTM）可应用于时间序列预测。

有许多类型的 LSTM 模型可用于每种特定类型的时间序列预测问题。

在本教程中，您将了解如何针对一系列标准时间序列预测问题开发一套 LSTM 模型。

本教程的目的是为每种类型的时间序列问题提供每个模型的独立示例，作为模板，您可以根据特定的时间序列预测问题进行复制和调整。

完成本教程后，您将了解：

*   如何开发 LSTM 模型进行单变量时间序列预测。
*   如何开发多变量时间序列预测的 LSTM 模型。
*   如何开发 LSTM 模型进行多步时间序列预测。

这是一个庞大而重要的职位;您可能希望将其加入书签以供将来参考。

让我们开始吧。

![How to Develop LSTM Models for Time Series Forecasting](img/d886deae66a94f53af0a8769f359935c.jpg)

如何开发用于时间序列预测的 LSTM 模型
照片由 [N i c o l a](https://www.flickr.com/photos/15216811@N06/6704346543/) ，保留一些权利。

## 教程概述

在本教程中，我们将探索如何为时间序列预测开发一套不同类型的 LSTM 模型。

这些模型在小型设计的时间序列问题上进行了演示，旨在解决时间序列问题类型的风格。所选择的模型配置是任意的，并未针对每个问题进行优化;那不是目标。

本教程分为四个部分;他们是：

1.  单变量 LSTM 模型
2.  多变量 LSTM 模型
3.  多步 LSTM 模型
4.  多变量多步 LSTM 模型

## 单变量 LSTM 模型

LSTM 可用于模拟单变量时间序列预测问题。

这些问题包括一系列观察，并且需要模型来从过去的一系列观察中学习以预测序列中的下一个值。

我们将演示 LSTM 模型的多种变体，用于单变量时间序列预测。

本节分为六个部分;他们是：

1.  数据准备
2.  香草 LSTM
3.  堆叠式 LSTM
4.  双向 LSTM
5.  CNN LSTM
6.  ConvLSTM

这些模型中的每一个都被演示为一步式单变量时间序列预测，但可以很容易地进行调整并用作其他类型的时间序列预测问题的模型的输入部分。

### 数据准备

在对单变量系列进行建模之前，必须准备好它。

LSTM 模型将学习一种函数，该函数将过去观察序列作为输入映射到输出观察。因此，必须将观察序列转换为 LSTM 可以学习的多个示例。

考虑给定的单变量序列：

```py
[10, 20, 30, 40, 50, 60, 70, 80, 90]
```

我们可以将序列划分为多个称为样本的输入/输出模式，其中三个时间步长用作输入，一个时间步长用作正在学习的一步预测的输出。

```py
X,				y
10, 20, 30		40
20, 30, 40		50
30, 40, 50		60
...
```

下面的`split_sequence()`函数实现了这种行为，并将给定的单变量序列分成多个样本，其中每个样本具有指定的时间步长，输出是单个时间步长。

```py
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以在上面的小型人为数据集上演示这个功能。

下面列出了完整的示例。

```py
# univariate data preparation
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

运行该示例将单变量系列分成六个样本，其中每个样本具有三个输入时间步长和一个输出时间步长。

```py
[10 20 30] 40
[20 30 40] 50
[30 40 50] 60
[40 50 60] 70
[50 60 70] 80
[60 70 80] 90
```

现在我们已经知道如何准备用于建模的单变量系列，让我们看看开发 LSTM 模型，它可以学习输入到输出的映射，从 Vanilla LSTM 开始。

### 香草 LSTM

Vanilla LSTM 是 LSTM 模型，具有单个隐藏的 LSTM 单元层，以及用于做出预测的输出层。

我们可以如下定义用于单变量时间序列预测的 Vanilla LSTM。

```py
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

定义的关键是输入的形状;这就是模型期望的每个样本的输入，包括时间步数和特征数。

我们正在使用单变量系列，因此对于一个变量，要素的数量是一个。

输入的时间步数是我们在准备数据集时选择的数字，作为`split_sequence()`函数的参数。

每个样本的输入形状在第一个隐藏层定义的`input_shape`参数中指定。

我们几乎总是有多个样本，因此，模型将期望训练数据的输入组件具有尺寸或形状：

```py
[samples, timesteps, features]
```

我们在上一节中的`split_sequence()`函数输出具有[_ 样本，时间步长 _]形状​​的 X，因此我们可以轻松地对其进行整形，以便为一个特征提供额外的维度。

```py
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
```

在这种情况下，我们定义隐藏层中具有 50 个 LSTM 单元的模型和预测单个数值的输出层。

使用随机梯度下降的有效 [Adam 版本拟合该模型，并使用均方误差或'`mse`'损失函数进行优化。](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

定义模型后，我们可以将其放在训练数据集上。

```py
# fit model
model.fit(X, y, epochs=200, verbose=0)
```

在模型拟合后，我们可以使用它来做出预测。

我们可以通过提供输入来预测序列中的下一个值：

```py
[70, 80, 90]
```

并期望模型预测如下：

```py
[100]
```

该模型期望输入形状为[_ 样本，时间步长，特征 _]三维，因此，我们必须在做出预测之前对单个输入样本进行整形。

```py
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
```

我们可以将所有这些结合在一起并演示如何开发用于单变量时间序列预测的 Vanilla LSTM 并进行单一预测。

```py
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例准备数据，拟合模型并做出预测。

鉴于算法的随机性，您的结果可能会有所不同;尝试运行几次这个例子。

我们可以看到模型预测序列中的下一个值。

```py
[[102.09213]]
```

### 堆叠式 LSTM

多个隐藏的 LSTM 层可以在所谓的堆叠 LSTM 模型中一个堆叠在另一个之上。

LSTM 层需要三维输入，默认情况下，LSTM 将产生二维输出作为序列末尾的解释。

我们可以通过在层上设置 _return_sequences = True_ 参数，为输入数据中的每个时间步长输出 LSTM 来解决这个问题。这允许我们将隐藏的 LSTM 层的 3D 输出作为下一个输入。

因此，我们可以如下定义 Stacked LSTM。

```py
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

我们可以将它们联系起来;完整的代码示例如下所示。

```py
# univariate stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例预测序列中的下一个值，我们预期该值为 100。

```py
[[102.47341]]
```

### 双向 LSTM

在一些序列预测问题上，允许 LSTM 模型向前和向后学习输入序列并连接两种解释可能是有益的。

这称为[双向 LSTM](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/) 。

我们可以通过将第一个隐藏层包装在名为 Bidirectional 的包装层中来实现双向 LSTM 以进行单变量时间序列预测。

定义双向 LSTM 以向前和向后读取输入的示例如下。

```py
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

下面列出了用于单变量时间序列预测的双向 LSTM 的完整示例。

```py
# univariate bidirectional lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例预测序列中的下一个值，我们预期该值为 100。

```py
[[101.48093]]
```

### CNN LSTM

卷积神经网络（简称 CNN）是一种为处理二维图像数据而开发的神经网络。

CNN 可以非常有效地从一维序列数据（例如单变量时间序列数据）中自动提取和学习特征。

CNN 模型可以在具有 LSTM 后端的混合模型中使用，其中 CNN 用于解释输入的子序列，这些子序列一起作为序列提供给 LSTM 模型以进行解释。 [这种混合模型称为 CNN-LSTM](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/) 。

第一步是将输入序列分成可由 CNN 模型处理的子序列。例如，我们可以首先将单变量时间序列数据拆分为输入/输出样本，其中四个步骤作为输入，一个作为输出。然后可以将每个样品分成两个子样品，每个子样品具有两个时间步骤。 CNN 可以解释两个时间步的每个子序列，并提供对 LSTM 模型的子序列的时间序列解释以作为输入进行处理。

我们可以对此进行参数化，并将子序列的数量定义为`n_seq`，将每个子序列的时间步数定义为`n_steps`。然后可以将输入数据重新整形为具有所需的结构：

```py
[samples, subsequences, timesteps, features]
```

例如：

```py
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
```

我们希望在分别读取每个数据子序列时重用相同的 CNN 模型。

这可以通过将整个 CNN 模型包装在 [TimeDistributed 包装器](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)中来实现，该包装器将每个输入应用整个模型一次，在这种情况下，每个输入子序列一次。

CNN 模型首先具有卷积层，用于读取子序列，该子序列需要指定多个过滤器和内核大小。过滤器的数量是输入序列的读取或解释的数量。内核大小是输入序列的每个“读取”操作所包含的时间步数。

卷积层后面跟着一个最大池池，它将过滤器图谱提取到其大小的 1/4，包括最显着的特征。然后将这些结构展平为单个一维向量，以用作 LSTM 层的单个输入时间步长。

```py
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
```

接下来，我们可以定义模型的 LSTM 部分，该部分解释 CNN 模型对输入序列的读取并做出预测。

```py
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
```

我们可以将所有这些结合在一起;下面列出了用于单变量时间序列预测的 CNN-LSTM 模型的完整示例。

```py
# univariate cnn lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例预测序列中的下一个值，我们预期该值为 100。

```py
[[101.69263]]
```

### ConvLSTM

与 CNN-LSTM 相关的一种 LSTM 是 ConvLSTM，其中输入的卷积读取直接建立在每个 LSTM 单元中。

ConvLSTM 是为读取二维时空数据而开发的，但可以用于单变量时间序列预测。

该层期望输入为二维图像序列，因此输入数据的形状必须为：

```py
[samples, timesteps, rows, columns, features]
```

为了我们的目的，我们可以将每个样本分成时序将成为子序列数的子序列，或`n_seq`，并且列将是每个子序列的时间步数，或`n_steps`。当我们使用一维数据时，行数固定为 1。

我们现在可以将准备好的样品重新塑造成所需的结构。

```py
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
```

我们可以根据过滤器的数量将 ConvLSTM 定义为单个层，并根据（行，列）将二维内核大小定义为单层。当我们使用一维系列时，内核中的行数始终固定为 1。

然后必须将模型的输出展平，然后才能进行解释并做出预测。

```py
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
```

下面列出了用于一步式单变量时间序列预测的 ConvLSTM 的完整示例。

```py
# univariate convlstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例预测序列中的下一个值，我们预期该值为 100。

```py
[[103.68166]]
```

现在我们已经查看了单变量数据的 LSTM 模型，让我们将注意力转向多变量数据。

## 多变量 LSTM 模型

多变量时间序列数据是指每个时间步长有多个观察值的数据。

对于多变量时间序列数据，我们可能需要两种主要模型;他们是：

1.  多输入系列。
2.  多个并联系列。

让我们依次看看每一个。

### 多输入系列

问题可能有两个或更多并行输入时间序列和输出时间序列，这取决于输入时间序列。

输入时间序列是平行的，因为每个系列在同一时间步骤具有观察。

我们可以通过两个并行输入时间序列的简单示例来演示这一点，其中输出序列是输入序列的简单添加。

```py
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

我们可以将这三个数据数组重新整形为单个数据集，其中每一行都是一个时间步，每列都是一个单独的时间序列。这是将并行时间序列存储在 CSV 文件中的标准方法。

```py
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
```

下面列出了完整的示例。

```py
# multivariate data preparation
from numpy import array
from numpy import hstack
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
```

运行该示例将打印数据集，每个时间步长为一行，两个输入和一个输出并行时间序列分别为一列。

```py
[[ 10  15  25]
 [ 20  25  45]
 [ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]
 [ 80  85 165]
 [ 90  95 185]]
```

与单变量时间序列一样，我们必须将这些数据组织成具有输入和输出元素的样本。

LSTM 模型需要足够的上下文来学习从输入序列到输出值的映射。 LSTM 可以支持并行输入时间序列作为单独的变量或特征。因此，我们需要将数据分成样本，保持两个输入序列的观察顺序。

如果我们选择三个输入时间步长，那么第一个样本将如下所示：

输入：

```py
10, 15
20, 25
30, 35
```

输出：

```py
65
```

也就是说，每个并行系列的前三个时间步长被提供作为模型的输入，并且模型将其与第三时间步骤（在这种情况下为 65）的输出系列中的值相关联。

我们可以看到，在将时间序列转换为输入/输出样本以训练模型时，我们将不得不从输出时间序列中丢弃一些值，其中我们在先前时间步骤中没有输入时间序列中的值。反过来，选择输入时间步数的大小将对使用多少训练数据产生重要影响。

我们可以定义一个名为`split_sequences()`的函数，该函数将采用数据集，因为我们已经为时间步长和行定义了并行序列和返回输入/输出样本的列。

```py
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以使用每个输入时间序列的三个时间步长作为输入在我们的数据集上测试此函数。

下面列出了完整的示例。

```py
# multivariate data preparation
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

首先运行该示例将打印 X 和 y 组件的形状。

我们可以看到 X 组件具有三维结构。

第一个维度是样本数，在本例中为 7.第二个维度是每个样本的时间步数，在这种情况下为 3，即为函数指定的值。最后，最后一个维度指定并行时间序列的数量或变量的数量，在这种情况下，两个并行序列为 2。

这是 LSTM 作为输入所期望的精确三维结构。数据即可使用而无需进一步重塑。

然后我们可以看到每个样本的输入和输出都被打印出来，显示了两个输入序列中每个样本的三个时间步长以及每个样本的相关输出。

```py
(7, 3, 2) (7,)

[[10 15]
 [20 25]
 [30 35]] 65
[[20 25]
 [30 35]
 [40 45]] 85
[[30 35]
 [40 45]
 [50 55]] 105
[[40 45]
 [50 55]
 [60 65]] 125
[[50 55]
 [60 65]
 [70 75]] 145
[[60 65]
 [70 75]
 [80 85]] 165
[[70 75]
 [80 85]
 [90 95]] 185
```

我们现在准备在这些数据上安装 LSTM 模型。

可以使用前一节中的任何种类的 LSTM，例如香草，堆叠，双向，CNN 或 ConvLSTM 模型。

我们将使用 Vanilla LSTM，其中通过`input_shape`参数为输入层指定时间步数和并行系列（特征）。

```py
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

在做出预测时，模型需要两个输入时间序列的三个时间步长。

我们可以预测输出系列中的下一个值，提供以下输入值：

```py
80,	 85
90,	 95
100, 105
```

具有三个时间步长和两个变量的一个样本的形状必须是[1,3,2]。

我们希望序列中的下一个值为 100 + 105 或 205。

```py
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
```

下面列出了完整的示例。

```py
# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例准备数据，拟合模型并做出预测。

```py
[[208.13531]]
```

## 多个并联系列

另一个时间序列问题是存在多个并行时间序列并且必须为每个时间序列预测值的情况。

例如，给定上一节的数据：

```py
[[ 10  15  25]
 [ 20  25  45]
 [ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]
 [ 80  85 165]
 [ 90  95 185]]
```

我们可能想要预测下一个时间步的三个时间序列中的每一个的值。

这可以称为多变量预测。

同样，必须将数据分成输入/输出样本以训练模型。

该数据集的第一个示例是：

输入：

```py
10, 15, 25
20, 25, 45
30, 35, 65
```

输出：

```py
40, 45, 85
```

下面的`split_sequences()`函数将分割多个并行时间序列，其中时间步长为行，每列一个系列为所需的输入/输出形状。

```py
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以在人为的问题上证明这一点;下面列出了完整的示例。

```py
# multivariate output data prep
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

首先运行该示例打印准备好的 X 和 y 组件的形状。

X 的形状是三维的，包括样品的数量（6），每个样品选择的时间步数（3），以及平行时间序列或特征的数量（3）。

y 的形状是二维的，正如我们可能期望的样本数量（6）和每个样本的时间变量数量（3）。

数据已准备好在 LSTM 模型中使用，该模型需要三维输入和每个样本的 X 和 y 分量的二维输出形状。

然后，打印每个样本，显示每个样本的输入和输出分量。

```py
(6, 3, 3) (6, 3)

[[10 15 25]
 [20 25 45]
 [30 35 65]] [40 45 85]
[[20 25 45]
 [30 35 65]
 [40 45 85]] [ 50  55 105]
[[ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]] [ 60  65 125]
[[ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]] [ 70  75 145]
[[ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]] [ 80  85 165]
[[ 60  65 125]
 [ 70  75 145]
 [ 80  85 165]] [ 90  95 185]
```

我们现在准备在这些数据上安装 LSTM 模型。

可以使用前一节中的任何种类的 LSTM，例如香草，堆叠，双向，CNN 或 ConvLSTM 模型。

我们将使用 Stacked LSTM，其中通过`input_shape`参数为输入层指定时间步数和并行系列（特征）。并行序列的数量也用于指定输出层中模型预测的值的数量;再次，这是三个。

```py
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
```

我们可以通过为每个系列提供三个时间步长的输入来预测三个并行系列中的每一个的下一个值。

```py
70, 75, 145
80, 85, 165
90, 95, 185
```

用于进行单个预测的输入的形状必须是 1 个样本，3 个时间步长和 3 个特征，或者[1,3,3]

```py
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
```

我们希望向量输出为：

```py
[100, 105, 205]
```

我们可以将所有这些结合在一起并演示下面的多变量输出时间序列预测的 Stacked LSTM。

```py
# multivariate output stacked lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=400, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例准备数据，拟合模型并做出预测。

```py
[[101.76599 108.730484 206.63577 ]]
```

## 多步 LSTM 模型

需要预测未来多个时间步长的时间序列预测问题可以称为多步时间序列预测。

具体而言，这些是预测范围或间隔超过一个时间步长的问题。

有两种主要类型的 LSTM 模型可用于多步预测;他们是：

1.  向量输出模型
2.  编解码器模型

在我们查看这些模型之前，让我们首先看一下多步骤预测的数据准备。

### 数据准备

与一步预测一样，用于多步时间序列预测的时间序列必须分为带有输入和输出组件的样本。

输入和输出组件都将包含多个时间步长，并且可以具有或不具有相同数量的步骤。

例如，给定单变量时间序列：

```py
[10, 20, 30, 40, 50, 60, 70, 80, 90]
```

我们可以使用最后三个时间步作为输入并预测接下来的两个时间步。

第一个样本如下：

输入：

```py
[10, 20, 30]
```

输出：

```py
[40, 50]
```

下面的`split_sequence()`函数实现了这种行为，并将给定的单变量时间序列分割为具有指定数量的输入和输出时间步长的样本。

```py
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以在小型设计数据集上演示此功能。

下面列出了完整的示例。

```py
# multi-step data preparation
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

运行该示例将单变量系列拆分为输入和输出时间步骤，并打印每个系列的输入和输出组件。

```py
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]
```

既然我们知道如何为多步预测准备数据，那么让我们看看一些可以学习这种映射的 LSTM 模型。

### 向量输出模型

与其他类型的神经网络模型一样，LSTM 可以直接输出向量，可以解释为多步预测。

在前一节中看到这种方法是每个输出时间序列的一个时间步骤被预测为向量。

与前一节中单变量数据的 LSTM 一样，必须首先对准备好的样本进行重新整形。 LSTM 期望数据具有[_ 样本，时间步长，特征 _]的三维结构，在这种情况下，我们只有一个特征，因此重塑是直截了当的。

```py
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
```

通过`n_steps_in`和`n_steps_out`变量中指定的输入和输出步数，我们可以定义一个多步骤时间序列预测模型。

可以使用任何呈现的 LSTM 模型类型，例如香草，堆叠，双向，CNN-LSTM 或 ConvLSTM。下面定义了用于多步预测的 Stacked LSTM。

```py
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

该模型可以对单个样本做出预测。我们可以通过提供输入来预测数据集末尾之后的下两个步骤：

```py
[70, 80, 90]
```

我们希望预测的输出为：

```py
[100, 110]
```

正如模型所预期的那样，做出预测时输入数据的单个样本的形状对于 1 个样本，输入的 3 个时间步长和单个特征必须是[1,3,1]。

```py
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
```

将所有这些结合在一起，下面列出了具有单变量时间序列的用于多步预测的 Stacked LSTM。

```py
# univariate multi-step vector-output stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=50, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行示例预测并打印序列中的后两个时间步骤。

```py
[[100.98096 113.28924]]
```

### 编解码器模型

专门为预测可变长度输出序列而开发的模型称为[编解码器 LSTM](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/) 。

该模型设计用于预测问题，其中存在输入和输出序列，即所谓的序列到序列或 seq2seq 问题，例如将文本从一种语言翻译成另一种语言。

该模型可用于多步时间序列预测。

顾名思义，该模型由两个子模型组成：编码器和解码器。

编码器是负责读取和解释输入序列的模型。编码器的输出是固定长度的向量，表示模型对序列的解释。编码器传统上是 Vanilla LSTM 模型，但也可以使用其他编码器模型，例如 Stacked，Bidirectional 和 CNN 模型。

```py
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
```

解码器使用编码器的输出作为输入。

首先，对输出序列中的每个所需时间步长重复一次编码器的固定长度输出。

```py
model.add(RepeatVector(n_steps_out))
```

然后将该序列提供给 LSTM 解码器模型。模型必须为输出时间步骤中的每个值输出一个值，该值可由单个输出模型解释。

```py
model.add(LSTM(100, activation='relu', return_sequences=True))
```

我们可以使用相同的一个或多个输出层在输出序列中进行每个一步预测。这可以通过将模型的输出部分包装在 [TimeDistributed 包装器](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)中来实现。

```py
model.add(TimeDistributed(Dense(1)))
```

下面列出了用于多步时间序列预测的编解码器模型的完整定义。

```py
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
```

与其他 LSTM 模型一样，输入数据必须重新整形为[_ 样本，时间步长，特征 _]的预期三维形状。

```py
X = X.reshape((X.shape[0], X.shape[1], n_features))
```

在编解码器模型的情况下，训练数据集的输出或 y 部分也必须具有该形状。这是因为模型将使用每个输入样本的给定数量的特征预测给定数量的时间步长。

```py
y = y.reshape((y.shape[0], y.shape[1], n_features))
```

下面列出了用于多步时间序列预测的编解码器 LSTM 的完整示例。

```py
# univariate multi-step encoder-decoder lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行示例预测并打印序列中的后两个时间步骤。

```py
[[[101.9736  
  [116.213615]]]
```

## 多变量多步 LSTM 模型

在前面的部分中，我们研究了单变量，多变量和多步骤时间序列预测。

可以混合和匹配到目前为止针对不同问题呈现的不同类型的 LSTM 模型。这也适用于涉及多变量和多步预测的时间序列预测问题，但可能更具挑战性。

在本节中，我们将提供多个多步骤时间序列预测的数据准备和建模的简短示例，作为模板来缓解这一挑战，具体来说：

1.  多输入多步输出。
2.  多个并行输入和多步输出。

也许最大的绊脚石是准备数据，所以这是我们关注的重点。

### 多输入多步输出

存在多变量时间序列预测问题，其中输出序列是分开的但取决于输入时间序列，并且输出序列需要多个时间步长。

例如，考虑前一部分的多变量时间序列：

```py
[[ 10  15  25]
 [ 20  25  45]
 [ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]
 [ 80  85 165]
 [ 90  95 185]]
```

我们可以使用两个输入时间序列中的每一个的三个先前时间步骤来预测输出时间序列的两个时间步长。

输入：

```py
10, 15
20, 25
30, 35
```

输出：

```py
65
85
```

下面的`split_sequences()`函数实现了这种行为。

```py
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以在我们设计的数据集上证明这一点。

下面列出了完整的示例。

```py
# multivariate multi-step data preparation
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

首先运行该示例打印准备好的训练数据的形状。

我们可以看到样本输入部分的形状是三维的，由六个样本组成，有三个时间步长，两个变量用于两个输入时间序列。

样本的输出部分对于六个样本是二维的，并且每个样本的两个时间步长是预测的。

然后打印制备的样品以确认数据是按照我们指定的方式制备的。

```py
(6, 3, 2) (6, 2)

[[10 15]
 [20 25]
 [30 35]] [65 85]
[[20 25]
 [30 35]
 [40 45]] [ 85 105]
[[30 35]
 [40 45]
 [50 55]] [105 125]
[[40 45]
 [50 55]
 [60 65]] [125 145]
[[50 55]
 [60 65]
 [70 75]] [145 165]
[[60 65]
 [70 75]
 [80 85]] [165 185]
```

我们现在可以开发用于多步预测的 LSTM 模型。

可以使用向量输出或编解码器模型。在这种情况下，我们将使用 Stacked LSTM 演示向量输出。

下面列出了完整的示例。

```py
# multivariate multi-step stacked lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例适合模型并预测输出序列的下两个时间步骤超出数据集。

我们希望接下来的两个步骤是：[185,205]

这是一个具有挑战性的问题框架，数据非常少，模型的任意配置版本也很接近。

```py
[[188.70619 210.16513]]
```

### 多个并行输入和多步输出

并行时间序列的问题可能需要预测每个时间序列的多个时间步长。

例如，考虑前一部分的多变量时间序列：

```py
[[ 10  15  25]
 [ 20  25  45]
 [ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]
 [ 80  85 165]
 [ 90  95 185]]
```

我们可以使用三个时间序列中的每一个的最后三个步骤作为模型的输入，并预测三个时间序列中的每一个的下一个时间步长作为输出。

训练数据集中的第一个样本如下。

输入：

```py
10, 15, 25
20, 25, 45
30, 35, 65
```

输出：

```py
40, 45, 85
50, 55, 105
```

下面的`split_sequences()`函数实现了这种行为。

```py
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

我们可以在小型设计数据集上演示此功能。

下面列出了完整的示例。

```py
# multivariate multi-step data preparation
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
```

首先运行该示例打印准备好的训练数据集的形状。

我们可以看到数据集的输入（X）和输出（Y）元素分别是样本数，时间步长和变量或并行时间序列的三维。

然后将每个系列的输入和输出元素并排打印，以便我们可以确认数据是按照我们的预期准备的。

```py
(5, 3, 3) (5, 2, 3)

[[10 15 25]
 [20 25 45]
 [30 35 65]] [[ 40  45  85]
 [ 50  55 105]]
[[20 25 45]
 [30 35 65]
 [40 45 85]] [[ 50  55 105]
 [ 60  65 125]]
[[ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]] [[ 60  65 125]
 [ 70  75 145]]
[[ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]] [[ 70  75 145]
 [ 80  85 165]]
[[ 50  55 105]
 [ 60  65 125]
 [ 70  75 145]] [[ 80  85 165]
 [ 90  95 185]]
```

我们可以使用向量输出或编码器解码器 LSTM 来模拟这个问题。在这种情况下，我们将使用编解码器模型。

下面列出了完整的示例。

```py
# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=0)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例适合模型并预测超出数据集末尾的下两个时间步的三个时间步中的每一个的值。

我们希望这些系列和时间步骤的值如下：

```py
90, 95, 185
100, 105, 205
```

我们可以看到模型预测合理地接近预期值。

```py
[[[ 91.86044   97.77231  189.66768 ]
  [103.299355 109.18123  212.6863  ]]]
```

## 摘要

在本教程中，您了解了如何针对一系列标准时间序列预测问题开发一套 LSTM 模型。

具体来说，你学到了：

*   如何开发 LSTM 模型进行单变量时间序列预测。
*   如何开发多变量时间序列预测的 LSTM 模型。
*   如何开发 LSTM 模型进行多步时间序列预测。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。