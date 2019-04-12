# 如何利用 Python 处理序列预测问题中的缺失时间步长

> 原文： [https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/](https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)

通常缺少来自序列数据的观察结果。

数据可能已损坏或不可用，但根据定义，您的数据也可能具有可变长度序列。具有较少时间步长的那些序列可被认为具有缺失值。

在本教程中，您将了解如何使用 Keras 深度学习库处理 Python 中序列预测问题缺失值的数据。

完成本教程后，您将了解：

*   如何删除包含缺少时间步长的行。
*   如何标记丢失的时间步骤并强制网络了解其含义。
*   如何屏蔽缺失的时间步长并将其从模型中的计算中排除。

让我们开始吧。

![A Gentle Introduction to Linear Algebra](img/a3d40bba50bf998fdd0ac5f7625becad.jpg)

线性代数的温和介绍
[Steve Corey](https://www.flickr.com/photos/stevecorey/13939447959/) 的照片，保留一些权利。

## 概观

本节分为 3 部分;他们是：

1.  回波序列预测问题
2.  处理缺失的序列数据
3.  学习缺少序列值

### 环境

本教程假定您已安装 Python SciPy 环境。您可以在此示例中使用 Python 2 或 3。

本教程假设您使用 TensorFlow（v1.1.0 +）或 Theano（v0.9 +）后端安装了 Keras（v2.0.4 +）。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您在设置 Python 环境时需要帮助，请参阅以下帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 回波序列预测问题

回声问题是一个人为的序列预测问题，其目标是在固定的先前时间步长处记住和预测观察，称为滞后观察。

例如，最简单的情况是预测从前一个时间步的观察结果，即回显它。例如：

```
Time 1: Input 45
Time 2: Input 23, Output 45
Time 3: Input 73, Output 23
...
```

问题是，我们如何处理时间步 1？

我们可以在 Python 中实现回声序列预测问题。

这涉及两个步骤：随机序列的生成和随机序列到有监督学习问题的转换。

### 生成随机序列

我们可以使用随机模块中的 [random（）函数](https://docs.python.org/3/library/random.html)生成 0 到 1 之间的随机值序列。

我们可以将它放在一个名为 generate_sequence（）的函数中，该函数将为所需的时间步长生成一系列随机浮点值。

此功能如下所列。

```
# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]
```

### 框架作为监督学习

在使用神经网络时，必须将序列框定为监督学习问题。

这意味着序列需要分为输入和输出对。

该问题可以被构造为基于当前和先前时间步的函数进行预测。

或者更正式地说：

```
y(t) = f(X(t), X(t-1))
```

其中 y（t）是当前时间步长的期望输出，f（）是我们寻求用神经网络逼近的函数，X（t）和 X（t-1）是当前和之前的观测值时间步长。

输出可以等于先前的观察值，例如，y（t）= X（t-1），但是它可以很容易地是 y（t）= X（t）。我们针对这个问题进行训练的模型并不知道真正的表述，必须学习这种关系。

这模拟了真实的序列预测问题，其中我们将模型指定为一组固定的顺序时间步长的函数，但我们不知道从过去的观察到期望的输出值的实际函数关系。

我们可以将这个回声问题的框架实现为 python 中的监督学习问题。

[Pandas shift（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html)可用于创建序列的移位版本，可用于表示先前时间步的观测值。这可以与原始序列连接以提供 X（t-1）和 X（t）输入值。

```
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
```

然后我们可以将 Pandas DataFrame 中的值作为输入序列（X），并使用第一列作为输出序列（y）。

```
# specify input and output data
X, y = values, values[:, 0]
```

综上所述，我们可以定义一个函数，它将时间步数作为参数，并返回名为 generate_data（）的序列学习的 X，y 数据。

```
# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 0]
	return X, y
```

### 序列问题演示

我们可以将 generate_sequence（）和 generate_data（）代码绑定到一个工作示例中。

下面列出了完整的示例。

```
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame

# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 0]
	return X, y

# generate sequence
n_timesteps = 10
X, y = generate_data(n_timesteps)
# print sequence
for i in range(n_timesteps):
	print(X[i], '=>', y[i])
```

运行此示例会生成一个序列，将其转换为监督表示，并打印每个 X，Y 对。

```
[ nan 0.18961404] => nan
[ 0.18961404 0.25956078] => 0.189614044109
[ 0.25956078 0.30322084] => 0.259560776929
[ 0.30322084 0.72581287] => 0.303220844801
[ 0.72581287 0.02916655] => 0.725812865047
[ 0.02916655 0.88711086] => 0.0291665472554
[ 0.88711086 0.34267107] => 0.88711086298
[ 0.34267107 0.3844453 ] => 0.342671068373
[ 0.3844453 0.89759621] => 0.384445299683
[ 0.89759621 0.95278264] => 0.897596208691
```

我们可以看到第一行有 NaN 值。

这是因为我们没有事先观察序列中的第一个值。我们必须用一些东西填补这个空间。

但我们无法使用 NaN 输入拟合模型。

## 处理缺失的序列数据

处理缺失的序列数据有两种主要方法。

它们将删除缺少数据的行，并使用其他值填充缺少的时间步。

有关处理缺失数据的更常用方法，请参阅帖子：

*   [如何使用 Python 处理丢失的数据](http://machinelearningmastery.com/handle-missing-data-python/)

处理缺失序列数据的最佳方法取决于您的问题和您选择的网络配置。我建议探索每种方法，看看哪种方法效果最好。

### 删除缺失的序列数据

在我们回显前一个时间步骤中的观察的情况下，第一行数据不包含任何有用的信息。

也就是说，在上面的例子中，给定输入：

```
[        nan  0.18961404]
```

和输出：

```
nan
```

没有任何有意义的东西可以学习或预测。

这里最好的情况是删除这一行。

我们可以通过删除包含 NaN 值的所有行，在序列的制定过程中将其作为监督学习问题。具体来说，可以在将数据拆分为 X 和 y 分量之前调用 [dropna（）函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)。

完整示例如下：

```
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame

# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# remove rows with missing values
	df.dropna(inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 0]
	return X, y

# generate sequence
n_timesteps = 10
X, y = generate_data(n_timesteps)
# print sequence
for i in range(len(X)):
	print(X[i], '=>', y[i])
```

运行该示例会导致 9 X，y 对而不是 10 对，并删除第一行。

```
[ 0.60619475  0.24408238] => 0.606194746194
[ 0.24408238  0.44873712] => 0.244082383195
[ 0.44873712  0.92939547] => 0.448737123424
[ 0.92939547  0.74481645] => 0.929395472523
[ 0.74481645  0.69891311] => 0.744816453809
[ 0.69891311  0.8420314 ] => 0.69891310578
[ 0.8420314   0.58627624] => 0.842031399202
[ 0.58627624  0.48125348] => 0.586276240292
[ 0.48125348  0.75057094] => 0.481253484036
```

### 替换缺失的序列数据

在回声问题被配置为在当前时间步骤回显观察的情况下，第一行将包含有意义的信息。

例如，我们可以将 y 的定义从值[：，0]更改为值[：，1]并重新运行演示以生成此问题的示例，如下所示：

```
[        nan  0.50513289] => 0.505132894821
[ 0.50513289  0.22879667] => 0.228796667421
[ 0.22879667  0.66980995] => 0.669809946421
[ 0.66980995  0.10445146] => 0.104451463568
[ 0.10445146  0.70642423] => 0.70642422679
[ 0.70642423  0.10198636] => 0.101986362328
[ 0.10198636  0.49648033] => 0.496480332278
[ 0.49648033  0.06201137] => 0.0620113728356
[ 0.06201137  0.40653087] => 0.406530870804
[ 0.40653087  0.63299264] => 0.632992635565
```

我们可以看到第一行给出了输入：

```
[        nan  0.50513289]
```

和输出：

```
0.505132894821
```

这可以从输入中学到。

问题是，我们仍然需要处理 NaN 值。

我们可以用输入中不会自然出现的特定值（例如-1）替换所有 NaN 值，而不是删除具有 NaN 值的行。为此，我们可以使用 [fillna（）Pandas 函数](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)。

完整示例如下：

```
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame

# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]
	return X, y

# generate sequence
n_timesteps = 10
X, y = generate_data(n_timesteps)
# print sequence
for i in range(len(X)):
	print(X[i], '=>', y[i])
```

运行该示例，我们可以看到第一行的第一列中的 NaN 值被替换为-1 值。

```
[-1\. 0.94641256] => 0.946412559807
[ 0.94641256 0.11958645] => 0.119586451733
[ 0.11958645 0.50597771] => 0.505977714614
[ 0.50597771 0.92496641] => 0.924966407025
[ 0.92496641 0.15011979] => 0.150119790096
[ 0.15011979 0.69387197] => 0.693871974256
[ 0.69387197 0.9194518 ] => 0.919451802966
[ 0.9194518 0.78690337] => 0.786903370269
[ 0.78690337 0.17017999] => 0.170179993691
[ 0.17017999 0.82286572] => 0.822865722747
```

## 学习缺少序列值

在学习具有标记缺失值的序列预测问题时，有两个主要选项。

该问题可以按原样建模，我们可以鼓励模型了解特定值意味着“缺失”。或者，可以屏蔽特殊缺失值并从预测计算中明确排除。

我们将通过两个输入来看看这两个案例的人为“回应当前观察”问题。

### 学习缺失的价值观

我们可以为预测问题开发 LSTM。

输入由 2 个时间步长和 1 个特征定义。在第一隐藏层中定义具有 5 个存储器单元的小 LSTM，并且具有线性激活功能的单个输出层。

使用均方误差丢失函数和具有默认配置的高效 ADAM 优化算法，网络将适合。

```
# define model
model = Sequential()
model.add(LSTM(5, input_shape=(2, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

为了确保模型学习问题的广义解，即始终将输入作为输出返回（y（t）== X（t）），我们将在每个时期生成一个新的随机序列。该网络将适合 500 个时期，并且将在每个序列中的每个样本之后执行更新（batch_size = 1）。

```
# fit model
for i in range(500):
	X, y = generate_data(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
```

一旦拟合，将生成另一个随机序列，并将来自模型的预测与预期值进行比较。这将提供模型技能的具体概念。

```
# evaluate model on new data
X, y = generate_data(n_timesteps)
yhat = model.predict(X)
for i in range(len(X)):
	print('Expected', y[i,0], 'Predicted', yhat[i,0])
```

将所有这些结合在一起，下面提供了完整的代码清单。

```
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]
	# reshape
	X = X.reshape(len(X), 2, 1)
	y = y.reshape(len(y), 1)
	return X, y

n_timesteps = 10
# define model
model = Sequential()
model.add(LSTM(5, input_shape=(2, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
for i in range(500):
	X, y = generate_data(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate model on new data
X, y = generate_data(n_timesteps)
yhat = model.predict(X)
for i in range(len(X)):
	print('Expected', y[i,0], 'Predicted', yhat[i,0])
```

运行该示例将打印每个时期的损失，并在运行结束时比较一个序列的预期输出与预测输出。

回顾最终预测，我们可以看到网络已经了解了问题并预测了“足够好”的输出，即使存在缺失值。

```
...
Epoch 1/1
0s - loss: 1.5992e-04
Epoch 1/1
0s - loss: 1.3409e-04
Epoch 1/1
0s - loss: 1.1581e-04
Epoch 1/1
0s - loss: 2.6176e-04
Epoch 1/1
0s - loss: 8.8303e-05
Expected 0.390784174343 Predicted 0.394238
Expected 0.688580469278 Predicted 0.690463
Expected 0.347155799665 Predicted 0.329972
Expected 0.345075533266 Predicted 0.333037
Expected 0.456591840482 Predicted 0.450145
Expected 0.842125610156 Predicted 0.839923
Expected 0.354087132135 Predicted 0.342418
Expected 0.601406667694 Predicted 0.60228
Expected 0.368929815424 Predicted 0.351224
Expected 0.716420996314 Predicted 0.719275
```

您可以进一步尝试此示例，并将给定序列的 t-1 观察值的 50％标记为-1，并查看它如何影响模型的技能随时间的变化。

### 掩盖缺失的价值观

可以从网络中的所有计算中屏蔽标记的缺失输入值。

我们可以通过使用 [Masking 层](https://keras.io/layers/core/#masking)作为网络的第一层来实现。

定义层时，我们可以指定要屏蔽的输入中的哪个值。如果时间步长的所有要素都包含蒙版值，则整个时间步长将从计算中排除。

这为完全排除行并强制网络了解标记缺失值的影响提供了一个中间立场。

由于 Masking 层是网络中的第一个，因此必须指定输入的预期形状，如下所示：

```
model.add(Masking(mask_value=-1, input_shape=(2, 1)))
```

我们可以将所有这些结合起来并重新运行示例。完整的代码清单如下。

```
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Masking

# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]
	# reshape
	X = X.reshape(len(X), 2, 1)
	y = y.reshape(len(y), 1)
	return X, y

n_timesteps = 10
# define model
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(2, 1)))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
for i in range(500):
	X, y = generate_data(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate model on new data
X, y = generate_data(n_timesteps)
yhat = model.predict(X)
for i in range(len(X)):
	print('Expected', y[i,0], 'Predicted', yhat[i,0])
```

同样，每个时期打印损失，并将预测与最终序列的预期值进行比较。

同样，预测看起来足够小到几位小数。

```
...
Epoch 1/1
0s - loss: 1.0252e-04
Epoch 1/1
0s - loss: 6.5545e-05
Epoch 1/1
0s - loss: 3.0831e-05
Epoch 1/1
0s - loss: 1.8548e-04
Epoch 1/1
0s - loss: 7.4286e-05
Expected 0.550889403319 Predicted 0.538004
Expected 0.24252028132 Predicted 0.243288
Expected 0.718869927574 Predicted 0.724669
Expected 0.355185878917 Predicted 0.347479
Expected 0.240554707978 Predicted 0.242719
Expected 0.769765554707 Predicted 0.776608
Expected 0.660782450416 Predicted 0.656321
Expected 0.692962017672 Predicted 0.694851
Expected 0.0485233839401 Predicted 0.0722362
Expected 0.35192019185 Predicted 0.339201
```

### 选择哪种方法？

这些一次性实验不足以评估在简单回波序列预测问题上最有效的方法。

他们提供的模板可以用于您自己的问题。

我鼓励您探索在序列预测问题中处理缺失值的 3 种不同方法。他们是：

*   删除缺少值的行。
*   标记并学习缺失值。
*   掩盖和学习没有遗漏的价值观。

尝试针对序列预测问题的每种方法，并对看起来效果最好的方法进行加倍研究。

## 摘要

如果序列具有可变长度，则通常在序列预测问题中具有缺失值。

在本教程中，您了解了如何使用 Keras 处理 Python 中序列预测问题中的缺失数据。

具体来说，你学到了：

*   如何删除包含缺失值的行。
*   如何标记缺失值并强制模型了解其含义。
*   如何屏蔽缺失值以将其从模型中的计算中排除。

您对处理丢失的序列数据有任何疑问吗？
在评论中提出您的问题，我会尽力回答。