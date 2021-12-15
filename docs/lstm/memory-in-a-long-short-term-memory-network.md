# 使用Python中的长短期记忆网络演示内存

> 原文： [https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/](https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/)

长短期记忆（LSTM）网络是一种能够学习长序列的循环神经网络。

这使它们与没有记忆的常规多层神经网络区分开来，并且只能学习输入和输出模式之间的映射。

重要的是要理解像LSTM这样的复杂神经网络在小型设计问题上的能力，因为这种理解将帮助您将网络扩展到大型甚至是非常大的问题。

在本教程中，您将发现LSTM记忆和回忆的功能。

完成本教程后，您将了解：

*   如何定义一个小序列预测问题，只有像LSTM这样的RNN可以使用内存来解决。
*   如何转换问题表示，使其适合LSTM学习。
*   如何设计LSTM来正确解决问题。

让我们开始吧。

![A Demonstration of Memory in a Long Short-Term Memory Network](img/c25fa76217d2aaa3d066421ffc6a5238.jpg)

在长期短期记忆网络中的记忆演示
照片由 [crazlei](https://www.flickr.com/photos/leifeng/4175290209/) ，保留一些权利。

## 环境

本教程假设您使用带有TensorFlow或Theano后端的SciPy，Keras 2.0或更高版本的Python 2或3环境。

有关设置Python环境的帮助，请参阅帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 序列问题描述

问题是一次预测一个序列的值。

给定序列中的一个值，模型必须预测序列中的下一个值。例如，给定值“0”作为输入，模型必须预测值“1”。

模型必须学习和正确预测有两种不同的序列。

皱纹是两个序列之间存在冲突的信息，并且模型必须知道每个一步预测的上下文（例如，它当前正在预测的序列），以便正确地预测每个完整序列。

这种皱纹对于防止模型记忆每个序列中的每个单步输入 - 输出值对非常重要，因为序列未知模型可能倾向于这样做。

要学习的两个序列如下：

*   3,0,1,2,3
*   4,0,1,2,4

我们可以看到序列的第一个值重复作为序列的最后一个值。这是指示器为模型提供关于它正在处理的序列的上下文。

冲突是从每个序列中的第二个项目到最后一个项目的过渡。在序列1中，给出“2”作为输入并且必须预测“3”，而在序列2中，给出“2”作为输入并且必须预测“4”。

这是多层感知器和其他非循环神经网络无法学习的问题。

这是“_实验2_ ”的简化版本，用于证明Hochreiter和Schmidhuber 1997年论文[长期短期记忆](http://dl.acm.org/citation.cfm?id=1246450)（ [PDF](http://www.bioinf.jku.at/publications/older/2604.pdf) ）中的LSTM长期记忆能力。

## 问题表征

本节分为3部分;他们是：

1.  一个热编码
2.  输入输出对
3.  重塑数据

### 一个热编码

我们将使用一个热编码来表示LSTM的学习问题。

也就是说，每个输入和输出值将表示为具有5个元素的二进制向量，因为问题的字母表是5个唯一值。

例如，[0,1,2,3,4]的5个值表示为以下5个二进制向量：

```py
0: [1, 0, 0, 0, 0]
1: [0, 1, 0, 0, 0]
2: [0, 0, 1, 0, 0]
3: [0, 0, 0, 1, 0]
4: [0, 0, 0, 0, 1]
```

我们可以使用一个简单的函数来执行此操作，该函数将获取序列并返回序列中每个值的二进制向量列表。下面的函数 _encode（）_实现了这种行为。

```py
# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded
```

我们可以在第一个序列上测试它并打印结果的二进制向量列表。下面列出了完整的示例。

```py
# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

seq1 = [3, 0, 1, 2, 3]
encoded = encode(seq1, 5)
for vector in encoded:
	print(vector)
```

运行该示例打印每个二进制向量。请注意，我们使用浮点值0.0和1.0，因为它们将用作模型的输入和输出。

```py
[0.0, 0.0, 0.0, 1.0, 0.0]
[1.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 1.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 1.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 1.0, 0.0]
```

### 输入输出对

下一步是将一系列编码值拆分为输入 - 输出对。

这是问题的监督学习表示，使得机器学习问题可以学习如何将输入模式（`X`）映射到输出模式（`y`）。

例如，第一个序列具有以下要学习的输入 - 输出对：

```py
X,	y
3,	0
0,	1
1,	2
2,	3
```

我们必须从一个热编码的二进制向量中创建这些映射对，而不是原始数字。

例如，3-&gt; 0的第一输入 - 输出对将是：

```py
X,			y
[0, 0, 0, 1, 0]		[1, 0, 0, 0, 0]
```

下面是一个名为 _to_xy_pairs（）_的函数，它将在给定编码二进制向量列表的情况下创建`X`和`y`模式的列表。

```py
# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y
```

我们可以将它与上面的一个热编码函数放在一起，并打印第一个序列的编码输入和输出对。

```py
# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

seq1 = [3, 0, 1, 2, 3]
encoded = encode(seq1, 5)
X, y = to_xy_pairs(encoded)
for i in range(len(X)):
	print(X[i], y[i])
```

运行该示例将打印序列中每个步骤的输入和输出对。

```py
[0.0, 0.0, 0.0, 1.0, 0.0] [1.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 0.0, 0.0, 0.0, 0.0] [0.0, 1.0, 0.0, 0.0, 0.0]
[0.0, 1.0, 0.0, 0.0, 0.0] [0.0, 0.0, 1.0, 0.0, 0.0]
[0.0, 0.0, 1.0, 0.0, 0.0] [0.0, 0.0, 0.0, 1.0, 0.0]
```

### 重塑数据

最后一步是重新整形数据，以便LSTM网络可以直接使用它。

Keras LSTM期望输入模式（X）为具有[_样本，时间步长，特征_]维度的三维NumPy阵列。

在一个输入数据序列的情况下，维度将是[4,1,5]，因为我们有4行数据，每行有1个时间步长，每行有5列。

我们可以从X模式列表中创建2D NumPy数组，然后将其重新整形为所需的3D格式。例如：

```py
df = DataFrame(X)
values = df.values
array = values.reshape(4, 1, 5)
```

我们还必须将输出模式列表（y）转换为2D NumPy数组。

下面是一个名为 _to_lstm_dataset（）_的函数，它将序列作为输入和序列字母表的大小，并返回准备使用的`X`和`y`数据集与LSTM。它在重新整形数据之前执行所需的序列转换为单热编码和输入输出对。

```py
# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = DataFrame(X), DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY
```

可以使用以下每个序列调用此函数：

```py
seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
n_unique = len(set(seq1 + seq2))

seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
```

我们现在拥有为LSTM准备数据的所有部分。

## 使用LSTM学习序列

在本节中，我们将定义LSTM以学习输入序列。

本节分为4个部分：

1.  LSTM配置
2.  LSTM训练
3.  LSTM评估
4.  LSTM完整示例

### LSTM配置

我们希望LSTM进行一步预测，我们已经在数据集的格式和形状中定义了这些预测。我们还希望在每个时间步之后更新LSTM错误，这意味着我们需要使用批量大小的一个。

默认情况下，Keras LSTM在批次之间不具有状态。我们可以通过将LSTM层上的_有状态_参数设置为`True`并手动管理训练时期来确保LSTM的内部状态在每个序列之后被重置，从而使它们成为有状态。

我们必须使用`batch_input_shape`参数定义批次的形状，其中3维[_批量大小，时间步长和特征_]分别为1,1和5。

网络拓扑将配置一个具有20个单元的隐藏LSTM层和一个具有5个输出的普通密集层，用于输出模式中的每个5列。由于二进制输出，将在输出层上使用sigmoid（逻辑）激活函数，并且将在LSTM层上使用默认的tanh（双曲正切）激活函数。

由于二进制输出，因此在拟合网络时将优化对数（交叉熵）损失函数，并且将使用有效的ADAM优化算法与所有默认参数。

下面列出了为此问题定义LSTM网络的Keras代码。

```py
model = Sequential()
model.add(LSTM(20, batch_input_shape=(1, 1, 5), stateful=True))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
```

### LSTM训练

我们必须一次手动拟合模型一个时期。

在一个时期内，我们可以在每个序列上拟合模型，确保在每个序列之后重置状态。

鉴于问题的简单性，该模型不需要长时间训练;在这种情况下，只需要250个迭代。

下面是一个示例，说明模型如何适应所有时期的每个序列。

```py
# train LSTM
for i in range(250):
	model.fit(seq1X, seq1Y, epochs=1, batch_size=1, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq2X, seq2Y, epochs=1, batch_size=1, verbose=0, shuffle=False)
	model.reset_states()
```

我希望在安装网络时能看到关于损失函数的一些反馈，因此从其中一个序列开启详细输出，而不是另一个序列。

### LSTM评估

接下来，我们可以通过预测学习序列的每个步骤来评估拟合模型。

我们可以通过预测每个序列的输出来做到这一点。

_predict_classes（）_函数可以用于直接预测类的LSTM模型。它通过在输出二进制向量上执行 _argmax（）_并返回具有最大输出的预测列的索引来完成此操作。输出索引完美地映射到序列中使用的整数（通过上面的仔细设计）。下面列出了做出预测的示例：

```py
result = model.predict_classes(seq1X, batch_size=1, verbose=0)
```

我们可以做出预测，然后在输入模式的上下文和序列的每个步骤的预期输出模式中打印结果。

### LSTM完整示例

我们现在可以将整个教程结合在一起。

完整的代码清单如下。

首先，准备数据，然后拟合模型并打印两个序列的预测。

```py
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = DataFrame(X), DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

# define sequences
seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
# convert sequences into required data format
n_unique = len(set(seq1 + seq2))
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
# define LSTM configuration
n_neurons = 20
n_batch = 1
n_epoch = 250
n_features = n_unique
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_features), stateful=True))
model.add(Dense(n_unique, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
# train LSTM
for i in range(n_epoch):
	model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq2X, seq2Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()

# test LSTM on sequence 1
print('Sequence 1')
result = model.predict_classes(seq1X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq1[i], seq1[i+1], result[i]))

# test LSTM on sequence 2
print('Sequence 2')
result = model.predict_classes(seq2X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq2[i], seq2[i+1], result[i]))
```

运行该示例提供关于模型在每个时期的第一序列上的损失的反馈。

在运行结束时，每个序列都在预测的上下文中打印。

```py
...
4/4 [==============================] - 0s - loss: 0.0930
Epoch 1/1
4/4 [==============================] - 0s - loss: 0.0927
Epoch 1/1
4/4 [==============================] - 0s - loss: 0.0925
Sequence 1
X=3.0 y=0.0, yhat=0.0
X=0.0 y=1.0, yhat=1.0
X=1.0 y=2.0, yhat=2.0
X=2.0 y=3.0, yhat=3.0
Sequence 2
X=4.0 y=0.0, yhat=0.0
X=0.0 y=1.0, yhat=1.0
X=1.0 y=2.0, yhat=2.0
X=2.0 y=4.0, yhat=4.0
```

结果显示了两件重要的事情：

*   LSTM一次一步地正确学习每个序列。
*   LSTM使用每个序列的上下文来正确地解析冲突的输入对。

本质上，LSTM能够记住3个时间步前序列开始处的输入模式，以正确预测序列中的最后一个值。

这种记忆和LSTM能够及时关联观测的能力是使LSTM如此强大以及它们如此广泛使用的关键能力。

虽然这个例子很简单，但LSTM能够在100秒甚至1000秒的时间步长中展示出同样的能力。

## 扩展

本节列出了本教程中示例扩展的思路。

*   **调整**。经过一些试验和错误后，选择了LSTM（时期，单位等）的配置。更简单的配置可能会在此问题上获得相同的结果。需要搜索一些参数。
*   **任意字母**。 5个整数的字母表是任意选择的。这可以更改为其他符号和更大的字母。
*   **长序列**。本例中使用的序列非常短。 LSTM能够在更长的100s和1000s时间步长序列上展示相同的能力。
*   **随机序列**。本教程中使用的序列呈线性增长。可以创建新的随机值序列，允许LSTM设计一个通用解决方案，而不是专门用于本教程中使用的两个序列的解决方案。
*   **批量学习**。每个时间步后都对LSTM进行了更新。探索使用批量更新，看看这是否会改善学习。
*   **Shuffle Epoch** 。序列在训练期间的每个时期以相同的顺序显示，并且在评估期间再次显示。随机化序列的顺序，使得序列1和2适合一个时期，这可以改善模型对具有相同字母表的新看不见的序列的概括。

你有没有探索过这些扩展？
在下面的评论中分享您的结果。我很想看看你想出了什么。

## 进一步阅读

我强烈建议阅读Hochreiter和Schmidhuber的1997年原始LSTM论文;这很棒。

*   [长期短记忆](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)，1997 [ [PDF](http://www.bioinf.jku.at/publications/older/2604.pdf) ]

## 摘要

在本教程中，您发现了LSTM能够记住多个时间步的关键功能。

具体来说，你学到了：

*   如何定义一个小序列预测问题，只有像LSTM这样的RNN可以使用内存来解决。
*   如何转换问题表示，使其适合LSTM学习。
*   如何设计LSTM来正确解决问题。

你有任何问题吗？
在下面的评论中发表您的问题，我会尽力回答。