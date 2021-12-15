# 学习使用编码器解码器LSTM循环神经网络添加数字

> 原文： [https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/](https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/)

长短期记忆（LSTM）网络是一种循环神经网络（RNN），能够学习输入序列中元素之间的关系。

LSTM的一个很好的演示是学习如何使用诸如和的数学运算将多个项组合在一起并输出计算结果。

初学者常犯的一个错误就是简单地学习从输入词到输出词的映射函数。关于这种问题的LSTM的良好演示涉及学习字符的排序输入（“50 + 11”）和以字符（“61”）预测序列输出。使用序列到序列或seq2seq（编码器 - 解码器），堆叠LSTM配置的LSTM可以学习这个难题。

在本教程中，您将了解如何使用LSTM解决添加随机生成的整数序列的问题。

完成本教程后，您将了解：

*   如何学习输入项的朴素映射函数来输出加法项。
*   如何构建添加问题（和类似问题）并适当地编码输入和输出。
*   如何使用seq2seq范例解决真正的序列预测添加问题。

让我们开始吧。

*   **更新Aug / 2018** ：修正了模型配置描述中的拼写错误。

![How to Learn to Add Numbers with seq2seq Recurrent Neural Networks](img/ba6b2b767f7039240f86040b472bd4d2.jpg)

如何学习使用seq2seq循环神经网络添加数字
照片由 [Lima Pix](https://www.flickr.com/photos/minhocos/11161305703/) ，保留一些权利。

## 教程概述

本教程分为3个部分;他们是：

1.  添加数字
2.  作为映射问题的添加（初学者的错误）
3.  作为seq2seq问题添加

### 环境

本教程假设安装了SciPy，NumPy，Pandas的Python 2或Python 3开发环境。

本教程还假设scikit-learn和Keras v2.0 +与Theano或TensorFlow后端一起安装。

如果您需要有关环境的帮助，请参阅帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 添加数字

任务是，给定一系列随机选择的整数，返回那些整数的总和。

例如，给定10 + 5，模型应输出15。

该模型将在随机生成的示例中进行训练和测试，以便学习添加数字的一般问题，而不是记住特定情况。

## 作为映射问题添加
（初学者的错误）

在本节中，我们将解决问题并使用LSTM解决它，并说明使初学者犯错误并且不利用循环神经网络的能力是多么容易。

### 数据生成

让我们首先定义一个函数来生成随机整数序列及其总和作为训练和测试数据。

我们可以使用 [randint（）](https://docs.python.org/3/library/random.html)函数生成最小值和最大值之间的随机整数，例如介于1和100之间。然后我们可以对序列求和。该过程可以重复固定次数，以创建数字输入序列对和匹配的输出求和值。

例如，此代码段将创建100个在1到100之间添加2个数字的示例：

```py
from random import seed
from random import randint

seed(1)
X, y = list(), list()
for i in range(100):
	in_pattern = [randint(1,100) for _ in range(2)]
	out_pattern = sum(in_pattern)
	print(in_pattern, out_pattern)
	X.append(in_pattern)
	y.append(out_pattern)
```

运行该示例将打印每个输入 - 输出对。

```py
...
[2, 97] 99
[97, 36] 133
[32, 35] 67
[15, 80] 95
[24, 45] 69
[38, 9] 47
[22, 21] 43
```

一旦我们有了模式，我们就可以将列表转换为NumPy Arrays并重新调整值。我们必须重新调整值以适应LSTM使用的激活范围。

例如：

```py
# format as NumPy arrays
X,y = array(X), array(y)
# normalize
X = X.astype('float') / float(100 * 2)
y = y.astype('float') / float(100 * 2)
```

综上所述，我们可以定义函数 _random_sum_pairs（）_，它接受指定数量的示例，每个序列中的一些整数，以及生成和返回X，y对数据的最大整数造型。

```py
from random import randint
from numpy import array

# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	# format as NumPy arrays
	X,y = array(X), array(y)
	# normalize
	X = X.astype('float') / float(largest * n_numbers)
	y = y.astype('float') / float(largest * n_numbers)
	return X, y
```

我们可能希望稍后反转数字的重新缩放。这将有助于将预测值与预期值进行比较，并以与原始数据相同的单位获得错误分数的概念。

下面的 _invert（）_函数反转了传入的预测值和期望值的标准化。

```py
# invert normalization
def invert(value, n_numbers, largest):
	return round(value * float(largest * n_numbers))
```

### 配置LSTM

我们现在可以定义一个LSTM来模拟这个问题。

这是一个相对简单的问题，因此模型不需要非常大。输入层将需要1个输入功能和2个时间步长（在添加两个数字的情况下）。

定义了两个隐藏的LSTM层，第一个具有6个单元，第二个具有2个单元，接着是完全连接的输出层，其返回单个总和值。

在给定网络的实值输出的情况下，使用有效的ADAM优化算法来拟合模型以及均方误差损失函数。

```py
# create LSTM
model = Sequential()
model.add(LSTM(6, input_shape=(n_numbers, 1), return_sequences=True))
model.add(LSTM(6))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

该网络适用于100个时期，每个时期生成新的示例，并且在每个批次结束时执行重量更新。

```py
# train LSTM
for _ in range(n_epoch):
	X, y = random_sum_pairs(n_examples, n_numbers, largest)
	X = X.reshape(n_examples, n_numbers, 1)
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)
```

### LSTM评估

我们在100个新模式上评估网络。

生成这些并且为每个预测总和值。实际和预测的和值都被重新调整到原始范围，并且计算出具有与原始值相同的比例的均方根误差（RMSE）分数。最后，列出了约20个预期值和预测值的示例作为示例。

最后，列出了20个预期值和预测值的示例作为示例。

```py
# evaluate on some new patterns
X, y = random_sum_pairs(n_examples, n_numbers, largest)
X = X.reshape(n_examples, n_numbers, 1)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:,0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))
```

## 完整的例子

我们可以将这一切联系起来。完整的代码示例如下所示。

```py
from random import seed
from random import randint
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	# format as NumPy arrays
	X,y = array(X), array(y)
	# normalize
	X = X.astype('float') / float(largest * n_numbers)
	y = y.astype('float') / float(largest * n_numbers)
	return X, y

# invert normalization
def invert(value, n_numbers, largest):
	return round(value * float(largest * n_numbers))

# generate training data
seed(1)
n_examples = 100
n_numbers = 2
largest = 100
# define LSTM configuration
n_batch = 1
n_epoch = 100
# create LSTM
model = Sequential()
model.add(LSTM(10, input_shape=(n_numbers, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# train LSTM
for _ in range(n_epoch):
	X, y = random_sum_pairs(n_examples, n_numbers, largest)
	X = X.reshape(n_examples, n_numbers, 1)
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = random_sum_pairs(n_examples, n_numbers, largest)
X = X.reshape(n_examples, n_numbers, 1)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:,0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))
```

运行该示例会在每个时期打印一些损失信息，并通过打印运行的RMSE和一些示例输出来完成。

结果并不完美，但很多例子都是正确预测的。

鉴于神经网络的随机性，您的具体输出可能会有所不同。

```py
RMSE: 0.565685
Expected=110, Predicted=110 (err=0)
Expected=122, Predicted=123 (err=-1)
Expected=104, Predicted=104 (err=0)
Expected=103, Predicted=103 (err=0)
Expected=163, Predicted=163 (err=0)
Expected=100, Predicted=100 (err=0)
Expected=56, Predicted=57 (err=-1)
Expected=61, Predicted=62 (err=-1)
Expected=109, Predicted=109 (err=0)
Expected=129, Predicted=130 (err=-1)
Expected=98, Predicted=98 (err=0)
Expected=60, Predicted=61 (err=-1)
Expected=66, Predicted=67 (err=-1)
Expected=63, Predicted=63 (err=0)
Expected=84, Predicted=84 (err=0)
Expected=148, Predicted=149 (err=-1)
Expected=96, Predicted=96 (err=0)
Expected=33, Predicted=34 (err=-1)
Expected=75, Predicted=75 (err=0)
Expected=64, Predicted=64 (err=0)
```

### 初学者的错误

一切都完成了吧？

错误。

我们解决的问题有多个输入但技术上不是序列预测问题。

实际上，您可以使用多层感知器（MLP）轻松解决它。例如：

```py
from random import seed
from random import randint
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	# format as NumPy arrays
	X,y = array(X), array(y)
	# normalize
	X = X.astype('float') / float(largest * n_numbers)
	y = y.astype('float') / float(largest * n_numbers)
	return X, y

# invert normalization
def invert(value, n_numbers, largest):
	return round(value * float(largest * n_numbers))

# generate training data
seed(1)
n_examples = 100
n_numbers = 2
largest = 100
# define LSTM configuration
n_batch = 2
n_epoch = 50
# create LSTM
model = Sequential()
model.add(Dense(4, input_dim=n_numbers))
model.add(Dense(2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# train LSTM
for _ in range(n_epoch):
	X, y = random_sum_pairs(n_examples, n_numbers, largest)
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = random_sum_pairs(n_examples, n_numbers, largest)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:,0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))
```

运行该示例可以完美地解决问题，并且可以在更少的时期内解决问题。

```py
RMSE: 0.000000
Expected=108, Predicted=108 (err=0)
Expected=143, Predicted=143 (err=0)
Expected=109, Predicted=109 (err=0)
Expected=16, Predicted=16 (err=0)
Expected=152, Predicted=152 (err=0)
Expected=59, Predicted=59 (err=0)
Expected=95, Predicted=95 (err=0)
Expected=113, Predicted=113 (err=0)
Expected=90, Predicted=90 (err=0)
Expected=104, Predicted=104 (err=0)
Expected=123, Predicted=123 (err=0)
Expected=92, Predicted=92 (err=0)
Expected=150, Predicted=150 (err=0)
Expected=136, Predicted=136 (err=0)
Expected=130, Predicted=130 (err=0)
Expected=76, Predicted=76 (err=0)
Expected=112, Predicted=112 (err=0)
Expected=129, Predicted=129 (err=0)
Expected=171, Predicted=171 (err=0)
Expected=127, Predicted=127 (err=0)
```

问题是我们将这么多的域编码到问题中，它将问题从序列预测问题转变为函数映射问题。

也就是说，输入的顺序不再重要。我们可以按照我们想要的任何方式改变它，并且仍然可以解决问题。

MLP旨在学习映射功能，并且可以轻松解决学习如何添加数字的问题。

一方面，这是一种更好的方法来解决添加数字的具体问题，因为模型更简单，结果更好。另一方面，它是反复神经网络的可怕用法。

这是一个初学者的错误，我看到在网络上的许多“`LSTMs`”的介绍中被​​复制了。

## 作为序列预测问题的添加

帧添加的另一种方法使其成为明确的序列预测问题，反过来又使其难以解决。

我们可以将添加框架作为输入和输出字符串，让模型找出字符的含义。整个添加问题可以被构造为一串字符，例如输出“62”的“12 + 50”，或者更具体地说：

*   输入：['1'，'2'，'+'，'5'，'0']
*   输出：['6'，'2']

该模型不仅必须学习字符的整数性质，还要学习要执行的数学运算的性质。

注意序列现在如何重要，并且随机改组输入将创建一个与输出序列无关的无意义序列。

还要注意问题如何转换为同时具有输入和输出序列。这称为序列到序列预测问题，或称为seq2seq问题。

我们可以通过添加两个数字来保持简单，但我们可以看到这可以如何缩放到可变数量的术语和数学运算，可以作为模型的输入供学习和概括。

请注意，这个形式和本例的其余部分受到了Keras项目中[添加seq2seq示例](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py)的启发，尽管我从头开始重新开发它。

### Data Generation

seq2seq定义问题的数据生成涉及更多。

我们将每件作为独立功能开发，以便您可以使用它们并了解它们的工作原理。在那里挂。

第一步是生成随机整数序列及其总和，如前所述，但没有归一化。我们可以把它放在一个名为 _random_sum_pairs（）_的函数中，如下所示。

```py
from random import seed
from random import randint

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y

seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)
```

仅运行此函数会打印一个在1到10之间添加两个随机整数的示例。

```py
[[3, 10]] [13]
```

下一步是将整数转换为字符串。输入字符串的格式为'99 +99'，输出字符串的格式为'99'。

此函数的关键是数字填充，以确保每个输入和输出序列具有相同的字符数。填充字符应与数据不同，因此模型可以学习忽略它们。在这种情况下，我们使用空格字符填充（''）并填充左侧的字符串，将信息保存在最右侧。

还有其他填充方法，例如单独填充每个术语。尝试一下，看看它是否会带来更好的表现。在下面的评论中报告您的结果。

填充需要我们知道最长序列可能有多长。我们可以通过获取我们可以生成的最大整数的 _log10（）_和该数字的上限来轻松计算这个数，以了解每个数字需要多少个字符。我们将最大数字加1，以确保我们期望3个字符而不是2个字符，用于圆形最大数字的情况，例如200.我们需要添加正确数量的加号。

```py
max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
```

在输出序列上重复类似的过程，当然没有加号。

```py
max_length = ceil(log10(n_numbers * (largest+1)))
```

下面的示例添加 _to_string（）_函数，并使用单个输入/输出对演示其用法。

```py
from random import seed
from random import randint
from math import ceil
from math import log10

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		Xstr.append(strp)
	max_length = ceil(log10(n_numbers * (largest+1)))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		ystr.append(strp)
	return Xstr, ystr

seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)
# convert to strings
X, y = to_string(X, y, n_numbers, largest)
print(X, y)
```

运行此示例首先打印整数序列和相同序列的填充字符串表示。

```py
[[3, 10]] [13]
[' 3+10'] ['13']
```

接下来，我们需要将字符串中的每个字符编码为整数值。毕竟我们必须处理神经网络中的数字，而不是字符。

整数编码将问题转换为分类问题，其中输出序列可以被视为具有11个可能值的类输出。这恰好是具有一些序数关系的整数（前10个类值）。

要执行此编码，我们必须定义可能出现在字符串编码中的完整符号字母，如下所示：

```py
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
```

然后，整数编码变成了一个简单的过程，即将字符的查找表构建为整数偏移量，并逐个转换每个字符串的每个字符。

下面的示例为整数编码提供了 _integer_encode（）_函数，并演示了如何使用它。

```py
from random import seed
from random import randint
from math import ceil
from math import log10

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		Xstr.append(strp)
	max_length = ceil(log10(n_numbers * (largest+1)))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		ystr.append(strp)
	return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
		yenc.append(integer_encoded)
	return Xenc, yenc

seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)
# convert to strings
X, y = to_string(X, y, n_numbers, largest)
print(X, y)
# integer encode
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
print(X, y)
```

运行该示例将打印每个字符串编码模式的整数编码版本。

我们可以看到空格字符（''）用11编码，三个字符（'3'）编码为3，依此类推。

```py
[[3, 10]] [13]
[' 3+10'] ['13']
[[11, 3, 10, 1, 0]] [[1, 3]]
```

下一步是对整数编码序列进行二进制编码。

这涉及将每个整数转换为具有与字母表相同长度的二进制向量，并用1标记特定整数。

例如，0整数表示'0'字符，并且将被编码为二元向量，其中11元素向量的第0个位置为1：[1,0,0,0,0,0,0,0， 0,0,0,0]。

下面的示例为二进制编码定义了 _one_hot_encode（）_函数，并演示了如何使用它。

```py
from random import seed
from random import randint
from math import ceil
from math import log10

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		Xstr.append(strp)
	max_length = ceil(log10(n_numbers * (largest+1)))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		ystr.append(strp)
	return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
		yenc.append(integer_encoded)
	return Xenc, yenc

# one hot encode
def one_hot_encode(X, y, max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		yenc.append(pattern)
	return Xenc, yenc

seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)
# convert to strings
X, y = to_string(X, y, n_numbers, largest)
print(X, y)
# integer encode
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
print(X, y)
# one hot encode
X, y = one_hot_encode(X, y, len(alphabet))
print(X, y)
```

运行该示例为每个整数编码打印二进制编码序列。

我添加了一些新行，使输入和输出二进制编码更清晰。

您可以看到单个和模式变为5个二进制编码向量的序列，每个向量具有11个元素。输出或总和变为2个二进制编码向量的序列，每个向量再次具有11个元素。

```py
[[3, 10]] [13]
[' 3+10'] ['13']
[[11, 3, 10, 1, 0]] [[1, 3]]
[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
 [[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]]
```

我们可以将所有这些步骤绑定到一个名为 _generate_data（）_的函数中，如下所示。

```py
# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
	# generate pairs
	X, y = random_sum_pairs(n_samples, n_numbers, largest)
	# convert to strings
	X, y = to_string(X, y, n_numbers, largest)
	# integer encode
	X, y = integer_encode(X, y, alphabet)
	# one hot encode
	X, y = one_hot_encode(X, y, len(alphabet))
	# return as numpy arrays
	X, y = array(X), array(y)
	return X, y
```

最后，我们需要反转编码以将输出向量转换回数字，以便我们可以将预期的输出整数与预测的整数进行比较。

下面的 _invert（）_功能执行此操作。关键是首先使用 _argmax（）_函数将二进制编码转换回整数，然后使用整数反向映射到字母表中的字符将整数转换回字符。

```py
# invert encoding
def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)
```

我们现在拥有为此示例准备数据所需的一切。

注意，这些函数是为这篇文章编写的，我没有编写任何单元测试，也没有用各种输入对它们进行战斗测试。如果您发现或发现明显的错误，请在下面的评论中告诉我。

### 配置并调整seq2seq LSTM模型

我们现在可以在这个问题上使用LSTM模型。

我们可以认为该模型由两个关键部分组成：编码器和解码器。

首先，输入序列一次向网络显示一个编码字符。我们需要一个编码级别来学习输入序列中的步骤之间的关系，并开发这些关系的内部表示。

网络的输入（对于两个数字的情况）是一系列5个编码字符（每个整数2个，“+”一个），其中每个向量包含11个可能字符的11个特征，序列中的每个项目可能是。

编码器将使用具有100个单位的单个LSTM隐藏层。

```py
model = Sequential()
model.add(LSTM(100, input_shape=(5, 11)))
```

解码器必须将输入序列的学习内部表示转换为正确的输出序列。为此，我们将使用具有50个单位的隐藏层LSTM，然后是输出层。

该问题被定义为需要两个输出字符的二进制输出向量。我们将使用相同的完全连接层（Dense）来输出每个二进制向量。要两次使用同一层，我们将它包装在TimeDistributed（）包装层中。

输出完全连接层将使用 [softmax激活函数](https://en.wikipedia.org/wiki/Softmax_function)输出[0,1]范围内的值。

```py
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(11, activation='softmax')))
```

但是有一个问题。

我们必须将编码器连接到解码器，它们不适合。

也就是说，编码器将为5个向量序列中的每个输入产生100个输出的2维矩阵。解码器是LSTM层，其期望[样本，时间步长，特征]的3D输入，以便产生具有2个时间步长的每个具有11个特征的1个样本的解码序列。

如果您尝试将这些碎片强制在一起，则会出现如下错误：

```py
ValueError: Input 0 is incompatible with layer lstm_2: expected ndim=3, found ndim=2
```

正如我们所期望的那样。

我们可以使用 [RepeatVector](https://keras.io/layers/core/#repeatvector) 层来解决这个问题。该层简单地重复提供的2D输入n次以创建3D输出。

RepeatVector层可以像适配器一样使用，以将网络的编码器和解码器部分组合在一起。我们可以配置RepeatVector重复输入2次。这将创建一个3D输出，包括两个来自编码器的序列输出副本，我们可以使用相同的完全连接层对两个所需输出向量中的每一个进行两次解码。

```py
model.add(RepeatVector(2))
```

该问题被定义为11类的分类问题，因此我们可以优化对数损失（`categorical_crossentropy`）函数，甚至可以跟踪每个训练时期的准确率和损失。

把这些放在一起，我们有：

```py
# define LSTM configuration
n_batch = 10
n_epoch = 30
# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# train LSTM
for i in range(n_epoch):
	X, y = generate_data(n_samples, n_numbers, largest, alphabet)
	print(i)
	model.fit(X, y, epochs=1, batch_size=n_batch)
```

### 为什么使用RepeatVector层？

为什么不将编码器的序列输出作为解码器的输入返回？

也就是说，每个输入序列时间步长的每个LSTM的一个输出而不是整个输入序列的每个LSTM的一个输出。

```py
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars), return_sequences=True))
```

输入序列的每个步骤的输出使解码器在每个步骤访问输入序列的中间表示。这可能有用也可能没用。在输入序列的末尾提供最终LSTM输出可能更符合逻辑，因为它捕获有关整个输入序列的信息，准备映射到或计算输出。

此外，这不会在网络中留下任何内容来指定除输入之外的解码器的大小，为输入序列的每个时间步长提供一个输出值（5而不是2）。

您可以将输出重构为由空格填充的5个字符的序列。网络将完成比所需更多的工作，并且可能失去编码器 - 解码器范例提供的一些压缩类型能力。试试看吧。

标题为“[”的问题是序列到序列学习吗？](https://github.com/fchollet/keras/issues/395) “关于Keras GitHub项目提供了一些你可以使用的替代表示的良好讨论。

### 评估LSTM模型

和以前一样，我们可以生成一批新的示例，并在算法适合后对其进行评估。

我们可以根据预测计算RMSE分数，尽管我在这里为了简单起见而将其排除在外。

```py
# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]
# show some examples
for i in range(20):
	print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
```

### 完整的例子

总而言之，下面列出了完整的示例。

```py
from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		Xstr.append(strp)
	max_length = ceil(log10(n_numbers * (largest+1)))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
		ystr.append(strp)
	return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
		yenc.append(integer_encoded)
	return Xenc, yenc

# one hot encode
def one_hot_encode(X, y, max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		yenc.append(pattern)
	return Xenc, yenc

# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
	# generate pairs
	X, y = random_sum_pairs(n_samples, n_numbers, largest)
	# convert to strings
	X, y = to_string(X, y, n_numbers, largest)
	# integer encode
	X, y = integer_encode(X, y, alphabet)
	# one hot encode
	X, y = one_hot_encode(X, y, len(alphabet))
	# return as numpy arrays
	X, y = array(X), array(y)
	return X, y

# invert encoding
def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)

# define dataset
seed(1)
n_samples = 1000
n_numbers = 2
largest = 10
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest+1)))
# define LSTM configuration
n_batch = 10
n_epoch = 30
# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# train LSTM
for i in range(n_epoch):
	X, y = generate_data(n_samples, n_numbers, largest, alphabet)
	print(i)
	model.fit(X, y, epochs=1, batch_size=n_batch)

# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]
# show some examples
for i in range(20):
	print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
```

运行示例几乎完全符合问题。事实上，运行更多的迭代或增加每个迭代的重量更新（ _batch_size = 1_ ）会让你到达那里，但需要10倍的时间来训练。

我们可以看到预测的结果与我们看到的前20个例子的预期结果相符。

```py
...
Epoch 1/1
1000/1000 [==============================] - 2s - loss: 0.0579 - acc: 0.9940
Expected=13, Predicted=13
Expected=13, Predicted=13
Expected=13, Predicted=13
Expected= 9, Predicted= 9
Expected=11, Predicted=11
Expected=18, Predicted=18
Expected=15, Predicted=15
Expected=14, Predicted=14
Expected= 6, Predicted= 6
Expected=15, Predicted=15
Expected= 9, Predicted= 9
Expected=10, Predicted=10
Expected= 8, Predicted= 8
Expected=14, Predicted=14
Expected=14, Predicted=14
Expected=19, Predicted=19
Expected= 4, Predicted= 4
Expected=13, Predicted=13
Expected= 9, Predicted= 9
Expected=12, Predicted=12
```

## 扩展

本节列出了您可能希望探索的本教程的一些自然扩展。

*   **整数编码**。探索问题是否可以单独使用整数编码来更好地了解问题。大多数输入之间的序数关系可能非常有用。
*   **变量号**。更改示例以在每个输入序列上支持可变数量的术语。只要执行足够的填充，这应该是直截了当的。
*   **可变数学运算**。更改示例以改变数学运算，以允许网络进一步概括。
*   **括号**。允许使用括号和其他数学运算。

你尝试过这些扩展吗？
在评论中分享您的发现;我很想看到你发现了什么。

## 进一步阅读

本节列出了一些可供进一步阅读的资源以及您可能觉得有用的其他相关示例。

### 文件

*   [神经网络序列学习](https://arxiv.org/pdf/1409.3215.pdf)，2014 [PDF]
*   [使用RNN编码器 - 解码器进行统计机器翻译的学习短语表示](https://arxiv.org/pdf/1406.1078.pdf)，2014 [PDF]
*   [LSTM可以解决困难的长时滞问题](https://papers.nips.cc/paper/1215-lstm-can-solve-hard-long-time-lag-problems.pdf) [PDF]
*   [学会执行](https://arxiv.org/pdf/1410.4615.pdf)，2014 [PDF]

### 代码和帖子

*   [Keras加法示例](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py)
*   [烤宽面条中的加法示例](https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py)
*   [RNN加成（一年级）](http://projects.rajivshah.com/blog/2016/04/05/rnn_addition/)和[笔记本](https://gist.github.com/rajshah4/aa6c67944f4a43a7c9a1204301788e0c)
*   [任何人都可以学习用Python编写LSTM-RNN（第1部分：RNN）](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)
*   [Tensorflow中50行LSTM的简单实现](https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23)

## 摘要

在本教程中，您了解了如何开发LSTM网络以了解如何使用seq2seq堆叠LSTM范例将随机整数添加到一起。

具体来说，你学到了：

*   如何学习输入项的朴素映射函数来输出加法项。
*   如何构建添加问题（和类似问题）并适当地编码输入和输出。
*   如何使用seq2seq范例解决真正的序列预测添加问题。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。