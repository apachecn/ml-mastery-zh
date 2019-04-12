# 用 Keras 理解 Python 中的有状态 LSTM 循环神经网络

> 原文： [https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)

强大且流行的递归神经网络是长期短期模型网络或 LSTM。

它被广泛使用，因为该体系结构克服了困扰所有递归神经网络的消失和暴露梯度问题，允许创建非常大且非常深的网络。

与其他递归神经网络一样，LSTM 网络维持状态，并且在 Keras 框架中如何实现这一点的具体细节可能会令人困惑。

在这篇文章中，您将通过 Keras 深度学习库确切了解 LSTM 网络中的状态。

阅读这篇文章后你会知道：

*   如何为序列预测问题开发一个朴素的 LSTM 网络。
*   如何通过 LSTM 网络批量管理状态和功能。
*   如何在 LSTM 网络中手动管理状态以进行状态预测。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 Aug / 2018** ：更新了 Python 3 的示例，更新了有状态示例以获得 100％的准确性。
*   **更新 Mar / 2019** ：修正了有状态示例中的拼写错误。

![Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](img/37f8b123630baa4ea9ced3b21e1ebed6.png)

使用 Keras 了解 Python 中的有状态 LSTM 回归神经网络
[Martin Abegglen](https://www.flickr.com/photos/twicepix/7923674788/) 的照片，保留一些权利。

## 问题描述：学习字母表

在本教程中，我们将开发和对比许多不同的 LSTM 递归神经网络模型。

这些比较的背景将是学习字母表的简单序列预测问题。也就是说，给定一个字母表的字母，预测字母表的下一个字母。

这是一个简单的序列预测问题，一旦理解就可以推广到其他序列预测问题，如时间序列预测和序列分类。

让我们用一些 python 代码来准备问题，我们可以从示例到示例重用这些代码。

首先，让我们导入我们计划在本教程中使用的所有类和函数。

```py
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
```

接下来，我们可以为随机数生成器播种，以确保每次执行代码时结果都相同。

```py
# fix random seed for reproducibility
numpy.random.seed(7)
```

我们现在可以定义我们的数据集，即字母表。为了便于阅读，我们用大写字母定义字母表。

神经网络模型编号，因此我们需要将字母表的字母映射为整数值。我们可以通过创建字符索引的字典（map）来轻松完成此操作。我们还可以创建反向查找，以便将预测转换回字符以便以后使用。

```py
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

现在我们需要创建输入和输出对来训练我们的神经网络。我们可以通过定义输入序列长度，然后从输入字母序列中读取序列来完成此操作。

例如，我们使用输入长度 1.从原始输入数据的开头开始，我们可以读出第一个字母“A”和下一个字母作为预测“B”。我们沿着一个角色移动并重复直到我们达到“Z”的预测。

```py
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
```

我们还打印出输入对以进行健全性检查。

将代码运行到此点将产生以下输出，总结长度为 1 的输入序列和单个输出字符。

```py
A -> B
B -> C
C -> D
D -> E
E -> F
F -> G
G -> H
H -> I
I -> J
J -> K
K -> L
L -> M
M -> N
N -> O
O -> P
P -> Q
Q -> R
R -> S
S -> T
T -> U
U -> V
V -> W
W -> X
X -> Y
Y -> Z
```

我们需要将 NumPy 阵列重新整形为 LSTM 网络所期望的格式，即[_ 样本，时间步长，特征 _]。

```py
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
```

一旦重新整形，我们就可以将输入整数归一化到 0 到 1 的范围，即 LSTM 网络使用的 S 形激活函数的范围。

```py
# normalize
X = X / float(len(alphabet))
```

最后，我们可以将此问题视为序列分类任务，其中 26 个字母中的每一个代表不同的类。因此，我们可以使用 Keras 内置函数 **to_categorical（）**将输出（y）转换为一个热编码。

```py
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
```

我们现在准备适应不同的 LSTM 模型。

## 用于学习 One-Char 到 One-Char 映射的 Naive LSTM

让我们从设计一个简单的 LSTM 开始，学习如何在给定一个字符的上下文的情况下预测字母表中的下一个字符。

我们将问题框架化为单字母输入到单字母输出对的随机集合。正如我们将看到的那样，这是 LSTM 学习问题的难点框架。

让我们定义一个具有 32 个单元的 LSTM 网络和一个具有 softmax 激活功能的输出层，用于进行预测。因为这是一个多类分类问题，我们可以使用日志丢失函数（在 Keras 中称为“ **categorical_crossentropy** ”），并使用 ADAM 优化函数优化网络。

该模型适用于 500 个时期，批量大小为 1。

```py
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
```

在我们拟合模型之后，我们可以评估和总结整个训练数据集的表现。

```py
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

然后，我们可以通过网络重新运行训练数据并生成预测，将输入和输出对转换回原始字符格式，以便直观地了解网络如何了解问题。

```py
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

下面提供了整个代码清单，以确保完整性。

```py
# Naive LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

运行此示例将生成以下输出。

```py
Model Accuracy: 84.00%
['A'] -> B
['B'] -> C
['C'] -> D
['D'] -> E
['E'] -> F
['F'] -> G
['G'] -> H
['H'] -> I
['I'] -> J
['J'] -> K
['K'] -> L
['L'] -> M
['M'] -> N
['N'] -> O
['O'] -> P
['P'] -> Q
['Q'] -> R
['R'] -> S
['S'] -> T
['T'] -> U
['U'] -> W
['V'] -> Y
['W'] -> Z
['X'] -> Z
['Y'] -> Z
```

我们可以看到这个问题对于网络来说确实很难学习。

原因是，糟糕的 LSTM 单位没有任何上下文可以使用。每个输入 - 输出模式以随机顺序显示给网络，并且在每个模式（每个批次包含一个模式的每个批次）之后重置网络状态。

这是滥用 LSTM 网络架构，将其视为标准的多层 Perceptron。

接下来，让我们尝试不同的问题框架，以便为网络提供更多的顺序来学习。

## Naive LSTM 用于三字符特征窗口到单字符映射

为多层 Perceptrons 添加更多上下文数据的流行方法是使用 window 方法。

这是序列中的先前步骤作为网络的附加输入功能提供的地方。我们可以尝试相同的技巧，为 LSTM 网络提供更多上下文。

在这里，我们将序列长度从 1 增加到 3，例如：

```py
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
```

这创建了以下训练模式：

```py
ABC -> D
BCD -> E
CDE -> F
```

然后，序列中的每个元素作为新的输入特征提供给网络。这需要修改数据准备步骤中输入序列的重新形成方式：

```py
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
```

在演示模型的预测时，还需要修改样本模式的重新整形方式。

```py
x = numpy.reshape(pattern, (1, 1, len(pattern)))
```

下面提供了整个代码清单，以确保完整性。

```py
# Naive LSTM to learn three-char window to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, 1, len(pattern)))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

运行此示例提供以下输出。

```py
Model Accuracy: 86.96%
['A', 'B', 'C'] -> D
['B', 'C', 'D'] -> E
['C', 'D', 'E'] -> F
['D', 'E', 'F'] -> G
['E', 'F', 'G'] -> H
['F', 'G', 'H'] -> I
['G', 'H', 'I'] -> J
['H', 'I', 'J'] -> K
['I', 'J', 'K'] -> L
['J', 'K', 'L'] -> M
['K', 'L', 'M'] -> N
['L', 'M', 'N'] -> O
['M', 'N', 'O'] -> P
['N', 'O', 'P'] -> Q
['O', 'P', 'Q'] -> R
['P', 'Q', 'R'] -> S
['Q', 'R', 'S'] -> T
['R', 'S', 'T'] -> U
['S', 'T', 'U'] -> V
['T', 'U', 'V'] -> Y
['U', 'V', 'W'] -> Z
['V', 'W', 'X'] -> Z
['W', 'X', 'Y'] -> Z
```

我们可以看到表现上的小幅提升可能是也可能不是真实的。这是一个简单的问题，即使使用窗口方法，我们仍然无法用 LSTM 学习。

同样，这是对问题的不良框架的 LSTM 网络的滥用。实际上，字母序列是一个特征的时间步长，而不是单独特征的一个时间步长。我们已经为网络提供了更多的上下文，但没有像预期的那样更多的序列。

在下一节中，我们将以时间步长的形式为网络提供更多上下文。

## 用于单字符映射的三字符时间步长窗口的朴素 LSTM

在 Keras 中，LSTM 的预期用途是以时间步长的形式提供上下文，而不是像其他网络类型那样提供窗口化功能。

我们可以采用我们的第一个例子，只需将序列长度从 1 更改为 3。

```py
seq_length = 3
```

同样，这会创建输入 - 输出对，如下所示：

```py
ABC -> D
BCD -> E
CDE -> F
DEF -> G
```

不同之处在于输入数据的重新整形将序列作为一个特征的时间步长序列，而不是多个特征的单个时间步长。

```py
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
```

这是为 Keras 中的 LSTM 提供序列上下文的正确用途。完整性代码示例如下所示。

```py
# Naive LSTM to learn three-char time steps to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

运行此示例提供以下输出。

```py
Model Accuracy: 100.00%
['A', 'B', 'C'] -> D
['B', 'C', 'D'] -> E
['C', 'D', 'E'] -> F
['D', 'E', 'F'] -> G
['E', 'F', 'G'] -> H
['F', 'G', 'H'] -> I
['G', 'H', 'I'] -> J
['H', 'I', 'J'] -> K
['I', 'J', 'K'] -> L
['J', 'K', 'L'] -> M
['K', 'L', 'M'] -> N
['L', 'M', 'N'] -> O
['M', 'N', 'O'] -> P
['N', 'O', 'P'] -> Q
['O', 'P', 'Q'] -> R
['P', 'Q', 'R'] -> S
['Q', 'R', 'S'] -> T
['R', 'S', 'T'] -> U
['S', 'T', 'U'] -> V
['T', 'U', 'V'] -> W
['U', 'V', 'W'] -> X
['V', 'W', 'X'] -> Y
['W', 'X', 'Y'] -> Z
```

我们可以看到模型完美地学习了问题，如模型评估和示例预测所证明的那样。

但它已经学到了一个更简单的问题。具体来说，它学会了从字母表中的三个字母序列预测下一个字母。它可以显示字母表中任意三个字母的随机序列，并预测下一个字母。

它实际上不能枚举字母表。我希望更大的多层感知网络可以使用窗口方法学习相同的映射。

LSTM 网络是有状态的。他们应该能够学习整个字母顺序，但默认情况下，Keras 实现会在每个训练批次之后重置网络状态。

## 批量生产中的 LSTM 状态

LSTM 的 Keras 实现在每批之后重置网络状态。

这表明，如果我们的批量大小足以容纳所有输入模式，并且如果所有输入模式都是按顺序排序的，那么 LSTM 可以使用批量中序列的上下文来更好地学习序列。

我们可以通过修改学习一对一映射的第一个示例并将批量大小从 1 增加到训练数据集的大小来轻松演示这一点。

此外，Keras 在每个训练时期之前对训练数据集进行混洗。为确保训练数据模式保持连续，我们可以禁用此改组。

```py
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)
```

网络将使用批内序列学习字符映射，但在进行预测时，网络将无法使用此上下文。我们可以评估网络随机和按顺序进行预测的能力。

完整性代码示例如下所示。

```py
# Naive LSTM to learn one-char to one-char mapping with all data in each batch
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
# convert list of lists to array and pad sequences if needed
X = pad_sequences(dataX, maxlen=seq_length, dtype='float32')
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (X.shape[0], seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
# demonstrate predicting random patterns
print("Test a Random Pattern:")
for i in range(0,20):
	pattern_index = numpy.random.randint(len(dataX))
	pattern = dataX[pattern_index]
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

运行该示例提供以下输出。

```py
Model Accuracy: 100.00%
['A'] -> B
['B'] -> C
['C'] -> D
['D'] -> E
['E'] -> F
['F'] -> G
['G'] -> H
['H'] -> I
['I'] -> J
['J'] -> K
['K'] -> L
['L'] -> M
['M'] -> N
['N'] -> O
['O'] -> P
['P'] -> Q
['Q'] -> R
['R'] -> S
['S'] -> T
['T'] -> U
['U'] -> V
['V'] -> W
['W'] -> X
['X'] -> Y
['Y'] -> Z
Test a Random Pattern:
['T'] -> U
['V'] -> W
['M'] -> N
['Q'] -> R
['D'] -> E
['V'] -> W
['T'] -> U
['U'] -> V
['J'] -> K
['F'] -> G
['N'] -> O
['B'] -> C
['M'] -> N
['F'] -> G
['F'] -> G
['P'] -> Q
['A'] -> B
['K'] -> L
['W'] -> X
['E'] -> F
```

正如我们所料，网络能够使用序列内上下文来学习字母表，从而实现训练数据的 100％准确性。

重要的是，网络可以对随机选择的字符中的下一个字母进行准确的预测。非常令人印象深刻。

## 用于单字符到单字符映射的有状态 LSTM

我们已经看到，我们可以将原始数据分解为固定大小的序列，并且这种表示可以由 LSTM 学习，但仅用于学习 3 个字符到 1 个字符的随机映射。

我们还看到，我们可以通过批量大小来为网络提供更多序列，但仅限于训练期间。

理想情况下，我们希望将网络暴露给整个序列，让它学习相互依赖关系，而不是在问题框架中明确定义这些依赖关系。

我们可以在 Keras 中通过使 LSTM 层有状态并在时期结束时手动重置网络状态来执行此操作，这也是训练序列的结束。

这确实是如何使用 LSTM 网络的。

我们首先需要将 LSTM 层定义为有状态。这样，我们必须明确指定批量大小作为输入形状的维度。这也意味着，当我们评估网络或进行预测时，我们还必须指定并遵守相同的批量大小。现在这不是一个问题，因为我们使用批量大小为 1.当批量大小不是一个时，这可能会在进行预测时带来困难，因为需要批量和按顺序进行预测。

```py
batch_size = 1
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
```

训练有状态 LSTM 的一个重要区别是我们一次手动训练一个时期并在每个时期后重置状态。我们可以在 for 循环中执行此操作。同样，我们不会改变输入，保留输入训练数据的创建顺序。

```py
for i in range(300):
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
```

如上所述，我们在评估整个训练数据集的网络表现时指定批量大小。

```py
# summarize performance of the model
scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

最后，我们可以证明网络确实学会了整个字母表。我们可以用第一个字母“A”播种它，请求预测，将预测反馈作为输入，并一直重复该过程到“Z”。

```py
# demonstrate some model predictions
seed = [char_to_int[alphabet[0]]]
for i in range(0, len(alphabet)-1):
	x = numpy.reshape(seed, (1, len(seed), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	print(int_to_char[seed[0]], "->", int_to_char[index])
	seed = [index]
model.reset_states()
```

我们还可以看到网络是否可以从任意字母开始进行预测。

```py
# demonstrate a random starting point
letter = "K"
seed = [char_to_int[letter]]
print("New start: ", letter)
for i in range(0, 5):
	x = numpy.reshape(seed, (1, len(seed), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	print(int_to_char[seed[0]], "->", int_to_char[index])
	seed = [index]
model.reset_states()
```

下面提供了整个代码清单，以确保完整性。

```py
# Stateful LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
batch_size = 1
model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for i in range(300):
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# summarize performance of the model
scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
seed = [char_to_int[alphabet[0]]]
for i in range(0, len(alphabet)-1):
	x = numpy.reshape(seed, (1, len(seed), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	print(int_to_char[seed[0]], "->", int_to_char[index])
	seed = [index]
model.reset_states()
# demonstrate a random starting point
letter = "K"
seed = [char_to_int[letter]]
print("New start: ", letter)
for i in range(0, 5):
	x = numpy.reshape(seed, (1, len(seed), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	print(int_to_char[seed[0]], "->", int_to_char[index])
	seed = [index]
model.reset_states()
```

运行该示例提供以下输出。

```py
Model Accuracy: 100.00%
A -> B
B -> C
C -> D
D -> E
E -> F
F -> G
G -> H
H -> I
I -> J
J -> K
K -> L
L -> M
M -> N
N -> O
O -> P
P -> Q
Q -> R
R -> S
S -> T
T -> U
U -> V
V -> W
W -> X
X -> Y
Y -> Z
New start:  K
K -> B
B -> C
C -> D
D -> E
E -> F
```

我们可以看到网络完全记住了整个字母表。它使用了样本本身的上下文，并学习了预测序列中下一个字符所需的依赖性。

我们还可以看到，如果我们用第一个字母为网络播种，那么它可以正确地敲击字母表的其余部分。

我们还可以看到，它只是从冷启动中学习了完整的字母序列。当被要求预测来自“K”的下一个字母时，它预测“B”并且重新回到整个字母表的反刍。

为了真实地预测“K”，需要将网络的状态反复加热，将字母从“A”加到“J”。这告诉我们，通过准备以下训练数据，我们可以通过“无状态”LSTM 实现相同的效果：

```py
---a -> b
--ab -> c
-abc -> d
abcd -> e
```

输入序列固定为 25（a-to-y 预测 z）并且模式以零填充为前缀。

最后，这提出了使用可变长度输入序列训练 LSTM 网络以预测下一个字符的问题。

## 具有可变长度输入到单字符输出的 LSTM

在上一节中，我们发现 Keras“有状态”LSTM 实际上只是重放第一个 n 序列的捷径，但并没有真正帮助我们学习字母表的通用模型。

在本节中，我们将探索“无状态”LSTM 的变体，它可以学习字母表的随机子序列，并努力构建一个可以给出任意字母或字母子序列的模型，并预测字母表中的下一个字母。

首先，我们正在改变问题的框架。为简化起见，我们将定义最大输入序列长度并将其设置为小值，如 5，以加快训练速度。这定义了为训练绘制的字母表子序列的最大长度。在扩展中，如果我们允许循环回到序列的开头，这可以设置为完整字母表（26）或更长。

我们还需要定义要创建的随机序列的数量，在本例中为 1000.这也可能更多或更少。我希望实际上需要更少的模式。

```py
# prepare the dataset of input to output pairs encoded as integers
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range(num_inputs):
	start = numpy.random.randint(len(alphabet)-2)
	end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
	sequence_in = alphabet[start:end+1]
	sequence_out = alphabet[end + 1]
	dataX.append([char_to_int[char] for char in sequence_in])
	dataY.append(char_to_int[sequence_out])
	print(sequence_in, '->', sequence_out)
```

在更广泛的上下文中运行此代码将创建如下所示的输入模式：

```py
PQRST -> U
W -> X
O -> P
OPQ -> R
IJKLM -> N
QRSTU -> V
ABCD -> E
X -> Y
GHIJ -> K
```

输入序列的长度在 1 和 **max_len** 之间变化，因此需要零填充。这里，我们使用左侧（前缀）填充和 **pad_sequences（）**函数中内置的 Keras。

```py
X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
```

在随机选择的输入模式上评估训练的模型。这可能很容易成为新的随机生成的字符序列。我也相信这也可以是一个带有“A”的线性序列，输出 fes 作为单个字符输入。

完整性代码清单如下所示。

```py
# LSTM with Variable Length Input Sequences to One Character Output
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from theano.tensor.shared_randomstreams import RandomStreams
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range(num_inputs):
	start = numpy.random.randint(len(alphabet)-2)
	end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
	sequence_in = alphabet[start:end+1]
	sequence_out = alphabet[end + 1]
	dataX.append([char_to_int[char] for char in sequence_in])
	dataY.append(char_to_int[sequence_out])
	print(sequence_in, '->', sequence_out)
# convert list of lists to array and pad sequences if needed
X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
# reshape X to be [samples, time steps, features]
X = numpy.reshape(X, (X.shape[0], max_len, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
batch_size = 1
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for i in range(20):
	pattern_index = numpy.random.randint(len(dataX))
	pattern = dataX[pattern_index]
	x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
	x = numpy.reshape(x, (1, max_len, 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
```

运行此代码将生成以下输出：

```py
Model Accuracy: 98.90%
['Q', 'R'] -> S
['W', 'X'] -> Y
['W', 'X'] -> Y
['C', 'D'] -> E
['E'] -> F
['S', 'T', 'U'] -> V
['G', 'H', 'I', 'J', 'K'] -> L
['O', 'P', 'Q', 'R', 'S'] -> T
['C', 'D'] -> E
['O'] -> P
['N', 'O', 'P'] -> Q
['D', 'E', 'F', 'G', 'H'] -> I
['X'] -> Y
['K'] -> L
['M'] -> N
['R'] -> T
['K'] -> L
['E', 'F', 'G'] -> H
['Q'] -> R
['Q', 'R', 'S'] -> T
```

我们可以看到，尽管模型没有从随机生成的子序列中完美地学习字母表，但它确实做得很好。该模型未经过调整，可能需要更多训练或更大的网络，或两者兼而有之（为读者练习）。

这是“_ 所有顺序输入示例中每个批次 _”字母模型的一个很好的自然扩展，它可以处理即席查询，但这次任意序列长度（最大长度） 。

## 摘要

在这篇文章中，您发现了 Keras 中的 LSTM 循环神经网络以及它们如何管理状态。

具体来说，你学到了：

*   如何为一个字符到一个字符的预测开发一个朴素的 LSTM 网络。
*   如何配置一个朴素的 LSTM 来学习样本中跨时间步的序列。
*   如何通过手动管理状态来配置 LSTM 以跨样本学习序列。

您对管理 LSTM 州或此帖有任何疑问吗？
在评论中提出您的问题，我会尽力回答。