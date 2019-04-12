# 如何在 Keras 中开发基于字符的神经语言模型

> 原文： [https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/](https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/)

语言模型根据序列中前面的特定单词预测序列中的下一个单词。

还可以使用神经网络在角色级别开发语言模型。基于字符的语言模型的好处是它们在处理任何单词，标点符号和其他文档结构时的小词汇量和灵活性。这需要以较慢的训练需要更大的模型为代价。

然而，在神经语言模型领域，基于字符的模型为语言建模的一般，灵活和强大的方法提供了许多希望。

在本教程中，您将了解如何开发基于字符的神经语言模型。

完成本教程后，您将了解：

*   如何为基于字符的语言建模准备文本。
*   如何使用 LSTM 开发基于字符的语言模型。
*   如何使用训练有素的基于字符的语言模型来生成文本。

让我们开始吧。

*   **2018 年 2 月更新**：Keras 2.1.3 中针对 API 更改生成的次要更新。

![How to Develop a Character-Based Neural Language Model in Keras](img/f5b42db5f9585614acf505c93ccca994.jpg)

如何在 Keras
中开发基于角色的神经语言模型 [hedera.baltica](https://www.flickr.com/photos/hedera_baltica/33907382116/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  唱一首六便士之歌
2.  数据准备
3.  训练语言模型
4.  生成文本

## 唱一首六便士之歌

童谣“[唱一首六便士之歌](https://en.wikipedia.org/wiki/Sing_a_Song_of_Sixpence)”在西方是众所周知的。

第一节是常见的，但也有一个 4 节版本，我们将用它来开发基于角色的语言模型。

它很短，所以适合模型会很快，但不会太短，以至于我们看不到任何有趣的东西。

我们将用作源文本的完整 4 节版本如下所示。

```py
Sing a song of sixpence,
A pocket full of rye.
Four and twenty blackbirds,
Baked in a pie.

When the pie was opened
The birds began to sing;
Wasn't that a dainty dish,
To set before the king.

The king was in his counting house,
Counting out his money;
The queen was in the parlour,
Eating bread and honey.

The maid was in the garden,
Hanging out the clothes,
When down came a blackbird
And pecked off her nose.
```

复制文本并将其保存在当前工作目录中的新文件中，文件名为“ _rhyme.txt_ ”。

## 数据准备

第一步是准备文本数据。

我们将从定义语言模型的类型开始。

### 语言模型设计

必须在文本上训练语言模型，对于基于字符的语言模型，输入和输出序列必须是字符。

用作输入的字符数也将定义需要提供给模型的字符数，以便引出第一个预测字符。

生成第一个字符后，可将其附加到输入序列并用作模型的输入以生成下一个字符。

较长的序列为模型提供了更多的上下文，以便了解接下来要输出的字符，但是在生成文本时需要更长的时间来训练并增加模型播种的负担。

我们将为此模型使用任意长度的 10 个字符。

没有很多文字，10 个字是几个字。

我们现在可以将原始文本转换为我们的模型可以学习的形式;特别是，输入和输出字符序列。

### 加载文字

我们必须将文本加载到内存中，以便我们可以使用它。

下面是一个名为 _load_doc（）_ 的函数，它将加载给定文件名的文本文件并返回加载的文本。

```py
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
```

我们可以使用童谣' _rhyme.txt_ '的文件名调用此函数，将文本加载到内存中。然后将文件的内容作为完整性检查打印到屏幕。

```py
# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)
```

### 干净的文字

接下来，我们需要清理加载的文本。

我们在这里不会做太多。具体来说，我们将删除所有新行字符，以便我们只有一个由空格分隔的长字符序列。

```py
# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)
```

您可能希望探索其他数据清理方法，例如将案例规范化为小写或删除标点符号以努力减少最终词汇量大小并开发更小更精简的模型。

### 创建序列

现在我们有了很长的字符列表，我们可以创建用于训练模型的输入输出序列。

每个输入序列将是 10 个字符，带有一个输出字符，使每个序列长 11 个字符。

我们可以通过枚举文本中的字符来创建序列，从索引 10 处的第 11 个字符开始。

```py
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
```

运行此片段，我们可以看到我们最终只有不到 400 个字符序列来训练我们的语言模型。

```py
Total Sequences: 399
```

### 保存序列

最后，我们可以将准备好的数据保存到文件中，以便我们可以在开发模型时加载它。

下面是一个函数 _save_doc（）_，给定一个字符串列表和一个文件名，将字符串保存到文件，每行一个。

```py
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

我们可以调用这个函数并将我们准备好的序列保存到我们当前工作目录中的文件名' _char_sequences.txt_ '。

```py
# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)
```

### 完整的例子

将所有这些结合在一起，下面提供了完整的代码清单。

```py
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)
```

运行该示例以创建' _char_seqiences.txt_ '文件。

看看里面你应该看到如下内容：

```py
Sing a song
ing a song
ng a song o
g a song of
 a song of
a song of s
 song of si
song of six
ong of sixp
ng of sixpe
...
```

我们现在准备训练基于角色的神经语言模型。

## 训练语言模型

在本节中，我们将为准备好的序列数据开发神经语言模型。

该模型将读取编码字符并预测序列中的下一个字符。将使用长短期记忆递归神经网络隐藏层来从输入序列学习上下文以进行预测。

### 加载数据

第一步是从' _char_sequences.txt_ '加载准备好的字符序列数据。

我们可以使用上一节中开发的相同 _load_doc（）_ 函数。加载后，我们按新行分割文本，以提供准备编码的序列列表。

```py
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
```

### 编码序列

字符序列必须编码为整数。

这意味着将为每个唯一字符分配一个特定的整数值，并且每个字符序列将被编码为整数序列。

我们可以在原始输入数据中给定一组排序的唯一字符来创建映射。映射是字符值到整数值的字典。

```py
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
```

接下来，我们可以一次处理一个字符序列，并使用字典映射查找每个字符的整数值。

```py
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
```

结果是整数列表的列表。

我们稍后需要知道词汇量的大小。我们可以将其检索为字典映射的大小。

```py
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
```

运行这一段，我们可以看到输入序列数据中有 38 个唯一字符。

```py
Vocabulary Size: 38
```

### 拆分输入和输出

现在序列已经整数编码，我们可以将列分成输入和输出字符序列。

我们可以使用简单的数组切片来完成此操作。

```py
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
```

接下来，我们需要对每个字符进行一次热编码。也就是说，只要词汇表（38 个元素）标记为特定字符，每个字符就变成一个向量。这为网络提供了更精确的输入表示。它还为网络预测提供了明确的目标，其中模型可以输出字符的概率分布，并与所有 0 值的理想情况进行比较，实际的下一个字符为 1。

我们可以使用 Keras API 中的 _to_categorical（）_ 函数对输入和输出序列进行热编码。

```py
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
```

我们现在已准备好适应该模型。

### 适合模型

该模型由输入层定义，该输入层采用具有 10 个时间步长的序列和用于一个热编码输入序列的 38 个特征。

我们在 X 输入数据上使用第二维和第三维，而不是指定这些数字。这样，如果我们更改序列的长度或词汇表的大小，我们就不需要更改模型定义。

该模型具有单个 LSTM 隐藏层，具有 75 个存储单元，通过一些试验和错误选择。

该模型具有完全连接的输出层，该输出层输出一个向量，其中概率分布跨越词汇表中的所有字符。在输出层上使用 softmax 激活函数以确保输出具有概率分布的属性。

```py
# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
```

运行此命令会将已定义网络的摘要打印为完整性检查。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 75)                34200
_________________________________________________________________
dense_1 (Dense)              (None, 38)                2888
=================================================================
Total params: 37,088
Trainable params: 37,088
Non-trainable params: 0
_________________________________________________________________
```

该模型正在学习多类分类问题，因此我们使用针对此类问题的分类日志丢失。梯度下降的有效 Adam 实现用于优化模型，并且在每次批量更新结束时报告准确性。

该模型适用于 100 个训练时期，再次通过一些试验和错误找到。

```py
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)
```

### 保存模型

模型适合后，我们将其保存到文件中供以后使用。

Keras 模型 API 提供 _save（）_ 函数，我们可以使用它将模型保存到单个文件，包括权重和拓扑信息。

```py
# save the model to file
model.save('model.h5')
```

我们还保存了从字符到整数的映射，在使用模型和解码模型的任何输出时，我们需要对任何输入进行编码。

```py
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))
```

### 完整的例子

将所有这些结合在一起，下面列出了适合基于字符的神经语言模型的完整代码清单。

```py
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))
```

运行示例可能需要一分钟。

你会看到模型很好地学习了这个问题，也许是为了生成令人惊讶的字符序列。

```py
...
Epoch 96/100
0s - loss: 0.2193 - acc: 0.9950
Epoch 97/100
0s - loss: 0.2124 - acc: 0.9950
Epoch 98/100
0s - loss: 0.2054 - acc: 0.9950
Epoch 99/100
0s - loss: 0.1982 - acc: 0.9950
Epoch 100/100
0s - loss: 0.1910 - acc: 0.9950
```

在运行结束时，您将有两个文件保存到当前工作目录，特别是 _model.h5_ 和 _mapping.pkl_ 。

接下来，我们可以看一下使用学习模型。

## 生成文本

我们将使用学习的语言模型生成具有相同统计特性的新文本序列。

### 加载模型

第一步是将保存的模型加载到文件' _model.h5_ '中。

我们可以使用 Keras API 中的 _load_model（）_ 函数。

```py
# load the model
model = load_model('model.h5')
```

我们还需要加载 pickle 字典，用于将字符映射到文件' _mapping.pkl_ '中的整数。我们将使用 Pickle API 加载对象。

```py
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
```

我们现在准备使用加载的模型。

### 生成角色

我们必须提供 10 个字符的序列作为模型的输入，以便开始生成过程。我们将手动选择这些。

需要以与为模型准备训练数据相同的方式准备给定的输入序列。

首先，必须使用加载的映射对字符序列进行整数编码。

```py
# encode the characters as integers
encoded = [mapping[char] for char in in_text]
```

接下来，序列需要使用 _to_categorical（）_ Keras 函数进行热编码。

```py
# one hot encode
encoded = to_categorical(encoded, num_classes=len(mapping))
```

然后我们可以使用该模型来预测序列中的下一个字符。

我们使用 _predict_classes（）_ 而不是 _predict（）_ 来直接选择具有最高概率的字符的整数，而不是在整个字符集中获得完整的概率分布。

```py
# predict character
yhat = model.predict_classes(encoded, verbose=0)
```

然后，我们可以通过查找映射来解码此整数，以查看它映射到的字符。

```py
out_char = ''
for char, index in mapping.items():
	if index == yhat:
		out_char = char
		break
```

然后可以将此字符添加到输入序列中。然后，我们需要通过截断输入序列文本中的第一个字符来确保输入序列是 10 个字符。

我们可以使用 Keras API 中的 _pad_sequences（）_ 函数来执行此截断操作。

将所有这些放在一起，我们可以定义一个名为 _generate_seq（）_ 的新函数，用于使用加载的模型生成新的文本序列。

```py
# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text
```

### 完整的例子

将所有这些结合在一起，下面列出了使用拟合神经语言模型生成文本的完整示例。

```py
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))
```

运行该示例会生成三个文本序列。

第一个是测试模型在从押韵开始时的作用。第二个是测试，看看它在一行开头的表现如何。最后一个例子是一个测试，看看它对前面从未见过的一系列字符有多好。

```py
Sing a song of sixpence, A poc
king was in his counting house
hello worls e pake wofey. The
```

我们可以看到，正如我们所期望的那样，模型在前两个示例中表现得非常好。我们还可以看到模型仍然为新文本生成了一些东西，但这是无稽之谈。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **填充**。更新示例以仅逐行提供序列，并使用填充将每个序列填充到最大行长度。
*   **序列长度**。尝试不同的序列长度，看看它们如何影响模型的行为。
*   **调谐模型**。尝试不同的模型配置，例如内存单元和时期的数量，并尝试为更少的资源开发更好的模型。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [在维基百科上演六便士之歌](https://en.wikipedia.org/wiki/Sing_a_Song_of_Sixpence)
*   [使用 Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) 在 Python 中使用 LSTM 回归神经网络生成文本
*   [Keras Utils API](https://keras.io/utils/)
*   [Keras 序列处理 API](https://keras.io/preprocessing/sequence/)

## 摘要

在本教程中，您了解了如何开发基于字符的神经语言模型。

具体来说，你学到了：

*   如何为基于字符的语言建模准备文本。
*   如何使用 LSTM 开发基于字符的语言模型。
*   如何使用训练有素的基于字符的语言模型来生成文本。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。