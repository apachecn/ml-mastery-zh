# 如何在 Python 中用 Keras 开发基于单词的神经语言模型

> 原文： [https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/](https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/)

语言建模涉及在已经存在的单词序列的情况下预测序列中的下一个单词。

语言模型是许多自然语言处理模型中的关键元素，例如机器翻译和语音识别。语言模型的框架选择必须与语言模型的使用方式相匹配。

在本教程中，您将了解在从童谣中生成短序列时，语言模型的框架如何影响模型的技能。

完成本教程后，您将了解：

*   为给定的应用程序开发基于单词的语言模型的良好框架的挑战。
*   如何为基于单词的语言模型开发单字，双字和基于行的框架。
*   如何使用拟合语言模型生成序列。

让我们开始吧。

![How to Develop Word-Based Neural Language Models in Python with Keras](img/d1aa5edf765e5e408fc694194fef5048.jpg)

如何使用 Keras 在 Python 中开发基于 Word 的神经语言模型
照片由 [Stephanie Chapman](https://www.flickr.com/photos/imcountingufoz/5602273537/) 保留，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  框架语言建模
2.  杰克和吉尔童谣
3.  模型 1：单字输入，单字输出序列
4.  模型 2：逐行序列
5.  模型 3：双字输入，单字输出序列

## 框架语言建模

从原始文本中学习统计语言模型，并且在给定已经存在于序列中的单词的情况下预测序列中下一个单词的概率。

语言模型是大型模型中的关键组件，用于挑战自然语言处理问题，如机器翻译和语音识别。它们也可以作为独立模型开发，并用于生成与源文本具有相同统计属性的新序列。

语言模型一次学习和预测一个单词。网络的训练涉及提供单词序列作为输入，每次处理一个单词，其中可以为每个输入序列进行预测和学习。

类似地，在进行预测时，可以用一个或几个单词播种该过程，然后可以收集预测的单词并将其作为后续预测的输入呈现，以便建立生成的输出序列

因此，每个模型将涉及将源文本分成输入和输出序列，使得模型可以学习预测单词。

有许多方法可以从源文本中构建序列以进行语言建模。

在本教程中，我们将探讨在 Keras 深度学习库中开发基于单词的语言模型的 3 种不同方法。

没有单一的最佳方法，只是可能适合不同应用的不同框架。

## 杰克和吉尔童谣

杰克和吉尔是一个简单的童谣。

它由 4 行组成，如下所示：

> 杰克和吉尔上山
> 去取一桶水
> 杰克摔倒了，打破了他的王冠
> 吉尔跌倒了之后

我们将使用它作为我们的源文本来探索基于单词的语言模型的不同框架。

我们可以在 Python 中定义这个文本如下：

```py
# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
```

## 模型 1：单字输入，单字输出序列

我们可以从一个非常简单的模型开始。

给定一个单词作为输入，模型将学习预测序列中的下一个单词。

例如：

```py
X,		y
Jack, 	and
and,	Jill
Jill,	went
...
```

第一步是将文本编码为整数。

源文本中的每个小写字都被赋予一个唯一的整数，我们可以将单词序列转换为整数序列。

Keras 提供了 [Tokenizer](https://keras.io/preprocessing/text/#tokenizer) 类，可用于执行此编码。首先，Tokenizer 适合源文本，以开发从单词到唯一整数的映射。然后通过调用`texts_to_sequences()`函数将文本序列转换为整数序列。

```py
# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
```

我们稍后需要知道词汇表的大小，以便在模型中定义单词嵌入层，以及使用一个热编码对输出单词进行编码。

通过访问`word_index`属性，可以从训练好的 Tokenizer 中检索词汇表的大小。

```py
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
```

运行这个例子，我们可以看到词汇量的大小是 21 个单词。

我们添加一个，因为我们需要将最大编码字的整数指定为数组索引，例如单词编码 1 到 21，数组指示 0 到 21 或 22 个位置。

接下来，我们需要创建单词序列以适合模型，其中一个单词作为输入，一个单词作为输出。

```py
# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
```

运行这一部分表明我们总共有 24 个输入输出对来训练网络。

```py
Total Sequences: 24
```

然后我们可以将序列分成输入（`X`）和输出元素（`y`）。这很简单，因为我们在数据中只有两列。

```py
# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
```

我们将使用我们的模型来预测词汇表中所有单词的概率分布。这意味着我们需要将输出元素从单个整数转换为一个热编码，对于词汇表中的每个单词都为 0，对于值的实际单词为 1。这为网络提供了一个基本事实，我们可以从中计算错误并更新模型。

Keras 提供`to_categorical()`函数，我们可以使用它将整数转换为一个热编码，同时指定类的数量作为词汇表大小。

```py
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
```

我们现在准备定义神经网络模型。

该模型使用嵌入在输入层中的学习单词。这对于词汇表中的每个单词具有一个实值向量，其中每个单词向量具有指定的长度。在这种情况下，我们将使用 10 维投影。输入序列包含单个字，因此 _input_length = 1_ 。

该模型具有单个隐藏的 LSTM 层，具有 50 个单元。这远远超过了需要。输出层由词汇表中每个单词的一个神经元组成，并使用 softmax 激活函数来确保输出被标准化为看起来像概率。

```py
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
```

网络结构可归纳如下：

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 1, 10)             220
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                12200
_________________________________________________________________
dense_1 (Dense)              (None, 22)                1122
=================================================================
Total params: 13,542
Trainable params: 13,542
Non-trainable params: 0
_________________________________________________________________
```

对于本教程中的每个示例，我们将使用相同的通用网络结构，对学习的嵌入层进行微小更改。

接下来，我们可以在编码的文本数据上编译和拟合网络。从技术上讲，我们正在建模一个多类分类问题（预测词汇表中的单词），因此使用分类交叉熵损失函数。我们在每个时代结束时使用有效的 Adam 实现梯度下降和跟踪精度。该模型适用于 500 个训练时期，也许比需要更多。

网络配置没有针对此和后续实验进行调整;选择了一个过度规定的配置，以确保我们可以专注于语言模型的框架。

```py
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
```

在模型拟合之后，我们通过从词汇表中传递给定的单词并让模型预测下一个单词来测试它。在这里我们通过编码传递'`Jack`'并调用 _model.predict_classes（）_ 来获得预测单词的整数输出。然后在词汇表映射中查找，以提供相关的单词。

```py
# evaluate
in_text = 'Jack'
print(in_text)
encoded = tokenizer.texts_to_sequences([in_text])[0]
encoded = array(encoded)
yhat = model.predict_classes(encoded, verbose=0)
for word, index in tokenizer.word_index.items():
	if index == yhat:
		print(word)
```

然后可以重复该过程几次以建立生成的单词序列。

为了使这更容易，我们将函数包含在一个函数中，我们可以通过传入模型和种子字来调用它。

```py
# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = array(encoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result
```

我们可以把所有这些放在一起。完整的代码清单如下。

```py
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = array(encoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate
print(generate_seq(model, tokenizer, 'Jack', 6))
```

运行该示例打印每个训练时期的损失和准确性。

```py
...
Epoch 496/500
0s - loss: 0.2358 - acc: 0.8750
Epoch 497/500
0s - loss: 0.2355 - acc: 0.8750
Epoch 498/500
0s - loss: 0.2352 - acc: 0.8750
Epoch 499/500
0s - loss: 0.2349 - acc: 0.8750
Epoch 500/500
0s - loss: 0.2346 - acc: 0.8750
```

我们可以看到模型没有记住源序列，可能是因为输入序列中存在一些模糊性，例如：

```py
jack => and
jack => fell
```

等等。

在运行结束时，传入'`Jack`'并生成预测或新序列。

我们得到一个合理的序列作为输出，它有一些源的元素。

```py
Jack and jill came tumbling after down
```

这是一个很好的第一个切割语言模型，但没有充分利用 LSTM 处理输入序列的能力，并通过使用更广泛的上下文消除一些模糊的成对序列的歧义。

## 模型 2：逐行序列

另一种方法是逐行分割源文本，然后将每一行分解为一系列构建的单词。

例如：

```py
X,									y
_, _, _, _, _, Jack, 				and
_, _, _, _, Jack, and 				Jill
_, _, _, Jack, and, Jill,			went
_, _, Jack, and, Jill, went,		up
_, Jack, and, Jill, went, up,		the
Jack, and, Jill, went, up, the,		hill
```

这种方法可以允许模型在一个简单的单字输入和输出模型产生歧义的情况下使用每一行的上下文来帮助模型。

在这种情况下，这是以跨行预测单词为代价的，如果我们只对建模和生成文本行感兴趣，那么现在可能没问题。

请注意，在此表示中，我们将需要填充序列以确保它们满足固定长度输入。这是使用 Keras 时的要求。

首先，我们可以使用已经适合源文本的 Tokenizer 逐行创建整数序列。

```py
# create line-based sequences
sequences = list()
for line in data.split('\n'):
	encoded = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(encoded)):
		sequence = encoded[:i+1]
		sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
```

接下来，我们可以填充准备好的序列。我们可以使用 Keras 中提供的 [pad_sequences（）](https://keras.io/preprocessing/sequence/#pad_sequences)函数来完成此操作。这首先涉及找到最长的序列，然后使用它作为填充所有其他序列的长度。

```py
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
```

接下来，我们可以将序列拆分为输入和输出元素，就像之前一样。

```py
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
```

然后可以像之前一样定义模型，除了输入序列现在比单个字长。具体来说，它们的长度为 _max_length-1_ ，-1 因为当我们计算序列的最大长度时，它们包括输入和输出元素。

```py
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
```

我们可以像以前一样使用该模型生成新序列。通过在每次迭代中将预测添加到输入词列表中，可以更新`generate_seq()`函数以建立输入序列。

```py
# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text
```

将所有这些结合在一起，下面提供了完整的代码示例。

```py
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for line in data.split('\n'):
	encoded = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(encoded)):
		sequence = encoded[:i+1]
		sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'Jack', 4))
print(generate_seq(model, tokenizer, max_length-1, 'Jill', 4))
```

运行该示例可以更好地适应源数据。添加的上下文允许模型消除一些示例的歧义。

仍有两行文字以“`Jack`”开头，可能仍然是网络的问题。

```py
...
Epoch 496/500
0s - loss: 0.1039 - acc: 0.9524
Epoch 497/500
0s - loss: 0.1037 - acc: 0.9524
Epoch 498/500
0s - loss: 0.1035 - acc: 0.9524
Epoch 499/500
0s - loss: 0.1033 - acc: 0.9524
Epoch 500/500
0s - loss: 0.1032 - acc: 0.9524
```

在运行结束时，我们生成两个具有不同种子词的序列：'`Jack`'和'`Jill`'。

第一个生成的行看起来很好，直接匹配源文本。第二个有点奇怪。这是有道理的，因为网络只在输入序列中看到'`Jill`'，而不是在序列的开头，所以它强制输出使用'`Jill`这个词'，即押韵的最后一行。

```py
Jack fell down and broke
Jill jill came tumbling after
```

这是一个很好的例子，说明框架可能如何产生更好的新线条，但不是良好的部分输入线条。

## 模型 3：双字输入，单字输出序列

我们可以使用单词输入和全句子方法之间的中间，并传入单词的子序列作为输入。

这将在两个框架之间进行权衡，允许生成新线并在中线拾取生成。

我们将使用 3 个单词作为输入来预测一个单词作为输出。序列的准备与第一个示例非常相似，只是源序列数组中的偏移量不同，如下所示：

```py
# encode 2 words -> 1 word
sequences = list()
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
```

下面列出了完整的示例

```py
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 5))
print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))
print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))
```

再次运行示例可以很好地适应源文本，准确度大约为 95％。

```py
...
Epoch 496/500
0s - loss: 0.0685 - acc: 0.9565
Epoch 497/500
0s - loss: 0.0685 - acc: 0.9565
Epoch 498/500
0s - loss: 0.0684 - acc: 0.9565
Epoch 499/500
0s - loss: 0.0684 - acc: 0.9565
Epoch 500/500
0s - loss: 0.0684 - acc: 0.9565
```

我们看一下 4 代示例，两个线路起始线和两个起始中线。

```py
Jack and jill went up the hill
And Jill went up the
fell down and broke his crown and
pail of water jack fell down and
```

第一次启动行案例正确生成，但第二次没有生成。第二种情况是第 4 行的一个例子，它与第一行的内容含糊不清。也许进一步扩展到 3 个输入单词会更好。

正确生成了两个中线生成示例，与源文本匹配。

我们可以看到，语言模型的框架选择以及模型的使用要求必须兼容。一般情况下使用语言模型时需要仔细设计，或许通过序列生成进行现场测试，以确认模型要求已得到满足。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **全韵序列**。考虑更新上述示例中的一个以构建整个押韵作为输入序列。该模型应该能够在给定第一个单词的种子的情况下生成整个事物，并证明这一点。
*   **预训练嵌入**。在嵌入中使用预先训练的单词向量进行探索，而不是将嵌入作为模型的一部分进行学习。这样一个小的源文本不需要这样做，但可能是一个好习惯。
*   **角色模型**。探索使用基于字符的语言模型来源文本而不是本教程中演示的基于单词的方法。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [杰克和吉尔在维基百科](https://en.wikipedia.org/wiki/Jack_and_Jill_(nursery_rhyme))
*   维基百科上的[语言模型](https://en.wikipedia.org/wiki/Language_model)
*   [Keras 嵌入层 API](https://keras.io/layers/embeddings/#embedding)
*   [Keras 文本处理 API](https://keras.io/preprocessing/text/)
*   [Keras 序列处理 API](https://keras.io/preprocessing/sequence/)
*   [Keras Utils API](https://keras.io/utils/)

## 摘要

在本教程中，您了解了如何为简单的童谣开发不同的基于单词的语言模型。

具体来说，你学到了：

*   为给定的应用程序开发基于单词的语言模型的良好框架的挑战。
*   如何为基于单词的语言模型开发单字，双字和基于行的框架。
*   如何使用拟合语言模型生成序列。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。