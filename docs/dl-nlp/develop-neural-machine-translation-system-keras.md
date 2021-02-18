# 如何从零开始开发神经机器翻译系统

> 原文： [https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)

#### 自动开发深度学习模型
使用 Keras 逐步将 Python 从德语翻译成英语。

机器翻译是一项具有挑战性的任务，传统上涉及使用高度复杂的语言知识开发的大型统计模型。

[神经机器翻译](https://machinelearningmastery.com/introduction-neural-machine-translation/)是利用深度神经网络解决机器翻译问题。

在本教程中，您将了解如何开发用于将德语短语翻译成英语的神经机器翻译系统。

完成本教程后，您将了解：

*   如何清理和准备数据准备训练神经机器翻译系统。
*   如何开发机器翻译的编码器 - 解码器模型。
*   如何使用训练有素的模型推断新的输入短语并评估模型技巧。

让我们开始吧。

**注**：摘录自：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。
看一下，如果你想要更多的分步教程，在使用文本数据时充分利用深度学习方法。

![How to Develop a Neural Machine Translation System in Keras](img/0f92cb4ebcdf0a35d478ceb006527e87.jpg)

如何在 Keras
中开发神经机器翻译系统[BjörnGroß](https://www.flickr.com/photos/damescalito/34527830324/)，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  德语到英语翻译数据集
2.  准备文本数据
3.  训练神经翻译模型
4.  评估神经翻译模型

### Python 环境

本教程假定您已安装 Python 3 SciPy 环境。

您必须安装带有 TensorFlow 或 Theano 后端的 Keras（2.0 或更高版本）。

本教程还假设您已安装 NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

这样的教程不需要 GPU，但是，您可以在 Amazon Web Services 上廉价地访问 GPU。在本教程中学习如何：

*   [如何设置 Amazon AWS EC2 GPU 以训练 Keras 深度学习模型（循序渐进）](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

让我们潜入。

## 德语到英语翻译数据集

在本教程中，我们将使用德语到英语术语的数据集作为语言学习的抽认卡的基础。

该数据集可从 [ManyThings.org](http://www.manythings.org) 网站获得，其中的例子来自 [Tatoeba Project](http://tatoeba.org/home) 。该数据集由德语短语及其英语对应组成，旨在与 [Anki 闪卡软件](https://apps.ankiweb.net/)一起使用。

该页面提供了许多语言对的列表，我建议您探索其他语言：

*   [制表符分隔的双语句子对](http://www.manythings.org/anki/)

我们将在本教程中使用的数据集可在此处下载：

*   [德语 - 英语 deu-eng.zip](http://www.manythings.org/anki/deu-eng.zip)

将数据集下载到当前工作目录并解压缩;例如：

```py
unzip deu-eng.zip
```

您将拥有一个名为 _deu.txt_ 的文件，其中包含 152,820 对英语到德语阶段，每行一对，并带有分隔语言的选项卡。

例如，文件的前 5 行如下所示：

```py
Hi.	Hallo!
Hi.	Grüß Gott!
Run!	Lauf!
Wow!	Potzdonner!
Wow!	Donnerwetter!
```

我们将预测问题框定为德语中的一系列单词作为输入，翻译或预测英语单词的序列。

我们将开发的模型适用于一些初学德语短语。

## 准备文本数据

下一步是准备好文本数据以进行建模。

如果您不熟悉清理文本数据，请参阅此帖子：

*   [如何使用 Python 清理机器学习文本](https://machinelearningmastery.com/clean-text-machine-learning-python/)

查看原始数据并记下您在数据清理操作中可能需要处理的内容。

例如，以下是我在审核原始数据时注意到的一些观察结果：

*   有标点符号。
*   该文本包含大写和小写。
*   德语中有特殊字符。
*   英语中有重复的短语，德语有不同的翻译。
*   文件按句子长度排序，文件末尾有很长的句子。

你有没有注意到其他重要的事情？
请在下面的评论中告诉我。

良好的文本清理程序可以处理这些观察中的一些或全部。

数据准备分为两个小节：

1.  干净的文字
2.  拆分文字

### 1.清洁文字

首先，我们必须以保留 Unicode 德语字符的方式加载数据。下面的函数 _load_doc（）_ 将把文件加载为一团文本。

```py
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
```

每行包含一对短语，首先是英语，然后是德语，由制表符分隔。

我们必须逐行拆分加载的文本，然后按短语拆分。下面的函数 _to_pairs（）_ 将拆分加载的文本。

```py
# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs
```

我们现在准备清理每一句话。我们将执行的具体清洁操作如下：

*   删除所有不可打印的字符。
*   删除所有标点字符。
*   将所有 Unicode 字符规范化为 ASCII（例如拉丁字符）。
*   将案例规范化为小写。
*   删除任何非字母的剩余令牌。

我们将对加载的数据集中每对的每个短语执行这些操作。

下面的 _clean_pairs（）_ 函数实现了这些操作。

```py
# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)
```

最后，既然已经清理了数据，我们可以将短语对列表保存到准备使用的文件中。

函数 _save_clean_data（）_ 使用 pickle API 将干净文本列表保存到文件中。

将所有这些结合在一起，下面列出了完整的示例。

```py
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
```

运行该示例在当前工作目录中创建一个新文件，其中包含名为 _english-german.pkl_ 的已清理文本。

打印清洁文本的一些示例供我们在运行结束时进行评估，以确认清洁操作是按预期执行的。

```py
[hi] => [hallo]
[hi] => [gru gott]
[run] => [lauf]
[wow] => [potzdonner]
[wow] => [donnerwetter]
[fire] => [feuer]
[help] => [hilfe]
[help] => [zu hulf]
[stop] => [stopp]
[wait] => [warte]
...
```

### 2.分割文字

干净的数据包含超过 150,000 个短语对，并且文件末尾的一些对非常长。

这是开发小型翻译模型的大量示例。模型的复杂性随着示例的数量，短语的长度和词汇的大小而增加。

虽然我们有一个很好的数据集用于建模翻译，但我们会稍微简化问题，以大幅减少所需模型的大小，进而缩短适合模型所需的训练时间。

您可以探索在更全面的数据集上开发模型作为扩展;我很想听听你的表现。

我们将通过将数据集减少到文件中的前 10,000 个示例来简化问题;这些将是数据集中最短的短语。

此外，我们将把前 9,000 个作为训练示例，其余 1,000 个例子用于测试拟合模型。

下面是加载干净数据，拆分数据并将数据拆分部分保存到新文件的完整示例。

```py
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')
```

运行该示例将创建三个新文件： _english-german-both.pkl_ ，其中包含我们可用于定义问题参数的所有训练和测试示例，例如最大短语长度和词汇，以及训练和测试数据集的 _english-german-train.pkl_ 和 _english-german-test.pkl_ 文件。

我们现在准备开始开发我们的翻译模型。

## 训练神经翻译模型

在本节中，我们将开发神经翻译模型。

如果您不熟悉神经翻译模型，请参阅帖子：

*   [神经机器翻译的温和介绍](https://machinelearningmastery.com/introduction-neural-machine-translation/)

这涉及加载和准备准备好建模的清洁文本数据，以及在准备好的数据上定义和训练模型。

让我们从加载数据集开始，以便我们可以准备数据。以下名为 _load_clean_sentences（）_ 的函数可用于依次加载 train，test 和两个数据集。

```py
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
```

我们将使用“两者”或训练和测试数据集的组合来定义问题的最大长度和词汇。

这是为了简单起见。或者，我们可以单独从训练数据集定义这些属性，并截断测试集中的例子，这些例子太长或者词汇不在词汇表中。

我们可以根据建模需要使用 Keras`Tokenize`类将单词映射到整数。我们将为英语序列和德语序列使用单独的分词器。下面命名为 _create_tokenizer（）_ 的函数将在短语列表上训练一个分词器。

```py
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

类似地，下面名为 _max_length（）_ 的函数将找到短语列表中最长序列的长度。

```py
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
```

我们可以使用组合数据集调用这些函数来为英语和德语短语准备标记符，词汇表大小和最大长度。

```py
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))
```

我们现在准备准备训练数据集。

每个输入和输出序列必须编码为整数并填充到最大短语长度。这是因为我们将对输入序列使用字嵌入，并对输出序列进行热编码。以下名为 _encode_sequences（）_ 的函数将执行这些操作并返回结果。

```py
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
```

输出序列需要进行单热编码。这是因为模型将预测词汇表中每个单词作为输出的概率。

下面的函数 _encode_output（）_ 将对英文输出序列进行单热编码。

```py
# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
```

我们可以利用这两个函数并准备训练模型的训练和测试数据集。

```py
# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)
```

我们现在准备定义模型。

我们将在这个问题上使用编码器 - 解码器 LSTM 模型。在这种架构中，输入序列由称为编码器的前端模型编码，然后由称为解码器的后端模型逐字解码。

下面的函数 _define_model（）_ 定义了模型，并采用了许多用于配置模型的参数，例如输入和输出词汇的大小，输入和输出短语的最大长度以及数字用于配置模型的内存单元。

该模型使用有效的 Adam 方法训练随机梯度下降并最小化分类损失函数，因为我们将预测问题框定为多类分类。

模型配置未针对此问题进行优化，这意味着您有足够的机会对其进行调整并提升翻译技能。我很想看看你能想出什么。

有关配置神经机器翻译模型的更多建议，请参阅帖子：

*   [如何为神经机器翻译配置编码器 - 解码器模型](https://machinelearningmastery.com/configure-encoder-decoder-model-neural-machine-translation/)

```py
# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
```

最后，我们可以训练模型。

我们训练了 30 个时期的模型和 64 个样本的批量大小。

我们使用检查点来确保每次测试集上的模型技能得到改进时，模型都会保存到文件中。

```py
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
```

我们可以将所有这些结合在一起并适合神经翻译模型。

完整的工作示例如下所示。

```py
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
```

首先运行该示例将打印数据集参数的摘要，例如词汇表大小和最大短语长度。

```py
English Vocabulary Size: 2404
English Max Length: 5
German Vocabulary Size: 3856
German Max Length: 10
```

接下来，打印已定义模型的摘要，允许我们确认模型配置。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 10, 256)           987136
_________________________________________________________________
lstm_1 (LSTM)                (None, 256)               525312
_________________________________________________________________
repeat_vector_1 (RepeatVecto (None, 5, 256)            0
_________________________________________________________________
lstm_2 (LSTM)                (None, 5, 256)            525312
_________________________________________________________________
time_distributed_1 (TimeDist (None, 5, 2404)           617828
=================================================================
Total params: 2,655,588
Trainable params: 2,655,588
Non-trainable params: 0
_________________________________________________________________
```

还创建了模型图，提供了模型配置的另一个视角。

![Plot of Model Graph for NMT](img/4319d3f70ffb87578d739d5d94a563f7.jpg)

NMT 模型图的图

接下来，训练模型。

在现代 CPU 硬件上，每个时期大约需要 30 秒;不需要 GPU。

在运行期间，模型将保存到文件 _model.h5_ ，准备在下一步中进行推理。

```py
...
Epoch 26/30
Epoch 00025: val_loss improved from 2.20048 to 2.19976, saving model to model.h5
17s - loss: 0.7114 - val_loss: 2.1998
Epoch 27/30
Epoch 00026: val_loss improved from 2.19976 to 2.18255, saving model to model.h5
17s - loss: 0.6532 - val_loss: 2.1826
Epoch 28/30
Epoch 00027: val_loss did not improve
17s - loss: 0.5970 - val_loss: 2.1970
Epoch 29/30
Epoch 00028: val_loss improved from 2.18255 to 2.17872, saving model to model.h5
17s - loss: 0.5474 - val_loss: 2.1787
Epoch 30/30
Epoch 00029: val_loss did not improve
17s - loss: 0.5023 - val_loss: 2.1823
```

## 评估神经翻译模型

我们将评估训练上的模型和测试数据集。

该模型应该在训练数据集上表现很好，并且理想情况下已被推广以在测试数据集上表现良好。

理想情况下，我们将使用单独的验证数据集来帮助在训练期间选择模型而不是测试集。您可以尝试将其作为扩展名。

必须像以前一样加载和准备干净的数据集。

```py
...
# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
```

接下来，必须加载训练期间保存的最佳模型。

```py
# load model
model = load_model('model.h5')
```

评估涉及两个步骤：首先生成翻译的输出序列，然后针对许多输入示例重复此过程，并在多个案例中总结模型的技能。

从推理开始，模型可以以一次性方式预测整个输出序列。

```py
translation = model.predict(source, verbose=0)
```

这将是一个整数序列，我们可以在 tokenizer 中枚举和查找以映射回单词。

以下函数名为 _word_for_id（）_，将执行此反向映射。

```py
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
```

我们可以为转换中的每个整数执行此映射，并将结果作为一个单词串返回。

下面的函数 _predict_sequence（）_ 对单个编码的源短语执行此操作。

```py
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
```

接下来，我们可以对数据集中的每个源短语重复此操作，并将预测结果与英语中的预期目标短语进行比较。

我们可以将这些比较中的一些打印到屏幕上，以了解模型在实践中的表现。

我们还将计算 BLEU 分数，以获得模型表现良好的定量概念。

您可以在此处了解有关 BLEU 分数的更多信息：

*   [计算 Python 中文本的 BLEU 分数的温和介绍](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)

下面的 _evaluate_model（）_ 函数实现了这一点，为提供的数据集中的每个短语调用上述 _predict_sequence（）_ 函数。

```py
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
```

我们可以将所有这些结合在一起，并在训练和测试数据集上评估加载的模型。

完整的代码清单如下。

```py
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

# load model
model = load_model('model.h5')
# test on some training sequences
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)
```

首先运行示例打印源文本，预期和预测翻译的示例，以及训练数据集的分数，然后是测试数据集。

鉴于数据集的随机改组和神经网络的随机性，您的具体结果会有所不同。

首先查看测试数据集的结果，我们可以看到翻译是可读的并且大部分都是正确的。

例如：“ _ich liebe dich_ ”被正确翻译为“_ 我爱你 _”。

我们还可以看到翻译并不完美，“ _ich konnte nicht gehen_ ”翻译为“_ 我不能 _”而不是预期的“_ 我无法行走 _ ]“。

我们还可以看到 BLEU-4 得分为 0.51，它提供了我们对此模型的预期上限。

```py
src=[ich liebe dich], target=[i love you], predicted=[i love you]
src=[ich sagte du sollst den mund halten], target=[i said shut up], predicted=[i said stop up]
src=[wie geht es eurem vater], target=[hows your dad], predicted=[hows your dad]
src=[das gefallt mir], target=[i like that], predicted=[i like that]
src=[ich gehe immer zu fu], target=[i always walk], predicted=[i will to]
src=[ich konnte nicht gehen], target=[i couldnt walk], predicted=[i cant go]
src=[er ist sehr jung], target=[he is very young], predicted=[he is very young]
src=[versucht es doch einfach], target=[just try it], predicted=[just try it]
src=[sie sind jung], target=[youre young], predicted=[youre young]
src=[er ging surfen], target=[he went surfing], predicted=[he went surfing]

BLEU-1: 0.085682
BLEU-2: 0.284191
BLEU-3: 0.459090
BLEU-4: 0.517571
```

查看测试集上的结果，确实看到可读的翻译，这不是一件容易的事。

例如，我们看到“ _ich mag dich nicht_ ”正确翻译为“_ 我不喜欢你 _”。

我们还看到一些不良的翻译以及该模型可能受到进一步调整的好例子，例如“ _ich bin etwas beschwipst_ ”翻译为“ _ia bit bit_ ”而不是预期“_ 我有点醉了 _”

BLEU-4 得分为 0.076238，提供了基线技能，可以进一步改进模型。

```py
src=[tom erblasste], target=[tom turned pale], predicted=[tom went pale]
src=[bring mich nach hause], target=[take me home], predicted=[let us at]
src=[ich bin etwas beschwipst], target=[im a bit tipsy], predicted=[i a bit bit]
src=[das ist eine frucht], target=[its a fruit], predicted=[thats a a]
src=[ich bin pazifist], target=[im a pacifist], predicted=[im am]
src=[unser plan ist aufgegangen], target=[our plan worked], predicted=[who is a man]
src=[hallo tom], target=[hi tom], predicted=[hello tom]
src=[sei nicht nervos], target=[dont be nervous], predicted=[dont be crazy]
src=[ich mag dich nicht], target=[i dont like you], predicted=[i dont like you]
src=[tom stellte eine falle], target=[tom set a trap], predicted=[tom has a cough]

BLEU-1: 0.082088
BLEU-2: 0.006182
BLEU-3: 0.046129
BLEU-4: 0.076238
```

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **数据清理**。可以对数据执行不同的数据清理操作，例如不删除标点符号或标准化案例，或者可能删除重复的英语短语。
*   **词汇**。可以改进词汇表，可能删除在数据集中使用少于 5 或 10 次的单词并替换为“`unk`”。
*   **更多数据**。用于拟合模型的数据集可以扩展到 50,000,1000 个短语或更多。
*   **输入订单**。输入短语的顺序可以颠倒，据报道可提升技能，或者可以使用双向输入层。
*   **层**。编码器和/或解码器模型可以通过附加层进行扩展，并针对更多时期进行训练，从而为模型提供更多的代表表现力。
*   **单位**。可以增加编码器和解码器中的存储器单元的数量，从而为模型提供更多的代表性容量。
*   **正规化**。该模型可以使用正则化，例如权重或激活正则化，或在 LSTM 层上使用压差。
*   **预训练的单词向量**。可以在模型中使用预训练的单词向量。
*   **递归模型**。可以使用模型的递归公式，其中输出序列中的下一个字可以以输入序列和到目前为止生成的输出序列为条件。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [制表符分隔的双语句子对](http://www.manythings.org/anki/)
*   [德语 - 英语 deu-eng.zip](http://www.manythings.org/anki/deu-eng.zip)
*   [编码器 - 解码器长短期存储器网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

## 摘要

在本教程中，您了解了如何开发用于将德语短语翻译成英语的神经机器翻译系统。

具体来说，你学到了：

*   如何清理和准备数据准备训练神经机器翻译系统。
*   如何开发机器翻译的编码器 - 解码器模型。
*   如何使用训练有素的模型推断新的输入短语并评估模型技巧。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。

**注**：这篇文章摘录自：“[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。看一下，如果您想要在使用文本数据时获得有关深入学习方法的更多分步教程。