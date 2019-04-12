# 使用 Keras 在 Python 中使用 LSTM 循环神经网络生成文本

> 原文： [https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

递归神经网络也可以用作生成模型。

这意味着除了用于预测模型（进行预测）之外，他们还可以学习问题的序列，然后为问题域生成全新的合理序列。

像这样的生成模型不仅可用于研究模型学习问题的程度，还可以了解有关问题领域本身的更多信息。

在这篇文章中，您将了解如何使用 Keras 中的 Python 中的 LSTM 循环神经网络逐个字符地创建文本的生成模型。

阅读这篇文章后你会知道：

*   在哪里下载免费的文本语料库，您可以使用它来训练文本生成模型。
*   如何将文本序列问题构建为递归神经网络生成模型。
*   如何开发 LSTM 以针对给定问题生成合理的文本序列。

让我们开始吧。

**注意**：LSTM 递归神经网络训练速度很慢，强烈建议您在 GPU 硬件上进行训练。您可以使用 Amazon Web Services 非常便宜地访问云中的 GPU 硬件，[请参阅此处的教程](http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)。

*   **2016 年 10 月更新**：修复了代码中的一些小错误拼写错误。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![Text Generation With LSTM Recurrent Neural Networks in Python with Keras](img/ce1bbf908214dba8ac5ef35fd8c2b3e6.jpg)

用 Keras
在 Python 中使用 LSTM 回归神经网络生成文本 [Russ Sanderlin](https://www.flickr.com/photos/tearstone/5028273685/) ，保留一些权利。

## 问题描述：古腾堡项目

许多经典文本不再受版权保护。

这意味着您可以免费下载这些书籍的所有文本，并在实验中使用它们，例如创建生成模型。也许获取不受版权保护的免费书籍的最佳地点是 [Project Gutenberg](https://www.gutenberg.org) 。

在本教程中，我们将使用童年时代最喜欢的书作为数据集：[刘易斯卡罗尔的爱丽丝梦游仙境](https://www.gutenberg.org/ebooks/11)。

我们将学习字符之间的依赖关系和序列中字符的条件概率，这样我们就可以生成全新的原始字符序列。

这很有趣，我建议用 Project Gutenberg 的其他书重复这些实验，[这里是网站上最受欢迎的书籍列表](https://www.gutenberg.org/ebooks/search/%3Fsort_order%3Ddownloads)。

这些实验不仅限于文本，您还可以尝试其他 ASCII 数据，例如计算机源代码，LaTeX 中标记的文档，HTML 或 Markdown 等。

您可以[免费下载本书的 ASCII 格式](http://www.gutenberg.org/cache/epub/11/pg11.txt)（纯文本 UTF-8）全文，并将其放在工作目录中，文件名为 **wonderland.txt** 。

现在我们需要准备好数据集以进行建模。

Project Gutenberg 为每本书添加了标准页眉和页脚，这不是原始文本的一部分。在文本编辑器中打开文件并删除页眉和页脚。

标题很明显，以文字结尾：

```
*** START OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***
```

页脚是文本行后面的所有文本：

```
THE END
```

您应该留下一个包含大约 3,330 行文本的文本文件。

## 开发小型 LSTM 回归神经网络

在本节中，我们将开发一个简单的 LSTM 网络，以学习 Alice in Wonderland 中的角色序列。在下一节中，我们将使用此模型生成新的字符序列。

让我们首先导入我们打算用来训练模型的类和函数。

```
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
```

接下来，我们需要将书籍的 ASCII 文本加载到内存中，并将所有字符转换为小写，以减少网络必须学习的词汇量。

```
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
```

既然本书已加载，我们必须准备数据以供神经网络建模。我们不能直接对字符进行建模，而是必须将字符转换为整数。

我们可以通过首先在书中创建一组所有不同的字符，然后创建每个字符到唯一整数的映射来轻松完成此操作。

```
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

例如，书中唯一排序的小写字符列表如下：

```
['\n', '\r', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xbb', '\xbf', '\xef']
```

您可以看到，我们可能会删除某些字符以进一步清理数据集，从而减少词汇量并可能改进建模过程。

现在已经加载了本书并准备了映射，我们可以总结数据集。

```
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
```

将代码运行到此点会产生以下输出。

```
Total Characters:  147674
Total Vocab:  47
```

我们可以看到这本书的字符数不到 150,000，当转换为小写时，网络词汇表中只有 47 个不同的字符供网络学习。远远超过字母表中的 26。

我们现在需要定义网络的训练数据。在训练过程中，如何选择拆分文本并将其暴露给网络，有很多灵活性。

在本教程中，我们将书本文本拆分为子序列，其长度固定为 100 个字符，任意长度。我们可以轻松地按句子分割数据并填充较短的序列并截断较长的序列。

网络的每个训练模式由 100 个时间步长组成，一个字符（X）后跟一个字符输出（y）。在创建这些序列时，我们一次一个字符地沿着整本书滑动这个窗口，允许每个角色从它前面的 100 个字符中学习（当然前 100 个字符除外）。

例如，如果序列长度为 5（为简单起见），则前两个训练模式如下：

```
CHAPT -> E
HAPTE -> R
```

当我们将书分成这些序列时，我们使用我们之前准备的查找表将字符转换为整数。

```
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
```

运行代码到这一点向我们展示了当我们将数据集拆分为网络的训练数据时，我们知道我们只有不到 150,000 个训练模式。这有意义，因为排除前 100 个字符，我们有一个训练模式来预测每个剩余的字符。

```
Total Patterns:  147574
```

现在我们已经准备好了训练数据，我们需要对其进行转换，以便它适合与 Keras 一起使用。

首先，我们必须将输入序列列表转换为 LSTM 网络所期望的 _[样本，时间步长，特征]_ 形式。

接下来，我们需要将整数重新缩放到 0 到 1 的范围，以使默认情况下使用 sigmoid 激活函数的 LSTM 网络更容易学习模式。

最后，我们需要将输出模式（转换为整数的单个字符）转换为一个热编码。这样我们就可以配置网络来预测词汇表中 47 个不同字符中每个字符的概率（更容易表示），而不是试图强制它准确地预测下一个字符。每个 y 值都被转换为一个长度为 47 的稀疏向量，除了在模式所代表的字母（整数）的列中有 1 之外，它们都是零。

例如，当“n”（整数值 31）是一个热编码时，它看起来如下：

```
[ 0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0.
  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  1\.  0\.  0\.  0\.  0.
  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0.]
```

我们可以执行以下步骤。

```
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
```

我们现在可以定义我们的 LSTM 模型。在这里，我们定义了一个具有 256 个内存单元的隐藏 LSTM 层。网络使用概率为 20 的丢失。输出层是密集层，使用 softmax 激活函数输出 0 和 1 之间的 47 个字符中的每一个的概率预测。

问题实际上是 47 个类的单个字符分类问题，因此被定义为优化日志损失（交叉熵），这里使用 ADAM 优化算法来提高速度。

```
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

没有测试数据集。我们正在对整个训练数据集进行建模，以了解序列中每个字符的概率。

我们对训练数据集的最准确（分类准确性）模型不感兴趣。这将是一个完美预测训练数据集中每个角色的模型。相反，我们感兴趣的是最小化所选损失函数的数据集的概括。我们正在寻求在泛化和过度拟合之间取得平衡，但缺乏记忆。

网络训练缓慢（Nvidia K520 GPU 上每个迭代约 300 秒）。由于速度缓慢以及由于我们的优化要求，我们将使用模型检查点来记录每次在时期结束时观察到损失改善时的所有网络权重。我们将在下一节中使用最佳权重集（最低损失）来实例化我们的生成模型。

```
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
```

我们现在可以将模型与数据相匹配。在这里，我们使用适度数量的 20 个时期和 128 个模式的大批量大小。

```
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
```

完整性代码清单如下所示。

```
# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
```

由于模型的随机性，您将看到不同的结果，并且因为很难为 LSTM 模型修复随机种子以获得 100％可重复的结果。这不是这个生成模型的关注点。

运行该示例后，您应该在本地目录中有许多权重检查点文件。

除了丢失值最小的那个之外，您可以删除它们。例如，当我运行这个例子时，下面是我实现的损失最小的检查点。

```
weights-improvement-19-1.9435.hdf5
```

网络损失几乎每个时代都在减少，我预计网络可以从更多时代的训练中受益。

在下一节中，我们将介绍如何使用此模型生成新的文本序列。

## 使用 LSTM 网络生成文本

使用经过训练的 LSTM 网络生成文本相对简单。

首先，我们以完全相同的方式加载数据并定义网络，除了从检查点文件加载网络权重并且不需要训练网络。

```
# load the network weights
filename = "weights-improvement-19-1.9435.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

此外，在准备将唯一字符映射到整数时，我们还必须创建一个反向映射，我们可以使用它将整数转换回字符，以便我们可以理解预测。

```
int_to_char = dict((i, c) for i, c in enumerate(chars))
```

最后，我们需要实际做出预测。

使用 Keras LSTM 模型进行预测的最简单方法是首先以种子序列作为输入开始，生成下一个字符然后更新种子序列以在末尾添加生成的字符并修剪第一个字符。只要我们想要预测新字符（例如，长度为 1,000 个字符的序列），就重复该过程。

我们可以选择随机输入模式作为种子序列，然后在生成它们时打印生成的字符。

```
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
```

下面列出了使用加载的 LSTM 模型生成文本的完整代码示例，以确保完整性。

```
# Load LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-19-1.9435.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
```

运行此示例首先输出所选的随机种子，然后输出生成的每个字符。

例如，下面是此文本生成器的一次运行的结果。随机种子是：

```
be no mistake about it: it was neither more nor less than a pig, and she
felt that it would be quit
```

随机种子生成的文本（清理后用于演示）是：

```
be no mistake about it: it was neither more nor less than a pig, and she
felt that it would be quit e aelin that she was a little want oe toiet
ano a grtpersent to the tas a little war th tee the tase oa teettee
the had been tinhgtt a little toiee at the cadl in a long tuiee aedun
thet sheer was a little tare gereen to be a gentle of the tabdit  soenee
the gad  ouw ie the tay a tirt of toiet at the was a little 
anonersen, and thiu had been woite io a lott of tueh a tiie  and taede
bot her aeain  she cere thth the bene tith the tere bane to tee
toaete to tee the harter was a little tire the same oare cade an anl ano
the garee and the was so seat the was a little gareen and the sabdit,
and the white rabbit wese tilel an the caoe and the sabbit se teeteer,
and the white rabbit wese tilel an the cade in a lonk tfne the sabdi
ano aroing to tea the was sf teet whitg the was a little tane oo thete
the sabeit  she was a little tartig to the tar tf tee the tame of the
cagd, and the white rabbit was a little toiee to be anle tite thete ofs
and the tabdit was the wiite rabbit, and
```

我们可以注意到有关生成文本的一些观察。

*   它通常符合原始文本中观察到的行格式，在新行之前少于 80 个字符。
*   字符被分成单词组，大多数组是实际的英语单词（例如“the”，“little”和“was”），但许多组不是（例如“lott”，“tiie”和“taede”）。
*   顺序中的一些词是有意义的（例如“_ 和白兔 _”），但许多词没有（例如“ _wese tilel_ ”）。

这本基于角色的本书模型产生这样的输出这一事实令人印象深刻。它让您了解 LSTM 网络的学习能力。

结果并不完美。在下一节中，我们将通过开发更大的 LSTM 网络来提高结果的质量。

## 更大的 LSTM 递归神经网络

我们得到了结果，但在上一节中没有出色的结果。现在，我们可以尝试通过创建更大的网络来提高生成文本的质量。

我们将内存单元的数量保持为 256，但添加第二层。

```
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

我们还将更改检查点权重的文件名，以便我们可以区分此网络和之前的权重（通过在文件名中附加“更大”一词）。

```
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
```

最后，我们将训练时期的数量从 20 个增加到 50 个，并将批量大小从 128 个减少到 64 个，以便为网络提供更多的机会进行更新和学习。

完整代码清单如下所示。

```
# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
```

运行此示例需要一些时间，每个时期至少 700 秒。

运行此示例后，您可能会损失大约 1.2。例如，我通过运行此模型获得的最佳结果存储在一个名称为的检查点文件中：

```
weights-improvement-47-1.2219-bigger.hdf5
```

在 47 迭代实现亏损 1.2219。

与上一节一样，我们可以使用运行中的最佳模型来生成文本。

我们需要对上一节中的文本生成脚本进行的唯一更改是在网络拓扑的规范中以及从哪个文件中为网络权重设定种子。

完整性代码清单如下所示。

```
# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-47-1.2219-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
```

运行此文本生成脚本的一个示例生成下面的输出。

随机选择的种子文本是：

```
d herself lying on the bank, with her
head in the lap of her sister, who was gently brushing away s
```

生成的文本与种子（清理用于演示）是：

```
herself lying on the bank, with her
head in the lap of her sister, who was gently brushing away
so siee, and she sabbit said to herself and the sabbit said to herself and the sood
way of the was a little that she was a little lad good to the garden,
and the sood of the mock turtle said to herself, 'it was a little that
the mock turtle said to see it said to sea it said to sea it say it
the marge hard sat hn a little that she was so sereated to herself, and
she sabbit said to herself, 'it was a little little shated of the sooe
of the coomouse it was a little lad good to the little gooder head. and
said to herself, 'it was a little little shated of the mouse of the
good of the courte, and it was a little little shated in a little that
the was a little little shated of the thmee said to see it was a little
book of the was a little that she was so sereated to hare a little the
began sitee of the was of the was a little that she was so seally and
the sabbit was a little lad good to the little gooder head of the gad
seared to see it was a little lad good to the little good
```

我们可以看到，通常拼写错误较少，文本看起来更逼真，但仍然是非常荒谬的。

例如，相同的短语一次又一次地重复，例如“_ 对自己说 _”和“_ 少 _”。行情已经开启但尚未平仓。

这些都是更好的结果，但仍有很大的改进空间。

## 改进模型的 10 个扩展思路

以下是可以进一步改进您可以尝试的模型的 10 个想法：

*   预测少于 1,000 个字符作为给定种子的输出。
*   从源文本中删除所有标点符号，从而从模型的词汇表中删除。
*   尝试对输入序列进行热编码。
*   在填充句子而不是随机字符序列上训练模型。
*   将训练时期的数量增加到 100 或数百。
*   将 dropout 添加到可见输入图层并考虑调整丢失百分比。
*   调整批量大小，尝试批量大小为 1 作为（非常慢）基线，并从那里开始更大的尺寸。
*   向层和/或更多层添加更多内存单元。
*   在解释预测概率时，对比例因子（[温度](https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning)）进行实验。
*   将 LSTM 图层更改为“有状态”以维护批次之间的状态。

你尝试过这些扩展吗？在评论中分享您的结果。

## 资源

该字符文本模型是使用递归神经网络生成文本的流行方式。

如果您有兴趣深入了解，下面是一些关于该主题的更多资源和教程。也许最受欢迎的是 Andrej Karpathy 的教程，题为“[回归神经网络的不合理效力](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)”。

*   [使用递归神经网络生成文本](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf) [pdf]，2011
*   [用于文本生成的 LSTM 的 Keras 代码示例](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)。
*   [用于文本生成的 LSTM 的烤宽面条代码示例](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py)。
*   [MXNet 教程，用于使用 LSTM 进行文本生成](http://mxnetjl.readthedocs.io/en/latest/tutorial/char-lstm.html)。
*   [使用递归神经网络](https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/)自动生成 Clickbait。

## 摘要

在这篇文章中，您了解了如何使用 Keras 深度学习库开发用于 Python 文本生成的 LSTM 循环神经网络。

阅读这篇文章后你知道：

*   在哪里可以免费下载经典书籍的 ASCII 文本，以便进行训练。
*   如何在文本序列上训练 LSTM 网络以及如何使用训练有素的网络生成新序列。
*   如何开发堆叠 LSTM 网络并提升模型的表现。

您对 LSTM 网络或此帖子的文本生成有任何疑问吗？在下面的评论中提出您的问题，我会尽力回答。