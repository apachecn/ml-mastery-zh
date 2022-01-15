# 如何开发一个单词级神经语言模型并用它来生成文本

> 原文： [https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)

语言模型可以基于序列中已经观察到的单词来预测序列中下一个单词的概率。

神经网络模型是开发统计语言模型的首选方法，因为它们可以使用分布式表示，其中具有相似含义的不同单词具有相似的表示，并且因为它们在做出预测时可以使用最近观察到的单词的大的上下文。

在本教程中，您将了解如何使用 Python 中的深度学习开发统计语言模型。

完成本教程后，您将了解：

*   如何准备文本以开发基于单词的语言模型。
*   如何设计和拟合具有学习嵌入和 LSTM 隐藏层的神经语言模型。
*   如何使用学习的语言模型生成具有与源文本类似的统计属性的新文本。

让我们开始吧。

*   **Update Apr / 2018** ：修正了模型描述中 100 个输入字与实际模型中 50 个输入字之间的不匹配。

![How to Develop a Word-Level Neural Language Model and Use it to Generate Text](img/d0ee6a8a9a2fc43e0ab26fc902881264.jpg)

如何开发一个单词级神经语言模型并用它来生成文本
照片由 [Carlo Raso](https://www.flickr.com/photos/70125105@N06/32512473990/) 拍摄，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  柏拉图共和国
2.  数据准备
3.  训练语言模型
4.  使用语言模型

## 柏拉图共和国

[共和国](https://en.wikipedia.org/wiki/Republic_(Plato))是古典希腊哲学家柏拉图最着名的作品。

它被构建为关于城市国家内秩序和正义主题的对话（例如对话）

整个文本在公共领域免费提供。它可以在 [Project Gutenberg 网站](https://www.gutenberg.org/ebooks/1497)上以多种格式获得。

您可以在此处下载整本书（或书籍）的 ASCII 文本版本：

*   [下载柏拉图共和国](http://www.gutenberg.org/cache/epub/1497/pg1497.txt)

下载书籍文本并将其直接放在当前工作中，文件名为“`republic.txt`”

在文本编辑器中打开文件并删除前后问题。这包括开头的书籍详细信息，长篇分析以及最后的许可证信息。

案文应以：

> 书 I.
> 
> 我昨天和阿里斯顿的儿子格劳孔一起去了比雷埃夫斯，
> ......

结束

> ......
> 在这一生和我们一直描述的千年朝圣中，我们都应该好好相处。

将清理后的版本保存为' _republic_clean。当前工作目录中的 _ txt'。该文件应该是大约 15,802 行文本。

现在我们可以从这个文本开发一个语言模型。

## 数据准备

我们将从准备建模数据开始。

第一步是查看数据。

### 查看文本

在编辑器中打开文本，然后查看文本数据。

例如，这是第一个对话框：

> 书 I.
> 
> 我昨天和阿里斯顿的儿子 Glaucon 一起去了比雷埃夫斯，
> 我可以向女神祈祷（Bendis，Thracian
> Artemis。）;而且因为我想看看他们以什么样的方式
> 庆祝这个节日，这是一个新事物。我很满意居民的
> 游行;但是色雷斯人的情况也是如此，
> 即使不是更多，也是美丽的。当我们完成祈祷并观看
> 景观时，我们转向了城市的方向;在那一瞬间
> Cephalus 的儿子 Polemarchus 偶然在我们回家的路上从
> 的距离看到了我们，并告诉他的仆人
> 跑去让我们等他。仆人背后披着斗篷
> 抓住我，并说：Polemarchus 希望你等。
> 
> 我转过身，问他的主人在哪里。
> 
> 他说，如果你只是等待，那么他就是那个年轻人。
> 
> 当然，我们会，Glaucon 说。几分钟后，Polemarchus
> 出现了，并与他一起出现了 Glaucon 的兄弟 Adeimantus，Nicias 的儿子 Niceratus 以及其他几位参加过游行的人。
> 
> Polemarchus 对我说：我认为，苏格拉底，你和你的
> 同伴已经在前往城市的路上。
> 
> 我说，你没错。
> 
> ...

您认为我们在准备数据时需要处理什么？

以下是我从快速浏览中看到的内容：

*   书/章标题（例如“BOOK I.”）。
*   英国英语拼写（例如“荣幸”）
*   标点符号很多（例如“ - ”，“; - ”，“？ - ”等）
*   奇怪的名字（例如“Polemarchus”）。
*   一些漫长的独白，持续数百行。
*   一些引用的对话框（例如'...'）

这些观察以及更多建议以我们可能希望准备文本数据的方式提出。

我们准备数据的具体方式实际上取决于我们打算如何对其进行建模，而这又取决于我们打算如何使用它。

### 语言模型设计

在本教程中，我们将开发一个文本模型，然后我们可以使用它来生成新的文本序列。

语言模型将是统计的，并且将预测给定输入文本序列的每个单词的概率。预测的单词将作为输入输入，进而生成下一个单词。

关键的设计决策是输入序列应该有多长。它们需要足够长以允许模型学习要预测的单词的上下文。此输入长度还将定义在使用模型时用于生成新序列的种子文本的长度。

没有正确的答案。有了足够的时间和资源，我们就可以探索模型用不同大小的输入序列学习的能力。

相反，我们将为输入序列的长度选择 50 个字的长度，有点任意。

我们可以处理数据，以便模型只处理自包含的句子并填充或截断文本以满足每个输入序列的这一要求。您可以将此作为本教程的扩展进行探索。

相反，为了使示例保持简洁，我们将让所有文本一起流动并训练模型以预测文本中句子，段落甚至书籍或章节中的下一个单词。

现在我们有一个模型设计，我们可以看看将原始文本转换为 50 个输入字到 1 个输出字的序列，准备好适合模型。

### 加载文字

第一步是将文本加载到内存中。

我们可以开发一个小函数来将整个文本文件加载到内存中并返回它。该函数名为 _load_doc（）_，如下所示。给定文件名，它返回一个加载文本序列。

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

使用此函数，我们可以在文件'`republic_clean.txt`'中加载文档的清洁版本，如下所示：

```py
# load document
in_filename = 'republic_clean.txt'
doc = load_doc(in_filename)
print(doc[:200])
```

运行此代码段会加载文档并打印前 200 个字符作为完整性检查。

> 书 I.
> 
> 我昨天和阿里斯顿的儿子 Glaucon 一起去了比雷埃夫斯，
> 我可以向女神祈祷（Bendis，Thracian
> Artemis。）;还因为我想知道什么

到现在为止还挺好。接下来，让我们清理文本。

### 干净的文字

我们需要将原始文本转换为一系列令牌或单词，我们可以将其用作训练模型的源。

基于查看原始文本（上文），下面是我们将执行的一些特定操作来清理文本。您可能希望自己探索更多清洁操作作为扩展。

*   将' - '替换为空格，以便我们可以更好地分割单词。
*   基于空白区域的分词。
*   从单词中删除所有标点符号以减少词汇量大小（例如'What？'变为'What'）。
*   删除所有非字母的单词以删除独立的标点符号。
*   将所有单词标准化为小写以减少词汇量。

词汇量大小与语言建模有很大关系。较小的词汇量会导致较小的模型更快地训练。

我们可以在一个函数中按此顺序实现每个清理操作。下面是函数 _clean_doc（）_，它将加载的文档作为参数并返回一个干净的标记数组。

```py
import string

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens
```

我们可以在加载的文档上运行此清理操作，并打印出一些标记和统计量作为完整性检查。

```py
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
```

首先，我们可以看到一个很好的令牌列表，它看起来比原始文本更清晰。我们可以删除' _Book I_ '章节标记等等，但这是一个好的开始。

```py
['book', 'i', 'i', 'went', 'down', 'yesterday', 'to', 'the', 'piraeus', 'with', 'glaucon', 'the', 'son', 'of', 'ariston', 'that', 'i', 'might', 'offer', 'up', 'my', 'prayers', 'to', 'the', 'goddess', 'bendis', 'the', 'thracian', 'artemis', 'and', 'also', 'because', 'i', 'wanted', 'to', 'see', 'in', 'what', 'manner', 'they', 'would', 'celebrate', 'the', 'festival', 'which', 'was', 'a', 'new', 'thing', 'i', 'was', 'delighted', 'with', 'the', 'procession', 'of', 'the', 'inhabitants', 'but', 'that', 'of', 'the', 'thracians', 'was', 'equally', 'if', 'not', 'more', 'beautiful', 'when', 'we', 'had', 'finished', 'our', 'prayers', 'and', 'viewed', 'the', 'spectacle', 'we', 'turned', 'in', 'the', 'direction', 'of', 'the', 'city', 'and', 'at', 'that', 'instant', 'polemarchus', 'the', 'son', 'of', 'cephalus', 'chanced', 'to', 'catch', 'sight', 'of', 'us', 'from', 'a', 'distance', 'as', 'we', 'were', 'starting', 'on', 'our', 'way', 'home', 'and', 'told', 'his', 'servant', 'to', 'run', 'and', 'bid', 'us', 'wait', 'for', 'him', 'the', 'servant', 'took', 'hold', 'of', 'me', 'by', 'the', 'cloak', 'behind', 'and', 'said', 'polemarchus', 'desires', 'you', 'to', 'wait', 'i', 'turned', 'round', 'and', 'asked', 'him', 'where', 'his', 'master', 'was', 'there', 'he', 'is', 'said', 'the', 'youth', 'coming', 'after', 'you', 'if', 'you', 'will', 'only', 'wait', 'certainly', 'we', 'will', 'said', 'glaucon', 'and', 'in', 'a', 'few', 'minutes', 'polemarchus', 'appeared', 'and', 'with', 'him', 'adeimantus', 'glaucons', 'brother', 'niceratus', 'the', 'son', 'of', 'nicias', 'and', 'several', 'others', 'who', 'had', 'been', 'at', 'the', 'procession', 'polemarchus', 'said']
```

我们还获得了有关干净文档的一些统计量。

我们可以看到，干净的文字中只有不到 120,000 个单词，而且词汇量不到 7,500 个单词。这个很小，适合这些数据的模型应该可以在适度的硬件上进行管理。

```py
Total Tokens: 118684
Unique Tokens: 7409
```

接下来，我们可以看看将标记整形为序列并将它们保存到文件中。

### 保存干净的文字

我们可以将长令牌列表组织成 50 个输入字和 1 个输出字的序列。

也就是说，51 个单词的序列。

我们可以通过从令牌 51 开始迭代令牌列表并将先前的 50 个令牌作为序列进行迭代，然后将该过程重复到令牌列表的末尾。

我们将令牌转换为以空格分隔的字符串，以便以后存储在文件中。

下面列出了将清洁令牌列表拆分为长度为 51 令牌的序列的代码。

```py
# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))
```

运行此片段会创建一长串的行。

在列表上打印统计量，我们可以看到我们将有 118,633 种训练模式来适应我们的模型。

```py
Total Sequences: 118633
```

接下来，我们可以将序列保存到新文件中以便以后加载。

我们可以定义一个新函数来保存文本行到文件。这个新函数叫做 _save_doc（）_，如下所示。它将行和文件名列表作为输入。这些行以 ASCII 格式写入，每行一行。

```py
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

我们可以调用此函数并将训练序列保存到文件'`republic_sequences.txt`'。

```py
# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)
```

使用文本编辑器查看文件。

你会看到每一行都沿着一个单词移动，最后一个新单词被预测;例如，以下是截断形式的前 3 行：

> 我知道了......看到了
> 我去了......看到我们
> 我从
> 下来......

### 完整的例子

将所有这些结合在一起，下面提供了完整的代码清单。

```py
import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load document
in_filename = 'republic_clean.txt'
doc = load_doc(in_filename)
print(doc[:200])

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)
```

您现在应该将训练数据存储在当前工作目录中的文件'`republic_sequences.txt`'中。

接下来，让我们看看如何使语言模型适合这些数据。

## 训练语言模型

我们现在可以从准备好的数据中训练统计语言模型。

我们将训练的模型是神经语言模型。它有一些独特的特点：

*   它使用单词的分布式表示，以便具有相似含义的不同单词具有相似的表示。
*   它在学习模型的同时学习表示。
*   它学会使用最后 100 个单词的上下文预测下一个单词的概率。

具体来说，我们将使用嵌入层来学习单词的表示，并使用长期短期记忆（LSTM）循环神经网络来学习根据其上下文预测单词。

让我们从加载我们的训练数据开始。

### 加载序列

我们可以使用我们在上一节中开发的`load_doc()`函数加载我们的训练数据。

加载后，我们可以通过基于新行的拆分将数据拆分为单独的训练序列。

下面的代码段将从当前工作目录加载'`republic_sequences.txt`'数据文件。

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
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
```

接下来，我们可以编码训练数据。

### 编码序列

单词嵌入层要求输入序列由整数组成。

我们可以将词汇表中的每个单词映射到一个唯一的整数，并对输入序列进行编码。之后，当我们做出预测时，我们可以将预测转换为数字并在同一映射中查找其关联的单词。

要进行此编码，我们将使用 Keras API 中的 [Tokenizer 类](https://keras.io/preprocessing/text/#tokenizer)。

首先，必须在整个训练数据集上训练 Tokenizer，这意味着它会找到数据中的所有唯一单词并为每个单词分配一个唯一的整数。

然后我们可以使用 fit Tokenizer 对所有训练序列进行编码，将每个序列从单词列表转换为整数列表。

```py
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
```

我们可以访问单词到整数的映射，作为 Tokenizer 对象上名为 word_index 的字典属性。

我们需要知道稍后定义嵌入层的词汇表的大小。我们可以通过计算映射字典的大小来确定词汇表。

为单词分配从 1 到单词总数的值（例如 7,409）。嵌入层需要为此词汇表中的每个单词分配一个向量表示，从索引 1 到最大索引，并且因为数组的索引是零偏移，所以词汇结尾的单词索引将是 7,409;这意味着数组的长度必须为 7,409 + 1。

因此，在为嵌入层指定词汇表大小时，我们将其指定为比实际词汇大 1。

```py
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
```

### 序列输入和输出

现在我们已经编码了输入序列，我们需要将它们分成输入（`X`）和输出（`y`）元素。

我们可以通过数组切片来做到这一点。

分离后，我们需要对输出字进行热编码。这意味着将它从整数转换为 0 值的向量，一个用于词汇表中的每个单词，用 1 表示单词整数值索引处的特定单词。

这样，模型学习预测下一个单词的概率分布，并且除了接下来的实际单词之外，所有单词的学习基础真实为 0。

Keras 提供 _to_categorical（）_，可用于对每个输入 - 输出序列对的输出字进行热编码。

最后，我们需要为嵌入层指定输入序列的长度。我们知道有 50 个单词，因为我们设计了模型，但指定的一个很好的通用方法是使用输入数据形状的第二个维度（列数）。这样，如果在准备数据时更改序列的长度，则无需更改此数据加载代码;它是通用的。

```py
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
```

### 适合模型

我们现在可以在训练数据上定义和拟合我们的语言模型。

如前所述，学习嵌入需要知道词汇表的大小和输入序列的长度。它还有一个参数来指定用于表示每个单词的维度。也就是说，嵌入向量空间的大小。

常用值为 50,100 和 300.我们在这里使用 50，但考虑测试更小或更大的值。

我们将使用两个 LSTM 隐藏层，每层有 100 个存储单元。更多的存储单元和更深的网络可以获得更好的结果。

具有 100 个神经元的密集完全连接层连接到 LSTM 隐藏层以解释从序列提取的特征。输出层将下一个单词预测为单个向量，即词汇表的大小，其中词汇表中的每个单词具有概率。 softmax 激活函数用于确保输出具有归一化概率的特征。

```py
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
```

定义网络的摘要打印为完整性检查，以确保我们构建了我们的预期。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 50, 50)            370500
_________________________________________________________________
lstm_1 (LSTM)                (None, 50, 100)           60400
_________________________________________________________________
lstm_2 (LSTM)                (None, 100)               80400
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_2 (Dense)              (None, 7410)              748410
=================================================================
Total params: 1,269,810
Trainable params: 1,269,810
Non-trainable params: 0
_________________________________________________________________
```

接下来，编译模型，指定拟合模型所需的分类交叉熵损失。从技术上讲，该模型正在学习多分类，这是此类问题的合适损失函数。使用有效的 Adam 实现到小批量梯度下降并且评估模型的准确率。

最后，该模型适用于 100 个训练时期的数据，适当的批量大小为 128，以加快速度。

没有 GPU 的现代硬件上的训练可能需要几个小时。您可以使用更大的批量大小和/或更少的训练时期加快速度。

```py
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)
```

在训练期间，您将看到表现摘要，包括在每次批次更新结束时从训练数据评估的损失和准确率。

你会得到不同的结果，但是预测序列中下一个单词的准确度可能只有 50％以上，这也不错。我们的目标不是 100％准确（例如记忆文本的模型），而是一种捕捉文本本质的模型。

```py
...
Epoch 96/100
118633/118633 [==============================] - 265s - loss: 2.0324 - acc: 0.5187
Epoch 97/100
118633/118633 [==============================] - 265s - loss: 2.0136 - acc: 0.5247
Epoch 98/100
118633/118633 [==============================] - 267s - loss: 1.9956 - acc: 0.5262
Epoch 99/100
118633/118633 [==============================] - 266s - loss: 1.9812 - acc: 0.5291
Epoch 100/100
118633/118633 [==============================] - 270s - loss: 1.9709 - acc: 0.5315
```

### 保存模型

在运行结束时，训练的模型将保存到文件中。

在这里，我们使用 Keras 模型 API 将模型保存到当前工作目录中的文件'`model.h5`'。

之后，当我们加载模型做出预测时，我们还需要将单词映射到整数。这是在 Tokenizer 对象中，我们也可以使用 Pickle 保存它。

```py
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

### 完整的例子

我们可以把所有这些放在一起;下面列出了拟合语言模型的完整示例。

```py
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

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
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

## 使用语言模型

既然我们有一个训练有素的语言模型，我们就可以使用它。

在这种情况下，我们可以使用它来生成与源文本具有相同统计属性的新文本序列。

这是不切实际的，至少不是这个例子，但它给出了语言模型学到的具体例子。

我们将再次加载训练序列。

### 加载数据

我们可以使用上一节中的相同代码来加载文本的训练数据序列。

具体来说，`load_doc()`功能。

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

# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
```

我们需要文本，以便我们可以选择源序列作为模型的输入，以生成新的文本序列。

该模型将需要 100 个单词作为输入。

稍后，我们需要指定预期的输入长度。我们可以通过计算加载数据的一行的长度从输入序列中确定这一点，并且对于同一行上的预期输出字减去 1。

```py
seq_length = len(lines[0].split()) - 1
```

### 加载模型

我们现在可以从文件加载模型。

Keras 提供`load_model()`功能，用于加载模型，随时可以使用。

```py
# load the model
model = load_model('model.h5')
```

我们还可以使用 Pickle API 从文件加载 tokenizer。

```py
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
```

我们准备使用加载的模型。

### 生成文本

生成文本的第一步是准备种子输入。

为此，我们将从输入文本中选择一行随机文本。一旦选定，我们将打印它，以便我们对使用的内容有所了解。

```py
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
```

接下来，我们可以一次创建一个新单词。

首先，必须使用我们在训练模型时使用的相同标记器将种子文本编码为整数。

```py
encoded = tokenizer.texts_to_sequences([seed_text])[0]
```

该模型可以通过调用`model.predict_classes()`直接预测下一个单词，该模型将返回具有最高概率的单词的索引。

```py
# predict probabilities for each word
yhat = model.predict_classes(encoded, verbose=0)
```

然后，我们可以在 Tokenizers 映射中查找索引以获取关联的单词。

```py
out_word = ''
for word, index in tokenizer.word_index.items():
	if index == yhat:
		out_word = word
		break
```

然后，我们可以将此单词附加到种子文本并重复该过程。

重要的是，输入序列将变得太长。在输入序列编码为整数后，我们可以将其截断为所需的长度。 Keras 提供了`pad_sequences()`函数，我们可以使用它来执行此截断。

```py
encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
```

我们可以将所有这些包装成一个名为`generate_seq()`的函数，该函数将模型，标记生成器，输入序列长度，种子文本和要生成的单词数作为输入。然后它返回由模型生成的一系列单词。

```py
# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
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
		result.append(out_word)
	return ' '.join(result)
```

我们现在准备在给出一些种子文本的情况下生成一系列新单词。

```py
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
```

综上所述，下面列出了从学习语言模型生成文本的完整代码清单。

```py
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
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
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
```

首先运行示例打印种子文本。

> 当他说一个人长大后可以学到很多东西，因为他不能学到更多东西，因为他可以跑得很多青春是时候任何特殊的辛劳，因此计算和几何以及所有其他教学要素都是一个

然后打印 50 个生成的文本。

> 辩证法的准备应该以怠惰挥霍者的名义呈现，其他人是多方面的，不公正的，是最好的，另一个是高兴的灵魂灵魂的开放，绣花者必须在

你会得到不同的结果。尝试运行几代产品。

你可以看到文字看似合理。实际上，添加连接将有助于解释种子和生成的文本。然而，生成的文本以正确的顺序获得正确的单词。

尝试运行几次示例以查看生成文本的其他示例。如果你看到有趣的话，请在下面的评论中告诉我。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   **Sentence-Wise Model** 。基于句子分割原始数据并将每个句子填充到固定长度（例如，最长的句子长度）。
*   **简化词汇**。探索一个更简单的词汇，可能会删除词干或停止词。
*   **调谐模型**。调整模型，例如隐藏层中嵌入的大小或存储单元的数量，以查看是否可以开发更好的模型。
*   **更深的型号**。扩展模型以具有多个 LSTM 隐藏层，可能具有丢失以查看是否可以开发更好的模型。
*   **预训练单词嵌入**。扩展模型以使用预先训练的 word2vec 或 GloVe 向量来查看它是否会产生更好的模型。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [Gutenberg 项目](https://www.gutenberg.org/)
*   [柏拉图共和国古腾堡项目](https://www.gutenberg.org/ebooks/1497)
*   维基百科上的[共和国（柏拉图）](https://en.wikipedia.org/wiki/Republic_(Plato))
*   维基百科上的[语言模型](https://en.wikipedia.org/wiki/Language_model)

## 摘要

在本教程中，您了解了如何使用单词嵌入和循环神经网络开发基于单词的语言模型。

具体来说，你学到了：

*   如何准备文本以开发基于单词的语言模型。
*   如何设计和拟合具有学习嵌入和 LSTM 隐藏层的神经语言模型。
*   如何使用学习的语言模型生成具有与源文本类似的统计属性的新文本。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。