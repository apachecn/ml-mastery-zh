# 如何为情感分析准备电影评论数据

> 原文： [https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/](https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/)

每个问题的文本数据准备都不同。

准备工作从简单的步骤开始，例如加载数据，但是对于您正在使用的数据非常具体的清理任务很快就会变得困难。您需要有关从何处开始以及从原始数据到准备建模的数据的步骤的工作顺序的帮助。

在本教程中，您将逐步了解如何为情绪分析准备电影评论文本数据。

完成本教程后，您将了解：

*   如何加载文本数据并清除它以删除标点符号和其他非单词。
*   如何开发词汇表，定制它并将其保存到文件中。
*   如何使用清洁和预定义的词汇表准备电影评论，并将其保存到准备建模的新文件中。

让我们开始吧。

*   **2017 年 10 月更新**：修正了跳过不匹配文件的小错误，感谢 Jan Zett。
*   **2017 年 12 月更新**：修复了完整示例中的小错字，感谢 Ray 和 Zain。

![How to Prepare Movie Review Data for Sentiment Analysis](img/94ff9955426ec7150d7216dce6dc3f89.jpg)

如何为情感分析准备电影评论数据
[Kenneth Lu](https://www.flickr.com/photos/toasty/1125019024/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  电影评论数据集
2.  加载文本数据
3.  清理文本数据
4.  培养词汇量
5.  保存准备好的数据

## 1.电影评论数据集

电影评论数据是 Bo Pang 和 Lillian Lee 在 21 世纪初从 imdb.com 网站上检索到的电影评论的集合。收集的评论作为他们自然语言处理研究的一部分。

评论最初于 2002 年发布，但更新和清理版本于 2004 年发布，称为“ _v2.0_ ”。

该数据集包含 1,000 个正面和 1,000 个负面电影评论，这些评论来自 [IMDB](http://reviews.imdb.com/Reviews) 托管的 rec.arts.movi​​es.reviews 新闻组的存档。作者将该数据集称为“_ 极性数据集 _”。

> 我们的数据包含 2000 年之前写的 1000 份正面和 1000 份负面评论，每位作者的评论上限为 20（每位作者共 312 位）。我们将此语料库称为极性数据集。

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

数据已经有所清理，例如：

*   数据集仅包含英语评论。
*   所有文本都已转换为小写。
*   标点符号周围有空格，如句号，逗号和括号。
*   文本每行被分成一个句子。

该数据已用于一些相关的自然语言处理任务。对于分类，经典模型（例如支持向量机）对数据的性能在高 70％至低 80％（例如 78％至 82％）的范围内。

更复杂的数据准备可以看到高达 86％的结果，交叉验证 10 倍。如果我们想在现代方法的实验中使用这个数据集，这给了我们 80 年代中期的球场。

> ...根据下游极性分类器的选择，我们可以实现高度统计上的显着改善（从 82.8％到 86.4％）

- [感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。

您可以从此处下载数据集：

*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （review_polarity.tar.gz，3MB）

解压缩文件后，您将有一个名为“ _txt_sentoken_ ”的目录，其中包含两个子目录，其中包含文本“ _neg_ ”和“ _pos_ ”的负数和积极的评论。对于 neg 和 pos 中的每一个，每个文件存储一个评论约定 _cv000_ 到 _cv999_ 。

接下来，我们来看看加载文本数据。

## 2.加载文本数据

在本节中，我们将介绍加载单个文本文件，然后处理文件目录。

我们假设审查数据已下载并在文件夹“ _txt_sentoken_ ”的当前工作目录中可用。

我们可以通过打开它，读取 ASCII 文本和关闭文件来加载单个文本文件。这是标准的文件处理。例如，我们可以加载第一个负面评论文件“ _cv000_29416.txt_ ”，如下所示：

```
# load one file
filename = 'txt_sentoken/neg/cv000_29416.txt'
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
```

这会将文档加载为 ASCII 并保留任何空白区域，如新行。

我们可以把它变成一个名为 load_doc（）的函数，它接受文档的文件名加载并返回文本。

```
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

我们有两个目录，每个目录有 1,000 个文档。我们可以依次使用 [listdir（）函数](https://docs.python.org/3/library/os.html#os.listdir)获取目录中的文件列表来依次处理每个目录，然后依次加载每个文件。

例如，我们可以使用 _load_doc（）_ 函数在负目录中加载每个文档来进行实际加载。

```
from os import listdir

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# specify directory to load
directory = 'txt_sentoken/neg'
# walk through all files in the folder
for filename in listdir(directory):
	# skip files that do not have the right extension
	if not filename.endswith(".txt"):
		continue
	# create the full path of the file to open
	path = directory + '/' + filename
	# load document
	doc = load_doc(path)
	print('Loaded %s' % filename)
```

运行此示例会在加载后打印每个评论的文件名。

```
...
Loaded cv995_23113.txt
Loaded cv996_12447.txt
Loaded cv997_5152.txt
Loaded cv998_15691.txt
Loaded cv999_14636.txt
```

我们也可以将文档的处理转换为函数，稍后将其用作模板，以开发清除文件夹中所有文档的函数。例如，下面我们定义一个 _process_docs（）_ 函数来做同样的事情。

```
from os import listdir

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load all docs in a directory
def process_docs(directory):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load document
		doc = load_doc(path)
		print('Loaded %s' % filename)

# specify directory to load
directory = 'txt_sentoken/neg'
process_docs(directory)
```

现在我们知道了如何加载电影评论文本数据，让我们看一下清理它。

## 3.清理文本数据

在本节中，我们将了解我们可能要对电影评论数据进行哪些数据清理。

我们假设我们将使用一个词袋模型或者可能是一个不需要太多准备的单词嵌入。

分成代币

首先，让我们加载一个文档，然后查看由空格分割的原始标记。我们将使用上一节中开发的 _load_doc（）_ 函数。我们可以使用 _split（）_ 函数将加载的文档拆分为由空格分隔的标记。

```
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load the document
filename = 'txt_sentoken/neg/cv000_29416.txt'
text = load_doc(filename)
# split into tokens by white space
tokens = text.split()
print(tokens)
```

运行该示例从文档中提供了很长的原始令牌列表。

```
...
'years', 'ago', 'and', 'has', 'been', 'sitting', 'on', 'the', 'shelves', 'ever', 'since', '.', 'whatever', '.', '.', '.', 'skip', 'it', '!', "where's", 'joblo', 'coming', 'from', '?', 'a', 'nightmare', 'of', 'elm', 'street', '3', '(', '7/10', ')', '-', 'blair', 'witch', '2', '(', '7/10', ')', '-', 'the', 'crow', '(', '9/10', ')', '-', 'the', 'crow', ':', 'salvation', '(', '4/10', ')', '-', 'lost', 'highway', '(', '10/10', ')', '-', 'memento', '(', '10/10', ')', '-', 'the', 'others', '(', '9/10', ')', '-', 'stir', 'of', 'echoes', '(', '8/10', ')']
```

只要查看原始令牌就可以给我们提供很多想法的想法，例如：

*   从单词中删除标点符号（例如“what's”）。
*   删除只是标点符号的标记（例如“ - ”）。
*   删除包含数字的标记（例如'10 / 10'）。
*   删除具有一个字符（例如“a”）的令牌。
*   删除没有多大意义的令牌（例如'和'）

一些想法：

*   我们可以使用字符串 _translate（）_ 函数从标记中过滤出标点符号。
*   我们可以通过对每个标记使用 _isalpha（）_ 检查来删除只是标点符号或包含数字的标记。
*   我们可以使用 NLTK 加载的列表删除英语停用词。
*   我们可以通过检查短标记来过滤掉短标记。

以下是清洁此评论的更新版本。

```
from nltk.corpus import stopwords
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

# load the document
filename = 'txt_sentoken/neg/cv000_29416.txt'
text = load_doc(filename)
# split into tokens by white space
tokens = text.split()
# remove punctuation from each token
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
tokens = [word for word in tokens if len(word) > 1]
print(tokens)
```

运行该示例可以提供更清晰的令牌列表

```
...
'explanation', 'craziness', 'came', 'oh', 'way', 'horror', 'teen', 'slasher', 'flick', 'packaged', 'look', 'way', 'someone', 'apparently', 'assuming', 'genre', 'still', 'hot', 'kids', 'also', 'wrapped', 'production', 'two', 'years', 'ago', 'sitting', 'shelves', 'ever', 'since', 'whatever', 'skip', 'wheres', 'joblo', 'coming', 'nightmare', 'elm', 'street', 'blair', 'witch', 'crow', 'crow', 'salvation', 'lost', 'highway', 'memento', 'others', 'stir', 'echoes']
```

我们可以将它放入一个名为 _clean_doc（）_ 的函数中，并在另一个评论中测试它，这次是一个积极的评论。

```
from nltk.corpus import stopwords
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
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
```

同样，清洁程序似乎产生了一组良好的令牌，至少作为第一次切割。

```
...
'comic', 'oscar', 'winner', 'martin', 'childs', 'shakespeare', 'love', 'production', 'design', 'turns', 'original', 'prague', 'surroundings', 'one', 'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']
```

我们可以采取更多的清洁步骤，让我们想象一下。

接下来，让我们看看如何管理一个首选的令牌词汇表。

## 4.培养词汇量

当使用文本的预测模型时，比如词袋模型，存在减小词汇量大小的压力。

词汇量越大，每个单词或文档的表示越稀疏。

为情绪分析准备文本的一部分涉及定义和定制模型支持的单词的词汇表。

我们可以通过加载数据集中的所有文档并构建一组单词来完成此操作。我们可能会决定支持所有这些词，或者可能会丢弃一些词。然后可以将最终选择的词汇表保存到文件中供以后使用，例如将来在新文档中过滤单词。

我们可以在[计数器](https://docs.python.org/3/library/collections.html#collections.Counter)中跟踪词汇，这是一个单词及其计数字典，带有一些额外的便利功能。

我们需要开发一个新函数来处理文档并将其添加到词汇表中。该函数需要通过调用先前开发的 _load_doc（）_ 函数来加载文档。它需要使用先前开发的 _clean_doc（）_ 函数清理加载的文档，然后需要将所有标记添加到 Counter，并更新计数。我们可以通过调用计数器对象上的 _update（）_ 函数来完成最后一步。

下面是一个名为 _add_doc_to_vocab（）_ 的函数，它将文档文件名和计数器词汇表作为参数。

```
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)
```

最后，我们可以使用上面的模板处理名为 process_docs（）的目录中的所有文档，并将其更新为调用 _add_doc_to_vocab（）_。

```
# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)
```

我们可以将所有这些放在一起，并从数据集中的所有文档开发完整的词汇表。

```
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

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
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
```

运行该示例将创建包含数据集中所有文档的词汇表，包括正面和负面评论。

我们可以看到所有评论中有超过 46,000 个独特单词，前 3 个单词是'_ 电影 _'，' _one_ '和'_ 电影 _ ”。

```
46557
[('film', 8860), ('one', 5521), ('movie', 5440), ('like', 3553), ('even', 2555), ('good', 2320), ('time', 2283), ('story', 2118), ('films', 2102), ('would', 2042), ('much', 2024), ('also', 1965), ('characters', 1947), ('get', 1921), ('character', 1906), ('two', 1825), ('first', 1768), ('see', 1730), ('well', 1694), ('way', 1668), ('make', 1590), ('really', 1563), ('little', 1491), ('life', 1472), ('plot', 1451), ('people', 1420), ('movies', 1416), ('could', 1395), ('bad', 1374), ('scene', 1373), ('never', 1364), ('best', 1301), ('new', 1277), ('many', 1268), ('doesnt', 1267), ('man', 1266), ('scenes', 1265), ('dont', 1210), ('know', 1207), ('hes', 1150), ('great', 1141), ('another', 1111), ('love', 1089), ('action', 1078), ('go', 1075), ('us', 1065), ('director', 1056), ('something', 1048), ('end', 1047), ('still', 1038)]
```

也许最不常见的单词，那些仅在所有评论中出现一次的单词，都不具有预测性。也许一些最常见的词也没用。

这些都是好问题，应该用特定的预测模型进行测试。

一般来说，在 2000 条评论中只出现一次或几次的单词可能不具有预测性，可以从词汇表中删除，大大减少了我们需要建模的标记。

我们可以通过单词和它们的计数来执行此操作，并且只保留计数高于所选阈值的计数。这里我们将使用 5 次。

```
# keep tokens with > 5 occurrence
min_occurane = 5
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
```

这将词汇量从 46,557 减少到 14,803 个单词，这是一个巨大的下降。也许至少 5 次发生过于激进;你可以尝试不同的价值观。

然后，我们可以将选择的单词词汇保存到新文件中。我喜欢将词汇表保存为 ASCII，每行一个单词。

下面定义了一个名为 _save_list（）_ 的函数来保存项目列表，在这种情况下，标记为文件，每行一个。

```
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

下面列出了定义和保存词汇表的完整示例。

```
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

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
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with > 5 occurrence
min_occurane = 5
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

在创建词汇表后运行此最终片段会将所选单词保存到文件中。

最好先查看，甚至研究您选择的词汇表，以便获得更好地准备这些数据或未来文本数据的想法。

```
hasnt
updating
figuratively
symphony
civilians
might
fisherman
hokum
witch
buffoons
...
```

接下来，我们可以看一下使用词汇表来创建电影评论数据集的准备版本。

## 5.保存准备好的数据

我们可以使用数据清理和选择的词汇表来准备每个电影评论，并保存准备好的评论版本以备建​​模。

这是一个很好的做法，因为它将数据准备与建模分离，如果您有新想法，则可以专注于建模并循环回数据准备。

我们可以从' _vocab.txt_ '加载词汇开始。

```
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load vocabulary
vocab_filename = 'review_polarity/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

接下来，我们可以清理评论，使用加载的词汇来过滤掉不需要的令牌，并将干净的评论保存在新文件中。

一种方法可以是将所有正面评论保存在一个文件中，将所有负面评论保存在另一个文件中，将过滤后的标记用空格分隔，以便在单独的行上进行每次评审。

首先，我们可以定义一个函数来处理文档，清理它，过滤它，然后将它作为可以保存在文件中的单行返回。下面定义 _doc_to_line（）_ 函数，将文件名和词汇（作为一组）作为参数。

它调用先前定义的 _load_doc（）_ 函数来加载文档，调用 _clean_doc（）_ 来标记文档。

```
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)
```

接下来，我们可以定义新版本的 _process_docs（）_ 来逐步浏览文件夹中的所有评论，并通过为每个文档调用 _doc_to_line（）_ 将它们转换为行。然后返回行列表。

```
# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines
```

然后我们可以为正面和负面评论的目录调用 _process_docs（）_，然后从上一节调用 _save_list（）_ 将每个处理过的评论列表保存到文件中。

完整的代码清单如下。

```
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

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
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# save list to file
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# prepare negative reviews
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
# prepare positive reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')
```

运行该示例将保存两个新文件，' _negative.txt_ '和' _positive.txt_ '，分别包含准备好的负面和正面评论。

数据已准备好用于单词包甚至单词嵌入模型。

## 扩展

本节列出了您可能希望探索的一些扩展。

*   **Stemming** 。我们可以使用像 Porter stemmer 这样的词干算法将文档中的每个单词减少到它们的词干。
*   **N-Grams** 。我们可以使用词汇对词汇，而不是处理单个词汇。我们还可以研究使用更大的群体，例如三胞胎（三卦）和更多（n-gram）。
*   **编码字**。我们可以保存单词的整数编码，而不是按原样保存标记，其中词汇表中单词的索引表示单词的唯一整数。这将使建模时更容易处理数据。
*   **编码文件**。我们可以使用词袋模型对文档进行编码，并将每个单词编码为布尔存在/不存在标记或使用更复杂的评分，例如 TF-IDF，而不是在文档中保存标记。

如果你尝试任何这些扩展，我很想知道。
在下面的评论中分享您的结果。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 数据集

*   [电影评论数据](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
*   [一种感伤教育：基于最小削减的主观性总结的情感分析](http://xxx.lanl.gov/abs/cs/0409058)，2004。
*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （。tgz）
*   数据集自述文件 [v2.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt) 和 [v1.1](http://www.cs.cornell.edu/people/pabo/movie-review-data/README.1.1) 。

### 蜜蜂

*   [nltk.tokenize 包 API](http://www.nltk.org/api/nltk.tokenize.html)
*   [第 2 章，访问文本语料库和词汇资源](http://www.nltk.org/book/ch02.html)
*   [os API 其他操作系统接口](https://docs.python.org/3/library/os.html)
*   [集合 API - 容器数据类型](https://docs.python.org/3/library/collections.html)

## 摘要

在本教程中，您逐步了解了如何为情绪分析准备电影评论文本数据。

具体来说，你学到了：

*   如何加载文本数据并清除它以删除标点符号和其他非单词。
*   如何开发词汇表，定制它并将其保存到文件中。
*   如何使用清洁和预定义词汇表准备电影评论，并将其保存到准备建模的新文件中。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。