# 如何为机器翻译准备法语到英语的数据集

> 原文： [https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/](https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/)

机器翻译是将文本从源语言转换为目标语言中的连贯和匹配文本的挑战性任务。

诸如编码器 - 解码器循环神经网络之类的神经机器翻译系统正在通过直接在源语言和目标语言上训练的单个端到端系统实现机器翻译的最先进结果。

需要标准数据集来开发，探索和熟悉如何开发神经机器翻译系统。

在本教程中，您将发现 Europarl 标准机器翻译数据集以及如何准备数据以进行建模。

完成本教程后，您将了解：

*   Europarl 数据集由欧洲议会以 11 种语言提供的程序组成。
*   如何加载和清理准备在神经机器翻译系统中建模的平行法语和英语成绩单。
*   如何减少法语和英语数据的词汇量，以降低翻译任务的复杂性。

让我们开始吧。

![How to Prepare a French-to-English Dataset for Machine Translation](img/8792dfc88847977ba9077568a855a5fb.jpg)

如何为机器翻译准备法语 - 英语数据集
[Giuseppe Milo](https://www.flickr.com/photos/giuseppemilo/15366744101/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  Europarl 机器翻译数据集
2.  下载法语 - 英语数据集
3.  加载数据集
4.  清理数据集
5.  减少词汇量

### Python 环境

本教程假设您安装了安装了 Python 3 的 Python SciPy 环境。

本教程还假设您安装了 scikit-learn，Pandas，NumPy 和 Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## Europarl 机器翻译数据集

Europarl 是用于统计机器翻译的标准数据集，最近是神经机器翻译。

它由欧洲议会的议事程序组成，因此数据集的名称为收缩`Europarl`。

诉讼程序是欧洲议会发言人的抄本，翻译成 11 种不同的语言。

> 它是欧洲议会议事录的集合，可追溯到 1996 年。总共包括欧盟 11 种官方语言中每种语言约 3000 万字的语料库。

- [Europarl：统计机器翻译平行语料库](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf)，2005。

原始数据可在[欧洲议会网站](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf)上以 HTML 格式获得。

数据集的创建由 [Philipp Koehn](http://www.cs.jhu.edu/~phi/) 领导，该书是“[统计机器翻译](http://amzn.to/2xbAuwx)”一书的作者。

该数据集在网站“[欧洲议会会议论文集平行语料库 1996-2011](http://www.statmt.org/europarl/) ”上免费提供给研究人员，并且经常作为机器翻译挑战的一部分出现，例如[机器翻译任务](http://www.statmt.org/wmt14/translation-task.html)在 2014 年统计机器翻译研讨会上。

最新版本的数据集是 2012 年发布的版本 7，包含 1996 年至 2011 年的数据。

## 下载法语 - 英语数据集

我们将专注于平行的法语 - 英语数据集。

这是 1996 年至 2011 年间记录的法语和英语对齐语料库。

数据集具有以下统计信息：

*   句子：2,007,723
*   法语单词：51,388,643
*   英语单词：50,196,035

您可以从此处下载数据集：

*   [平行语料库法语 - 英语](http://www.statmt.org/europarl/v7/fr-en.tgz)（194 兆字节）

下载后，您当前的工作目录中应该有“ _fr-en.tgz_ ”文件。

您可以使用 tar 命令解压缩此存档文件，如下所示：

```py
tar zxvf fr-en.tgz
```

您现在将拥有两个文件，如下所示：

*   英语：europarl-v7.fr-en.en（288M）
*   法语：europarl-v7.fr-en.fr（331M）

以下是英文文件的示例。

```py
Resumption of the session
I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.
Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful.
You have requested a debate on this subject in the course of the next few days, during this part-session.
In the meantime, I should like to observe a minute' s silence, as a number of Members have requested, on behalf of all the victims concerned, particularly those of the terrible storms, in the various countries of the European Union.
```

以下是法语文件的示例。

```py
Reprise de la session
Je déclare reprise la session du Parlement européen qui avait été interrompue le vendredi 17 décembre dernier et je vous renouvelle tous mes vux en espérant que vous avez passé de bonnes vacances.
Comme vous avez pu le constater, le grand "bogue de l'an 2000" ne s'est pas produit. En revanche, les citoyens d'un certain nombre de nos pays ont été victimes de catastrophes naturelles qui ont vraiment été terribles.
Vous avez souhaité un débat à ce sujet dans les prochains jours, au cours de cette période de session.
En attendant, je souhaiterais, comme un certain nombre de collègues me l'ont demandé, que nous observions une minute de silence pour toutes les victimes, des tempêtes notamment, dans les différents pays de l'Union européenne qui ont été touchés.
```

## 加载数据集

让我们从加载数据文件开始。

我们可以将每个文件作为字符串加载。由于文件包含 unicode 字符，因此在将文件作为文本加载时必须指定编码。在这种情况下，我们将使用 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 来轻松处理两个文件中的 unicode 字符。

下面的函数名为 _load_doc（）_，它将加载一个给定的文件并将其作为一个文本块返回。

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

接下来，我们可以将文件拆分成句子。

通常，在每一行上存储一个话语。我们可以将它们视为句子并用新行字符拆分文件。下面的函数`to_sentences()`将拆分加载的文档。

```py
# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')
```

在以后准备我们的模型时，我们需要知道数据集中句子的长度。我们可以写一个简短的函数来计算最短和最长的句子。

```py
# shortest and longest sentence lengths
def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)
```

我们可以将所有这些结合在一起，以加载和汇总英语和法语数据文件。下面列出了完整的示例。

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

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')

# shortest and longest sentence lengths
def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)

# load English data
filename = 'europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

# load French data
filename = 'europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('French data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))
```

运行该示例总结了每个文件中的行数或句子数以及每个文件中最长和最短行的长度。

```py
English data: sentences=2007723, min=0, max=668
French data: sentences=2007723, min=0, max=693
```

重要的是，我们可以看到 2,007,723 行符合预期。

## 清理数据集

在用于训练神经翻译模型之前，数据需要一些最小的清洁。

查看一些文本样本，一些最小的文本清理可能包括：

*   用空格标记文本。
*   将大小写归一化为小写。
*   从每个单词中删除标点符号。
*   删除不可打印的字符。
*   将法语字符转换为拉丁字符。
*   删除包含非字母字符的单词。

这些只是一些基本操作作为起点;您可能知道或需要更复杂的数据清理操作。

下面的函数`clean_lines()`实现了这些清理操作。一些说明：

*   我们使用 unicode API 来规范化 unicode 字符，将法语字符转换为拉丁语字符。
*   我们使用逆正则表达式匹配来仅保留可打印单词中的那些字符。
*   我们使用转换表按原样翻译字符，但不包括所有标点字符。

```py
# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned
```

标准化后，我们使用 pickle API 直接以二进制格式保存简洁行列表。这将加快后期和未来的进一步操作的加载。

重用前面部分中开发的加载和拆分功能，下面列出了完整的示例。

```py
import string
import re
from pickle import dump
from unicodedata import normalize

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
def to_sentences(doc):
	return doc.strip().split('\n')

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load English data
filename = 'europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'english.pkl')
# spot check
for i in range(10):
	print(sentences[i])

# load French data
filename = 'europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'french.pkl')
# spot check
for i in range(10):
	print(sentences[i])
```

运行后，干净的句子分别保存在`english.pkl`和`french.pkl`文件中。

作为运行的一部分，我们还打印每个清晰句子列表的前几行，转载如下。

英语：

```py
resumption of the session
i declare resumed the session of the european parliament adjourned on friday december and i would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period
although as you will have seen the dreaded millennium bug failed to materialise still the people in a number of countries suffered a series of natural disasters that truly were dreadful
you have requested a debate on this subject in the course of the next few days during this partsession
in the meantime i should like to observe a minute s silence as a number of members have requested on behalf of all the victims concerned particularly those of the terrible storms in the various countries of the european union
please rise then for this minute s silence
the house rose and observed a minute s silence
madam president on a point of order
you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka
one of the people assassinated very recently in sri lanka was mr kumar ponnambalam who had visited the european parliament just a few months ago
```

法国：

```py
reprise de la session
je declare reprise la session du parlement europeen qui avait ete interrompue le vendredi decembre dernier et je vous renouvelle tous mes vux en esperant que vous avez passe de bonnes vacances
comme vous avez pu le constater le grand bogue de lan ne sest pas produit en revanche les citoyens dun certain nombre de nos pays ont ete victimes de catastrophes naturelles qui ont vraiment ete terribles
vous avez souhaite un debat a ce sujet dans les prochains jours au cours de cette periode de session
en attendant je souhaiterais comme un certain nombre de collegues me lont demande que nous observions une minute de silence pour toutes les victimes des tempetes notamment dans les differents pays de lunion europeenne qui ont ete touches
je vous invite a vous lever pour cette minute de silence
le parlement debout observe une minute de silence
madame la presidente cest une motion de procedure
vous avez probablement appris par la presse et par la television que plusieurs attentats a la bombe et crimes ont ete perpetres au sri lanka
lune des personnes qui vient detre assassinee au sri lanka est m kumar ponnambalam qui avait rendu visite au parlement europeen il y a quelques mois a peine
```

我对法语的阅读非常有限，但至少就英语而言，可以进一步改进，例如丢弃或连接复数的''字符。

## 减少词汇量

作为数据清理的一部分，限制源语言和目标语言的词汇量非常重要。

翻译任务的难度与词汇量的大小成比例，这反过来影响模型训练时间和使模型可行所需的数据集的大小。

在本节中，我们将减少英语和法语文本的词汇量，并使用特殊标记标记所有词汇（OOV）单词。

我们可以从加载上一节保存的酸洗干净线开始。下面的`load_clean_sentences()`函数将加载并返回给定文件名的列表。

```py
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
```

接下来，我们可以计算数据集中每个单词的出现次数。为此，我们可以使用`Counter`对象，这是一个键入单词的 Python 字典，每次添加每个单词的新出现时都会更新计数。

下面的`to_vocab()`函数为给定的句子列表创建词汇表。

```py
# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab
```

然后，我们可以处理创建的词汇表，并从计数器中删除出现低于特定阈值的所有单词。

下面的`trim_vocab()`函数执行此操作并接受最小出现次数作为参数并返回更新的词汇表。

```py
# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	return set(tokens)
```

最后，我们可以更新句子，删除不在修剪词汇表中的所有单词，并用特殊标记标记它们的删除，在本例中为字符串“`unk`”。

下面的`update_dataset()`函数执行此操作并返回更新行的列表，然后可以将其保存到新文件中。

```py
# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines
```

我们可以将所有这些结合在一起，减少英语和法语数据集的词汇量，并将结果保存到新的数据文件中。

我们将使用最小值 5，但您可以自由探索适合您的应用的其他最小值。

完整的代码示例如下所示。

```py
from pickle import load
from pickle import dump
from collections import Counter

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

# load English dataset
filename = 'english.pkl'
lines = load_clean_sentences(filename)
# calculate vocabulary
vocab = to_vocab(lines)
print('English Vocabulary: %d' % len(vocab))
# reduce vocabulary
vocab = trim_vocab(vocab, 5)
print('New English Vocabulary: %d' % len(vocab))
# mark out of vocabulary words
lines = update_dataset(lines, vocab)
# save updated dataset
filename = 'english_vocab.pkl'
save_clean_sentences(lines, filename)
# spot check
for i in range(10):
	print(lines[i])

# load French dataset
filename = 'french.pkl'
lines = load_clean_sentences(filename)
# calculate vocabulary
vocab = to_vocab(lines)
print('French Vocabulary: %d' % len(vocab))
# reduce vocabulary
vocab = trim_vocab(vocab, 5)
print('New French Vocabulary: %d' % len(vocab))
# mark out of vocabulary words
lines = update_dataset(lines, vocab)
# save updated dataset
filename = 'french_vocab.pkl'
save_clean_sentences(lines, filename)
# spot check
for i in range(10):
	print(lines[i])
```

首先，报告英语词汇的大小，然后是更新的大小。更新的数据集将保存到文件'`english_vocab.pkl`'，并打印一些更新的示例的现场检查，其中包含用“`unk`”替换的词汇单词。

```py
English Vocabulary: 105357
New English Vocabulary: 41746
Saved: english_vocab.pkl
```

我们可以看到词汇量的大小缩减了一半到 40,000 多个单词。

```py
resumption of the session
i declare resumed the session of the european parliament adjourned on friday december and i would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period
although as you will have seen the dreaded millennium bug failed to materialise still the people in a number of countries suffered a series of natural disasters that truly were dreadful
you have requested a debate on this subject in the course of the next few days during this partsession
in the meantime i should like to observe a minute s silence as a number of members have requested on behalf of all the victims concerned particularly those of the terrible storms in the various countries of the european union
please rise then for this minute s silence
the house rose and observed a minute s silence
madam president on a point of order
you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka
one of the people assassinated very recently in sri lanka was mr unk unk who had visited the european parliament just a few months ago
```

然后对 French 数据集执行相同的过程，将结果保存到文件'`french_vocab.pkl`'。

```py
French Vocabulary: 141642
New French Vocabulary: 58800
Saved: french_vocab.pkl
```

我们看到法语词汇量大小相似缩小。

```py
reprise de la session
je declare reprise la session du parlement europeen qui avait ete interrompue le vendredi decembre dernier et je vous renouvelle tous mes vux en esperant que vous avez passe de bonnes vacances
comme vous avez pu le constater le grand bogue de lan ne sest pas produit en revanche les citoyens dun certain nombre de nos pays ont ete victimes de catastrophes naturelles qui ont vraiment ete terribles
vous avez souhaite un debat a ce sujet dans les prochains jours au cours de cette periode de session
en attendant je souhaiterais comme un certain nombre de collegues me lont demande que nous observions une minute de silence pour toutes les victimes des tempetes notamment dans les differents pays de lunion europeenne qui ont ete touches
je vous invite a vous lever pour cette minute de silence
le parlement debout observe une minute de silence
madame la presidente cest une motion de procedure
vous avez probablement appris par la presse et par la television que plusieurs attentats a la bombe et crimes ont ete perpetres au sri lanka
lune des personnes qui vient detre assassinee au sri lanka est m unk unk qui avait rendu visite au parlement europeen il y a quelques mois a peine
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [Europarl：统计机器翻译平行语料库](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf)，2005。
*   [欧洲议会诉讼平行语料库 1996-2011 主页](http://www.statmt.org/europarl/)
*   [维基百科上的 Europarl Corpus](https://en.wikipedia.org/wiki/Europarl_Corpus)

## 摘要

在本教程中，您发现了 Europarl 机器翻译数据集以及如何准备数据以便进行建模。

具体来说，你学到了：

*   Europarl 数据集由欧洲议会以 11 种语言提供的程序组成。
*   如何加载和清理准备在神经机器翻译系统中建模的平行法语和英语成绩单。
*   如何减少法语和英语数据的词汇量，以降低翻译任务的复杂性。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。