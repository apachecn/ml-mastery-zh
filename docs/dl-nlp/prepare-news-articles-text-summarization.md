# 如何为文本摘要准备新闻文章

> 原文： [https://machinelearningmastery.com/prepare-news-articles-text-summarization/](https://machinelearningmastery.com/prepare-news-articles-text-summarization/)

文本摘要是创建文章的简短，准确和流畅的摘要的任务。

CNN 新闻故事数据集是一种流行的免费数据集，用于深度学习方法的文本摘要实验。

在本教程中，您将了解如何准备 CNN 新闻数据集以进行文本摘要。

完成本教程后，您将了解：

*   关于 CNN 新闻数据集以及如何将故事数据下载到您的工作站。
*   如何加载数据集并将每篇文章拆分为故事文本和突出显示。
*   如何清理准备建模的数据集并将清理后的数据保存到文件中供以后使用。

让我们开始吧。

![How to Prepare News Articles for Text Summarization](img/8629787a201fe5d6aadfbd08b965f80c.jpg)

如何为文本摘要准备新闻文章
[DieselDemon](https://www.flickr.com/photos/28096801@N05/6252168841/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  CNN 新闻故事数据集
2.  检查数据集
3.  加载数据
4.  数据清理
5.  保存清洁数据

## CNN 新闻故事数据集

DeepMind Q＆amp; A 数据集是来自 CNN 和每日邮报的大量新闻文章以及相关问题。

该数据集是作为深度学习的问题和回答任务而开发的，并在 2015 年的论文“[教学机器中进行了阅读和理解](https://arxiv.org/abs/1506.03340)”。

该数据集已用于文本摘要中，其中汇总了来自新闻文章的句子。值得注意的例子是论文：

*   [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。
*   [达到要点：利用指针生成器网络汇总](https://arxiv.org/abs/1704.04368)，2017 年。

Kyunghyun Cho 是纽约大学的学者，已经提供了下载数据集：

*   [DeepMind Q＆amp; A 数据集](http://cs.nyu.edu/~kcho/DMQA/)

在本教程中，我们将使用 CNN 数据集，特别是下载此处提供的新闻报道的 ASCII 文本：

*   [cnn_stories.tgz](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ) （151 兆字节）

此数据集包含超过 93,000 篇新闻文章，其中每篇文章都存储在单个“ _.story_ ”文件中。

将此数据集下载到您的工作站并解压缩。下载后，您可以在命令行上解压缩存档，如下所示：

```
tar xvf cnn_stories.tgz
```

这将创建一个 _cnn / stories /_ 目录，其中包含 _.story_ 文件。

例如，我们可以在命令行上计算故事文件的数量，如下所示：

```
ls -ltr | wc -l
```

这向我们展示了我们共有 92,580 家商店。

```
92580
```

## 检查数据集

使用文本编辑器，查看一些故事并记下准备这些数据的一些想法。

例如，下面是一个故事的例子，为简洁起见，身体被截断。

```
(CNN) -- If you travel by plane and arriving on time makes a difference, try to book on Hawaiian Airlines. In 2012, passengers got where they needed to go without delay on the carrier more than nine times out of 10, according to a study released on Monday.

In fact, Hawaiian got even better from 2011, when it had a 92.8% on-time performance. Last year, it improved to 93.4%.

[...]

@highlight

Hawaiian Airlines again lands at No. 1 in on-time performance

@highlight

The Airline Quality Rankings Report looks at the 14 largest U.S. airlines

@highlight

ExpressJet and American Airlines had the worst on-time performance

@highlight

Virgin America had the best baggage handling; Southwest had lowest complaint rate
```

我注意到数据集的一般结构是让故事文本后跟一些“_ 突出显示 _”点。

回顾 CNN 网站上的文章，我可以看到这种模式仍然很常见。

![Example of a CNN News Article With Highlights from cnn.com](img/cc79667253e4757c2223dd24295aec31.jpg)

来自 [cnn.com](http://edition.cnn.com/2017/08/28/politics/donald-trump-hurricane-harvey-response-texas/index.html) 的重点介绍 CNN 新闻文章的例子

ASCII 文本不包括文章标题，但我们可以使用这些人工编写的“_ 重点 _”作为每篇新闻文章的多个参考摘要。

我还可以看到许多文章都是从源信息开始的，可能是创建故事的 CNN 办公室;例如：

```
(CNN) --
Gaza City (CNN) --
Los Angeles (CNN) --
```

这些可以完全删除。

数据清理是一个具有挑战性的问题，必须根据系统的特定应用进行定制。

如果我们通常对开发新闻文章摘要系统感兴趣，那么我们可以清理文本以通过减小词汇量来简化学习问题。

这些数据的一些数据清理思路包括。

*   将大小写归一化为小写（例如“An Italian”）。
*   删除标点符号（例如“准时”）。

我们还可以进一步减少词汇量来加速测试模型，例如：

*   删除号码（例如“93.4％”）。
*   删除名称等低频词（例如“Tom Watkins”）。
*   将故事截断为前 5 或 10 个句子。

## 加载数据

第一步是加载数据。

我们可以先编写一个函数来加载给定文件名的单个文档。数据有一些 unicode 字符，因此我们将通过强制编码为 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 来加载数据集。

下面名为 _load_doc（）_ 的函数将加载单个文档作为给定文件名的文本。

```
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
```

接下来，我们需要跳过 stories 目录中的每个文件名并加载它们。

我们可以使用 _listdir（）_ 函数加载目录中的所有文件名，然后依次加载每个文件名。以下名为 _load_stories（）_ 的函数实现了此行为，并为准备加载的文档提供了一个起点。

```
# load all stories in a directory
def load_stories(directory):
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
```

每个文档可以分为新闻故事文本和精彩部分或摘要文本。

这两点的分割是第一次出现' _@highlight_ '令牌。拆分后，我们可以将亮点组织到列表中。

以下名为 _split_story（）_ 的函数实现了此行为，并将给定的已加载文档文本拆分为故事和高亮列表。

```
# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights
```

我们现在可以更新 _load_stories（）_ 函数，为每个加载的文档调用 _split_story（）_ 函数，然后将结果存储在列表中。

```
# load all stories in a directory
def load_stories(directory):
	all_stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		all_stories.append({'story':story, 'highlights':highlights})
	return all_stories
```

将所有这些结合在一起，下面列出了加载整个数据集的完整示例。

```
from os import listdir

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# load all stories in a directory
def load_stories(directory):
	stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		stories.append({'story':story, 'highlights':highlights})
	return stories

# load stories
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))
```

运行该示例将打印已加载故事的数量。

```
Loaded Stories 92,579
```

我们现在可以访问加载的故事并突出显示数据，例如：

```
print(stories[4]['story'])
print(stories[4]['highlights'])
```

## 数据清理

现在我们可以加载故事数据，我们可以通过清理它来预处理文本。

我们可以逐行处理故事，并在每个高亮线上使用相同的清洁操作。

对于给定的行，我们将执行以下操作：

删除 CNN 办公室信息。

```
# strip source cnn office if it exists
index = line.find('(CNN) -- ')
if index > -1:
	line = line[index+len('(CNN)'):]
```

使用空格标记拆分线：

```
# tokenize on white space
line = line.split()
```

将案例规范化为小写。

```
# convert to lower case
line = [word.lower() for word in line]
```

从每个标记中删除所有标点符号（特定于 Python 3）。

```
# prepare a translation table to remove punctuation
table = str.maketrans('', '', string.punctuation)
# remove punctuation from each token
line = [w.translate(table) for w in line]
```

删除任何具有非字母字符的单词。

```
# remove tokens with numbers in them
line = [word for word in line if word.isalpha()]
```

将这一切放在一起，下面是一个名为 _clean_lines（）_ 的新函数，它接受一行文本行并返回一个简洁的文本行列表。

```
# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned
```

我们可以通过首先将其转换为一行文本来将其称为故事。可以在高亮列表上直接调用该函数。

```
example['story'] = clean_lines(example['story'].split('\n'))
example['highlights'] = clean_lines(example['highlights'])
```

下面列出了加载和清理数据集的完整示例。

```
from os import listdir
import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# load all stories in a directory
def load_stories(directory):
	stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		stories.append({'story':story, 'highlights':highlights})
	return stories

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned

# load stories
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
for example in stories:
	example['story'] = clean_lines(example['story'].split('\n'))
	example['highlights'] = clean_lines(example['highlights'])
```

请注意，故事现在存储为一个简洁的行列表，名义上用句子分隔。

## 保存清洁数据

最后，既然已经清理了数据，我们可以将其保存到文件中。

保存清理数据的简便方法是选择故事和精彩部分列表。

例如：

```
# save to file
from pickle import dump
dump(stories, open('cnn_dataset.pkl', 'wb'))
```

这将创建一个名为 _cnn_dataset.pkl_ 的新文件，其中包含所有已清理的数据。该文件大小约为 374 兆字节。

然后我们可以稍后加载它并将其与文本摘要模型一起使用，如下所示：

```
# load from file
stories = load(open('cnn_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [DeepMind Q＆amp; A 数据集](http://cs.nyu.edu/~kcho/DMQA/)
*   [教学机器阅读和理解](https://arxiv.org/abs/1506.03340)，2015。
*   [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。
*   [达到要点：利用指针生成器网络汇总](https://arxiv.org/abs/1704.04368)，2017 年。

## 摘要

在本教程中，您了解了如何准备 CNN 新闻数据集以进行文本摘要。

具体来说，你学到了：

*   关于 CNN 新闻数据集以及如何将故事数据下载到您的工作站。
*   如何加载数据集并将每篇文章拆分为故事文本和突出显示。
*   如何清理准备建模的数据集并将清理后的数据保存到文件中供以后使用。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。