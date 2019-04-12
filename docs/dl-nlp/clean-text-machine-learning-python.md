# 如何用 Python 清理机器学习的文本

> 原文： [https://machinelearningmastery.com/clean-text-machine-learning-python/](https://machinelearningmastery.com/clean-text-machine-learning-python/)

你不能直接从原始文本到适合机器学习或深度学习模型。

您必须首先清理文本，这意味着将其拆分为单词并处理标点符号和大小写。

实际上，您可能需要使用一整套文本准备方法，方法的选择实际上取决于您的自然语言处理任务。

在本教程中，您将了解如何清理和准备文本，以便通过机器学习进行建模。

完成本教程后，您将了解：

*   如何开始开发自己非常简单的文本清理工具。
*   如何采取措施并使用 NLTK 库中更复杂的方法。
*   如何在使用像文字嵌入这样的现代文本表示方法时准备文本。

让我们开始吧。

*   **2017 年 11 月更新**：修正了“分裂为单词”部分中的代码拼写错误，感谢 David Comfort。

![How to Develop Multilayer Perceptron Models for Time Series Forecasting](img/3845e70194ea7d465b653bbb0d8b993a.jpg)

如何开发用于时间序列预测的多层感知器模型
照片由土地管理局提供，保留一些权利。

## 教程概述

本教程分为 6 个部分;他们是：

1.  弗兰兹卡夫卡的变态
2.  文本清理是特定于任务的
3.  手动标记
4.  使用 NLTK 进行标记和清理
5.  其他文字清理注意事项
6.  清除词嵌入文本的提示

## 弗兰兹卡夫卡的变态

让我们从选择数据集开始。

在本教程中，我们将使用 [Franz Kafka](https://en.wikipedia.org/wiki/Franz_Kafka) 的书 [Metamorphosis](https://en.wikipedia.org/wiki/The_Metamorphosis) 中的文本。没有具体的原因，除了它的简短，我喜欢它，你也可能喜欢它。我希望这是大多数学生在学校必读的经典之作。

Metamorphosis 的全文可从 Project Gutenberg 免费获得。

*   [Franz Kafka 对 Project Gutenberg 的变形](http://www.gutenberg.org/ebooks/5200)

您可以在此处下载文本的 ASCII 文本版本：

*   [变形由 Franz Kafka 纯文本 UTF-8](http://www.gutenberg.org/cache/epub/5200/pg5200.txt) （可能需要加载页面两次）。

下载文件并将其放在当前工作目录中，文件名为“ _metamorphosis.txt_ ”。

该文件包含我们不感兴趣的页眉和页脚信息，特别是版权和许可证信息。打开文件并删除页眉和页脚信息，并将文件另存为“ _metamorphosis_clean.txt_ ”。

clean 文件的开头应如下所示：

> 一天早上，当 Gregor Samsa 从困扰的梦中醒来时，他发现自己在床上变成了一个可怕的害虫。

该文件应以：

> 并且，好像在确认他们的新梦想和善意时，一旦他们到达目的地，Grete 就是第一个站起来伸展她年轻的身体的人。

穷格雷戈尔......

## 文本清理是特定于任务的

在实际掌握了您的文本数据之后，清理文本数据的第一步是对您要实现的目标有一个强烈的了解，并在该上下文中查看您的文本，看看究竟可能有什么帮助。

花点时间看看文字。你注意到什么？

这是我看到的：

*   它是纯文本，所以没有解析标记（耶！）。
*   原始德语的翻译使用英国英语（例如“_ 旅行 _”）。
*   这些线条是用约 70 个字符（meh）的新线条人工包裹的。
*   没有明显的拼写错误或拼写错误。
*   有标点符号，如逗号，撇号，引号，问号等。
*   有像盔甲一样的连字符描述。
*   有很多使用 em 破折号（“ - ”）继续句子（可能用逗号替换？）。
*   有名字（例如“ _Samsa 先生 _”）
*   似乎没有需要处理的数字（例如 1999）
*   有节标记（例如“II”和“III”），我们删除了第一个“I”。

我确信还有很多人会接受训练有素的眼睛。

我们将在本教程中查看一般文本清理步骤。

尽管如此，请考虑我们在处理此文本文档时可能遇到的一些目标。

例如：

*   如果我们有兴趣开发 [Kafkaesque](http://www.thefreedictionary.com/Kafkaesk) 语言模型，我们可能希望保留所有案例，引号和其他标点符号。
*   如果我们有兴趣将文件分类为“ _Kafka_ ”和“ _Not Kafka_ ”，那么我们可能会想要删除案例，标点符号，甚至修剪单词。

使用您的任务作为镜头，通过它选择如何准备文本数据。

## 手动标记

文本清理很难，但我们选择使用的文本已经非常干净了。

我们可以编写一些 Python 代码来手动清理它，这对于遇到的那些简单问题来说是一个很好的练习。像正则表达式和拆分字符串这样的工具可以帮到你很长的路。

### 1.加载数据

让我们加载文本数据，以便我们可以使用它。

文本很小，可以快速加载并轻松融入内存。情况并非总是如此，您可能需要将代码写入内存映射文件。像 NLTK 这样的工具（将在下一节中介绍）将使得处理大文件变得更加容易。

我们可以将整个“_ 变态 clean.text_ ”加载到内存中，如下所示：

```py
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

运行该示例将整个文件加载到可以使用的内存中。

### 2.按空白分割

清晰文本通常表示我们可以在机器学习模型中使用的单词或标记列表。

这意味着将原始文本转换为单词列表并再次保存。

一种非常简单的方法是使用空格分割文档，包括“”，新行，制表符等。我们可以在 Python 中使用 split（）函数在加载的字符串上执行此操作。

```py
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
print(words[:100])
```

运行该示例将文档拆分为一长串单词并打印前 100 个供我们查看。

我们可以看到标点符号被保留（例如“_ 不是 _”和“_ 盔甲式 _”），这很好。我们还可以看到句子标点符号的结尾与最后一个单词保持一致（例如“_ 认为 _。”），这不是很好。

```py
['One', 'morning,', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams,', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin.', 'He', 'lay', 'on', 'his', 'armour-like', 'back,', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly,', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections.', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment.', 'His', 'many', 'legs,', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him,', 'waved', 'about', 'helplessly', 'as', 'he', 'looked.', '"What\'s', 'happened', 'to', 'me?"', 'he', 'thought.', 'It', "wasn't", 'a', 'dream.', 'His', 'room,', 'a', 'proper', 'human']
```

### 3.选择单词

另一种方法可能是使用正则表达式模型（重新）并通过选择字母数字字符串（a-z，A-Z，0-9 和'_'）将文档拆分为单词。

例如：

```py
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
import re
words = re.split(r'\W+', text)
print(words[:100])
```

再次，运行示例我们可以看到我们得到了单词列表。这一次，我们可以看到“_ 盔甲式 _”现在是两个词“_ 装甲 _”和“_ 喜欢 _”（精）但是收缩像“ _]什么是 _“也是两个词”_ 什么 _“和” _s_ “（不是很好）。

```py
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin', 'He', 'lay', 'on', 'his', 'armour', 'like', 'back', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment', 'His', 'many', 'legs', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', 'waved', 'about', 'helplessly', 'as', 'he', 'looked', 'What', 's', 'happened', 'to', 'me', 'he', 'thought', 'It', 'wasn', 't', 'a', 'dream', 'His', 'room']
```

### 3.按空格分割并删除标点符号

注意：此示例是为 Python 3 编写的。

我们可能想要这些单词，但没有像逗号和引号那样的标点符号。我们也希望将宫缩保持在一起。

一种方法是通过空格将文档拆分为单词（如“ _2.按空白划分 _”），然后使用字符串翻译将所有标点符号替换为空（例如删除它）。

Python 提供了一个名为 _string.punctuation_ 的常量，它提供了一个很好的标点字符列表。例如：

```py
print(string.punctuation)
```

结果是：

```py
!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

Python 提供了一个名为 [translate（）](https://docs.python.org/3/library/stdtypes.html#str.translate)的函数，它将一组字符映射到另一组。

我们可以使用函数 [maketrans（）](https://docs.python.org/3/library/stdtypes.html#str.maketrans)来创建映射表。我们可以创建一个空的映射表，但是这个函数的第三个参数允许我们列出在翻译过程中要删除的所有字符。例如：

```py
table = str.maketrans('', '', string.punctuation)
```

我们可以将所有这些放在一起，加载文本文件，通过空格将其拆分为单词，然后翻译每个单词以删除标点符号。

```py
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in words]
print(stripped[:100])
```

我们可以看到，这主要是产生了预期的效果。

像“_ 什么 _”这样的收缩已成为“_ 什么 _”，但“_ 盔甲式 _”已成为“ _armourlike_ ”。

```py
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin', 'He', 'lay', 'on', 'his', 'armourlike', 'back', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment', 'His', 'many', 'legs', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', 'waved', 'about', 'helplessly', 'as', 'he', 'looked', 'Whats', 'happened', 'to', 'me', 'he', 'thought', 'It', 'wasnt', 'a', 'dream', 'His', 'room', 'a', 'proper', 'human']
```

如果您对正则表达式有所了解，那么您就知道事情可能会变得复杂。

### 4.规范化案例

将所有单词转换为一个案例是很常见的。

这意味着词汇量会缩小，但会丢失一些区别（例如“ _Apple_ ”公司与“ _apple_ ”水果是一个常用的例子）。

我们可以通过调用每个单词的 lower（）函数将所有单词转换为小写。

例如：

```py
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lower case
words = [word.lower() for word in words]
print(words[:100])
```

运行该示例，我们可以看到所有单词现在都是小写的。

```py
['one', 'morning,', 'when', 'gregor', 'samsa', 'woke', 'from', 'troubled', 'dreams,', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin.', 'he', 'lay', 'on', 'his', 'armour-like', 'back,', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly,', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections.', 'the', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment.', 'his', 'many', 'legs,', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him,', 'waved', 'about', 'helplessly', 'as', 'he', 'looked.', '"what\'s', 'happened', 'to', 'me?"', 'he', 'thought.', 'it', "wasn't", 'a', 'dream.', 'his', 'room,', 'a', 'proper', 'human']
```

### 注意

清理文本非常困难，特定于问题，并且充满了权衡。

记住，简单就是更好。

更简单的文本数据，更简单的模型，更小的词汇表。您可以随时将事情变得更复杂，看看它是否会带来更好的模型技能。

接下来，我们将介绍 NLTK 库中的一些工具，它们提供的不仅仅是简单的字符串拆分。

## 使用 NLTK 进行标记和清理

[自然语言工具包](http://www.nltk.org/)，简称 NLTK，是为工作和建模文本而编写的 Python 库。

它提供了用于加载和清理文本的良好工具，我们可以使用这些工具来准备我们的数据，以便使用机器学习和深度学习算法。

### 1.安装 NLTK

您可以使用自己喜欢的包管理器安装 NLTK，例如 pip：

```py
sudo pip install -U nltk
```

安装之后，您将需要安装库使用的数据，包括一组很好的文档，您可以在以后用它们来测试 NLTK 中的其他工具。

有几种方法可以做到这一点，例如在脚本中：

```py
import nltk
nltk.download()
```

或者从命令行：

```py
python -m nltk.downloader all
```

有关安装和设置 NLTK 的更多帮助，请参阅：

*   [安装 NLTK](http://www.nltk.org/install.html)
*   [安装 NLTK 数据](http://www.nltk.org/data.html)

### 2.分成句子

一个很好的有用的第一步是将文本分成句子。

一些建模任务更喜欢以段落或句子的形式输入，例如 word2vec。您可以先将文本拆分为句子，将每个句子分成单词，然后将每个句子保存到文件中，每行一个。

NLTK 提供 _sent_tokenize（）_ 函数将文本拆分成句子。

下面的示例将“ _metamorphosis_clean.txt_ ”文件加载到内存中，将其拆分为句子，然后打印第一个句子。

```py
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(text)
print(sentences[0])
```

运行这个例子，我们可以看到虽然文档被分成了句子，但每个句子仍然保留了原始文档中行的人工包装的新行。

> 一天早上，当格里高尔萨姆莎从困扰的梦中醒来时，他发现
> 自己在床上变成了一个可怕的害虫。

### 3.分成单词

NLTK 提供了一个名为 _word_tokenize（）_ 的函数，用于将字符串拆分为标记（名义上为单词）。

它根据空格和标点符号分割标记。例如，逗号和句点被视为单独的标记。收缩被分开（例如“_ 什么 _”变成“_ 什么 _”“' _s_ ”）。行情保留，等等。

例如：

```py
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens[:100])
```

运行代码，我们可以看到标点符号现在是我们可以决定专门过滤掉的标记。

```py
['One', 'morning', ',', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', ',', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin', '.', 'He', 'lay', 'on', 'his', 'armour-like', 'back', ',', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly', ',', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections', '.', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment', '.', 'His', 'many', 'legs', ',', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', ',', 'waved', 'about', 'helplessly', 'as', 'he', 'looked', '.', '``', 'What', "'s", 'happened', 'to']
```

### 4.过滤掉标点符号

我们可以过滤掉我们不感兴趣的所有令牌，例如所有独立标点符号。

这可以通过遍历所有令牌并且仅保留那些全部是字母的令牌来完成。 Python 具有可以使用的函数 [isalpha（）](https://docs.python.org/3/library/stdtypes.html#str.isalpha)。例如：

```py
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])
```

运行这个例子，你不仅可以看到标点符号，而且“_ 盔甲式 _”和“_ 的 _”等例子也被过滤掉了。

```py
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible', 'vermin', 'He', 'lay', 'on', 'his', 'back', 'and', 'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment', 'His', 'many', 'legs', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', 'waved', 'about', 'helplessly', 'as', 'he', 'looked', 'What', 'happened', 'to', 'me', 'he', 'thought', 'It', 'was', 'a', 'dream', 'His', 'room', 'a', 'proper', 'human', 'room']
```

### 5.过滤掉停用词（和管道）

[停用词](https://en.wikipedia.org/wiki/Stop_words)是那些对词组的深层含义没有贡献的词。

它们是最常见的词，例如：“”，“ _a_ ”和“_ 是 _”。

对于某些应用程序（如文档分类），删除停用词可能有意义。

NLTK 提供了各种语言（例如英语）共同商定的停用词列表。它们可以按如下方式加载：

```py
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
```

您可以看到完整列表，如下所示：

```py
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
```

您可以看到它们都是小写并删除了标点符号。

您可以将您的令牌与停用词进行比较并过滤掉它们，但您必须确保以相同的方式准备文本。

让我们通过一小段文本准备来演示这一点，包括：

1.  加载原始文本。
2.  分成代币。
3.  转换为小写。
4.  从每个令牌中删除标点符号。
5.  过滤掉非字母的剩余令牌。
6.  过滤掉停用词的令牌。

```py
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])
```

运行这个例子，我们可以看到除了所有其他变换之外，还删除了诸如“ _a_ ”和“_ 到 _”之类的停用词。

我注意到我们仍然留下像“ _nt_ ”这样的令牌。兔子洞很深;我们总能做得更多。

```py
['one', 'morning', 'gregor', 'samsa', 'woke', 'troubled', 'dreams', 'found', 'transformed', 'bed', 'horrible', 'vermin', 'lay', 'armourlike', 'back', 'lifted', 'head', 'little', 'could', 'see', 'brown', 'belly', 'slightly', 'domed', 'divided', 'arches', 'stiff', 'sections', 'bedding', 'hardly', 'able', 'cover', 'seemed', 'ready', 'slide', 'moment', 'many', 'legs', 'pitifully', 'thin', 'compared', 'size', 'rest', 'waved', 'helplessly', 'looked', 'happened', 'thought', 'nt', 'dream', 'room', 'proper', 'human', 'room', 'although', 'little', 'small', 'lay', 'peacefully', 'four', 'familiar', 'walls', 'collection', 'textile', 'samples', 'lay', 'spread', 'table', 'samsa', 'travelling', 'salesman', 'hung', 'picture', 'recently', 'cut', 'illustrated', 'magazine', 'housed', 'nice', 'gilded', 'frame', 'showed', 'lady', 'fitted', 'fur', 'hat', 'fur', 'boa', 'sat', 'upright', 'raising', 'heavy', 'fur', 'muff', 'covered', 'whole', 'lower', 'arm', 'towards', 'viewer']
```

### 6.词干

[词干](https://en.wikipedia.org/wiki/Stemming)指的是将每个单词缩减为其根或基数的过程。

例如“_ 钓鱼 _”，“_ 捕捞 _”，“ _fisher_ ”全部减少到茎“_ 鱼 _”。

一些应用程序，如文档分类，可以从词干分析中受益，以便既减少词汇量又专注于文档的感觉或情感，而不是更深层的含义。

有许多词干算法，尽管流行的和长期存在的方法是 Porter Stemming 算法。这种方法可以通过 [PorterStemmer](https://tartarus.org/martin/PorterStemmer/) 类在 NLTK 中使用。

例如：

```py
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])
```

运行这个例子，你可以看到单词已经减少到它们的词干，例如“ _trouble_ ”变成了“ _troubl_ ”。您还可以看到，词干实现还将令牌减少为小写，可能是字表中的内部查找。

您还可以看到，词干实现还将令牌减少为小写，可能是字表中的内部查找。

```py
['one', 'morn', ',', 'when', 'gregor', 'samsa', 'woke', 'from', 'troubl', 'dream', ',', 'he', 'found', 'himself', 'transform', 'in', 'hi', 'bed', 'into', 'a', 'horribl', 'vermin', '.', 'He', 'lay', 'on', 'hi', 'armour-lik', 'back', ',', 'and', 'if', 'he', 'lift', 'hi', 'head', 'a', 'littl', 'he', 'could', 'see', 'hi', 'brown', 'belli', ',', 'slightli', 'dome', 'and', 'divid', 'by', 'arch', 'into', 'stiff', 'section', '.', 'the', 'bed', 'wa', 'hardli', 'abl', 'to', 'cover', 'it', 'and', 'seem', 'readi', 'to', 'slide', 'off', 'ani', 'moment', '.', 'hi', 'mani', 'leg', ',', 'piti', 'thin', 'compar', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', ',', 'wave', 'about', 'helplessli', 'as', 'he', 'look', '.', '``', 'what', "'s", 'happen', 'to'
```

在 NLTK 中有一套很好的词干和词形还原算法可供选择，如果将词语缩减到它们的根目录就是你的项目需要的东西。

## 其他文字清理注意事项

我们才刚开始。

因为本教程的源文本开头是相当干净的，所以我们跳过了许多您可能需要在自己的项目中处理的文本清理问题。

以下是清理文本时的其他注意事项的简短列表：

*   处理不适合内存的大型文档和大量文本文档。
*   从标记中提取文本，如 HTML，PDF 或其他结构化文档格式。
*   从其他语言到英语的音译。
*   将 Unicode 字符解码为规范化形式，例如 UTF8。
*   处理特定领域的单词，短语和首字母缩略词。
*   处理或删除数字，例如日期和金额。
*   找出并纠正常见的拼写错误和拼写错误。
*   ...

这份名单可以继续使用。

希望您能够看到获得真正干净的文本是不可能的，我们真的可以根据我们拥有的时间，资源和知识做到最好。

“清洁”的概念实际上是由项目的特定任务或关注点定义的。

专家提示是在每次转换后不断检查您的令牌。我试图在本教程中表明，我希望你能理解这一点。

理想情况下，您可以在每次转换后保存新文件，以便花时间处理新表单中的所有数据。在花时间审查您的数据时，事情总是会突然发生。

你以前做过一些文字清理吗？您最喜欢的变换管道是什么？
请在下面的评论中告诉我。

## 清除词嵌入文本的提示

最近，自然语言处理领域已逐渐从单词模型和单词编码转向单词嵌入。

单词嵌入的好处在于，它们将每个单词编码为一个密集的向量，捕获有关其在训练文本中的相对含义的内容。

这意味着在嵌入空间中将自动学习诸如大小写，拼写，标点符号等单词的变体。反过来，这可能意味着您的文本所需的清洁量可能更少，也许与传统的文本清理完全不同。

例如，干缩词语或删除标点符号可能不再有意义。

Tomas Mikolov 是 word2vec 的开发者之一，word2vec 是一种流行的嵌入式方法。他建议在学习单词嵌入模型时只需要非常小的文本清理。

下面是他在回答有关如何最好地为 word2vec 准备文本数据的问题时的回答。

> 没有普遍的答案。这一切都取决于你打算使用的向量。根据我的经验，通常可以从单词中断开（或删除）标点符号，有时还会将所有字符转换为小写。人们也可以用一些单一的标记替换所有数字（可能大于某些常数），例如。
> 
> 所有这些预处理步骤都旨在减少词汇量，而不删除任何重要内容（在某些情况下，当你小写某些单词时可能不是这样，即'Bush'与'bush'不同，而'Another'通常有与“另一个”的意义相同。词汇量越小，内存复杂度越低，估计的词的参数越稳健。您还必须以相同的方式预处理测试数据。
> 
> ...
> 
> 简而言之，如果你要进行实验，你会更好地理解这一切。

[阅读 Google 网上论坛](https://groups.google.com/d/msg/word2vec-toolkit/jPfyP6FoB94/tGzZxScO0GsJ)的完整帖子。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [Franz Kafka 对 Project Gutenberg 的变形](http://www.gutenberg.org/ebooks/5200)
*   [nltk.tokenize 包 API](http://www.nltk.org/api/nltk.tokenize.html)
*   [nltk.stem 包 API](http://www.nltk.org/api/nltk.stem.html)
*   [第 3 章：使用 Python 处理原始文本，自然语言处理](http://www.nltk.org/book/ch03.html)

## 摘要

在本教程中，您了解了如何在 Python 中清理文本或机器学习。

具体来说，你学到了：

*   如何开始开发自己非常简单的文本清理工具。
*   如何采取措施并使用 NLTK 库中更复杂的方法。
*   如何在使用像文字嵌入这样的现代文本表示方法时准备文本。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。

你有清洁文字的经验吗？
请在下面的评论中分享您的经验。