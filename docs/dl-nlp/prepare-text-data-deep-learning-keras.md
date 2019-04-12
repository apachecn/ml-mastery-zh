# 如何使用 Keras 为深度学习准备文本数据

> 原文： [https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/](https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/)

您无法直接将原始文本提供给深度学习模型。

文本数据必须编码为数字，以用作机器学习和深度学习模型的输入或输出。

Keras 深度学习库提供了一些基本工具来帮助您准备文本数据。

在本教程中，您将了解如何使用 Keras 准备文本数据。

完成本教程后，您将了解：

*   关于可用于快速准备文本数据的便捷方法。
*   Tokenizer API，可以适用于训练数据，用于编码训练，验证和测试文档。
*   Tokenizer API 提供的 4 种不同文档编码方案的范围。

让我们开始吧。

![How to Prepare Text Data for Deep Learning with Keras](img/173941fdd7b41eb463da5a301be0b140.jpg)

如何使用 Keras
为深度学习准备文本数据照片来自 [ActiveSteve](https://www.flickr.com/photos/activesteve/36054715916/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  用 text_to_word_sequence 分隔单词。
2.  用 one_hot 编码。
3.  使用 hashing_trick 进行哈希编码。
4.  Tokenizer API

## 使用 text_to_word_sequence 分割单词

使用文本时，第一步是将其拆分为单词。

单词称为标记，将文本拆分为标记的过程称为分词。

Keras 提供 [text_to_word_sequence（）函数](https://keras.io/preprocessing/text/#text_to_word_sequence)，您可以使用它将文本拆分为单词列表。

默认情况下，此功能自动执行以下三项操作：

*   按空格拆分单词（split =“”）。
*   过滤掉标点符号（filters ='！“＃$％＆amp;（）* +， - 。/：;＆lt; =＆gt;？@ [\\] ^ _` {|}〜\ t \ n'）。
*   将文本转换为小写（lower = True）。

您可以通过将参数传递给函数来更改任何这些默认值。

下面是使用 text_to_word_sequence（）函数将文档（在本例中为简单字符串）拆分为单词列表的示例。

```py
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)
```

运行该示例将创建一个包含文档中所有单词的数组。打印单词列表以供审阅。

```py
['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
```

这是一个很好的第一步，但在使用文本之前需要进一步的预处理。

## 用 one_hot 编码

将文档表示为整数值序列是很流行的，其中文档中的每个单词都表示为唯一的整数。

Keras 提供 [one_hot（）函数](https://keras.io/preprocessing/text/#one_hot)，您可以使用它来一步对文本文档进行分词和整数编码。该名称表明它将创建文档的单热编码，但事实并非如此。

相反，该函数是下一节中描述的 hashing_trick（）函数的包装器。该函数返回文档的整数编码版本。散列函数的使用意味着可能存在冲突，并且不是所有单词都将被分配唯一的整数值。

与上一节中的 text_to_word_sequence（）函数一样，one_hot（）函数将使文本小写，过滤掉标点符号，并根据空格分割单词。

除文本外，还必须指定词汇量（总词数）。如果您打算编码包含其他单词的其他文档，则可以是文档中的单词总数或更多。词汇表的大小定义了散列单词的散列空间。理想情况下，这应该比词汇量大一些百分比（可能是 25％），以最大限度地减少碰撞次数。默认情况下，使用'hash'函数，虽然我们将在下一节中看到，但是在直接调用 hashing_trick（）函数时可以指定备用散列函数。

我们可以使用上一节中的 text_to_word_sequence（）函数将文档拆分为单词，然后使用集合仅表示文档中的唯一单词。该集的大小可用于估计一个文档的词汇表大小。

例如：

```py
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
```

我们可以将它与 one_hot（）函数放在一起，并对文档中的单词进行热编码。下面列出了完整的示例。

词汇大小增加三分之一，以最大限度地减少散列词时的冲突。

```py
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)
```

首先运行该示例将词汇表的大小打印为 8.然后将编码的文档打印为整数编码的单词数组。

```py
8
[5, 9, 8, 7, 9, 1, 5, 3, 8]
```

## 使用 hashing_trick 进行哈希编码

整数和计数基本编码的限制是它们必须保持单词的词汇表及其到整数的映射。

此方法的替代方法是使用单向散列函数将单词转换为整数。这避免了跟踪词汇表的需要，词汇表更快并且需要更少的内存。

Keras 提供了 [hashing_trick（）函数](https://keras.io/preprocessing/text/#hashing_trick)，它分词然后对文档进行整数编码，就像 one_hot（）函数一样。它提供了更大的灵活性，允许您将散列函数指定为“散列”（默认）或其他散列函数，例如内置的 md5 函数或您自己的函数。

下面是使用 md5 哈希函数对文档进行整数编码的示例。

```py
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)
```

运行该示例将打印词汇表的大小和整数编码的文档。

我们可以看到，使用不同的散列函数会导致单词的一致但不同的整数作为上一节中的 one_hot（）函数。

```py
8
[6, 4, 1, 2, 7, 5, 6, 2, 6]
```

## Tokenizer API

到目前为止，我们已经研究了使用 Keras 准备文本的一次性便捷方法。

Keras 提供了更复杂的 API，用于准备可以适合和重用以准备多个文本文档的文本。这可能是大型项目的首选方法。

Keras 提供 [Tokenizer 类](https://keras.io/preprocessing/text/#tokenizer)，用于为深度学习准备文本文档。必须构造 Tokenizer，然后将其放在原始文本文档或整数编码的文本文档上。

例如：

```py
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
```

适用后，Tokenizer 提供了 4 个属性，您可以使用这些属性查询有关文档的内容：

*   **word_counts** ：单词及其计数字典。
*   **word_docs** ：一个单词词典和每个出现的文档数量。
*   **word_index** ：单词字典及其唯一分配的整数。
*   **document_count** ：用于适合 Tokenizer 的文档总数的整数计数。

例如：

```py
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
```

一旦 Tokenizer 适合训练数据，它就可用于编码列车或测试数据集中的文档。

Tokenizer 上的 texts_to_matrix（）函数可用于为每个输入提供每个文档创建一个向量。向量的长度是词汇表的总大小。

此函数提供了一套标准的词袋模型文本编码方案，可以通过函数的模式参数提供。

可用的模式包括：

*   ' _binary_ '：文档中是否存在每个单词。这是默认值。
*   ' _count_ '：文档中每个单词的计数。
*   ' _tfidf_ '：文本频率 - 反向文档频率（TF-IDF）对文档中每个单词的评分。
*   ' _freq_ '：每个单词的频率，作为每个文档中单词的比例。

我们可以将所有这些与一个有效的例子放在一起。

```py
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)
```

运行该示例使 Tokenizer 与 5 个小文档相匹配。打印适合标记符的详细信息。然后使用字数对 5 个文档进行编码。

每个文档被编码为 9 元素向量，每个字具有一个位置，并且每个字位置具有所选择的编码方案值。在这种情况下，使用简单的字数计数模式。

```py
OrderedDict([('well', 1), ('done', 1), ('good', 1), ('work', 2), ('great', 1), ('effort', 1), ('nice', 1), ('excellent', 1)])
5
{'work': 1, 'effort': 6, 'done': 3, 'great': 5, 'good': 4, 'excellent': 8, 'well': 2, 'nice': 7}
{'work': 2, 'effort': 1, 'done': 1, 'well': 1, 'good': 1, 'great': 1, 'excellent': 1, 'nice': 1}
[[ 0\.  0\.  1\.  1\.  0\.  0\.  0\.  0\.  0.]
 [ 0\.  1\.  0\.  0\.  1\.  0\.  0\.  0\.  0.]
 [ 0\.  0\.  0\.  0\.  0\.  1\.  1\.  0\.  0.]
 [ 0\.  1\.  0\.  0\.  0\.  0\.  0\.  1\.  0.]
 [ 0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  1.]]
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [文本预处理 Keras API](https://keras.io/preprocessing/text/)
*   [text_to_word_sequence Keras API](https://keras.io/preprocessing/text/#text_to_word_sequence)
*   [one_hot Keras API](https://keras.io/preprocessing/text/#one_hot)
*   [hashing_trick Keras API](https://keras.io/preprocessing/text/#hashing_trick)
*   [Tokenizer Keras API](https://keras.io/preprocessing/text/#tokenizer)

## 摘要

在本教程中，您了解了如何使用 Keras API 为深度学习准备文本数据。

具体来说，你学到了：

*   关于可用于快速准备文本数据的便捷方法。
*   Tokenizer API，可以适用于训练数据，用于编码训练，验证和测试文档。
*   Tokenizer API 提供的 4 种不同文档编码方案的范围。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。