# 如何开始深度学习自然语言处理（7 天迷你课程）

> 原文： [https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/](https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/)

## NLP 速成班的深度学习。

#### 在 7 天内为您的文本数据项目带来深度学习方法。

我们充斥着文字，包括书籍，论文，博客，推文，新闻，以及来自口头发言的越来越多的文字。

处理文本很难，因为它需要利用来自不同领域的知识，如语言学，机器学习，统计方法，以及如今的深度学习。

深度学习方法开始在一些具有挑战性的自然语言处理问题上超越经典和统计方法，使用单一和简单的模型。

在本速成课程中，您将了解如何在 7 天内使用 Python 开始并自信地开发自然语言处理问题的深度学习。

这是一个重要且重要的帖子。您可能想要将其加入书签。

让我们开始吧。

![How to Get Started with Deep Learning for Natural Language Processing](img/66190c5e18dacc8bf2d7536ffde3013a.jpg)

如何开始深度学习自然语言处理
照片由 [Daniel R. Blume](https://www.flickr.com/photos/drb62/2054107736/) ，保留一些权利。

## 谁是这个速成课？

在我们开始之前，让我们确保您在正确的位置。

以下列表提供了有关本课程设计对象的一般指导原则。

如果你没有完全匹配这些点，请不要惊慌，你可能只需要在一个或另一个区域刷新以跟上。

**你需要知道：**

*   你需要了解基本的 Python，NumPy 和 Keras 的深度学习方法。

**你不需要知道：**

*   你不需要成为一个数学家！
*   你不需要成为一名深度学习专家！
*   你不需要成为一名语言学家！

这个速成课程将带您从了解机器学习的开发人员到可以为您自己的自然语言处理项目带来深度学习方法的开发人员。

注意：此速成课程假设您有一个有效的 Python 2 或 3 SciPy 环境，至少安装了 NumPy，Pandas，scikit-learn 和 Keras 2。如果您需要有关环境的帮助，可以按照此处的分步教程进行操作：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 速成课程概述

这个速成课程分为 7 节课。

您可以每天完成一节课（推荐）或在一天内完成所有课程（硬核）。这取决于你有空的时间和你的热情程度。

以下是 7 个课程，通过深入学习 Python 中的自然语言处理，可以帮助您开始并提高工作效率：

1.  **第 01 课**：深度学习和自然语言
2.  **第 02 课**：清理文本数据
3.  **第 03 课**：Bag-of-Words 模型
4.  **第 04 课**：词嵌入表示
5.  **第 05 课**：学习嵌入
6.  **第 06 课**：文本分类
7.  **第 07 课**：电影评论情感分析项目

每节课可能需要 60 秒或 30 分钟。花点时间，按照自己的进度完成课程。在下面的评论中提出问题甚至发布结果。

课程期望你去学习如何做事。我会给你提示，但每节课的部分内容是强迫你学习去哪里寻求帮助以及深入学习，自然语言处理和 Python 中最好的工具（提示，我直接在这个博客上有所有答案，使用搜索框）。

我确实以相关帖子的链接形式提供了更多帮助，因为我希望你建立一些信心和惯性。

在评论中发布您的结果，我会为你欢呼！

挂在那里，不要放弃。

**注**：这只是一个速成课程。有关更多细节和 30 个充实的教程，请参阅我的书，主题为“ _[深度学习自然语言处理](https://machinelearningmastery.com/deep-learning-for-nlp/)”。_

## 第一课：深度学习和自然语言

在本课程中，您将发现自然语言，深度学习的简明定义以及使用文本数据进行深度学习的承诺。

### 自然语言处理

自然语言处理（简称 NLP）被广义地定义为通过软件自动操纵自然语言，如语音和文本。

自然语言处理的研究已经存在了 50 多年，随着计算机的兴起，语言学领域逐渐兴起。

理解文本的问题没有解决，也可能永远不会，主要是因为语言混乱。规则很少。然而，我们可以在大多数时间轻松地相互理解。

### 深度学习

深度学习是机器学习的一个子领域，涉及受大脑结构和功能激发的算法，称为人工神经网络。

深度学习的一个特性是这些类型的模型的表现通过增加其深度或代表能力来训练它们的更多示例而得到改善。

除了可扩展性之外，深度学习模型的另一个经常被引用的好处是它们能够从原始数据执行自动特征提取，也称为特征学习。

### NLP 深度学习的承诺

深度学习方法在自然语言中很受欢迎，主要是因为它们兑现了他们的承诺。

深度学习的第一次大型演示是自然语言处理，特别是语音识别。最近在机器翻译。

自然语言处理深度学习的三个关键承诺如下：

*   **特色学习的承诺**。也就是说，深度学习方法可以从模型所需的自然语言中学习特征，而不是要求专家指定和提取特征。
*   **持续改进的承诺**。也就是说，自然语言处理中的深度学习的表现基于实际结果，并且改进似乎在继续并且可能加速。
*   **端到端模型的承诺**。也就是说，大型端到端深度学习模型可以适应自然语言问题，提供更通用，表现更好的方法。

自然语言处理不是“解决”，但需要深入学习才能使您掌握该领域中许多具有挑战性的问题的最新技术。

### 你的任务

在本课程中，您必须研究并列出深度学习方法在自然语言处理领域的 10 个令人印象深刻的应用。如果您可以链接到演示该示例的研究论文，则可获得奖励积分。

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [什么是自然语言处理？](https://machinelearningmastery.com/natural-language-processing/)
*   [什么是深度学习？](https://machinelearningmastery.com/what-is-deep-learning/)
*   [深度学习对自然语言处理的承诺](https://machinelearningmastery.com/promise-deep-learning-natural-language-processing/)
*   [7 深度学习在自然语言处理中的应用](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)

在下一课中，您将了解如何清理文本数据以便为建模做好准备。

## 第 02 课：清理文本数据

在本课程中，您将了解如何加载和清理文本数据，以便可以手动和使用 NLTK Python 库进行建模。

### 文字很乱

你不能直接从原始文本到适合机器学习或深度学习模型。

您必须首先清理文本，这意味着将其拆分为单词并规范化问题，例如：

*   大写和小写字符。
*   单词内部和周围的标点符号。
*   金额和日期等数字。
*   拼写错误和区域变化。
*   Unicode 字符
*   以及更多…

### 手动标记

一般来说，我们指的是将原始文本转换为我们可以建模为“分词”的东西的过程，其中我们留下了单词列表或“标记”。

我们可以手动开发 Python 代码来清理文本，并且这通常是一种很好的方法，因为每个文本数据集必须以独特的方式进行分词。

例如，下面的代码片段将加载文本文件，按空格分割标记并将每个标记转换为小写。

```py
filename = '...'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lowercase
words = [word.lower() for word in words]
```

您可以想象如何扩展此代码段以处理和规范化 Unicode 字符，删除标点符号等。

### NLTK 分词

许多用于标记原始文本的最佳实践已被捕获并在名为 Natural Language Toolkit 或 NLTK 的 Python 库中提供。

您可以使用 pip 在命令行上键入以下命令来安装此库：

```py
sudo pip install -U nltk
```

安装后，还必须通过 Python 脚本安装库使用的数据集：

```py
import nltk
nltk.download()
```

或通过命令行：

```py
python -m nltk.downloader all
```

安装后，您可以使用 API​​来标记文本。例如，下面的代码段将加载并分词 ASCII 文本文件。

```py
# load data
filename = '...'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
```

此库中有许多工具，您可以使用自己的手动方法进一步优化干净的令牌，例如删除标点，删除停用词，词干等等。

### 你的任务

您的任务是在 Project Gutenberg 网站上找到一本免费的经典书籍，下载书籍的 ASCII 版本并将文本分词并将结果保存到新文件中。探索手动和 NLTK 方法的加分点。

在下面的评论中发布您的代码。我很想看看你选择哪本书以及你如何选择它来标记它。

### 更多信息

*   [Gutenberg 项目](http://www.gutenberg.org/)
*   [nltk.tokenize 包 API](http://www.nltk.org/api/nltk.tokenize.html)
*   [如何使用 Python 清理机器学习文本](https://machinelearningmastery.com/clean-text-machine-learning-python/)

在下一课中，您将发现词袋模型。

## 第 03 课：词袋模型

在本课程中，您将发现单词模型包以及如何使用此模型对文本进行编码，以便您可以使用 scikit-learn 和 Keras Python 库来训练模型。

### 一袋词

词袋模型是一种在使用机器学习算法对文本建模时表示文本数据的方式。

该方法非常简单和灵活，并且可以以多种方式用于从文档中提取特征。

词袋是文本的表示，用于描述文档中单词的出现。

选择词汇表，其中可能丢弃一些不经常使用的词。然后使用对于词汇表中的每个单词具有一个位置的向量以及在文档中出现（或不出现）的每个已知单词的分数来表示给定的文本文档。

它被称为单词的“包”，因为有关文档中单词的顺序或结构的任何信息都被丢弃。该模型仅关注文档中是否出现已知单词，而不是文档中的位置。

## 带有 scikit-learn 的词汇

用于机器学习的 scikit-learn Python 库提供了用于为词袋模型编码文档的工具。

可以创建编码器的实例，在文本文档集上训练，然后反复使用以编码训练，测试，验证以及需要为您的模型编码的任何新数据。

有一个编码器根据他们的计数得分单词，称为 CountVectorizer，一个用于使用每个单词的哈希函数来减少称为 HashingVectorizer 的向量长度，以及一个使用基于文档中单词出现的得分和反向出现的单词。所有文件称为 TfidfVectorizer。

下面的代码段显示了如何训练 TfidfVectorizer 字袋编码器并使用它来编码多个小文本文档。

```py
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

### 与 Keras 的词袋

用于深度学习的 Keras Python 库还提供了使用 Tokenizer 类中的 bag-of words-model 对文本进行编码的工具。

如上所述，编码器必须在源文档上进行训练，然后可用于对将来的训练数据，测试数据和任何其他数据进行编码。 API 还具有在对单词进行编码之前执行基本分词的优点。

下面的代码段演示了如何使用 Keras API 和单词的“计数”类型评分来训练和编码一些小型文本文档。

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

### 你的任务

您在本课程中的任务是尝试使用 scikit-learn 和 Keras 方法为单词包模型编码小型设计文本文档。如果您使用文档的小型标准文本数据集进行练习并执行数据清理作为准备工作的一部分，则可获得奖励积分。

在下面的评论中发布您的代码。我很想看看您探索和演示的 API。

### 更多信息

*   [对词袋模型的温和介绍](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
*   [如何使用 scikit-learn](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/) 为机器学习准备文本数据
*   [如何使用 Keras](https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/) 为深度学习准备文本数据

在下一课中，您将发现单词嵌入。

## 第 04 课：词嵌入表示法

在本课程中，您将发现嵌入分布式表示的单词以及如何使用 Gensim Python 库开发单词嵌入。

### 词嵌入

词嵌入是一种单词表示，允许具有相似含义的单词具有相似的表示。

它们是文本的分布式表示，这可能是深度学习方法在挑战自然语言处理问题上令人印象深刻的表现的关键突破之一。

单词嵌入方法从文本语料库中学习预定义固定大小的词汇表的实值向量表示。

### 训练词嵌入

您可以使用 Gensim Python 库训练嵌入分布式表示的单词，以进行主题建模。

Gensim 提供了 word2vec 算法的实现，该算法是在 Google 开发的，用于快速训练来自文本文档的字嵌入表示，

您可以在命令行中键入以下内容，使用 pip 安装 Gensim：

```py
pip install -U gensim
```

下面的代码段显示了如何定义一些人为的句子并在 Gensim 中训练一个嵌入表示的单词。

```py
from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
```

### 使用嵌入

一旦经过训练，嵌入就可以保存到文件中，作为另一个模型的一部分，例如深度学习模型的前端。

您还可以绘制单词的分布式表示的投影，以了解模型如何相信单词的相关性。您可以使用的常见投影技术是主成分分析或 PCA，可在 scikit-learn 中使用。

下面的代码段显示了如何训练单词嵌入模型，然后绘制词汇表中所有单词的二维投影。

```py
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

### 你的任务

您在本课程中的任务是使用 Gensim 在文本文档上训练单词嵌入，例如来自 Project Gutenberg 的书籍。如果您可以生成常用单词的图表，则可以获得奖励积分。

在下面的评论中发布您的代码。我很想看看你选择哪本书以及你学习嵌入的任何细节。

### 更多信息

*   [什么是词嵌入文本？](https://machinelearningmastery.com/what-are-word-embeddings/)
*   [如何使用 Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/) 在 Python 中开发词嵌入
*   [Gutenberg 项目](http://www.gutenberg.org/)

在下一课中，您将了解如何将词嵌入作为深度学习模型的一部分进行学习。

## 第 05 课：学习嵌入

在本课程中，您将学习如何学习嵌入字的分布式表示的单词，作为拟合深度学习模型的一部分

### 嵌入层

Keras 提供了一个嵌入层，可用于文本数据的神经网络。

它要求输入数据是整数编码的，以便每个单词由唯一的整数表示。可以使用 Keras 提供的 Tokenizer API 来执行该数据准备步骤。

使用随机权重初始化嵌入层，并将学习训练数据集中所有单词的嵌入。你必须指定`input_dim`，这是词汇量的大小，`output_dim`是嵌入的向量空间的大小，可选择`input_length`是输入序列中的单词数。

```py
layer = Embedding(input_dim, output_dim, input_length=??)
```

或者，更具体地，200 个单词的词汇表，32 维的分布式表示和 50 个单词的输入长度。

```py
layer = Embedding(200, 32, input_length=50)
```

### 嵌入模型

嵌入层可以用作深度学习模型的前端，以提供丰富的单词分布式表示，重要的是，这种表示可以作为训练深度学习模型的一部分来学习。

例如，下面的代码片段将定义和编译具有嵌入输入层和密集输出层的神经网络，用于文档分类问题。

当模型被训练有关填充文档及其相关输出标签的示例时，网络权重和分布式表示将被调整到特定数据。

```py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define problem
vocab_size = 100
max_length = 32
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
```

还可以使用预先训练的权重来初始化嵌入层，例如由 Gensim 准备的权重，并将层配置为不可训练的。如果可以使用非常大的文本语料库来预先训练单词嵌入，则该方法可能是有用的。

### 你的任务

您在本课程中的任务是设计一个小型文档分类问题，其中包含 10 个文档，每个文档包含一个句子以及相关的正面和负面结果标签，并使用单词嵌入这些数据来训练网络。请注意，在使用 Keras pad_sequences（）函数训练模型之前，需要将每个句子填充到相同的最大长度。如果您加载使用 Gensim 准备的预训练单词嵌入，则可获得奖励积分。

在下面的评论中发布您的代码。我很想看看你设法的句子和模特的技巧。

### 更多信息

*   [可变长度输入序列的数据准备](https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)
*   [如何使用 Keras 深入学习使用词嵌入层](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

在下一课中，您将了解如何开发用于分类文本的深度学习模型。

## 第 06 课：文本分类

在本课程中，您将发现标准的深度学习模型，用于对文本情感分析等问题上使用的文本进行分类。

### 文件分类

文本分类描述了一类问题，例如预测推文和电影评论的情感，以及将电子邮件分类为垃圾邮件。

它是自然语言处理的一个重要领域，也是开始在文本数据上使用深度学习技术的好地方。

深度学习方法在文本分类方面证明非常好，在一系列标准学术基准问题上实现了最先进的结果。

### 嵌入+ CNN

文本分类的操作方法涉及使用单词嵌入来表示单词，使用卷积神经网络或 CNN 来学习如何区分分类问题的文档。

该架构由三个关键部分组成：

*   **单词嵌入模型**：单词的分布式表示，其中具有相似含义的不同单词（基于其用法）也具有相似的表示。
*   **卷积模型**：一种特征提取模型，用于学习从使用单词嵌入表示的文档中提取显着特征。
*   **完全连接模型**：根据预测输出解释提取的特征。

这种类型的模型可以在 Keras Python 深度学习库中定义。下面的代码段显示了一个深度学习模型示例，用于将文本文档分类为两个类之一。

```py
# define problem
vocab_size = 100
max_length = 200
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

### 你的任务

您在本课程中的任务是研究使用嵌入+ CNN 深度学习方法组合进行文本分类，并报告配置此模型的示例或最佳实践，例如层数，内核大小，词汇量大小等等。

如果您可以通过改变内核大小找到并描述支持 n-gram 或多组单词作为输入的变体，则可获得奖励积分。

在下面的评论中发布您的发现。我很乐意看到你发现了什么。

### 更多信息

*   [深度学习文档分类的最佳实践](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

在下一课中，您将了解如何处理情感分析预测问题。

## 第 07 课：电影评论情感分析项目

在本课程中，您将了解如何准备文本数据，开发和评估深度学习模型以预测电影评论的情感。

我希望您将在此速成课程中学到的所有内容联系在一起，并通过端到端的实际问题进行处理。

### 电影评论数据集

电影评论数据集是 Bo Pang 和 Lillian Lee 在 21 世纪初从 imdb.com 网站上检索到的电影评论的集合。收集的评论作为他们自然语言处理研究的一部分。

您可以从此处下载数据集：

*   [电影评论 Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) （review_polarity.tar.gz，3MB）

从该数据集中，您将开发情感分析深度学习模型，以预测给定的电影评论是正面还是负面。

### 你的任务

您在本课程中的任务是开发和评估电影评论数据集中的深度学习模型：

1.  下载并检查数据集。
2.  清理并标记文本并将结果保存到新文件。
3.  将干净的数据拆分为训练和测试数据集。
4.  在训练数据集上开发嵌入+ CNN 模型。
5.  评估测试数据集上的模型。

如果您可以通过对新的电影评论做出预测，设计或真实来展示您的模型，那么奖励积分。如果您可以将您的模型与神经词袋模型进行比较，则可获得额外奖励积分。

在下面的评论中发布您的代码和模型技能。我很想看看你能想出什么。更简单的模型是首选，但也尝试深入，看看会发生什么。

### 更多信息

*   [如何为情感分析准备电影评论数据](https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/)
*   [如何开发一种用于预测电影评论情感的深度学习词袋模型](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)
*   [如何开发用于预测电影评论情感的词嵌入模型](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)

## 结束！
（看你有多远）

你做到了。做得好！

花点时间回顾一下你到底有多远。

你发现：

*   什么是自然语言处理，以及深度学习对该领域的承诺和影响。
*   如何手动清理和标记原始文本数据，并使用 NLTK 使其为建模做好准备。
*   如何使用带有 scikit-learn 和 Keras 库的词袋模型对文本进行编码。
*   如何训练使用 Gensim 库嵌入单词的分布式表示的单词。
*   如何学习嵌入分布式表示的单词作为拟合深度学习模型的一部分。
*   如何使用卷积神经网络的词嵌入来解决文本分类问题。
*   如何使用深度学习方法端到端地处理真实的情感分析问题。

这只是您深入学习自然语言处理之旅的开始。继续练习和发展你的技能。

下一步，查看我的[关于 NLP](https://machinelearningmastery.com/deep-learning-for-nlp/) 深度学习的书。

## 摘要

**你是如何使用迷你课程的？**
你喜欢这个速成班吗？

**你有什么问题吗？有没有任何问题？**
让我知道。在下面发表评论。