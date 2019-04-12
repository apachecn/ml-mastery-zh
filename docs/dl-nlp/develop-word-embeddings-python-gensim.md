# 如何使用 Gensim 在 Python 中开发词嵌入

> 原文： [https://machinelearningmastery.com/develop-word-embeddings-python-gensim/](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)

词嵌入是一种在自然语言处理中表示文本的现代方法。

嵌入算法如 word2vec 和 GloVe 是神经网络模型在机器翻译等自然语言处理问题上获得的最先进结果的关键。

在本教程中，您将了解如何使用 Gensim 在 Python 中为自然语言处理应用程序训练和加载字嵌入模型。

完成本教程后，您将了解：

*   如何在文本数据上训练自己的 word2vec 单词嵌入模型。
*   如何使用主成分分析可视化训练的单词嵌入模型。
*   如何加载谷歌和斯坦福的预训练 word2vec 和 GloVe 字嵌入模型。

让我们开始吧。

![How to Develop Word Embeddings in Python with Gensim](img/33cc3e32854514592a90afc05c1aa9c8.jpg)

如何使用 Gensim
在 Python 中开发词嵌入照片由 [dilettantiquity](https://www.flickr.com/photos/flyingblogspot/15361704293/) ，保留一些权利。

## 教程概述

本教程分为 6 个部分;他们是：

1.  词嵌入
2.  Gensim 库
3.  开发 Word2Vec 嵌入
4.  可视化词嵌入
5.  加载 Google 的 Word2Vec 嵌入
6.  加载斯坦福的 GloVe 嵌入

## 词嵌入

单词嵌入是一种提供单词的密集向量表示的方法，可以捕获关于其含义的单词。

字嵌入是对简单的字袋模型字编码方案（如字数和频率）的改进，这些方案导致描述文档而不是字的含义的大且稀疏的向量（通常为 0 值）。

词嵌入通过使用算法来训练一组基于大的文本语料库的固定长度密集和连续值向量。每个单词由嵌入空间中的一个点表示，并且这些点基于围绕目标单词的单词被学习和移动。

它正在由公司定义一个单词，它保留了允许单词嵌入来学习单词含义的东西。单词的向量空间表示提供了一个投影，其中具有相似含义的单词在空间内局部聚类。

在其他文本表示中使用单词嵌入是导致机器翻译等问题的深度神经网络突破性表现的关键方法之一。

在本教程中，我们将研究斯坦福大学的研究人员如何使用 Google 和 GloVe 的研究人员使用两种不同的词嵌入方法 word2vec。

## Gensim Python 库

[Gensim](https://radimrehurek.com/gensim/index.html) 是一个用于自然语言处理的开源 Python 库，主要关注主题建模。

它被称为：

> 人类主题建模

Gensim 由捷克自然语言处理研究员[RadimŘehůřek](https://www.linkedin.com/in/radimrehurek/)及其公司 [RaRe Technologies](https://rare-technologies.com/) 开发并维护。

它不是一个包括厨房水槽的 NLP 研究库（如 NLTK）;相反，Gensim 是一个成熟，专注，高效的 NLP 工具套件，用于主题建模。最值得注意的是，本教程支持 Word2Vec 单词嵌入的实现，用于从文本中学习新的单词向量。

它还提供了用于加载几种格式的预训练单词嵌入以及使用和查询加载嵌入的工具。

我们将在本教程中使用 Gensim 库。

如果您没有 Python 环境设置，可以使用本教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

使用 _pip_ 或 _easy_install_ 可以轻松安装 Gensim。

例如，您可以通过在命令行上键入以下内容来使用 pip 安装 Gensim：

```py
pip install --upgrade gensim
```

如果您需要在系统上安装 Gensim 的帮助，可以查看 [Gensim 安装说明](https://radimrehurek.com/gensim/install.html)。

## 开发 Word2Vec 嵌入

[Word2vec](https://en.wikipedia.org/wiki/Word2vec) 是一种用于从文本语料库中学习单词嵌入的算法。

有两种主要的训练算法可用于学习从文本嵌入;它们是连续的单词（CBOW）和跳过克。

我们不会深入研究算法，只是说它们通常会查看每个目标词的单词窗口，以提供上下文，反过来又意味着单词。该方法由 [Tomas Mikolov](https://en.wikipedia.org/wiki/Word2vec) 开发，以前在谷歌，目前在 Facebook。

Word2Vec 模型需要大量文本，例如整个维基百科语料库。然而，我们将使用一个小的内存中的文本示例来演示原理。

Gensim 提供 [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) 类用于处理 Word2Vec 模型。

学习从文本嵌入的单词涉及将文本加载和组织成句子并将它们提供给新的 _Word2Vec（）_ 实例的构造函数。例如：

```py
sentences = ...
model = Word2Vec(sentences)
```

具体地，每个句子必须被分词，意味着被分成单词并准备好（例如，可能预先过滤并且可能转换为优选情况）。

句子可以是加载到内存中的文本，也可以是逐步加载文本的迭代器，这是非常大的文本语料库所必需的。

这个构造函数有很多参数;您可能希望配置的一些值得注意的参数是：

*   **尺寸** :(默认为 100）嵌入的尺寸数，例如：表示每个标记（单词）的密集向量的长度。
*   **窗口** :(默认值 5）目标字与目标字周围的字之间的最大距离。
*   **min_count** :(默认值 5）训练模型时要考虑的最小字数;出现小于此计数的单词将被忽略。
*   **worker** :(默认 3）训练时使用的线程数。
*   **sg** :(默认为 0 或 CBOW）训练算法，CBOW（0）或跳过克（1）。

刚开始时，默认值通常足够好。如果您拥有大量核心，就像大多数现代计算机那样，我强烈建议您增加工作人员以匹配核心数量（例如 8）。

训练模型后，可通过“ _wv_ ”属性访问该模型。这是可以进行查询的实际单词向量模型。

例如，您可以打印所学习的令牌（单词）词汇，如下所示：

```py
words = list(model.wv.vocab)
print(words)
```

您可以查看特定标记的嵌入向量，如下所示：

```py
print(model['word'])
```

最后，通过调用单词向量模型上的 _save_word2vec_format（）_ 函数，可以将训练好的模型保存到文件中。

默认情况下，模型以二进制格式保存以节省空间。例如：

```py
model.wv.save_word2vec_format('model.bin')
```

入门时，您可以将学习的模型保存为 ASCII 格式并查看内容。

您可以通过在调用 _save_word2vec_format（）_ 函数时设置 _binary = False_ 来执行此操作，例如：

```py
model.wv.save_word2vec_format('model.txt', binary=False)
```

然后可以通过调用 _Word2Vec.load（）_ 函数再次加载保存的模型。例如：

```py
model = Word2Vec.load('model.bin')
```

我们可以将所有这些与一个有效的例子结合在一起。

我们不会从文件中加载大型文本文档或语料库，而是使用预先分词的小型内存列表。训练模型并将单词的最小计数设置为 1，这样就不会忽略单词。

学习模型后，我们总结，打印词汇表，然后为单词'_ 句子 _'打印单个向量。

最后，模型以二进制格式保存到文件中，加载，然后进行汇总。

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
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
```

运行该示例将打印以下输出。

```py
Word2Vec(vocab=14, size=100, alpha=0.025)
['second', 'sentence', 'and', 'this', 'final', 'word2vec', 'for', 'another', 'one', 'first', 'more', 'the', 'yet', 'is']
[ -4.61881841e-03  -4.88735968e-03  -3.19508743e-03   4.08568839e-03
  -3.38211656e-03   1.93076557e-03   3.90265253e-03  -1.04349572e-03
   4.14286414e-03   1.55219622e-03   3.85653134e-03   2.22428422e-03
  -3.52565176e-03   2.82056746e-03  -2.11121864e-03  -1.38054823e-03
  -1.12888147e-03  -2.87318649e-03  -7.99703528e-04   3.67874932e-03
   2.68940022e-03   6.31021452e-04  -4.36326629e-03   2.38655557e-04
  -1.94210222e-03   4.87691024e-03  -4.04118607e-03  -3.17813386e-03
   4.94802603e-03   3.43150692e-03  -1.44031656e-03   4.25637932e-03
  -1.15106850e-04  -3.73274647e-03   2.50349124e-03   4.28692997e-03
  -3.57313151e-03  -7.24728088e-05  -3.46099050e-03  -3.39612062e-03
   3.54845310e-03   1.56780297e-03   4.58260969e-04   2.52689526e-04
   3.06256465e-03   2.37558200e-03   4.06933809e-03   2.94650183e-03
  -2.96231941e-03  -4.47433954e-03   2.89590308e-03  -2.16034567e-03
  -2.58548348e-03  -2.06163677e-04   1.72605237e-03  -2.27384618e-04
  -3.70194600e-03   2.11557443e-03   2.03793868e-03   3.09839356e-03
  -4.71800892e-03   2.32995977e-03  -6.70911541e-05   1.39375112e-03
  -3.84263694e-03  -1.03898917e-03   4.13251948e-03   1.06330717e-03
   1.38514000e-03  -1.18144893e-03  -2.60811858e-03   1.54952740e-03
   2.49916781e-03  -1.95435272e-03   8.86975031e-05   1.89820060e-03
  -3.41996481e-03  -4.08187555e-03   5.88635216e-04   4.13103355e-03
  -3.25899688e-03   1.02130906e-03  -3.61028523e-03   4.17646067e-03
   4.65870230e-03   3.64110398e-04   4.95479070e-03  -1.29743712e-03
  -5.03367570e-04  -2.52546836e-03   3.31060472e-03  -3.12870182e-03
  -1.14580349e-03  -4.34387522e-03  -4.62882593e-03   3.19007039e-03
   2.88707414e-03   1.62976081e-04  -6.05802808e-04  -1.06368808e-03]
Word2Vec(vocab=14, size=100, alpha=0.025)
```

您可以看到，通过一些准备文本文档的工作，您可以使用 Gensim 轻松创建自己的单词嵌入。

## 可视化词嵌入

在学习了文本数据的单词嵌入之后，可以通过可视化来探索它。

您可以使用经典投影方法将高维字向量缩减为二维图并将其绘制在图形上。

可视化可以为您的学习模型提供定性诊断。

我们可以从训练有素的模型中检索所有向量，如下所示：

```py
X = model[model.wv.vocab]
```

然后我们可以在向量上训练投影方法，例如 scikit-learn 中提供的那些方法，然后使用 matplotlib 将投影绘制为散点图。

让我们看一下使用 Principal Component Analysis 或 PCA 的示例。

### 使用 PCA 绘制单词向量

我们可以使用 scikit-learn [PCA 类](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)创建单词向量的二维 PCA 模型，如下所示。

```py
pca = PCA(n_components=2)
result = pca.fit_transform(X)
```

可以使用 matplotlib 如下绘制得到的投影，将两个维度拉出为 x 和 y 坐标。

```py
pyplot.scatter(result[:, 0], result[:, 1])
```

我们可以更进一步，用图标本身注释图表上的点。没有任何良好偏移的粗略版本如下所示。

```py
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
```

将这一切与上一节中的模型结合在一起，下面列出了完整的示例。

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
# fit a 2d PCA model to the vectors
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

运行该示例将创建一个散点图，其中的点用单词注释。

鉴于这样一个小小的语料库用于拟合模型，很难从图中得出很多意义。

![Scatter Plot of PCA Projection of Word2Vec Model](img/4d6dd74b98672c2171973dfd36fe09bb.jpg)

Word2Vec 模型 PCA 投影的散点图

## 加载 Google 的 Word2Vec 嵌入

训练自己的单词向量可能是给定 NLP 问题的最佳方法。

但它可能需要很长时间，一台具有大量 RAM 和磁盘空间的快速计算机，并且可能在输入数据和训练算法方面具有一些专业知识。

另一种方法是简单地使用现有的预训练单词嵌入。

除了 word2vec 的论文和代码，谷歌还在 [Word2Vec 谷歌代码项目](https://code.google.com/archive/p/word2vec/)上发布了预训练的 word2vec 模型。

预先训练的模型只不过是包含令牌及其相关词向量的文件。预先训练的 Google word2vec 模型接受了谷歌新闻数据（约 1000 亿字）的训练;它包含 300 万个单词和短语，并且使用 300 维单词向量。

它是一个 1.53 千兆字节的文件。你可以在这里下载：

*   [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

解压缩后，二进制文件（GoogleNews-vectors-negative300.bin）为 3.4 千兆字节。

Gensim 库提供了加载此文件的工具。具体来说，您可以调用 _KeyedVectors.load_word2vec_format（）_ 函数将此模型加载到内存中，例如：

```py
from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
```

在我的现代工作站上，加载大约需要 43 秒。

你可以做的另一个有趣的事情是用单词做一点线性代数算法。

例如，讲座和介绍文章中描述的一个流行的例子是：

```py
queen = (king - man) + woman
```

这就是女王这个词是最接近的一个词，因为人们从国王那里减去了男人的概念，并添加了女人这个词。国王的“男人”被“女人”所取代，给了我们女王。一个非常酷的概念。

Gensim 提供了一个接口，用于在训练或加载的模型上的 _most_similar（）_ 函数中执行这些类型的操作。

例如：

```py
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
```

我们可以将所有这些放在一起如下。

```py
from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
```

运行该示例加载 Google 预训练的 word2vec 模型，然后计算（王者）+女人=？对这些单词的单词向量进行操作。

正如我们所料，答案是女王。

```py
[('queen', 0.7118192315101624)]
```

请参阅更多阅读部分中的一些帖子，了解您可以探索的更有趣的算术示例。

## 加载斯坦福的 GloVe 嵌入

斯坦福大学的研究人员也有自己的单词嵌入算法，如 word2vec，称为[全局向量字表示](https://nlp.stanford.edu/projects/glove/)，或简称 GloVe。

我不会在这里详细介绍 word2vec 和 GloVe 之间的差异，但一般来说，NLP 从业者似乎更喜欢基于结果的 GloVe。

与 word2vec 一样，GloVe 研究人员也提供预训练的单词向量，在这种情况下，可供选择。

您可以下载 GloVe 预训练的单词向量，并使用 gensim 轻松加载它们。

第一步是将 GloVe 文件格式转换为 word2vec 文件格式。唯一的区别是添加了一个小标题行。这可以通过调用 _glove2word2vec（）_ 函数来完成。例如：

```py
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
```

转换后，文件可以像上面的 word2vec 文件一样加载。

让我们以一个例子来具体化。

您可以从 [GloVe 网站](https://nlp.stanford.edu/projects/glove/)下载最小的 GloVe 预训练模型。它是一个 822 兆字节的 zip 文件，有 4 种不同的模型（50,100,200 和 300 维向量），在维基百科数据上训练有 60 亿个令牌和 400,000 个单词词汇。

直接下载链接在这里：

*   [手套.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

使用模型的 100 维版本，我们可以将文件转换为 word2vec 格式，如下所示：

```py
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
```

您现在拥有 word2vec 格式的 GloVe 模型副本，文件名为 _glove.6B.100d.txt.word2vec_ 。

现在我们可以加载它并执行相同的（国王 - 男人）+女人=？按照上一节进行测试。完整的代码清单如下。请注意，转换后的文件是 ASCII 格式，而不是二进制格式，因此我们在加载时设置 _binary = False_ 。

```py
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
```

运行该示例将打印“queen”的相同结果。

```py
[('queen', 0.7698540687561035)]
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [词嵌入维基百科](https://en.wikipedia.org/wiki/Word2vec)
*   [维基百科上的 Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
*   [谷歌 word2vec 项目](https://code.google.com/archive/p/word2vec/)
*   [Stanford GloVe 项目](https://nlp.stanford.edu/projects/glove/)

### Gensim

*   [gensim Python Library](https://radimrehurek.com/gensim/index.html)
*   [models.word2vec gensim API](https://radimrehurek.com/gensim/models/keyedvectors.html)
*   [models.keyedvectors gensim API](https://radimrehurek.com/gensim/models/keyedvectors.html)
*   [scripts.glove2word2vec gensim API](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)

### 帖子

*   [使用 Word2vec](https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/) ，2016 年
*   [数字人文学科的向量空间模型](http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html)，2015
*   [Gensim Word2vec 教程](https://rare-technologies.com/word2vec-tutorial/)，2014

## 摘要

在本教程中，您了解了如何使用 Gensim 在 Python 中开发和加载字嵌入层。

具体来说，你学到了：

*   如何在文本数据上训练自己的 word2vec 单词嵌入模型。
*   如何使用主成分分析可视化训练的单词嵌入模型。
*   如何加载谷歌和斯坦福的预训练 word2vec 和 GloVe 字嵌入模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。