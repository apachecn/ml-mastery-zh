# 7 深度学习在自然语言处理中的应用

> 原文： [https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)

自然语言处理领域正在从统计方法转向神经网络方法。

在自然语言中仍有许多具有挑战性的问题需要解决。然而，深度学习方法正在某些特定语言问题上取得最新成果。

在最基本的问题上，最有趣的不仅仅是深度学习模型的表现;事实上，单个模型可以学习单词意义并执行语言任务，从而无需使用专门的手工制作方法。

在这篇文章中，您将发现 7 种有趣的自然语言处理任务，其中深度学习方法正在取得一些进展。

让我们开始吧。

![7 Applications of Deep Learning for Natural Language Processing](img/e22ba1f973034ee09669593794f429a6.jpg)

7 深度学习在自然语言处理中的应用
[Tim Gorman](https://www.flickr.com/photos/tim_gorman/7529595228/) 的照片，保留一些权利。

## 概观

在这篇文章中，我们将看看以下 7 种自然语言处理问题。

1.  文本分类
2.  语言建模
3.  语音识别
4.  标题生成
5.  机器翻译
6.  文件摘要
7.  问题回答

我试图关注你可能感兴趣的最终用户问题的类型，而不是更深入学习的学术或语言子问题，如词性标注，分块，命名实体识别，等等。

每个示例都提供了问题的描述，示例以及对演示方法和结果的论文的引用。大多数参考文献都来自 Goldberg 2015 年优秀的 [NLP 研究人员深度学习入门](https://arxiv.org/abs/1510.00726)。

您是否有未列出的最喜欢的深度学习 NLP 应用程序？
请在下面的评论中告诉我。

## 1.文本分类

给定文本示例，预测预定义的类标签。

> 文本分类的目标是对文档的主题或主题进行分类。

- 第 575 页，[统计自然语言处理基础](http://amzn.to/2ePBz9t)，1999。

一个流行的分类示例是[情绪分析](https://en.wikipedia.org/wiki/Sentiment_analysis)，其中类标签表示源文本的情绪基调，例如“_ 阳性 _”或“_ 阴性 _”。

以下是另外 3 个例子：

*   垃圾邮件过滤，将电子邮件文本分类为垃圾邮件。
*   语言识别，对源文本的语言进行分类。
*   流派分类，对虚构故事的类型进行分类。

此外，该问题可以以需要分配给文本的多个类的方式构成，即所谓的多标签分类。例如预测源推文的多个主题标签。

有关一般主题的更多信息，请参阅：

*   [Scholarpedia 上的文本分类](http://www.scholarpedia.org/article/Text_categorization)
*   [维基百科上的文件分类](https://en.wikipedia.org/wiki/Document_classification)

以下是用于文本分类的深度学习论文的 3 个示例：

*   腐烂番茄电影评论的情感分析。
    *   [深度无序构成对象文本分类的句法方法](https://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf)，2015。
*   亚马逊产品评论的情感分析，IMDB 电影评论和新闻文章的主题分类。
    *   [利用卷积神经网络有效使用词序进行文本分类](https://arxiv.org/abs/1412.1058)，2015。
*   电影评论的情感分析，将句子分类为主观或客观，分类问题类型，产品评论情绪等。
    *   [用于句子分类的卷积神经网络](https://arxiv.org/abs/1408.5882)，2014。

## 2.语言建模

语言建模实际上是更有趣的自然语言问题的子任务，特别是那些在其他输入上调整语言模型的问题。

> ......问题是预测下一个词给出的前一个词。该任务是语音或光学字符识别的基础，也用于拼写校正，手写识别和统计机器翻译。

- 第 191 页，[统计自然语言处理基础](http://amzn.to/2ePBz9t)，1999。

除了对语言建模的学术兴趣之外，它还是许多深度学习自然语言处理架构的关键组成部分。

语言模型学习单词之间的概率关系，使得可以生成与源文本在统计上一致的新的单词序列。

单独的语言模型可用于文本或语音生成;例如：

*   生成新文章标题。
*   生成新的句子，段落或文档。
*   生成建议的句子延续。

有关语言建模的更多信息，请参阅：

*   维基百科上的[语言模型](https://en.wikipedia.org/wiki/Language_model)
*   [回归神经网络的不合理有效性](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)，2015 年。
*   [基于生成模型的文本到语音合成](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%2010%20-%20Text%20to%20Speech.pdf)，第 10 讲，牛津，2017

以下是语言建模的深度学习示例（仅限）：

*   英语文本，书籍和新闻文章的语言模型。
    *   [神经概率语言模型](http://www.jmlr.org/papers/v3/bengio03a.html)，2003

## 3.语音识别

语音识别是理解所说内容的问题。

> 语音识别的任务是将包含口头自然语言话语的声学信号映射到由说话者预期的相应的单词序列中。

- 第 458 页，[深度学习](http://amzn.to/2uE7WvS)，2016 年。

给定文本作为音频数据的话语，模型必须产生人类可读的文本。

鉴于该过程的自动特性，该问题也可称为自动语音识别（ASR）。

语言模型用于创建以音频数据为条件的文本输出。

一些例子包括：

*   抄写演讲。
*   为电影或电视节目创建文字字幕。
*   在驾驶时向收音机发出命令。

有关语音识别的更多信息，请参阅：

*   [维基百科上的语音识别](https://en.wikipedia.org/wiki/Speech_recognition)

以下是语音识别深度学习的 3 个例子。

*   英语演讲文本。
    *   [连接主义时间分类：用递归神经网络标记未分段的序列数据](http://www.cs.toronto.edu/~graves/icml_2006.pdf)，2006。
*   英语演讲文本。
    *   [深度递归神经网络语音识别](https://arxiv.org/abs/1303.5778)，2013。
*   英语演讲文本。
    *   [探索用于语音识别的卷积神经网络结构和优化技术](https://www.microsoft.com/en-us/research/publication/exploring-convolutional-neural-network-structures-and-optimization-techniques-for-speech-recognition/)，2014。

## 4.标题生成

标题生成是描述图像内容的问题。

给定数字图像（例如照片），生成图像内容的文本描述。

语言模型用于创建以图像为条件的标题。

一些例子包括：

*   描述场景的内容。
*   为照片创建标题。
*   描述视频。

这不仅是针对听力受损者的应用，而且还是为可以搜索的图像和视频数据生成人类可读文本，例如在网络上。

以下是生成字幕的深度学习的 3 个示例：

*   生成照片的标题。
    *   [显示，参与和讲述：视觉注意的神经图像标题生成](https://arxiv.org/abs/1502.03044)，2016。
*   生成照片的标题。
    *   [显示和告诉：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。
*   为视频生成字幕。
    *   [序列到序列 - 视频到文本](https://arxiv.org/abs/1505.00487)，2015。

## 5.机器翻译

机器翻译是将一种语言的源文本转换为另一种语言的问题。

> 机器翻译，文本或语音从一种语言到另一种语言的自动翻译，是 NLP 最重要的应用之一。

- 第 463 页，[统计自然语言处理基础](http://amzn.to/2ePBz9t)，1999。

鉴于使用深度神经网络，该领域被称为神经机器翻译。

> 在机器翻译任务中，输入已经由某种语言的符号序列组成，并且计算机程序必须将其转换为另一种语言的符号序列。这通常适用于自然语言，例如从英语翻译成法语。深度学习最近开始对这类任务产生重要影响。

- 第 98 页，[深度学习](http://amzn.to/2uE7WvS)，2016 年。

语言模型用于以源文本为条件输出第二语言的目标文本。

一些例子包括：

*   将文本文档从法语翻译成英语。
*   将西班牙语音频翻译为德语文本。
*   将英文文本翻译为意大利语音频。

有关神经机器翻译的更多信息，请参阅：

*   [维基百科上的神经机器翻译](https://en.wikipedia.org/wiki/Neural_machine_translation)

以下是机器翻译深度学习的 3 个例子：

*   文本从英语翻译成法语。
    *   [用神经网络进行序列学习的序列](https://arxiv.org/abs/1409.3215)，2014。
*   文本从英语翻译成法语。
    *   [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014。
*   文本从英语翻译成法语。
    *   [联合语言和翻译建模与回归神经网络](https://www.microsoft.com/en-us/research/publication/joint-language-and-translation-modeling-with-recurrent-neural-networks/)，2013。

## 6.文件摘要

文档摘要是创建文本文档的简短描述的任务。

如上所述，语言模型用于输出以完整文档为条件的摘要。

文档摘要的一些示例包括：

*   为文档创建标题。
*   创建文档的摘要。

有关该主题的更多信息，请参阅：

*   [维基百科上的自动摘要](https://en.wikipedia.org/wiki/Automatic_summarization)。
*   [深度学习是否已应用于自动文本摘要（成功）？](https://www.quora.com/Has-Deep-Learning-been-applied-to-automatic-text-summarization-successfully)

以下是文档摘要深度学习的 3 个示例：

*   新闻文章中句子的总结。
    *   [抽象概括的神经注意模型](https://arxiv.org/abs/1509.00685)，2015。
*   新闻文章中句子的总结。
    *   [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。
*   新闻文章中句子的总结。
    *   [通过提取句子和单词进行神经总结](https://arxiv.org/abs/1603.07252)，2016。

## 7.问题回答

问题回答是给定主题（例如文本文档）回答关于主题的特定问题的问题。

> ...通过返回适当的无短语（例如位置，人或日期）来尝试回答以问题形式表达的用户查询的问答系统。例如，为何杀死肯尼迪总统的问题？可能会用名词短语 Oswald 回答

- 第 377 页，[统计自然语言处理基础](http://amzn.to/2ePBz9t)，1999。

一些例子包括：

*   [维基百科上的问题回答](https://en.wikipedia.org/wiki/Question_answering)

有关问题解答的更多信息，请参阅：

*   回答有关维基百科文章的问题。
*   回答有关新闻文章的问题。
*   回答有关医疗记录的问题。

以下是 3 个问答的深度学习示例：

*   回答有关新闻文章的问题。
    *   [教学机器阅读和理解](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend)，2015。
*   回答关于 freebase 文章的一般知识问题。
    *   [使用多列卷积神经网络回答 Freebase 问题](http://www.aclweb.org/anthology/P15-1026)，2015 年。
*   回答提供特定文件的事实问题。
    *   [答案句子选择的深度学习](https://arxiv.org/abs/1412.1632)，2015。

## 进一步阅读

如果您正在深入研究，本节将为 NLP 的深度学习应用程序提供更多资源。

*   [自然语言处理神经网络模型入门](https://arxiv.org/abs/1510.00726)，2015。
*   [自然语言处理（几乎）来自 Scratch](https://arxiv.org/abs/1103.0398) ，2011。
*   [自然语言处理的深度学习](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202b%20-%20Overview%20of%20the%20Practicals.pdf)，实用概述，牛津，2017 年
*   [哪些 NLP 问题已成功应用深度学习或神经网络？](https://www.quora.com/What-NLP-problems-has-deep-learning-or-neural-networks-been-applied-to-successfully)
*   [深度学习可以在自然语言处理方面取得类似的突破，就像它在视觉和视觉方面所做的那样。言语？](https://www.quora.com/Can-deep-learning-make-similar-breakthroughs-in-natural-language-processing-as-it-did-in-vision-speech)

## 摘要

在这篇文章中，您发现了 7 种深度学习应用于自然语言处理任务。

您最喜欢的 NLP 深度学习的例子是错过的吗？
请在评论中告诉我。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。