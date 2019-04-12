# 自然语言处理的数据集

> 原文： [https://machinelearningmastery.com/datasets-natural-language-processing/](https://machinelearningmastery.com/datasets-natural-language-processing/)

在开始深入学习自然语言处理任务时，您需要数据集来练习。

最好使用可以快速下载的小型数据集，并且不需要太长时间来适应模型。此外，使用易于理解和广泛使用的标准数据集也很有帮助，这样您就可以比较结果，看看您是否在取得进展。

在这篇文章中，您将发现一套用于自然语言处理任务的标准数据集，您可以在深入学习入门时使用这些数据集。

### 概观

这篇文章分为 7 个部分;他们是：

1.  文本分类
2.  语言建模
3.  图像标题
4.  机器翻译
5.  问题回答
6.  语音识别
7.  文件摘要

我试图提供一种混合的数据集，这些数据集很受欢迎，适用于规模适中的学术论文。

几乎所有数据集都可以免费下载。

如果您没有列出您最喜欢的数据集，或者您认为您知道应该列出的更好的数据集，请在下面的评论中告诉我。

让我们开始吧。

![Datasets for Natural Language Processing](img/5035bb8bcd75bf878dc5c012041baddc.jpg)

自然语言处理数据集
照[格兰特](https://www.flickr.com/photos/visual_dichotomy/2400003250/)，保留一些权利。

## 1.文本分类

文本分类是指标记句子或文档，例如电子邮件垃圾邮件分类和情感分析。

下面是一些很好的初学者文本分类数据集。

*   [路透社 Newswire 主题分类](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html)（路透社-21578）。 1987 年路透社出现的一系列新闻文件，按类别编制索引。 [另见 RCV1，RCV2 和 TRC2](http://trec.nist.gov/data/reuters/reuters.html) 。
*   [IMDB 电影评论情感分类](http://ai.stanford.edu/~amaas/data/sentiment/)（斯坦福）。来自网站 imdb.com 的一系列电影评论及其积极或消极的情感。
*   [新闻集团电影评论情感分类](http://www.cs.cornell.edu/people/pabo/movie-review-data/)（康奈尔）。来自网站 imdb.com 的一系列电影评论及其积极或消极的情感。

有关更多信息，请参阅帖子：

*   [单标签文本分类的数据集。](http://ana.cachopo.org/datasets-for-single-label-text-categorization)

## 2.语言建模

语言建模涉及开发一种统计模型，用于预测句子中的下一个单词或单词中的下一个单词。它是语音识别和机器翻译等任务中的前置任务。

它是语音识别和机器翻译等任务中的前置任务。

下面是一些很好的初学者语言建模数据集。

*   [Project Gutenberg](https://www.gutenberg.org/) ，这是一系列免费书籍，可以用纯文本检索各种语言。

还有更多正式的语料库得到了很好的研究;例如：

*   [布朗大学现代美国英语标准语料库](https://en.wikipedia.org/wiki/Brown_Corpus)。大量英语单词样本。
*   [谷歌 10 亿字语料库](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark)。

## 3.图像标题

图像字幕是为给定图像生成文本描述的任务。

下面是一些很好的初学者图像字幕数据集。

*   [上下文中的通用对象（COCO）](http://mscoco.org/dataset/#overview)。包含超过 12 万张描述图像的集合
*   [Flickr 8K](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html) 。从 flickr.com 获取的 8 千个描述图像的集合。
*   [Flickr 30K](http://shannon.cs.illinois.edu/DenotationGraph/) 。从 flickr.com 获取的 3 万个描述图像的集合。

欲了解更多，请看帖子：

*   [探索图像字幕数据集](http://sidgan.me/technical/2016/01/09/Exploring-Datasets)，2016 年

## 4.机器翻译

机器翻译是将文本从一种语言翻译成另一种语言的任务。

下面是一些很好的初学者机器翻译数据集。

*   [加拿大第 36 届议会的协调议长](https://www.isi.edu/natural-language/download/hansard/)。成对的英语和法语句子。
*   [欧洲议会诉讼平行语料库 1996-2011](http://www.statmt.org/europarl/) 。句子对一套欧洲语言。

有大量标准数据集用于年度机器翻译挑战;看到：

*   [统计机器翻译](http://www.statmt.org/)

## 5.问题回答

问答是一项任务，其中提供了一个句子或文本样本，从中提出问题并且必须回答问题。

下面是一些很好的初学者问题回答数据集。

*   [斯坦福问题答疑数据集（SQuAD）](https://rajpurkar.github.io/SQuAD-explorer/)。回答有关维基百科文章的问题。
*   [Deepmind Question Answering Corpus](https://github.com/deepmind/rc-data) 。从每日邮报回答有关新闻文章的问题。
*   [亚马逊问答数据](http://jmcauley.ucsd.edu/data/amazon/qa/)。回答关于亚马逊产品的问题。

有关更多信息，请参阅帖子：

*   [数据集：我如何获得问答网站的语料库，如 Quora 或 Yahoo Answers 或 Stack Overflow 来分析答案质量？](https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality)

## 6.语音识别

语音识别是将口语的音频转换为人类可读文本的任务。

下面是一些很好的初学者语音识别数据集。

*   [TIMIT 声 - 语音连续语音语料库](https://catalog.ldc.upenn.edu/LDC93S1)。不是免费的，但因其广泛使用而上市。口语美国英语和相关的转录。
*   [VoxForge](http://voxforge.org/) 。用于构建用于语音识别的开源数据库的项目。
*   [LibriSpeech ASR 语料库](http://www.openslr.org/12/)。从 [LibriVox](https://librivox.org/) 中收集的大量英语有声读物。

你知道一些更好的自动语音识别数据集吗？
请在评论中告诉我。

## 7.文件摘要

文档摘要是创建较大文档的简短有意义描述的任务。

下面是一些很好的初学者文档摘要数据集。

*   [法律案例报告数据集](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports)。收集了 4000 份法律案件及其摘要。
*   [TIPSTER 文本摘要评估会议语料库](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html)。收集了近 200 份文件及其摘要。
*   [英语新闻文本的 AQUAINT 语料库](https://catalog.ldc.upenn.edu/LDC2002T31)。不是免费的，而是广泛使用的。新闻文章的语料库。

欲了解更多信息：

*   [文件理解会议（DUC）任务](http://www-nlpir.nist.gov/projects/duc/data.html)。
*   [我在哪里可以找到文本摘要的好数据集？](https://www.quora.com/Where-can-I-find-good-data-sets-for-text-summarization)

## 进一步阅读

如果您希望更深入，本节提供了其他数据集列表。

*   [维基百科研究中使用的文本数据集](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research#Text_data)
*   [数据集：计算语言学家和自然语言处理研究人员使用的主要文本语料库是什么？](https://www.quora.com/Datasets-What-are-the-major-text-corpora-used-by-computational-linguists-and-natural-language-processing-researchers-and-what-are-the-characteristics-biases-of-each-corpus)
*   [斯坦福统计自然语言处理语料库](https://nlp.stanford.edu/links/statnlp.html#Corpora)
*   [按字母顺序排列的 NLP 数据集](https://github.com/niderhoff/nlp-datasets)
*   [NLTK Corpora](http://www.nltk.org/nltk_data/)
*   [DL4J 深度学习开放数据](https://deeplearning4j.org/opendata)

你知道其他任何自然语言处理数据集的好名单吗？
请在下面的评论中告诉我。

## 摘要

在这篇文章中，您发现了一套标准数据集，您可以在深入学习入门时用于自然语言处理任务。

你选择了一个数据集吗？您使用上述数据集之一吗？
请在下面的评论中告诉我。