# 使用于文本摘要的编码器 - 解码器深度学习模型

> 原文： [https://machinelearningmastery.com/encoder-decoder-deep-learning-models-text-summarization/](https://machinelearningmastery.com/encoder-decoder-deep-learning-models-text-summarization/)

文本摘要是从较大的文本文档创建简短，准确和流畅的摘要的任务。

最近深度学习方法已被证明在文本摘要的抽象方法中是有效的。

在这篇文章中，您将发现三种不同的模型，它们构建在有效的编码器 - 解码器架构之上，该架构是为机器翻译中的序列到序列预测而开发的。

阅读这篇文章后，你会知道：

*   Facebook AI Research 模型使用编码器 - 解码器模型和卷积神经网络编码器。
*   使用 Encoder-Decoder 模型的 IBM Watson 模型，具有指向和分层注意力。
*   斯坦福/谷歌模型使用带有指向和覆盖的编码器 - 解码器模型。

让我们开始吧。

![Encoder-Decoder Deep Learning Models for Text Summarization](img/3c52998a2f284f4cb4fabfab639712d7.jpg)

用于文本摘要的编码器 - 解码器深度学习模型
照片由[HiếuBùi](https://www.flickr.com/photos/thphoto1788/34624647625/)拍摄，保留一些权利。

## 型号概述

我们将查看三种不同的文本摘要模型，这些模型以撰写本文时作者所属的组织命名：

1.  Facebook 模型
2.  IBM 模型
3.  谷歌模型

## Facebook 模型

Alexander Rush 等人描述了这种方法。来自 Facebook AI Research（FAIR）的 2015 年论文“[用于抽象句子摘要的神经注意模型](https://arxiv.org/abs/1509.00685)”。

该模型是为句子摘要而开发的，具体为：

> 给定一个输入句子，目标是产生一个简明的摘要。 [...]摘要生成器将 x 作为输入并输出长度 N &lt;1 的缩短句子 y。 M.我们将假设摘要中的单词也来自相同的词汇表

这比完整的文档摘要更简单。

该方法遵循用于具有编码器和解码器的神经机器转换的一般方法。探索了三种不同的解码器：

*   **Bag-of-Words 编码器**。输入句子使用词袋模型编码，丢弃词序信息。
*   **卷积编码器**。使用字嵌入表示，然后使用跨字和汇集层的时间延迟卷积层。
*   **基于注意力的编码器**。单词嵌入表示与上下文向量一起使用简单的注意机制，在输入句子和输出摘要之间提供一种软对齐。

![Network Diagram of Encoder and Decoder Elements](img/bf03345040dee4e6c83f6bd3e5a3db62.jpg)

编码器和解码器元件的网络图
取自“用于抽象句子摘要的神经注意模型”。

然后，在生成文本摘要时使用波束搜索，这与机器翻译中使用的方法不同。

该模型在标准 [DUC-2014 数据集](http://duc.nist.gov/data.html)上进行评估，该数据集涉及为 500 篇新闻文章生成大约 14 个字的摘要。

这项任务的数据包括来自纽约时报和美联社有线服务的 500 篇新闻文章，每篇文章都配有 4 个不同的人工参考摘要（实际上不是头条新闻），上限为 75 字节。

该模型还在大约 950 万篇新闻文章的 [Gigaword 数据集](https://catalog.ldc.upenn.edu/LDC2012T21)上进行了评估，其中给出了新闻文章第一句的标题。

使用 ROUGE-1，ROUGE-2 和 ROUGE-L 测量结果报告了两个问题，并且调谐系统显示在 DUC-2004 数据集上实现了最先进的结果。

> 与几个强大的基线相比，该模型显示了 DUC-2004 共享任务的显着表现提升。

## IBM 模型

Ramesh Nallapati 等人描述了这种方法。来自 IBM Watson 的 2016 年论文“[使用序列到序列 RNN 和超越](https://arxiv.org/abs/1602.06023)的抽象文本摘要”。

该方法基于编码器 - 解码器循环神经网络，注重机器翻译。

> 我们的基线模型对应于 Bahdanau 等人使用的神经机器翻译模型。 （2014）。编码器由双向 GRU-RNN（Chung 等，2014）组成，而解码器由具有与编码器相同的隐藏状态大小的单向 GRU-RNN 和源上的关注机制组成。 - 隐藏状态和目标词汇表上的软最大层以生成单词。

除了用于标记的词性和离散的 TF 和 IDF 特征的嵌入之外，还使用用于输入词的词嵌入。这种更丰富的输入表示旨在使模型在识别源文本中的关键概念和实体方面具有更好的表现。

该模型还使用学习开关机制来决定是否生成输出字或指向输入序列中的字，用于处理稀有和低频字。

> ...解码器配有一个“开关”，用于决定在每个时间步使用发生器还是指针。如果开关打开，则解码器以正常方式从其目标词汇表中产生一个单词。但是，如果关闭开关，则解码器生成指向源中的一个字位置的指针。

最后，该模型是分层的，因为注意机制在编码输入数据上的单词级和句子级操作。

![Hierarchical encoder with hierarchical attention](img/c64351a71d47be92099f11ee7e621711.jpg)

具有分层关注的分层编码器。
取自“使用序列到序列的 RNN 及其后的抽象文本摘要”。

在 DUC-2003/2004 数据集和 Gigaword 数据集上评估了该方法的总共 6 种变体，两者都用于评估 Facebook 模型。

该模型还在来自 CNN 和 Daily Mail 网站的新的新闻文章集上进行了评估。

与 Facebook 方法和其他方法相比，IBM 方法在标准数据集上取得了令人瞩目的成果。

> ...我们将注意力编码器 - 解码器应用于抽象概括的任务，具有非常有希望的结果，在两个不同的数据集上显着优于最先进的结果。

## 谷歌模型

Abigail See 等人描述了这种方法。来自斯坦福大学 2017 年论文“[到达重点：利用指针生成器网络进行总结](https://arxiv.org/abs/1704.04368)。”

一个更好的名字可能是“斯坦福模型”，但我试图将这项工作与合作者 Peter Liu（谷歌大脑）2016 年帖子标题为“[文本摘要与 TensorFlow](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html) ”在谷歌上联系起来研究博客。

在他们的博客文章中，Peter Liu 等人。在 Google Brain 上引入了 [TensorFlow 模型](https://github.com/tensorflow/models/tree/master/textsum)，该模型直接将用于机器翻译的编码器 - 解码器模型应用于生成 Gigaword 数据集的短句的摘要。虽然没有在代码提供的文本文档之外提供结果的正式记录，但它们声称比模型的最新结果更好。

在他们的论文中，Abigail See 等人。描述了抽象文本摘要的深度学习方法的两个主要缺点：它们产生事实错误并且它们重复出现。

> 虽然这些系统很有前景，但它们表现出不良行为，例如不准确地复制事实细节，无法处理词汇外（OOV）词，以及重复自己

他们的方法旨在总结多个句子而不是单句概括，并应用于用于演示 IBM 模型的 CNN / Daily Mail 数据集。该数据集中的文章平均包含大约 39 个句子。

基线编码器 - 解码器模型与字嵌入，双向 LSTM 用于输入和注意一起使用。探索了一种扩展，它使用指向输入数据中的单词来解决词汇表单词，类似于 IBM 模型中使用的方法。最后，覆盖机制用于帮助减少输出中的重复。

![Pointer-generator model for Text Summarization](img/b1597fc975e0921630e8ec485d29844d.jpg)

用于文本摘要的指针生成器模型
取自“到达点：使用指针生成器网络进行汇总”。

使用 ROUGE 和 METEOR 得分报告结果，与其他抽象方法和挑战采掘模型的得分相比，显示出最先进的表现。

> 我们的指针生成器模型覆盖率进一步提高了 ROUGE 和 METEOR 得分，令人信服地超越了最佳[比较]抽象模型......

结果确实表明可以使用基线 seq-to-seq 模型（带注意的编码器 - 解码器），但不会产生竞争结果，显示了它们对方法的扩展的好处。

> 我们发现我们的基线模型在 ROUGE 和 METEOR 方面都表现不佳，实际上较大的词汇量（150k）似乎没有帮助。 ......事实细节经常被错误地复制，通常用一个更常见的替代词替换一个不常见的（但是词汇表中）词。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [抽象句子摘要的神经注意模型](https://arxiv.org/abs/1509.00685)（[见代码](https://github.com/facebook/NAMAS)），2015。
*   [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。
*   [达到要点：指针生成器网络汇总](https://arxiv.org/abs/1704.04368)（[见代码](https://github.com/abisee/pointer-generator)），2017 年。
*   [使用 TensorFlow 进行文本摘要](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html)（[参见代码](https://github.com/tensorflow/models/tree/master/textsum)），2016
*   [驯服循环神经网络以实现更好的总结](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html)，2017 年。

## 摘要

在这篇文章中，您发现了文本摘要的深度学习模型。

具体来说，你学到了：

*   使用编码器 - 解码器模型和卷积神经网络编码器的 Facebook AI Research 模型。
*   使用 Encoder-Decoder 模型的 IBM Watson 模型，具有指向和分层注意力。
*   斯坦福/谷歌模型使用带有指向和覆盖的编码器 - 解码器模型。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。