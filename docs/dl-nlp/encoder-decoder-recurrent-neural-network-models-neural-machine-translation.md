# 用于神经机器翻译的编解码器循环神经网络模型

> 原文： [https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)

用于循环神经网络的编解码器架构是标准的神经机器翻译方法，其可以与传统的统计机器翻译方法相媲美并且在某些情

这种架构非常新颖，仅在 2014 年率先推出，但已被采纳为 [Google 翻译服务](https://translate.google.com/)的核心技术。

在这篇文章中，您将发现用于神经机器翻译的编解码器模型的两个开创性示例。

阅读这篇文章后，你会知道：

*   编解码器循环神经网络架构是 Google 翻译服务中的核心技术。
*   用于直接端到端机器翻译的所谓“ _Sutskever 型号 _”。
*   所谓的“ _Cho 模型 _”通过 GRU 单元和注意机制扩展了架构。

让我们开始吧。

![Encoder-Decoder Recurrent Neural Network Models for Neural Machine Translation](img/37525e20edcb357c3f9a4f90b69ef230.jpg)

用于神经机器翻译的编解码器循环神经网络模型
[Fabio Pani](https://www.flickr.com/photos/fabiuxfabiux/34223907581/) 的照片，保留一些权利。

## 用于 NMT 的编解码器架构

具有循环神经网络的编解码器架构已成为神经机器翻译（NMT）和序列到序列（seq2seq）预测的有效和标准方法。

该方法的主要优点是能够直接在源语句和目标语句上训练单个端到端模型，以及处理可变长度输入和输出文本序列的能力。

作为该方法成功的证据，该架构是 [Google 翻译服务](https://translate.google.com/)的核心。

> 我们的模型遵循常见的序列到序列学习框架。它有三个组件：编码器网络，解码器网络和注意网络。

- [Google 的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016

在这篇文章中，我们将仔细研究两个不同的研究项目，这些项目在 2014 年同时开发了相同的编解码器架构，并取得了成为人们关注的方法。他们是：

*   Sutskever NMT 模型
*   Cho NMT 模型

有关架构的更多信息，请参阅帖子：

*   [编解码器长短期记忆网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

## Sutskever NMT 模型

在本节中，我们将研究 [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/) 等开发的神经机器翻译模型。正如 2014 年论文“[序列学习与神经网络](https://arxiv.org/abs/1409.3215)”所述。由于缺乏更好的名称，我们将其称为“ _Sutskever NMT 模型 _”。

这是一篇重要的论文，因为它是第一个引入用于机器翻译的编解码器模型以及更普遍的序列到序列学习的模型之一。

它是机器翻译领域的一个重要模型，因为它是第一个在大型翻译任务上胜过基线统计机器学习模型的神经机器翻译系统之一。

### 问题

该模型适用于英语到法语的翻译，特别是 [WMT 2014 翻译任务](http://www.statmt.org/wmt14/translation-task.html)。

翻译任务一次处理一个句子，并且在训练期间将序列结束（＆lt; EOS＆gt;）标记添加到输出序列的末尾以表示翻译序列的结束。这允许模型能够预测可变长度输出序列。

> 注意，我们要求每个句子以特殊的句末符号“＆lt; EOS＆gt;”结束，这使得模型能够定义所有可能长度的序列上的分布。

该模型在数据集中的 12 百万个句子的子集上进行训练，包括 348,000 个法语单词和 30400 万个英语单词。选择此集是因为它是预先标记的。

源词汇量减少到 160,000 个最频繁的源英语单词和 80,000 个最常见的目标法语单词。所有词汇外单词都被“UNK”标记取代。

### 模型

开发了编解码器架构，其中整个读取输入序列并将其编码为固定长度的内部表示。

然后，解码器网络使用该内部表示来输出字，直到到达序列令牌的末尾。 LSTM 网络用于编码器和解码器。

> 想法是使用一个 LSTM 读取输入序列，一次一个步骤，以获得大的固定维向量表示，然后使用另一个 LSTM 从该向量中提取输出序列

最终的模型是 5 个深度学习模型的集合。在推断翻译期间使用从左到右的光束搜索。

![Depiction of Sutskever Encoder-Decoder Model for Text Translation](img/315a778984788329e62f75784a273c90.jpg)

用于文本翻译的 Sutskever 编解码器模型的描述
取自“使用神经网络的序列到序列学习”，2014。

### 型号配置

*   输入序列被颠倒了。
*   使用 1000 维字嵌入层来表示输入字。
*   Softmax 用于输出层。
*   输入和输出模型有 4 层，每层 1,000 个单元。
*   该模型适合 7.5 个时期，其中进行了一些学习率衰减。
*   在训练期间使用批量大小的 128 个序列。
*   在训练期间使用梯度裁剪来减轻梯度爆炸的可能性。
*   批次由具有大致相同长度的句子组成，以加速计算。

该模型适用于 8-GPU 机器，其中每层在不同的 GPU 上运行。训练需要 10 天。

> 由此产生的实施速度达到每秒 6,300（英语和法语）的速度，小批量为 128.这种实施的训练大约需要 10 天。

### 结果

该系统的 BLEU 得分为 34.81，与使用 33.30 的统计机器翻译系统开发的基线得分相比，这是一个很好的分数。重要的是，这是神经机器翻译系统的第一个例子，它在大规模问题上胜过基于短语的统计机器翻译基线。

> ......我们获得了 34.81 的 BLEU 分数[...]这是迄今为止通过大型神经网络直接翻译获得的最佳结果。为了比较，该数据集上 SMT 基线的 BLEU 得分为 33.30

使用最终模型对[最佳翻译](http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/)列表进行评分，并将得分提高至 36.5，使其接近 37.0 时的最佳结果。

你可以在这里看到与论文[相关的谈话视频：](https://www.youtube.com/watch?v=-uyXE7dY5H0)

&lt;iframe allowfullscreen="" frameborder="0" gesture="media" height="281" src="https://www.youtube.com/embed/-uyXE7dY5H0?feature=oembed" width="500"&gt;&lt;/iframe&gt;

## Cho NMT 模型

在本节中，我们将讨论 [Kyunghyun Cho](http://www.kyunghyuncho.me/) 等人描述的神经机器翻译系统。他们在 2014 年的论文题为“[学习短语表示使用 RNN 编解码器进行统计机器翻译](https://arxiv.org/abs/1406.1078)。”我们将其称为“ _Cho NMT 模型 _”模型缺乏更好的名称。

重要的是，Cho 模型仅用于对候选翻译进行评分，并不像上面的 Sutskever 模型那样直接用于翻译。虽然对更好地诊断和改进模型的工作的扩展确实直接和单独使用它进行翻译。

### 问题

如上所述，问题是 WMT 2014 研讨会的英语到法语翻译任务。

源词汇量和目标词汇量仅限于最常见的 15,000 个法语和英语单词，涵盖了 93％的数据集，词汇单词中的单词被“UNK”取代。

### 模型

该模型使用相同的双模型方法，这里给出了编解码器架构的明确名称。

> ...称为 RNN 编解码器，由两个循环神经网络（RNN）组成。一个 RNN 将符号序列编码成固定长度的向量表示，而另一个 RNN 将该表示解码成另一个符号序列。

![Depiction of the Encoder-Decoder architecture](img/733acefebad1632bfbddd1e52fc8f434.jpg)

描述编解码器架构。
取自“使用 RNN 编解码器进行统计机器翻译的学习短语表示”。

实施不使用 LSTM 单位;相反，开发了一种更简单的循环神经网络单元，称为门控循环单元或 GRU。

> ......我们还提出了一种新型的隐藏单元，它受 LSTM 单元的推动，但计算和实现起来要简单得多。

### 型号配置

*   使用 100 维单词嵌入来表示输入单词。
*   编码器和解码器配置有 1 层 1000 GRU 单元。
*   在解码器之后使用 500 个 Maxout 单元汇集 2 个输入。
*   在训练期间使用 64 个句子的批量大小。

该模型训练了大约 2 天。

### 扩展

在论文“[关于神经机器翻译的性质：编解码器方法](https://arxiv.org/abs/1409.1259)”，Cho，et al。调查他们的模型的局限性。他们发现，随着输入句子长度的增加和词汇表之外的单词数量的增加，表现会迅速下降。

> 我们的分析表明神经机器翻译的表现受句子长度的影响很大。

它们提供了模型表现的有用图表，因为句子的长度增加，可以捕捉技能的优美损失，增加难度。

![Loss in model skill with increased sentence length](img/569776abf8da241c3eea080a14b4142d.jpg)

句子长度增加导致模特技能丧失。
取自“关于神经机器翻译的属性：编解码器方法”。

为了解决未知单词的问题，他们建议在训练期间大大增加已知单词的词汇量。

他们在一篇题为“[通过联合学习协调和翻译](https://arxiv.org/abs/1409.0473)的神经机器翻译”的后续论文中解决了句子长度的问题，其中他们建议使用注意机制。不是将输入语句编码为固定长度向量，而是保持编码输入的更全面表示，并且模型学习用于关注解码器输出的每个字的输入的不同部分。

> 每次所提出的模型在翻译中生成单词时，它（软）搜索源语句中的一组位置，其中最相关的信息被集中。然后，模型基于与这些源位置和所有先前生成的目标词相关联的上下文向量来预测目标词。

本文提供了大量技术细节;例如：

*   使用类似配置的模型，但具有双向层。
*   准备数据使得在词汇表中保留 30,000 个最常见的单词。
*   该模型首先使用长度最多为 20 个单词的句子进行训练，然后使用长度最多为 50 个单词的句子进行训练。
*   使用 80 个句子的批量大小，该模型适合 4-6 个时期。
*   在推理期间使用集束搜索来找到每个翻译的最可能的词序列。

这次模型需要大约 5 天的时间来训练。该后续工作的代码也是 [](https://github.com/lisa-groundhog/GroundHog) 。

与 Sutskever 一样，该模型在经典的基于短语的统计方法的范围内取得了成果。

> 也许更重要的是，所提出的方法实现了与现有的基于短语的统计机器翻译相当的翻译表现。考虑到所提出的架构或整个神经机器翻译系列仅在今年才被提出，这是一个引人注目的结果。我们相信这里提出的架构是朝着更好的机器翻译和更好地理解自然语言迈出的有希望的一步。

Kyunghyun Cho 也是 Nvidia 开发者博客 2015 年系列帖子的作者，该博客主题为神经机器翻译的编解码器架构，题为“ _GPU 神经机器翻译简介”。_ “该系列提供了对主题和模型的良好介绍;见[第 1 部分](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/)，[第 2 部分](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/)和[第 3 部分](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/)。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016。
*   [用神经网络进行序列学习的序列](https://arxiv.org/abs/1409.3215)，2014。
*   [用神经网络进行序列到序列学习的表示](https://www.youtube.com/watch?v=-uyXE7dY5H0)，2016。
*   [Ilya Sutskever 主页](http://www.cs.toronto.edu/~ilya/)
*   [使用 RNN 编解码器进行统计机器翻译的学习短语表示](https://arxiv.org/abs/1406.1078)，2014。
*   [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014。
*   [关于神经机器翻译的特性：编解码器方法](https://arxiv.org/abs/1409.1259)，2014。
*   [Kyunghyun Cho 主页](http://www.kyunghyuncho.me/)
*   用 GPU 进行神经机器翻译的介绍（ [part1](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/) ， [part2](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/) ， [part3](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/) ），2015。

## 摘要

在这篇文章中，您发现了两个用于神经机器翻译的编解码器模型的示例。

具体来说，你学到了：

*   编解码器循环神经网络架构是 Google 翻译服务中的核心技术。
*   用于直接端到端机器翻译的所谓“Sutskever 模型”。
*   所谓的“Cho 模型”，通过 GRU 单元和注意机制扩展了架构。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。