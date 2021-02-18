# Keras 中文本摘要的编码器 - 解码器模型

> 原文： [https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/)

文本摘要是自然语言处理中的一个问题，即创建源文档的简短，准确和流畅的摘要。

为机器翻译开发的编码器 - 解码器循环神经网络架构在应用于文本摘要问题时已被证明是有效的。

在 Keras 深度学习库中应用这种架构可能很困难，因为为了使库清洁，简单和易于使用而牺牲了一些灵活性。

在本教程中，您将了解如何在 Keras 中实现用于文本摘要的编码器 - 解码器架构。

完成本教程后，您将了解：

*   如何使用编码器 - 解码器循环神经网络架构来解决文本摘要。
*   如何针对该问题实现不同的编码器和解码器。
*   您可以使用三种模型在 Keras 中实现文本摘要的体系结构。

让我们开始吧。

![Encoder-Decoder Models for Text Summarization in Keras](img/bd7f0bdd354c591ea4a3577195c35dd1.jpg)

用于 Keras 中文本摘要的编码器 - 解码器模型
照片由 [Diogo Freire](https://www.flickr.com/photos/diogofreire/4766208168/) 拍摄，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  编码器 - 解码器架构
2.  文本摘要编码器
3.  文本摘要解码器
4.  阅读源文本
5.  实施模型

## 编码器 - 解码器架构

编码器 - 解码器架构是一种组织循环神经网络的方法，用于具有可变数量的输入，输出或两者输入和输出的序列预测问题。

该架构涉及两个组件：编码器和解码器。

*   **编码器**：编码器读取整个输入序列并将其编码为内部表示，通常是称为上下文向量的固定长度向量。
*   **解码器**：解码器从编码器读取编码的输入序列并生成输出序列。

有关编码器 - 解码器架构的更多信息，请参阅帖子：

*   [编码器 - 解码器长短期存储器网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

编码器和解码器子模型都是联合训练的，意思是同时进行。

这是一项非常壮观的事情，因为传统上，挑战自然语言问题需要开发单独的模型，这些模型后来被串入管道，允许错误在序列生成过程中累积。

整个编码输入用作生成输出中每个步骤的上下文。虽然这有效，但输入的固定长度编码限制了可以生成的输出序列的长度。

编码器 - 解码器架构的扩展是提供编码输入序列的更具表现形式，并允许解码器在生成输出序列的每个步骤时学习在何处关注编码输入。

这种体系结构的扩展称为注意。

有关编码器 - 解码器架构中的注意事项的更多信息，请参阅帖子：

*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项

编码器 - 解码器体系结构受到关注，是一组自然语言处理问题，它产生可变长度的输出序列，例如文本摘要。

体系结构在文本摘要中的应用如下：

*   **编码器**：编码器负责读取源文档并将其编码为内部表示。
*   **解码器**：解码器是一种语言模型，负责使用源文档的编码表示在输出摘要中生成每个单词。

## 文本摘要编码器

编码器是模型的复杂性所在，因为它负责捕获源文档的含义。

可以使用不同类型的编码器，但是更常用的是双向循环神经网络，例如 LSTM。在编码器中使用循环神经网络的情况下，使用字嵌入来提供字的分布式表示。

亚历山大拉什等人。使用一个简单的词袋编码器来丢弃单词顺序和卷积编码器，明确地尝试捕获 n-gram。

> 我们最基本的模型只使用嵌入到 H 大小的输入句子的词袋，而忽略原始顺序的属性或相邻单词之间的关系。 [...]为了解决一些词形的建模问题，我们还考虑使用深度卷积编码器来输入句子。

- [抽象句概括的神经注意模型](https://arxiv.org/abs/1509.00685)，2015。

Konstantin Lopyrev 使用深度堆叠的 4 个 LSTM 循环神经网络作为编码器。

> 编码器作为输入被输入一个单词一次的新闻文章的文本。每个单词首先通过嵌入层，该嵌入层将单词转换为分布式表示。然后使用多层神经网络组合该分布式表示

- [使用循环神经网络生成新闻标题](https://arxiv.org/abs/1512.01712)，2015 年。

Abigail See，et al。使用单层双向 LSTM 作为编码器。

> 将文章 w（i）的标记一个接一个地馈送到编码器（单层双向 LSTM）中，产生编码器隐藏状态序列 h（i）。

- [达到要点：利用指针生成器网络汇总](https://arxiv.org/abs/1704.04368)，2017 年。

Ramesh Nallapati，et al。在编码器中使用双向 GRU 循环神经网络，并在输入序列中包含有关每个字的附加信息。

> 编码器由双向 GRU-RNN 组成......

- [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。

## 文本摘要解码器

在给定两个信息源的情况下，解码器必须在输出序列中生成每个字：

1.  **上下文向量**：编码器提供的源文档的编码表示。
2.  **生成的序列**：已作为摘要生成的单词或单词序列。

上下文向量可以是如在简单编码器 - 解码器架构中的固定长度编码，或者可以是通过注意机制过滤的更具表现力的形式。

生成的序列提供很少的准备，例如通过字嵌入的每个生成的字的分布式表示。

> 在每个步骤 t，解码器（单层单向 LSTM）接收前一个字的字嵌入（在训练时，这是参考摘要的前一个字;在测试时它是解码器发出的前一个字）

- [达到要点：利用指针生成器网络汇总](https://arxiv.org/abs/1704.04368)，2017 年。

亚历山大拉什等人。在`x`是源文档的图表中干净地显示，`enc`是提供源文档内部表示的编码器，`yc`是先前的序列生成的单词。

![Example of inputs to the decoder for text summarization](img/2453ff0732cb22ffb4e8569f76989116.jpg)

用于文本摘要的解码器的输入示例。
取自“用于抽象句子摘要的神经注意模型”，2015 年。

一次生成一个单词需要运行模型，直到生成一些最大数量的摘要单词或达到特殊的序列结束标记。

必须通过为模型提供特殊的序列开始标记来启动该过程，以便生成第一个单词。

> 解码器将输入文本的最后一个单词后生成的隐藏层作为输入。首先，再次使用嵌入层将符号结束符号作为输入馈入，以将符号变换为分布式表示。 [...]。在生成下一个单词时，生成每个单词后输入相同的单词作为输入。

- [使用循环神经网络生成新闻标题](https://arxiv.org/abs/1512.01712)，2015 年。

Ramesh Nallapati，et al。使用 GRU 循环神经网络生成输出序列。

> ...解码器由单向 GRU-RNN 组成，其具有与编码器相同的隐藏状态大小

## 阅读源文本

根据所解决的特定文本摘要问题，该架构的应用具有灵活性。

大多数研究都集中在编码器中的一个或几个源句子上，但并非必须如此。

例如，编码器可以配置为以不同大小的块读取和编码源文档：

*   句子。
*   段。
*   页。
*   文献。

同样地，解码器可以被配置为汇总每个块或聚合编码的块并输出更广泛的概要。

亚历山大拉什等人在这条道路上做了一些工作。使用分层编码器模型，同时注意单词和句子级别。

> 该模型旨在使用源侧的两个双向 RNN 捕获这两个重要级别的概念，一个在单词级别，另一个在句子级别。注意机制同时在两个层面上运作

- [抽象句概括的神经注意模型](https://arxiv.org/abs/1509.00685)，2015。

## 实施模型

在本节中，我们将介绍如何在 Keras 深度学习库中实现用于文本摘要的编码器 - 解码器架构。

### 一般模型

模型的简单实现涉及具有嵌入输入的编码器，其后是 LSTM 隐藏层，其产生源文档的固定长度表示。

解码器读取表示和最后生成的单词的嵌入，并使用这些输入在输出摘要中生成每个单词。

![General Text Summarization Model in Keras](img/c48bfc3a2cf7965d3a39f815f6daa641.jpg)

Keras 中的一般文本摘要模型

这儿存在一个问题。

Keras 不允许递归循环，其中模型的输出自动作为输入提供给模型。

这意味着上面描述的模型不能直接在 Keras 中实现（但也许可以在像 TensorFlow 这样更灵活的平台中实现）。

相反，我们将看看我们可以在 Keras 中实现的模型的三种变体。

### 替代 1：一次性模型

第一种替代模型是以一次性方式生成整个输出序列。

也就是说，解码器仅使用上下文向量来生成输出序列。

![Alternate 1 - One-Shot Text Summarization Model](img/6c1cb97aa80bca4ee41e557e47448c35.jpg)

替代 1 - 一次性文本摘要模型

以下是使用功能 API 在 Keras 中使用此方法的示例代码。

```py
vocab_size = ...
src_txt_length = ...
sum_txt_length = ...
# encoder input model
inputs = Input(shape=(src_txt_length,))
encoder1 = Embedding(vocab_size, 128)(inputs)
encoder2 = LSTM(128)(encoder1)
encoder3 = RepeatVector(sum_txt_length)(encoder2)
# decoder output model
decoder1 = LSTM(128, return_sequences=True)(encoder3)
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)
# tie it together
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

这种模式给解码器带来了沉重的负担。

解码器可能没有足够的上下文来生成相干输出序列，因为它必须选择单词及其顺序。

### 备选 2：递归模型 A.

第二种替代模型是开发一种模型，该模型生成单个单词预测并递归调用。

也就是说，解码器使用上下文向量和到目前为止生成的所有单词的分布式表示作为输入，以便生成下一个单词。

语言模型可用于解释到目前为止生成的单词序列，以提供第二上下文向量以与源文档的表示相结合，以便生成序列中的下一个单词。

通过递归调用模型并使用先前生成的单词（或者更具体地说，在训练期间预期的前一个单词）来构建摘要。

可以将上下文向量集中或加在一起以为解码器提供更宽的上下文来解释和输出下一个字。

![Alternate 2 - Recursive Text Summarization Model A](img/f5989e00bdf4556d451c7eaf21e4ee68.jpg)

备用 2 - 递归文本摘要模型 A.

以下是使用功能 API 在 Keras 中使用此方法的示例代码。

```py
vocab_size = ...
src_txt_length = ...
sum_txt_length = ...
# source text input model
inputs1 = Input(shape=(src_txt_length,))
am1 = Embedding(vocab_size, 128)(inputs1)
am2 = LSTM(128)(am1)
# summary input model
inputs2 = Input(shape=(sum_txt_length,))
sm1 = = Embedding(vocab_size, 128)(inputs2)
sm2 = LSTM(128)(sm1)
# decoder output model
decoder1 = concatenate([am2, sm2])
outputs = Dense(vocab_size, activation='softmax')(decoder1)
# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

这更好，因为解码器有机会使用先前生成的单词和源文档作为生成下一个单词的上下文。

它确实给合并操作和解码器带来了负担，以解释它在生成输出序列时的位置。

### 备选 3：递归模型 B.

在该第三替代方案中，编码器生成源文档的上下文向量表示。

在生成的输出序列的每个步骤将该文档馈送到解码器。这允许解码器建立与用于在输出序列中生成单词的内部状态相同的内部状态，以便准备生成序列中的下一个单词。

然后通过对输出序列中的每个字一次又一次地调用模型来重复该过程，直到生成最大长度或序列结束标记。

![Alternate 3 - Recursive Text Summarization Model B](img/96538108e3739e0de8e4d8d543c7da60.jpg)

备用 3 - 递归文本摘要模型 B.

以下是使用功能 API 在 Keras 中使用此方法的示例代码。

```py
vocab_size = ...
src_txt_length = ...
sum_txt_length = ...
# article input model
inputs1 = Input(shape=(src_txt_length,))
article1 = Embedding(vocab_size, 128)(inputs1)
article2 = LSTM(128)(article1)
article3 = RepeatVector(sum_txt_length)(article2)
# summary input model
inputs2 = Input(shape=(sum_txt_length,))
summ1 = Embedding(vocab_size, 128)(inputs2)
# decoder model
decoder1 = concatenate([article3, summ1])
decoder2 = LSTM(128)(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

您还有其他替代实施方案吗？
请在下面的评论中告诉我。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [抽象句概括的神经注意模型](https://arxiv.org/abs/1509.00685)，2015。
*   [使用循环神经网络生成新闻标题](https://arxiv.org/abs/1512.01712)，2015 年。
*   [使用序列到序列 RNN 及其后的抽象文本摘要](https://arxiv.org/abs/1602.06023)，2016。
*   [达到要点：利用指针生成器网络汇总](https://arxiv.org/abs/1704.04368)，2017 年。

### 有关

*   [编码器 - 解码器长短期存储器网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项

## 摘要

在本教程中，您了解了如何在 Keras 深度学习库中实现用于文本摘要的编码器 - 解码器架构。

具体来说，你学到了：

*   如何使用编码器 - 解码器循环神经网络架构来解决文本摘要。
*   如何针对该问题实现不同的编码器和解码器。
*   您可以使用三种模型在 Keras 中实现文本摘要的体系结构。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。