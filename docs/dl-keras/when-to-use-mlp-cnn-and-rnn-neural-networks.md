# 何时使用 MLP，CNN 和 RNN 神经网络

> 原文： [https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/)
> 
> 校对：[linmeishang](https://github.com/linmeishang)

**什么神经网络适合您的预测性建模问题？**

对于初学者来说，深度学习领域很难知道要使用什么类型的网络。有许多类型的网络可供选择，每天都会发布和讨论新的方法。

更糟糕的是，大多数神经网络足够灵活，即使在使用错误类型的数据或预测问题时也可以工作（做出预测）。

在这篇文章中，您将发现三种主要类型的人工神经网络的建议用法。

阅读这篇文章后，您会知道：

*   在处理预测性建模问题时要关注哪种类型的神经网络。
*   何时使用，不使用，并可能尝试在项目中使用 MLP，CNN 和 RNN。
*   在选择模型之前，要考虑使用混合模型并清楚了解项目目标。

让我们开始吧。

![When to Use MLP, CNN, and RNN Neural Networks](img/070851788dabfc54104852fe22f56b44.png)

何时使用 MLP，CNN 和 RNN 神经网络
照片由 [PRODAVID S. FERRY III，DDS](https://www.flickr.com/photos/drdavidferry/15365735518/) ，保留一些权利。

## 概观

这篇文章分为五个部分;他们是：

1.  什么神经网络要关注？
2.  何时使用多层感知机？
3.  何时使用卷积神经网络？
4.  何时使用递归神经网络？
5.  混合网络模型

## 什么神经网络要关注？

[深度学习](https://machinelearningmastery.com/what-is-deep-learning/)是使用现代硬件的人工神经网络的应用。

它允许开发，训练和使用比以前认为可能更大（更多层）的神经网络。

研究人员提出了数千种类型的特定神经网络，作为对现有模型的修改或调整。有时是全新的方法。

作为一名从业者，我建议您等到模型出现后普遍适用。很难从每天或每周发布的大量出版物的噪音中梳理出一般效果良好的信号。

有三类人工神经网络我建议您一般关注。他们是：

*   多层感知机（Multilayer Perceptron，MLP）
*   卷积神经网络（Convolutional Neural Network，CNN）
*   递归神经网络（Recurrent Neural Network，RNN）

这三类网络提供了很大的灵活性，并且经过数十年的证明，它们在各种各样的问题中都是有用和可靠的。他们还有许多子类型来帮助他们专注于预测问题和不同数据集的不同框架的怪癖。

现在我们知道要关注哪些网络，让我们看看何时可以使用每一类神经网络。

## 何时使用多层感知机？

多层感知机（简称 MLP）是经典类型的神经网络。

它们由一层或多层神经元(neuron)组成。数据被馈送到输入层（input layer），之后可能有一个或多个隐藏层(hidden layer)来提供更高水平的抽象（abstraction）关系，并且在输出层(output layer)（也称为可见层(visiable layer)）上做出预测。

有关 MLP 的更多详细信息，请参阅帖子：

*   [多层感知机神经网络速成课程](https://machinelearningmastery.com/neural-networks-crash-course/)

![简单神经网络模型](img/98d0e7f8e58b0a5cb817d172e0256fe0.png)


MLP 适用于分类预测（classification prediction）问题，其中输入(input)被指定给一定的类（class）或标签(label)。

它们也适用于回归预测（regression prediction）问题，其中在给定一组输入的情况下预测实际值。数据通常以表格格式提供，例如 CSV 文件或电子表格。

**MLP 可用于：**

*   表格数据集
*   分类预测问题
*   回归预测问题

它们非常灵活，通常可用于学习输入和输出之间的关系。

这种灵活性允许它们应用于其他类型的数据。例如，图像的像素可以缩减为一行长数据并馈送到 MLP 中。文档的单词也可以缩减为一行长数据并馈送到 MLP。甚至对时间序列预测问题的滞后观察也可以减少为长行数据并馈送到 MLP。

因此，如果您的数据采用的不是表格数据集，例如图像、文档或时间序列，我建议至少测试一个 MLP 来解决您的问题。MLP 的结果可用作比较的基准，以确定那些可能看起来更适合的模型确实比 MLP 更好。

**可用MLP尝试如下问题：**

*   图像数据
*   文本数据
*   时间序列数据
*   其他类型的数据

## 何时使用卷积神经网络？

卷积神经网络（Convolutional Neural Network，CNN）用于将图像数据映射到一定的输出变量。

事实证明它们非常有效，如果您遇到涉及图像数据作为输入的任何类型的预测问题，它们是首选方法。

有关 CNN 的更多详细信息，请参阅帖子：

*   [用于机器学习的卷积神经网络的速成课程](https://machinelearningmastery.com/crash-course-convolutional-neural-networks/)

使用 CNN 的好处是它们能够开发二维图像的内部表征（representation）。这允许模型在各种不同的数据结构中学习图像中物体的位置和比例，这在处理图像时很重要。

**CNN  可用于：**

*   图像数据
*   分类预测问题
*   回归预测问题

更一般地，CNN 在具有空间关系的数据上表现良好。

CNN 的输入传统上是二维的，场或矩阵，但也可以改变为一维的，允许它学习一维序列的内部表征。

这允许 CNN 更普遍地用于具有空间关系的其他类型的数据。例如，文本文档中的单词之间存在顺序关系，时间序列的步长（time step）中存在有序关系。

虽然不是专门为非图像数据开发的，但 CNN 在诸如情绪分析中使用的文档分类和相关问题上实现了最先进的结果。

**可用CNN尝试如下问题：**

*   文字数据
*   时间序列数据
*   序列输入数据

## 何时使用递归神经网络？

递归神经网络（Recurrent Neural Network，RNN）用于处理序列预测问题。

序列预测问题有多种形式，我们最好用模型可支持的输入和输出的类型来描述这些预测问题的类型。

序列预测问题的一些例子包括：

*   **一对多**：一个观测点（observation）作为输入映射到多步（multiple steps）序列输出。
*   **多对一**：多步序列输入映射到一个类或数值的输出。
*   **多对多**：多步序列输入映射到多步序列输出。

多对多问题通常被称为序列到序列(sequence-to-sequence)，或简称为 seq2seq。

有关序列预测问题类型的更多详细信息，请参阅帖子：

*   [递归神经网络序列预测模型的简要介绍](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)

递归神经网络通常难以训练。

长短期记忆( Long Short-Term Memory, LSTM) 网络可能是最成功的 RNN，因为它克服了递归神经网络训练的难题，已经被广泛的应用了。

有关 RNN 的更多详细信息，请参阅帖子：

*   [深度学习递归神经网络速成课程](https://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)

一般而言，RNNs 和 LSTM 在处理单词和段落序列时最为成功，通常称为自然语言处理（natural language processing，NLP）。

这包括以时间序列表示的文本序列语音序列。它们还用作生成模型（generative model），需要生成一定的序列。可生成的序列不仅包括文本，还包括手写字迹等。

**RNN 可用于：**

*   文字数据
*   语音数据
*   分类预测问题
*   回归预测问题
*   生成模型

正如您在 CSV 文件或电子表格中看到的那样，递归神经网络不适用于表格数据集。它们也不适合图像数据输入。

**请勿使用 RNN 于：**

*   表格数据
*   图像数据

RNN 和 LSTM 已经被用在时间序列预测问题上进行了测试，但结果却很差。自回归方法，甚至线性方法通常表现得更好。 在同样的数据集上，甚至简单的 MLP 都比 LSTM 效果更好。

有关此主题的更多信息，请参阅帖子：

*   [关于长短期记忆网络对时间序列预测的适用性](https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/)

然而，它仍然是一个活跃的研究领域。

**也许可以用 RNN 尝试如下问题 ：**

*   时间序列数据

## 混合网络模型（Hybrid Network Models）

CNN 或 RNN 模型很少单独使用。

这些类型的网络可包括在更广泛的模型中作为 CNN 或 RNN 层，这种模型也可具有一个或多个 MLP 层。从技术上讲，这些是混合类型的神经网络架构。

也许最有趣的是将来自不同类型的网络混合在一起成为混合模型。

例如，设想这样一个模型：它使用堆叠的 CNN 作为输入层，中间为 LSTM 层，输出为 MLP 层。像这样的模型可以读取一系列图像输入，例如视频，并生成预测。这种模型被称为 [CNN LSTM 架构](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)。

网络类型也可以堆叠在特定的结构中，以此来解锁新功能。例如可重复使用的图像识别模型，这些模型使用非常深的 CNN 和 MLP 网络，它可以被添加到新的 LSTM 模型并用于给照片加标题。此外，编码器-解码器 LSTM 网络*（encoder-decoder LSTM networks）可用于具有不同长度的输入和输出序列。

重要的是要先了解您和利益相关者对项目的要求，然后寻找（或开发一个）满足特定项目需求的网络架构。

如果您需要一个鉴别数据和预测问题的框架，请参阅帖子：

*   [如何定义机器学习问题](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)

## 进一步阅读

如果您希望深入了解此话题，可参见有关该主题的更多资源：

*   [什么是深度学习？](https://machinelearningmastery.com/what-is-deep-learning/)
*   [多层感知机神经网络速成课程](https://machinelearningmastery.com/neural-networks-crash-course/)
*   [用于机器学习的卷积神经网络的速成课程](https://machinelearningmastery.com/crash-course-convolutional-neural-networks/)
*   [深度学习的递归神经网络速成课程](https://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)
*   [递归神经网络序列预测模型的简要介绍](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)
*   [如何定义机器学习问题](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)

## 总结

在这篇文章中，您发现了三种主流人工神经网络的推荐用法。

具体来说，您学到了：

*   在处理预测性建模问题时要关注哪种类型的神经网络。
*   何时使用、不使用或可以尝试在项目中使用 MLP，CNN 和 RNN。
*   在选择模型之前，可以考虑使用混合模型并清楚了解项目目标。

您有任何问题吗？
请在评论中区中提出您的问题，我会尽力解答。
