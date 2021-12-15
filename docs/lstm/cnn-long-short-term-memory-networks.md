# CNN 长短期记忆网络

> 原文： [https://machinelearningmastery.com/cnn-long-short-term-memory-networks/](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)

### 使用示例 Python 代码轻松介绍 CNN LSTM 循环神经网络
。

具有空间结构的输入（如图像）无法使用标准 Vanilla LSTM 轻松建模。

CNN 长短期记忆网络或简称 CNN LSTM 是一种 LSTM 架构，专门用于具有空间输入的序列预测问题，如图像或视频。

在这篇文章中，您将发现用于序列预测的 CNN LSTM 架构。

完成这篇文章后，你会知道：

*   关于用于序列预测的 CNN LSTM 模型架构的开发。
*   CNN LSTM 模型适合的问题类型的示例。
*   如何使用 Keras 在 Python 中实现 CNN LSTM 架构。

让我们开始吧。

![Convolutional Neural Network Long Short-Term Memory Networks](img/28aa9063f5dd83a2c8c2e7a9a66db246.jpg)

卷积神经网络长短期记忆网络
摄影： [Yair Aronshtam](https://www.flickr.com/photos/yairar/34484734116/) ，保留了一些权利。

## CNN LSTM 架构

CNN LSTM 架构涉及使用卷积神经网络（CNN）层对输入数据进行特征提取以及 LSTM 以支持序列预测。

CNN LSTM 是针对视觉时间序列预测问题以及从图像序列（例如视频）生成文本描述的应用而开发的。具体来说，问题是：

*   **活动识别**：生成在一系列图像中演示的活动的文本描述。
*   **图像说明**：生成单个图像的文本描述。
*   **视频说明**：生成图像序列的文本描述。

> [CNN LSTM]是一类在空间和时间上都很深的模型，并且可以灵活地应用于涉及顺序输入和输出的各种视觉任务

- [用于视觉识别和描述的长期循环卷积网络](https://arxiv.org/abs/1411.4389)，2015。

这种架构最初被称为长期循环卷积网络或 LRCN 模型，尽管我们将使用更通用的名称“CNN LSTM”来指代在本课程中使用 CNN 作为前端的 LSTM。

该架构用于生成图像的文本描述的任务。关键是使用在具有挑战性的图像分类任务上预先训练的 CNN，该任务被重新用作字幕生成问题的特征提取器。

> ...使用 CNN 作为图像“编码器”是很自然的，首先将其预训练用于图像分类任务，并使用最后隐藏层作为生成句子的 RNN 解码器的输入

- [Show and Tell：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。

该架构还用于语音识别和自然语言处理问题，其中 CNN 用作音频和文本输入数据上的 LSTM 的特征提取器。

此架构适用于以下问题：

*   在其输入中具有空间结构，例如 2D 结构或图像中的像素或句子，段落或文档中的单词的一维结构。
*   在其输入中具有时间结构，诸如视频中的图像的顺序或文本中的单词，或者需要生成具有时间结构的输出，诸如文本描述中的单词。

![Convolutional Neural Network Long Short-Term Memory Network Architecture](img/ae84a006384400ada510e876d69bc2a4.jpg)

卷积神经网络长短期记忆网络架构

## 在 Keras 实现 CNN LSTM

我们可以定义一个在 Keras 联合训练的 CNN LSTM 模型。

可以通过在前端添加 CNN 层，然后在输出上添加具有 Dense 层的 LSTM 层来定义 CNN LSTM。

将此架构视为定义两个子模型是有帮助的：用于特征提取的 CNN 模型和用于跨时间步骤解释特征的 LSTM 模型。

让我们在一系列 2D 输入的背景下看一下这两个子模型，我们假设它们是图像。

### CNN 模型

作为复习，我们可以定义一个 2D 卷积网络，它由 Conv2D 和 MaxPooling2D 层组成，这些层被排列成所需深度的栈。

Conv2D 将解释图像的快照（例如小方块），并且轮询层将合并或抽象解释。

例如，下面的片段期望读入具有 1 个通道（例如，黑色和白色）的 10×10 像素图像。 Conv2D 将以 2×2 快照读取图像，并输出一个新的 10×10 图像解释。 MaxPooling2D 将解释汇集为 2×2 块，将输出减少到 5×5 合并。 Flatten 层将采用单个 5×5 贴图并将其转换为 25 个元素的向量，准备用于处理其他层，例如用于输出预测的 Dense。

```py
cnn = Sequential()
cnn.add(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=(10,10,1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
```

这对于图像分类和其他计算机视觉任务是有意义的。

### LSTM 模型

上面的 CNN 模型仅能够处理单个图像，将其从输入像素变换为内部矩阵或向量表示。

我们需要跨多个图像重复此操作，并允许 LSTM 在输入图像的内部向量表示序列中使用 BPTT 建立内部状态和更新权重。

在使用现有的预训练模型（如 VGG）从图像中提取特征的情况下，可以固定 CNN。 CNN 可能未经过训练，我们可能希望通过将来自 LSTM 的错误反向传播到多个输入图像到 CNN 模型来训练它。

在这两种情况下，概念上存在单个 CNN 模型和一系列 LSTM 模型，每个时间步长一个。我们希望将 CNN 模型应用于每个输入图像，并将每个输入图像的输出作为单个时间步骤传递给 LSTM。

我们可以通过在 TimeDistributed 层中包装整个 CNN 输入模型（一层或多层）来实现这一点。该层实现了多次应用相同层的期望结果。在这种情况下，将其多次应用于多个输入时间步骤，并依次向 LSTM 模型提供一系列“图像解释”或“图像特征”以进行处理。

```py
model.add(TimeDistributed(...))
model.add(LSTM(...))
model.add(Dense(...))
```

我们现在有模型的两个元素;让我们把它们放在一起。

### CNN LSTM 模型

我们可以在 Keras 中定义 CNN LSTM 模型，首先定义一个或多个 CNN 层，将它们包装在 TimeDistributed 层中，然后定义 LSTM 和输出层。

我们有两种方法可以定义相同的模型，只是在品味上有所不同。

您可以首先定义 CNN 模型，然后通过将整个 CNN 层序列包装在 TimeDistributed 层中将其添加到 LSTM 模型，如下所示：

```py
# define CNN model
cnn = Sequential()
cnn.add(Conv2D(...))
cnn.add(MaxPooling2D(...))
cnn.add(Flatten())
# define LSTM model
model = Sequential()
model.add(TimeDistributed(cnn, ...))
model.add(LSTM(..))
model.add(Dense(...))
```

另一种也许更容易阅读的方法是在将 CNN 模型中的每个层添加到主模型时，将其包装在 TimeDistributed 层中。

```py
model = Sequential()
# define CNN model
model.add(TimeDistributed(Conv2D(...))
model.add(TimeDistributed(MaxPooling2D(...)))
model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(...))
model.add(Dense(...))
```

第二种方法的好处是所有层都出现在模型摘要中，因此现在是首选。

您可以选择您喜欢的方法。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### CNN LSTM 论文

*   [用于视觉识别和描述的长期循环卷积网络](https://arxiv.org/abs/1411.4389)，2015。
*   [Show and Tell：神经图像标题生成器](https://arxiv.org/abs/1411.4555)，2015。
*   [卷积，长短期记忆，完全连接的深度神经网络](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43455.pdf)，2015。
*   [字符意识神经语言模型](https://arxiv.org/abs/1508.06615)，2015。
*   [卷积 LSTM 网络：用于降水预报的机器学习方法](https://arxiv.org/abs/1506.04214)，2015。

### Keras API

*   [Conv2D Keras API](https://keras.io/layers/convolutional/#conv2d) 。
*   [MaxPooling2D Keras API](https://keras.io/layers/pooling/#maxpooling2d) 。
*   [Flatten Keras API](https://keras.io/layers/core/#flatten) 。
*   [TimeDistributed Keras API](https://keras.io/layers/wrappers/#timedistributed) 。

### 帖子

*   [用于机器学习的卷积神经网络的速成课程](http://machinelearningmastery.com/crash-course-convolutional-neural-networks/)
*   [用 Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) 在 Python 中用 LSTM 循环神经网络进行序列分类

## 摘要

在这篇文章中，您发现了 CNN LSTM 模型架构。

具体来说，你学到了：

*   关于用于序列预测的 CNN LSTM 模型架构的开发。
*   CNN LSTM 模型适合的问题类型的示例。
*   如何使用 Keras 在 Python 中实现 CNN LSTM 架构。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。