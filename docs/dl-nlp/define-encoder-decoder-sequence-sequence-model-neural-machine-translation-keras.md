# 如何定义 Keras 中神经机器翻译的编码器 - 解码器序列 - 序列模型

> 原文： [https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)

编码器 - 解码器模型提供了使用递归神经网络来解决具有挑战性的序列到序列预测问题（例如机器翻译）的模式。

可以在 Keras Python 深度学习库中开发编码器 - 解码器模型，并且在 Keras 博客上描述了使用该模型开发的神经机器翻译系统的示例，其中示例代码与 Keras 项目一起分发。

在本文中，您将了解如何定义用于机器翻译的编码器 - 解码器序列到序列预测模型，如 Keras 深度学习库的作者所述。

阅读这篇文章后，你会知道：

*   神奇机器翻译示例与 Keras 一起提供并在 Keras 博客上进行了描述。
*   如何正确定义编码器 - 解码器 LSTM 以训练神经机器翻译模型。
*   如何正确定义推理模型以使用经过训练的编码器 - 解码器模型来转换新序列。

让我们开始吧。

*   **更新 Apr / 2018** ：有关应用此复杂模型的示例，请参阅帖子：[如何开发 Keras 中序列到序列预测的编码器 - 解码器模型](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

![How to Define an Encoder-Decoder Sequence-to-Sequence Model for Neural Machine Translation in Keras](img/38cae6eb1536c9b0a1ba0bb9d1c6906e.jpg)

如何在 Keras
中定义用于神经机器翻译的编码器 - 解码器序列 - 序列模型 [Tom Lee](https://www.flickr.com/photos/68942208@N02/16012752622/) ，保留一些权利。

## Keras 中的序列到序列预测

[Keras 深度学习库的作者 Francois Chollet](https://twitter.com/fchollet) 最近发布了一篇博文，其中介绍了一个代码示例，用于开发一个序列到序列预测的编码器 - 解码器 LSTM，标题为“ [A ten - 对 Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 中序列到序列学习的细致介绍。

博客文章中开发的代码也已添加到 Keras 中，作为文件 [lstm_seq2seq.py](https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py) 中的示例。

该帖子开发了编码器 - 解码器 LSTM 的复杂实现，如关于该主题的规范论文中所述：

*   [用神经网络进行序列学习的序列](https://arxiv.org/abs/1409.3215)，2014。
*   [使用 RNN 编码器 - 解码器进行统计机器翻译的学习短语表示](https://arxiv.org/abs/1406.1078)，2014。

该模型适用于机器翻译问题，与首次描述该方法的源文件相同。从技术上讲，该模型是神经机器翻译模型。

Francois 的实现提供了一个模板，用于在编写本文时在 Keras 深度学习库中如何（正确地）实现序列到序列预测。

在这篇文章中，我们将详细了解训练和推理模型的设计方式以及它们的工作原理。

您将能够利用这种理解为您自己的序列到序列预测问题开发类似的模型。

## 机器翻译数据

该示例中使用的数据集涉及闪存卡软件 [Anki](https://apps.ankiweb.net/) 中使用的简短的法语和英语句子对。

该数据集被称为“[制表符分隔的双语句子对](http://www.manythings.org/anki/)”，并且是 [Tatoeba 项目](http://tatoeba.org/home)的一部分，并列在 [ManyThings.org](http://www.manythings.org/) 网站上，用于帮助英语作为第二语言学生。

可以从此处下载本教程中使用的数据集：

*   [法语 - 英语 fra-eng.zip](http://www.manythings.org/anki/fra-eng.zip)

下面是解压缩下载的存档后您将看到的 _fra.txt_ 数据文件的前 10 行示例。

```py
Go.		Va !
Run!	Cours !
Run!	Courez !
Wow!	Ça alors !
Fire!	Au feu !
Help!	À l'aide !
Jump.	Saute.
Stop!	Ça suffit !
Stop!	Stop !
Stop!	Arrête-toi !
```

该问题被定义为序列预测问题，其中字符的输入序列是英语并且输出的字符序列是法语。

数据集中使用了数据文件中近 150,000 个示例中的 10,000 个。准备数据的一些技术细节如下：

*   **输入序列**：填充最大长度为 16 个字符，词汇量为 71 个不同的字符（10000,16,71）。
*   **输出序列**：填充最大长度为 59 个字符，词汇量为 93 个不同的字符（10000,59,93）。

对训练数据进行框架化，使得模型的输入包括一个完整的英文字符输入序列和整个法语字符输出序列。模型的输出是整个法语字符序列，但向前偏移一个步骤。

例如（使用最小填充并且没有单热编码）：

*   输入 1：['G'，'o'，'。'，“]
*   输入 2：[“，'V'，'a'，'']
*   输出：['V'，'a'，''，'！']

## 机器翻译模型

神经翻译模型是编码器 - 解码器递归神经网络。

它由读取可变长度输入序列的编码器和预测可变长度输出序列的解码器组成。

在本节中，我们将逐步介绍模型定义的每个元素，代码直接来自 Keras 项目中的帖子和代码示例（在撰写本文时）。

该模型分为两个子模型：负责输出输入英语序列的固定长度编码的编码器，以及负责预测输出序列的解码器，每个输出时间步长一个字符。

第一步是定义编码器。

编码器的输入是一系列字符，每个字符编码为长度为 _num_encoder_tokens_ 的单热向量。

编码器中的 LSTM 层定义为 _return_state_ 参数设置为 _True_ 。这将返回 LSTM 图层返回的隐藏状态输出，以及图层中所有单元格的隐藏状态和单元格状态。这些在定义解码器时使用。

```py
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
```

接下来，我们定义解码器。

解码器输入被定义为法语字符一热编码到二元向量的序列，其长度为 _num_decoder_tokens_ 。

LSTM 层定义为返回序列和状态。忽略最终的隐藏和单元状态，仅引用隐藏状态的输出序列。

重要的是，编码器的最终隐藏和单元状态用于初始化解码器的状态。这意味着每次编码器模型对输入序列进行编码时，编码器模型的最终内部状态将用作输出输出序列中第一个字符的起始点。这也意味着编码器和解码器 LSTM 层必须具有相同数量的单元，在这种情况下为 256。

_Dense_ 输出层用于预测每个字符。该 _Dense_ 用于以一次性方式产生输出序列中的每个字符，而不是递归地，至少在训练期间。这是因为在训练期间已知输入模型所需的整个目标序列。

Dense 不需要包含在 _TimeDistributed_ 层中。

```py
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```

最后，使用编码器和解码器的输入以及输出目标序列来定义模型。

```py
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

我们可以在一个独立的示例中将所有这些组合在一起并修复配置并打印模型图。下面列出了定义模型的完整代码示例。

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)
```

运行该示例会创建已定义模型的图，可帮助您更好地了解所有内容是如何挂起的。

注意，编码器 LSTM 不直接将其输出作为输入传递给解码器 LSTM;如上所述，解码器使用最终隐藏和单元状态作为解码器的初始状态。

还要注意，解码器 LSTM 仅将隐藏状态序列传递给密集输出，而不是输出形状信息所建议的最终隐藏状态和单元状态。

![Graph of Encoder-Decoder Model For Training](img/2700043ef80aa99a679207e3c43f0a5e.jpg)

用于训练的编码器 - 解码器模型图

## 神经机器翻译推理

一旦定义的模型适合，它就可以用于进行预测。具体而言，输出英文源文本的法语翻译。

为训练定义的模型已经学习了此操作的权重，但模型的结构并非设计为递归调用以一次生成一个字符。

相反，预测步骤需要新模型，特别是用于编码英文输入字符序列的模型和模型，该模型采用到目前为止生成的法语字符序列和编码作为输入并预测序列中的下一个字符。

定义推理模型需要参考示例中用于训练的模型的元素。或者，可以定义具有相同形状的新模型并从文件加载权重。

编码器模型被定义为从训练模型中的编码器获取输入层（ _encoder_inputs_ ）并输出隐藏和单元状态张量（ _encoder_states_ ）。

```py
# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)
```

解码器更精细。

解码器需要来自编码器的隐藏和单元状态作为新定义的编码器模型的初始状态。由于解码器是一个单独的独立模型，因此这些状态将作为模型的输入提供，因此必须首先定义为输入。

```py
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
```

然后可以指定它们用作解码器 LSTM 层的初始状态。

```py
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
```

对于要在翻译序列中生成的每个字符，将递归地调用编码器和解码器。

在第一次调用时，来自编码器的隐藏和单元状态将用于初始化解码器 LSTM 层，作为模型的输入直接提供。

在随后对解码器的递归调用中，必须向模型提供最后隐藏和单元状态。这些状态值已经在解码器内;尽管如此，我们必须在每次调用时重新初始化状态，给定模型的定义方式，以便在第一次调用时从编码器中获取最终状态。

因此，解码器必须在每次调用时输出隐藏和单元状态以及预测字符，以便可以将这些状态分配给变量并在每个后续递归调用上用于要翻译的给定输入英语文本序列。

```py
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
```

考虑到一些元素的重用，我们可以将所有这些结合在一起，形成一个独立的代码示例，并结合上一节训练模型的定义。完整的代码清单如下。

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)
# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)
# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# summarize model
plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)
```

运行该示例定义了训练模型，推理编码器和推理解码器。

然后创建所有三个模型的图。

![Graph of Encoder Model For Inference](img/9bf817dc8602143f240fc3fd82f44a17.jpg)

用于推理的编码器模型图

编码器的图是直截了当的。

解码器显示解码翻译序列中单个字符所需的三个输入，到目前为止的编码转换输出，以及首先从编码器然后从解码器的输出提供的隐藏和单元状态，因为模型被递归调用给定的翻译。

![Graph of Decoder Model For Inference](img/35033e3c96830fc0f20c05871454d37f.jpg)

用于推理的解码器模型图

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [Francois Chollet 在 Twitter](https://twitter.com/fchollet)
*   [Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 中序列到序列学习的十分钟介绍
*   [Keras seq2seq 代码示例（lstm_seq2seq）](https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py)
*   [Keras 功能 API](https://keras.io/getting-started/functional-api-guide/)
*   [Keras 的 LSTM API](https://keras.io/layers/recurrent/#lstm)
*   [长期短期记忆](http://www.bioinf.jku.at/publications/older/2604.pdf)，1997 年。
*   [了解 LSTM 网络](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，2015 年。
*   [用神经网络进行序列学习的序列](https://arxiv.org/abs/1409.3215)，2014。
*   [使用 RNN 编码器 - 解码器进行统计机器翻译的学习短语表示](https://arxiv.org/abs/1406.1078)，2014。

**更新**

有关如何在独立问题上使用此模型的示例，请参阅此帖子：

*   [如何开发 Keras 中序列到序列预测的编码器 - 解码器模型](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

## 摘要

在这篇文章中，您发现了如何定义用于机器翻译的编码器 - 解码器序列到序列预测模型，如 Keras 深度学习库的作者所描述的。

具体来说，你学到了：

*   神奇机器翻译示例与 Keras 一起提供并在 Keras 博客上进行了描述。
*   如何正确定义编码器 - 解码器 LSTM 以训练神经机器翻译模型。
*   如何正确定义推理模型以使用经过训练的编码器 - 解码器模型来转换新序列。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。