# 如何在Keras中开发带有注意力的编解码器模型

> 原文： [https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)

用于循环神经网络的编解码器架构被证明在诸如机器翻译和字幕生成之类的自然语言处理领域中的大量序列到序列预测问题上是强大的。

注意力是一种解决编解码器架构在长序列上的限制的机制，并且通常加速学习并且提升模型的技能而没有序列来排序预测问题。

在本教程中，您将了解如何使用Keras在Python中开发一个编解码器循环神经网络。

完成本教程后，您将了解：

*   如何设计一个小的可配置问题来评估编解码器循环神经网络有无注意。
*   如何设计和评估编解码器网络，有和没有注意序列预测问题。
*   如何有力地比较编解码器网络的表现有没有注意。

让我们开始吧。

![How to Develop an Encoder-Decoder Model with Attention for Sequence-to-Sequence Prediction in Keras](img/a508d456c2630712158b024c6041c69e.jpg)

如何开发一个编解码器模型，注意Keras中的序列到序列预测
照片由 [Angela和Andrew](https://www.flickr.com/photos/150568953@N07/34585914155/) ，保留一些权利。

## 教程概述

本教程分为6个部分;他们是：

1.  注意编解码器
2.  注意力的测试问题
3.  编解码器没有注意
4.  自定义Keras注意层
5.  注意编解码器
6.  模型比较

### Python环境

本教程假定您已安装Python 3 SciPy环境。

您必须安装带有TensorFlow或Theano后端的Keras（2.0或更高版本）。

本教程还假设您安装了scikit-learn，Pandas，NumPy和Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 注意编解码器

用于循环神经网络的编解码器模型是用于序列到序列预测问题的架构。

它由两个子模型组成，顾名思义：

*   **编码器**：编码器负责逐步执行输入时间步长并将整个序列编码为称为上下文向量的固定长度向量。
*   **解码器**：解码器负责在从上下文向量读取时逐步执行输出时间步长。

该架构的问题在于长输入或输出序列的表现差。原因被认为是由于编码器使用的固定大小的内部表示。

注意是解决此限制的架构的扩展。它的工作原理是首先提供从编码器到解码器的更丰富的上下文和学习机制，其中解码器可以在预测输出序列中的每个时间步长时学习在更丰富的编码中注意的位置。

有关编解码器架构的更多关注，请参阅帖子：

*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项
*   [编解码器循环神经网络中的注意事项如何工作](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

## 注意力的测试问题

在我们开发注意力模型之前，我们将首先定义一个可设计的可扩展测试问题，我们可以用它来确定注意力是否提供任何好处。

在这个问题中，我们将生成随机整数序列作为输入和匹配输出序列，该输出序列由输入序列中的整数子集组成。

例如，输入序列可能是[1,6,2,7,3]，预期的输出序列可能是序列[1,6]中的前两个随机整数。

我们将定义问题，使输入和输出序列长度相同，并根据需要用“0”值填充输出序列。

首先，我们需要一个函数来生成随机整数序列。我们将使用Python [randint（）](https://docs.python.org/3/library/random.html)函数生成0和最大值之间的随机整数，并使用此范围作为问题的基数（例如，要素数或难度轴）。

下面的函数 _generate_sequence（）_将生成一个固定长度和指定基数的随机整数序列。

```py
from random import randint

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# generate random sequence
sequence = generate_sequence(5, 50)
print(sequence)
```

运行此示例将生成一个包含5个时间步长的序列，其中序列中的每个值都是0到49之间的随机整数。

```py
[43, 3, 28, 34, 33]
```

接下来，我们需要一个函数[将单热编码](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)的离散整数值转换成二进制向量。

如果使用50的基数，则每个整数将由0个值的50个元素向量和指定整数值的索引中的1表示。

下面的 _one_hot_encode（）_函数将对给定的整数序列进行热编码。

```py
# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

我们还需要能够解码编码序列。需要将预测从模型或编码的预期序列转换回我们可以读取和评估的整数序列。

下面的 _one_hot_decode（）_函数将单热编码序列解码回整数序列。

```py
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

我们可以在下面的例子中测试这些操作。

```py
from random import randint
from numpy import array
from numpy import argmax

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# generate random sequence
sequence = generate_sequence(5, 50)
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence, 50)
print(encoded)
# decode
decoded = one_hot_decode(encoded)
print(decoded)
```

首先运行该示例打印随机生成的序列，然后打印单热编码版本，最后再打印解码序列。

```py
[3, 18, 32, 11, 36]
[[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]]
[3, 18, 32, 11, 36]
```

最后，我们需要一个能够创建输入和输出序列对的函数来训练和评估模型。

下面命名为 _get_pair（）_的函数将返回一个输入和输出序列对，给定指定的输入长度，输出长度和基数。输入和输出序列的长度和输入序列的长度相同，但输出序列将作为输入序列的第一个`n`字符，并用零值填充到所需长度。

然后对整数序列进行编码，然后重新成形为循环神经网络所需的3D格式，其尺寸为：_样本_，_时间步长_和_特征_。在这种情况下，样本总是1，因为我们只生成一个输入 - 输出对，时间步长是输入序列长度，特征是每个时间步长的基数。

```py
# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
	# generate random sequence
	sequence_in = generate_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, n_unique)
	y = one_hot_encode(sequence_out, n_unique)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y
```

我们可以将它们放在一起并演示数据准备代码。

```py
from random import randint
from numpy import array
from numpy import argmax

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
	# generate random sequence
	sequence_in = generate_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, n_unique)
	y = one_hot_encode(sequence_out, n_unique)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# generate random sequence
X, y = get_pair(5, 2, 50)
print(X.shape, y.shape)
print('X=%s, y=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))
```

运行该示例生成单个输入 - 输出对并打印两个数组的形状。

然后以解码的形式打印生成的对，其中我们可以看到序列的前两个整数在输出序列中被再现，随后是零值的填充。

```py
(1, 5, 50) (1, 5, 50)
X=[12, 20, 36, 40, 12], y=[12, 20, 0, 0, 0]
```

## 编解码器没有注意

在本节中，我们将在没有注意的情况下使用编解码器模型开发关于问题表现的基线。

我们将在5个时间步的输入和输出序列中修复问题定义，输出序列中输入序列的前2个元素和基数为50。

```py
# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
```

我们可以通过从编码器LSTM模型获得输出，在Keras中开发一个简单的编解码器模型，对输出序列中的时间步长重复n次，然后使用解码器来预测输出序列。

有关如何在Keras中定义编解码器架构的更多详细信息，请参阅帖子：

*   [编解码器长短期记忆网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

我们将使用相同数量的单位配置编码器和解码器，在本例中为150.我们将使用梯度下降的有效Adam实现并优化分类交叉熵损失函数，因为该问题在技术上是一个多类别分类问题。

模型的配置是在经过一些试验和错误之后找到的，并未进行优化。

下面列出了Keras中编解码器架构的代码。

```py
# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

我们将在5,000个随机输入 - 输出对的整数序列上训练模型。

```py
# train LSTM
for epoch in range(5000):
	# generate new random sequence
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, verbose=2)
```

一旦经过训练，我们将在100个新的随机生成的整数序列上评估模型，并且只在整个输出序列与预期值匹配时才标记预测正确。

```py
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
```

最后，我们将打印10个预期输出序列和模型预测序列的例子。

将所有这些放在一起，下面列出了完整的示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(5000):
	# generate new random sequence
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, verbose=2)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
```

运行此示例不会花费很长时间，可能需要几分钟的CPU，不需要GPU。

据报道该模型的准确率略低于20％。鉴于[神经网络的随机性](https://machinelearningmastery.com/randomness-in-machine-learning/)，您的结果会有所不同;考虑运行几次示例并取平均值。

```py
Accuracy: 19.00%
```

我们可以从样本输出中看到，模型确实在输出序列中得到一个数字，对于大多数或所有情况都是正确的，并且只与第二个数字斗争。正确预测所有零填充值。

```py
Expected: [47, 0, 0, 0, 0] Predicted [47, 47, 0, 0, 0]
Expected: [43, 31, 0, 0, 0] Predicted [43, 31, 0, 0, 0]
Expected: [14, 22, 0, 0, 0] Predicted [14, 14, 0, 0, 0]
Expected: [39, 31, 0, 0, 0] Predicted [39, 39, 0, 0, 0]
Expected: [6, 4, 0, 0, 0] Predicted [6, 4, 0, 0, 0]
Expected: [47, 0, 0, 0, 0] Predicted [47, 47, 0, 0, 0]
Expected: [39, 33, 0, 0, 0] Predicted [39, 39, 0, 0, 0]
Expected: [23, 2, 0, 0, 0] Predicted [23, 23, 0, 0, 0]
Expected: [19, 28, 0, 0, 0] Predicted [19, 3, 0, 0, 0]
Expected: [32, 33, 0, 0, 0] Predicted [32, 32, 0, 0, 0]
```

## 自定义Keras注意层

现在我们需要关注编解码器模型。

在撰写本文时，Keras没有内置于库中的注意力，但很快就会 [](https://github.com/fchollet/keras/pull/7980)。

在Keras正式提供关注之前，我们可以开发自己的实现或使用现有的第三方实现。

为了加快速度，让我们使用现有的第三方实现。

[Zafarali Ahmed](http://www.zafarali.me/) [Datalogue](https://www.datalogue.io/) 的实习生为Keras开发了一个[自定义层](https://keras.io/layers/writing-your-own-keras-layers/)，提供了关注支持，在一篇名为“[如何可视化您的复发的帖子中提出2017年Keras](https://medium.com/datalogue/attention-in-keras-1892773a4f22) 中关注的神经网络和GitHub项目称为“ [keras-attention](https://github.com/datalogue/keras-attention) ”。

自定义注意层称为`AttentionDecoder`，可在GitHub项目的 [custom_recurrents.py](https://github.com/datalogue/keras-attention/blob/master/models/custom_recurrents.py) 文件中找到。我们可以在项目的 [GNU Affero通用公共许可证v3.0许可证](https://github.com/datalogue/keras-attention/blob/master/LICENSE)下重用此代码。

下面列出了自定义层的副本以确保完整性。将其复制并粘贴到当前工作目录中名为“`attention_decoder.py`”的新单独文件中。

```py
import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

我们可以通过以下方式导入项目中来使用此自定义层：

```py
from attention_decoder import AttentionDecoder
```

如Bahdanau等人所述，该层实现了关注。在他们的论文“[神经机器翻译中通过联合学习来对齐和翻译](https://arxiv.org/abs/1409.0473)。”

该代码在原始帖子中得到了很好的解释，并与LSTM和注意力方程相关联。

该实现的限制是它必须输出与输入序列长度相同的序列，编解码器架构被设计为克服的特定限制。

重要的是，新层管理由第二LSTM执行的重复解码，以及由编解码器模型中的密集输出层执行的模型的softmax输出，而没有注意。这极大地简化了模型的代码。

值得注意的是，自定义层建立在Keras的 [Recurrent](https://github.com/fchollet/keras/blob/master/keras/legacy/layers.py#L762) 层上，在编写本文时，它被标记为遗留代码，并且可能会在某个时候从项目中删除。

## 编解码器注意

现在我们已经可以使用我们的注意力实现，我们可以开发一个编解码器模型，注意我们设计的序列预测问题。

具有关注层的模型定义如下。我们可以看到该层处理编解码器模型本身的一些机制，使得定义模型更简单。

```py
# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_features))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

而已。示例的其余部分是相同的。

下面列出了完整的示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from attention_decoder import AttentionDecoder

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2

# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_features))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(5000):
	# generate new random sequence
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, verbose=2)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
```

运行该示例在100个随机生成的输入 - 输出对上打印模型的技能。在相同的资源和相同数量的训练下，关注的模型表现得更好。

鉴于神经网络的随机性，您的结果可能会有所不同。尝试运行几次示例。

```py
Accuracy: 95.00%
```

通过抽查一些样本输出和预测序列，我们可以看到很少的错误，即使在前两个元素中存在零值的情况下也是如此。

```py
Expected: [48, 47, 0, 0, 0] Predicted [48, 47, 0, 0, 0]
Expected: [7, 46, 0, 0, 0] Predicted [7, 46, 0, 0, 0]
Expected: [32, 30, 0, 0, 0] Predicted [32, 2, 0, 0, 0]
Expected: [3, 25, 0, 0, 0] Predicted [3, 25, 0, 0, 0]
Expected: [45, 4, 0, 0, 0] Predicted [45, 4, 0, 0, 0]
Expected: [49, 9, 0, 0, 0] Predicted [49, 9, 0, 0, 0]
Expected: [22, 23, 0, 0, 0] Predicted [22, 23, 0, 0, 0]
Expected: [29, 36, 0, 0, 0] Predicted [29, 36, 0, 0, 0]
Expected: [0, 29, 0, 0, 0] Predicted [0, 29, 0, 0, 0]
Expected: [11, 26, 0, 0, 0] Predicted [11, 26, 0, 0, 0]
```

## 模型比较

尽管我们从模型中获得了更好的结果，但是每个模型的单次运行都会报告结果。

在这种情况下，我们通过多次重复评估每个模型并报告这些运行的平均表现来寻求更稳健的发现。有关评估神经网络模型的这种强大方法的更多信息，请参阅帖子：

*   [如何评估深度学习模型的技巧](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

我们可以定义一个函数来创建每种类型的模型，如下所示。

```py
# define the encoder-decoder model
def baseline_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
	model.add(RepeatVector(n_timesteps_in))
	model.add(LSTM(150, return_sequences=True))
	model.add(TimeDistributed(Dense(n_features, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# define the encoder-decoder with attention model
def attention_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(AttentionDecoder(150, n_features))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model
```

然后，我们可以定义一个函数来拟合和评估拟合模型的准确率并返回准确度分数。

```py
# train and evaluate a model, return accuracy
def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
	# train LSTM
	for epoch in range(5000):
		# generate new random sequence
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		# fit model for one epoch on this sequence
		model.fit(X, y, epochs=1, verbose=0)
	# evaluate LSTM
	total, correct = 100, 0
	for _ in range(total):
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		yhat = model.predict(X, verbose=0)
		if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
			correct += 1
	return float(correct)/float(total)*100.0
```

综上所述，我们可以多次重复创建，训练和评估每种类型的模型，并报告重复的平均准确度。为了减少运行时间，我们将重复每次模型评估10次，但如果您有资源，则可以将此值增加到30或100次。

The complete example is listed below.

```py
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from attention_decoder import AttentionDecoder

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# define the encoder-decoder model
def baseline_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
	model.add(RepeatVector(n_timesteps_in))
	model.add(LSTM(150, return_sequences=True))
	model.add(TimeDistributed(Dense(n_features, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# define the encoder-decoder with attention model
def attention_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(AttentionDecoder(150, n_features))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# train and evaluate a model, return accuracy
def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
	# train LSTM
	for epoch in range(5000):
		# generate new random sequence
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		# fit model for one epoch on this sequence
		model.fit(X, y, epochs=1, verbose=0)
	# evaluate LSTM
	total, correct = 100, 0
	for _ in range(total):
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		yhat = model.predict(X, verbose=0)
		if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
			correct += 1
	return float(correct)/float(total)*100.0

# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
n_repeats = 10
# evaluate encoder-decoder model
print('Encoder-Decoder Model')
results = list()
for _ in range(n_repeats):
	model = baseline_model(n_timesteps_in, n_features)
	accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
	results.append(accuracy)
	print(accuracy)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
# evaluate encoder-decoder with attention model
print('Encoder-Decoder With Attention Model')
results = list()
for _ in range(n_repeats):
	model = attention_model(n_timesteps_in, n_features)
	accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
	results.append(accuracy)
	print(accuracy)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
```

运行此示例将打印每个模型重复的准确率，以便您了解运行的进度。

```py
Encoder-Decoder Model
20.0
23.0
23.0
18.0
28.000000000000004
28.999999999999996
23.0
26.0
21.0
20.0
Mean Accuracy: 23.10%

Encoder-Decoder With Attention Model
98.0
91.0
94.0
93.0
96.0
99.0
97.0
94.0
99.0
96.0
Mean Accuracy: 95.70%
```

我们可以看到，即使平均超过10次运行，注意模型仍然表现出比没有注意的编解码器模型更好的表现，23.10％对95.70％。

此评估的一个很好的扩展是捕获每个模型的每个时期的模型损失，取平均值，并比较损失如何随着时间的推移而变化，无论是否受到关注。我希望这种追踪能够比非注意力模型更快，更快地显示出更好的技能，进一步突出了这种方法的好处。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项
*   [编解码器循环神经网络中的注意事项如何工作](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
*   [编解码器长短期记忆网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
*   [如何评估深度学习模型的技巧](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)
*   [如何在Keras中注意循环神经网络](https://medium.com/datalogue/attention-in-keras-1892773a4f22)，2017。
*   [keras-attention GitHub Project](https://github.com/datalogue/keras-attention)
*   [通过共同学习对齐和翻译的神经机器翻译](https://github.com/datalogue/keras-attention)，2015。

## 摘要

在本教程中，您了解了如何使用Keras在Python中开发编解码器循环神经网络。

具体来说，你学到了：

*   如何设计一个小的可配置问题来评估编解码器循环神经网络有无注意。
*   如何设计和评估编解码器网络，有和没有注意序列预测问题。
*   如何有力地比较编解码器网络的表现有没有注意。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。