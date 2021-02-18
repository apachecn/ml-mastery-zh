# 如何开发Keras序列到序列预测的编码器 - 解码器模型

> 原文： [https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

编码器 - 解码器模型提供了使用循环神经网络来解决具有挑战性的序列到序列预测问题（例如机器翻译）的模式。

可以在Keras Python深度学习库中开发编码器 - 解码器模型，并且在Keras博客上描述了使用该模型开发的神经机器翻译系统的示例，其中示例代码与Keras项目一起分发。

这个例子可以为您自己的序列到序列预测问题提供编码器 - 解码器LSTM模型的基础。

在本教程中，您将了解如何使用Keras为序列到序列预测问题开发复杂的编码器 - 解码器循环神经网络。

完成本教程后，您将了解：

*   如何在Keras中正确定义复杂的编码器 - 解码器模型以进行序列到序列预测。
*   如何定义可用于评估编码器 - 解码器LSTM模型的人为但可扩展的序列到序列预测问题。
*   如何在Keras中应用编码器 - 解码器LSTM模型来解决可伸缩的整数序列到序列预测问题。

让我们开始吧。

![How to Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras](img/aa2a843b49a185a698da7748e1ce1131.jpg)

如何开发Keras序列到序列预测的编码器 - 解码器模型
照片由[BjörnGroß](https://www.flickr.com/photos/damescalito/35370558025/)，保留一些权利。

## 教程概述

本教程分为3个部分;他们是：

*   Keras中的编码器 - 解码器模型
*   可扩展的序列到序列问题
*   用于序列预测的编码器 - 解码器LSTM

### Python环境

本教程假定您已安装Python SciPy环境。您可以在本教程中使用Python 2或3。

您必须安装带有TensorFlow或Theano后端的Keras（2.0或更高版本）。

本教程还假设您安装了scikit-learn，Pandas，NumPy和Matplotlib。

如果您需要有关环境的帮助，请参阅此帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## Keras中的编码器 - 解码器模型

编码器 - 解码器模型是一种组织序列到序列预测问题的循环神经网络的方法。

它最初是为机器翻译问题而开发的，尽管它已经证明在相关的序列到序列预测问题上是成功的，例如文本摘要和问题回答。

该方法涉及两个循环神经网络，一个用于编码源序列，称为编码器，第二个用于将编码的源序列解码为目标序列，称为解码器。

Keras深度学习Python库提供了一个如何实现机器翻译的编码器 - 解码器模型的例子（ [lstm_seq2seq.py](https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py) ），由库创建者在帖子中描述：“[十分钟的介绍在Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 中进行序列到序列学习。“

有关此型号的详细分类，请参阅帖子：

*   [如何定义Keras神经机器翻译的编码器 - 解码器序列 - 序列模型](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)

有关使用return_state的更多信息，可能是您的新手，请参阅帖子：

*   [了解Keras中LSTM的返回序列和返回状态之间的差异](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)

有关Keras Functional API入门的更多帮助，请参阅帖子：

*   [如何使用Keras功能API进行深度学习](https://machinelearningmastery.com/keras-functional-api-deep-learning/)

使用该示例中的代码作为起点，我们可以开发一个通用函数来定义编码器 - 解码器循环神经网络。下面是这个名为 _define_models（）_的函数。

```py
# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model
```

该函数有3个参数，如下所示：

*   **n_input** ：输入序列的基数，例如每个时间步长的要素，单词或字符数。
*   **n_output** ：输出序列的基数，例如每个时间步长的要素，单词或字符数。
*   **n_units** ：在编码器和解码器模型中创建的单元数，例如128或256。

然后该函数创建并返回3个模型，如下所示：

*   **train** ：在给定源，目标和移位目标序列的情况下可以训练的模型。
*   **inference_encoder** ：在对新源序列进行预测时使用的编码器模型。
*   **inference_decoder** 解码器模型在对新源序列进行预测时使用。

在给定源序列和目标序列的情况下训练模型，其中模型将源序列和​​移位版本的目标序列作为输入并预测整个目标序列。

例如，一个源序列可以是[1,2,3]和靶序列[4,5,6]。训练期间模型的输入和输出将是：

```py
Input1: ['1', '2', '3']
Input2: ['_', '4', '5']
Output: ['4', '5', '6']
```

当为新源序列生成目标序列时，该模型旨在被递归地调用。

对源序列进行编码，并且使用诸如“_”的“序列开始”字符一次一个元素地生成目标序列以开始该过程。因此，在上述情况下，在训练期间会出现以下输入输出对：

```py
t, 	Input1,				Input2,		Output
1,  ['1', '2', '3'],	'_',		'4'
2,  ['1', '2', '3'],	'4',		'5'
3,  ['1', '2', '3'],	'5',		'6'
```

在这里，您可以看到如何使用模型的递归使用来构建输出序列。

在预测期间，`inference_encoder`模型用于编码输入序列，其返回用于初始化`inference_decoder`模型的状态。从那时起，`inference_decoder`模型用于逐步生成预测。

在训练模型以生成给定源序列的目标序列之后，可以使用下面名为 _predict_sequence（）_的函数。

```py
# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)
```

该函数有5个参数如下：

*   **infenc** ：在对新源序列进行预测时使用的编码器模型。
*   **infdec** ：在对新源序列进行预测时使用解码器模型。
*   **source** ：编码的源序列。
*   **n_steps** ：目标序列中的时间步数。
*   **基数**：输出序列的基数，例如每个时间步的要素，单词或字符的数量。

然后该函数返回包含目标序列的列表。

## 可扩展的序列到序列问题

在本节中，我们定义了一个人为的，可扩展的序列到序列预测问题。

源序列是一系列随机生成的整数值，例如[20,36,40,10,34,28]，目标序列是输入序列的反向预定义子集，例如前3个元素以相反的顺序[40,36,20]。

源序列的长度是可配置的;输入和输出序列的基数以及目标序列的长度也是如此。

我们将使用6个元素的源序列，基数为50，以及3个元素的目标序列。

下面是一些更具体的例子。

```py
Source,						Target
[13, 28, 18, 7, 9, 5]		[18, 28, 13]
[29, 44, 38, 15, 26, 22]	[38, 44, 29]
[27, 40, 31, 29, 32, 1]		[31, 40, 27]
...
```

我们鼓励您探索更大，更复杂的变体。在下面的评论中发布您的发现。

让我们首先定义一个函数来生成一系列随机整数。

我们将使用0的值作为填充或序列字符的开头，因此它是保留的，我们不能在源序列中使用它。为实现此目的，我们将为配置的基数添加1，以确保单热编码足够大（例如，值1映射到索引1中的'1'值）。

例如：

```py
n_features = 50 + 1
```

我们可以使用 _randint（）_ python函数生成1到1范围内的随机整数 - 减去问题基数的大小。下面的 _generate_sequence（）_生成一系列随机整数。

```py
# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]
```

接下来，我们需要在给定源序列的情况下创建相应的输出序列。

为了简单起见，我们将选择源序列的前n个元素作为目标序列并反转它们。

```py
# define target sequence
target = source[:n_out]
target.reverse()
```

我们还需要一个版本的输出序列向前移动一个步骤，我们可以将其用作到目前为止生成的模拟目标，包括第一个时间步骤中序列值的开始。我们可以直接从目标序列创建它。

```py
# create padded input target sequence
target_in = [0] + target[:-1]
```

现在已经定义了所有序列，我们可以对它们进行一次热编码，即将它们转换成二进制向量序列。我们可以使用内置 _to_categorical（）_函数的Keras来实现这一点。

我们可以将所有这些放入名为 _get_dataset（）_的函数中，该函数将生成特定数量的序列，我们可以使用它来训练模型。

```py
# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)
```

最后，我们需要能够解码一个热门的编码序列，使其再次可读。

这对于打印所生成的靶序列以及容易地比较完整预测的靶序列是否与预期的靶序列匹配是必需的。 _one_hot_decode（）_函数将解码编码序列。

```py
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

我们可以将所有这些结合在一起并测试这些功能。

下面列出了一个完整的工作示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# generate a single source and target sequence
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
print(X1.shape, X2.shape, y.shape)
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))
```

首先运行示例打印生成的数据集的形状，确保训练模型所需的3D形状符合我们的期望。

然后将生成的序列解码并打印到屏幕，证明源和目标序列的准备符合我们的意图并且解码操作正在起作用。

```py
(1, 6, 51) (1, 3, 51) (1, 3, 51)
X1=[32, 16, 12, 34, 25, 24], X2=[0, 12, 16], y=[12, 16, 32]
```

我们现在准备为这个序列到序列预测问题开发一个模型。

## 用于序列预测的编码器 - 解码器LSTM

在本节中，我们将第一部分中开发的编码器 - 解码器LSTM模型应用于第二部分中开发的序列到序列预测问题。

第一步是配置问题。

```py
# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
```

接下来，我们必须定义模型并编译训练模型。

```py
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
```

接下来，我们可以生成100,000个示例的训练数据集并训练模型。

```py
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
```

一旦模型被训练，我们就可以对其进行评估。我们将通过对100个源序列进行预测并计算正确预测的目标序列的数量来实现此目的。我们将在解码序列上使用numpy _array_equal（）_函数来检查是否相等。

```py
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
```

最后，我们将生成一些预测并打印解码的源，目标和预测的目标序列，以了解模型是否按预期工作。

将所有这些元素放在一起，下面列出了完整的代码示例。

```py
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define padded target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
```

首先运行该示例将打印准备好的数据集的形状。

```py
(100000, 6, 51) (100000, 3, 51) (100000, 3, 51)
```

接下来，该模型是合适的。您应该看到一个进度条，并且在现代多核CPU上运行应该不到一分钟。

```py
100000/100000 [==============================] - 50s - loss: 0.6344 - acc: 0.7968
```

接下来，评估模型并打印精度。我们可以看到该模型在新随机生成的示例上实现了100％的准确性。

```py
Accuracy: 100.00%
```

最后，生成10个新示例并预测目标序列。同样，我们可以看到模型在每种情况下正确地预测输出序列，并且期望值与源序列的反向前3个元素匹配。

```py
X=[22, 17, 23, 5, 29, 11] y=[23, 17, 22], yhat=[23, 17, 22]
X=[28, 2, 46, 12, 21, 6] y=[46, 2, 28], yhat=[46, 2, 28]
X=[12, 20, 45, 28, 18, 42] y=[45, 20, 12], yhat=[45, 20, 12]
X=[3, 43, 45, 4, 33, 27] y=[45, 43, 3], yhat=[45, 43, 3]
X=[34, 50, 21, 20, 11, 6] y=[21, 50, 34], yhat=[21, 50, 34]
X=[47, 42, 14, 2, 31, 6] y=[14, 42, 47], yhat=[14, 42, 47]
X=[20, 24, 34, 31, 37, 25] y=[34, 24, 20], yhat=[34, 24, 20]
X=[4, 35, 15, 14, 47, 33] y=[15, 35, 4], yhat=[15, 35, 4]
X=[20, 28, 21, 39, 5, 25] y=[21, 28, 20], yhat=[21, 28, 20]
X=[50, 38, 17, 25, 31, 48] y=[17, 38, 50], yhat=[17, 38, 50]
```

您现在有一个编码器 - 解码器LSTM模型的模板，您可以将其应用于您自己的序列到序列预测问题。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 相关文章

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)
*   [如何定义Keras神经机器翻译的编码器 - 解码器序列 - 序列模型](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
*   [了解Keras中LSTM的返回序列和返回状态之间的差异](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)
*   [如何使用Keras功能API进行深度学习](https://machinelearningmastery.com/keras-functional-api-deep-learning/)

### Keras资源

*   [Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 中序列到序列学习的十分钟介绍
*   [Keras seq2seq代码示例（lstm_seq2seq）](https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py)
*   [Keras功能API](https://keras.io/getting-started/functional-api-guide/)
*   [Keras的LSTM API](https://keras.io/layers/recurrent/#lstm)

## 摘要

在本教程中，您了解了如何使用Keras为序列到序列预测问题开发编码器 - 解码器循环神经网络。

具体来说，你学到了：

*   如何在Keras中正确定义复杂的编码器 - 解码器模型以进行序列到序列预测。
*   如何定义可用于评估编码器 - 解码器LSTM模型的人为但可扩展的序列到序列预测问题。
*   如何在Keras中应用编码器 - 解码器LSTM模型来解决可伸缩的整数序列到序列预测问题。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。