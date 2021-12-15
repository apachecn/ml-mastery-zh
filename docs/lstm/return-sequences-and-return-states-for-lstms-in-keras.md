# 了解 Keras 中 LSTM 的返回序列和返回状态之间的差异

> 原文： [https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)

Keras 深度学习库提供了长期短期记忆或 LSTM 循环神经网络的实现。

作为此实现的一部分，Keras API 提供对返回序列和返回状态的访问。在设计复杂的循环神经网络模型（例如编解码器模型）时，这些数据之间的使用和差异可能会令人困惑。

在本教程中，您将发现 Keras 深度学习库中 LSTM 层的返回序列和返回状态的差异和结果。

完成本教程后，您将了解：

*   返回序列返回每个输入时间步的隐藏状态输出。
*   该返回状态返回上一个输入时间步的隐藏状态输出和单元状态。
*   返回序列和返回状态可以同时使用。

让我们开始吧。

![Understand the Difference Between Return Sequences and Return States for LSTMs in Keras](img/c1155fe1f0943d49c96799fa94c66cb1.jpg)

理解 Keras 中 LSTM 的返回序列和返回状态之间的差异
照片由 [Adrian Curt Dannemann](https://www.flickr.com/photos/12327992@N06/33431042255/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  长短期记忆
2.  返回序列
3.  返回国家
4.  返回状态和序列

## 长短期记忆

长短期记忆（LSTM）是一种由内部门组成的循环神经网络。

与其他循环神经网络不同，网络的内部门允许使用[反向传播通过时间](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)或 BPTT 成功训练模型，并避免消失的梯度问题。

在 Keras 深度学习库中，可以使用 [LSTM（）类](https://keras.io/layers/recurrent/#lstm)创建 LSTM 层。

创建一层 LSTM 内存单元允许您指定层中的内存单元数。

层内的每个单元或单元具有内部单元状态，通常缩写为“`c`”，并输出隐藏状态，通常缩写为“`h`”。

Keras API 允许您访问这些数据，这在开发复杂的循环神经网络架构（如编解码器模型）时非常有用甚至是必需的。

在本教程的其余部分中，我们将查看用于访问这些数据的 API。

## 返回序列

每个 LSTM 单元将为每个输入输出一个隐藏状态`h`。

```py
h = LSTM(X)
```

我们可以在 Keras 中使用一个非常小的模型来演示这一点，该模型具有单个 LSTM 层，该层本身包含单个 LSTM 单元。

在这个例子中，我们将有一个带有 3 个时间步长的输入样本，并在每个时间步骤观察到一个特征：

```py
t1 = 0.1
t2 = 0.2
t3 = 0.3
```

下面列出了完整的示例。

注意：本文中的所有示例都使用 [Keras 功能 API](https://keras.io/getting-started/functional-api-guide/) 。

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

运行该示例为输入序列输出单个隐藏状态，具有 3 个时间步长。

鉴于 LSTM 权重和单元状态的随机初始化，您的特定输出值将有所不同。

```py
[[-0.0953151]]
```

可以访问每个输入时间步的隐藏状态输出。

这可以通过在定义 LSTM 层时将`return_sequences`属性设置为`True`来完成，如下所示：

```py
LSTM(1, return_sequences=True)
```

我们可以使用此更改更新上一个示例。

完整的代码清单如下。

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

运行该示例将返回一个 3 个值的序列，一个隐藏状态输出，用于层中单个 LSTM 单元的每个输入时间步长。

```py
[[[-0.02243521]
  [-0.06210149]
  [-0.11457888]]]
```

栈式 LSTM 层时必须设置 _return_sequences = True_ ，以便第二个 LSTM 层具有三维序列输入。有关更多详细信息，请参阅帖子：

*   [堆叠长短期记忆网络](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)

在使用包含在 TimeDistributed 层中的`Dense`输出层预测输出序列时，您可能还需要访问隐藏状态输出序列。有关详细信息，请参阅此帖子：

*   [如何在 Python](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) 中为长期短期记忆网络使用时间分布层

## 返回国家

LSTM 单元或单元层的输出称为隐藏状态。

这很令人困惑，因为每个 LSTM 单元都保留一个不输出的内部状态，称为单元状态，或`c`。

通常，我们不需要访问单元状态，除非我们正在开发复杂模型，其中后续层可能需要使用另一层的最终单元状态初始化其单元状态，例如在编解码器模型中。

Keras 为 LSTM 层提供了 return_state 参数，该参数将提供对隐藏状态输出（`state_h`）和单元状态（`state_c`）的访问。例如：

```py
lstm1, state_h, state_c = LSTM(1, return_state=True)
```

这可能看起来很混乱，因为 lstm1 和`state_h`都指向相同的隐藏状态输出。这两个张量分离的原因将在下一节中明确。

我们可以使用下面列出的工作示例演示对 LSTM 层中单元格的隐藏和单元格状态的访问。

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

运行该示例返回 3 个数组：

1.  最后一个步骤的 LSTM 隐藏状态输出。
2.  LSTM 隐藏状态输出为最后一个时间步骤（再次）。
3.  最后一个步骤的 LSTM 单元状态。

```py
[array([[ 0.10951342]], dtype=float32),
 array([[ 0.10951342]], dtype=float32),
 array([[ 0.24143776]], dtype=float32)]
```

隐藏状态和单元状态又可以用于初始化具有相同数量单元的另一个 LSTM 层的状态。

## 返回状态和序列

我们可以同时访问隐藏状态序列和单元状态。

这可以通过将 LSTM 层配置为返回序列和返回状态来完成。

```py
lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)
```

The complete example is listed below.

```py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

运行该示例，我们现在可以看到为什么 LSTM 输出张量和隐藏状态输出张量是可分离地声明的。

该层返回每个输入时间步的隐藏状态，然后分别返回上一个时间步的隐藏状态输出和最后一个输入时间步的单元状态。

这可以通过查看返回序列中的最后一个值（第一个数组）与隐藏状态（第二个数组）中的值匹配来确认。

```py
[array([[[-0.02145359],
        [-0.0540871 ],
        [-0.09228823]]], dtype=float32),
 array([[-0.09228823]], dtype=float32),
 array([[-0.19803026]], dtype=float32)]
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [Keras 功能 API](https://keras.io/getting-started/functional-api-guide/)
*   [Keras 的 LSTM API](https://keras.io/layers/recurrent/#lstm)
*   [长期短期记忆](http://www.bioinf.jku.at/publications/older/2604.pdf)，1997 年。
*   [了解 LSTM 网络](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，2015 年。
*   [Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 中序列到序列学习的十分钟介绍

## 摘要

在本教程中，您发现了 Keras 深度学习库中 LSTM 层的返回序列和返回状态的差异和结果。

具体来说，你学到了：

*   返回序列返回每个输入时间步的隐藏状态输出。
*   该返回状态返回上一个输入时间步的隐藏状态输出和单元状态。
*   返回序列和返回状态可以同时使用。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。