# 如何在Python中单热编码序列数据

> 原文： [https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)

机器学习算法不能直接使用分类数据。

必须将分类数据转换为数字。

当您处理序列分类类型问题并计划使用深度学习方法（如长期短期记忆循环神经网络）时，这适用。

在本教程中，您将了解如何将输入或输出序列数据转换为单热编码，以便在Python中使用深度学习进行序列分类问题。

完成本教程后，您将了解：

*   什么是整数编码和单热编码，以及为什么它们在机器学习中是必需的。
*   如何在Python中手动计算整数编码和单热编码。
*   如何使用scikit-learn和Keras库在Python中自编码序列数据。

让我们开始吧。

![How to One Hot Encode Sequence Classification Data in Python](img/8213edb29ce473720ce6bb7f2f628a83.jpg)

如何在Python中使用热编码序列分类数据
照片由 [Elias Levy](https://www.flickr.com/photos/elevy/6997586997/) 拍摄，保留一些权利。

## 教程概述

本教程分为4个部分;他们是：

1.  什么是单热编码？
2.  手动单热编码
3.  单热门编码与scikit-learn
4.  单热门编码与Keras

## 什么是单热编码？

一种热编码是将分类变量表示为二进制向量。

这首先要求将分类值映射到整数值。

然后，每个整数值表示为二进制向量，除了整数的索引外，它都是零值，用1标记。

### 单热编码的工作示例

让我们用一个有效的例子来具体化。

假设我们有一系列标签，其值为“红色”和“绿色”。

我们可以将'red'指定为整数值0，将'green'指定为整数值1.只要我们总是将这些数字指定给这些标签，就称为整数编码。一致性很重要，以便我们可以稍后反转编码并从整数值返回标签，例如在做出预测时。

接下来，我们可以创建一个二进制向量来表示每个整数值。对于2个可能的整数值，向量的长度为2。

编码为0的“红色”标签将用二进制向量[1,0]表示，其中第零个索引用值1标记。反过来，编码为1的“绿色”标签将用二进制向量[0,1]，其中第一个索引标记为值1。

如果我们有序列：

```py
'red', 'red', 'green'
```

我们可以用整数编码来表示它：

```py
0, 0, 1
```

和热门编码：

```py
[1, 0]
[1, 0]
[0, 1]
```

### 为什么要使用单热编码？

一种热编码允许分类数据的表示更具表现力。

许多机器学习算法不能直接使用分类数据。必须将类别转换为数字。这对于分类的输入和输出变量都是必需的。

我们可以直接使用整数编码，在需要的地方重缩放。这可能适用于类别之间存在自然序数关系的问题，反过来又是整数值，例如温度“冷”，“暖”和“热”的标签。

当没有顺序关系并且允许表示依赖于任何这样的关系可能有损于学习解决问题时可能存在问题。一个例子可能是标签'狗'和'猫'

在这些情况下，我们希望为网络提供更具表现力的能力，以便为每个可能的标签值学习类似概率的数字。这有助于使问题更容易让网络建模。当单热编码用于输出变量时，它可以提供比单个标签更细微的预测集。

## 手动单热编码

在这个例子中，我们假设我们有一个字母字母的示例字符串，但示例序列并未涵盖所有可能的示例。

我们将使用以下字符的输入序列：

```py
hello world
```

我们假设所有可能输入的范围是小写字符和空格的完整字母表。因此，我们将以此为借口演示如何推出自己的热门编码。

下面列出了完整的示例。

```py
from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)
```

首先运行该示例打印输入字符串。

从char值到整数值创建所有可能输入的映射。然后使用该映射对输入字符串进行编码。我们可以看到输入'h'中的第一个字母编码为7，或者可能输入值（字母表）数组中的索引7。

然后将整数编码转换为单热编码。这是一次完成一个整数编码字符。创建0值的列表，使用字母表的长度，以便可以表示任何预期的字符。

接下来，特定字符的索引标记为1.我们可以看到编码为7的第一个字母'h'整数由长度为27且第7个索引标记为1的二进制向量表示。

最后，我们反转第一个字母的编码并打印结果。我们通过使用NumPy argmax（）函数定位具有最大值的二进制向量中的索引，然后在字符值的反向查找表中使用整数值来实现此操作。

注意：输出已格式化以便于阅读。

```py
hello world

[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]

[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

h
```

现在我们已经看到了如何从零开始编写自己的热编码，让我们看看如何在输入序列完全捕获预期输入值范围的情况下，使用scikit-learn库自动执行此映射。

## 单热门编码与scikit-learn

在此示例中，我们假设您具有以下3个标签的输出序列：

```py
"cold"
"warm"
"hot"
```

10个时间步长的示例序列可以是：

```py
cold, cold, warm, cold, hot, hot, warm, cold, warm, hot
```

这将首先需要整数编码，例如1,2,3。接下来是一个整数的热编码到具有3个值的二进制向量，例如[1,0,0]。

该序列提供序列中每个可能值的至少一个示例。因此，我们可以使用自动方法来定义标签到整数和整数到二进制向量的映射。

在这个例子中，我们将使用scikit-learn库中的编码器。具体地， [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 创建标签的整数编码， [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 用于创建整数编码值的单热编码。

The complete example is listed below.

```py
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
```

首先运行该示例将打印标签序列。接下来是标签的整数编码，最后是单热编码。

训练数据包含所有可能示例的集合，因此我们可以依赖整数和单热编码变换来创建标签到编码的完整映射。

默认情况下，OneHotEncoder类将返回更有效的稀疏编码。这可能不适合某些应用程序，例如与Keras深度学习库一起使用。在这种情况下，我们通过设置 _sparse = False_ 参数来禁用稀疏返回类型。

如果我们在这个3值热编码中接收到预测，我们可以轻松地将变换反转回原始标签。

首先，我们可以使用argmax（）NumPy函数来定位具有最大值的列的索引。然后可以将其馈送到LabelEncoder以计算反向变换回文本标签。

这在示例的结尾处被证明，其中第单热编码示例的逆变换返回到标签值'cold'。

再次注意，输入的格式是为了便于阅读。

```py
['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']

[0 0 2 0 1 1 2 0 2 1]

[[ 1\.  0\.  0.]
 [ 1\.  0\.  0.]
 [ 0\.  0\.  1.]
 [ 1\.  0\.  0.]
 [ 0\.  1\.  0.]
 [ 0\.  1\.  0.]
 [ 0\.  0\.  1.]
 [ 1\.  0\.  0.]
 [ 0\.  0\.  1.]
 [ 0\.  1\.  0.]]

['cold']
```

在下一个示例中，我们将看看如何直接对一个整数值序列进行热编码。

## 单热门编码与Keras

您可能有一个已经整数编码的序列。

在进行一些缩放之后，您可以直接使用整数。或者，您可以直接对整数进行热编码。如果整数没有真正的序数关系并且实际上只是标签的占位符，则需要考虑这一点。

Keras库提供了一个名为 [to_categorical（）](https://keras.io/utils/#to_categorical)的函数，您可以将其用于单热编码整数数据。

在这个例子中，我们有4个整数值[0,1,2,3]，我们有以下10个数字的输入序列：

```py
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
```

序列有一个所有已知值的示例，因此我们可以直接使用to_categorical（）函数。或者，如果序列从0开始（从0开始）并且不代表所有可能的值，我们可以指定num_classes参数 _to_categorical（num_classes = 4）_。

下面列出了此功能的完整示例。

```py
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)
```

首先运行示例定义并打印输入序列。

然后将整数编码为二进制向量并打印。我们可以看到第一个整数值1被编码为[0,1,0,0]，就像我们期望的那样。

然后，我们通过在序列中的第一个值上使用NumPy argmax（）函数来反转编码，该函数返回第一个整数的期望值1。

```py
[1 3 2 0 3 2 2 1 0 1]

[[ 0\.  1\.  0\.  0.]
 [ 0\.  0\.  0\.  1.]
 [ 0\.  0\.  1\.  0.]
 [ 1\.  0\.  0\.  0.]
 [ 0\.  0\.  0\.  1.]
 [ 0\.  0\.  1\.  0.]
 [ 0\.  0\.  1\.  0.]
 [ 0\.  1\.  0\.  0.]
 [ 1\.  0\.  0\.  0.]
 [ 0\.  1\.  0\.  0.]]

1
```

## 进一步阅读

本节列出了一些可供进一步阅读的资源。

*   [什么是热门编码，什么时候用于数据科学？ Quora上的](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
*   [OneHotEncoder scikit-learn API文档](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
*   [LabelEncoder scikit-learn API文档](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
*   [to_categorical Keras API文档](https://keras.io/utils/#to_categorical)
*   [Python中使用XGBoost进行梯度提升的数据准备](http://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/)
*   [Keras深度学习库的多分类教程](http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)

## 摘要

在本教程中，您了解了如何使用Python中的单热编码对分类序列数据进行编码以进行深度学习。

具体来说，你学到了：

*   什么是整数编码和单热编码，以及为什么它们在机器学习中是必需的。
*   如何在Python中手动计算整数编码和单热编码。
*   如何使用scikit-learn和Keras库在Python中自编码序列数据。

您对准备序列数据有任何疑问吗？
在评论中提出您的问题，我会尽力回答。