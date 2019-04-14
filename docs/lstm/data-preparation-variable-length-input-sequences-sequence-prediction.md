# 可变长度输入序列的数据准备

> 原文： [https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/](https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)

深度学习库假设您的数据的向量化表示。

在可变长度序列预测问题的情况下，这需要转换您的数据，使得每个序列具有相同的长度。

此向量化允许代码有效地为您选择的深度学习算法批量执行矩阵运算。

在本教程中，您将发现可用于使用Keras在Python中为序列预测问题准备可变长度序列数据的技术。

完成本教程后，您将了解：

*   如何填充具有虚拟值的可变长度序列。
*   如何将可变长度序列填充到新的更长的期望长度。
*   如何将可变长度序列截断为更短的期望长度。

让我们开始吧。

![Data Preparation for Variable Length Input Sequences for Sequence Prediction](img/25631fd0ade9a217cc6ef99e6006f7f4.jpg)

用于序列预测的可变长度输入序列的数据准备
照片由 [Adam Bautz](https://www.flickr.com/photos/130811041@N04/19547744848/) ，保留一些权利。

## 概观

本节分为3部分;他们是：

1.  受控序列问题
2.  序列填充
3.  序列截断

### 环境

本教程假定您已安装Python SciPy环境。您可以在此示例中使用Python 2或3。

本教程假设您使用TensorFlow（v1.1.0 +）或Theano（v0.9 +）后端安装了Keras（v2.0.4 +）。

本教程还假设您安装了scikit-learn，Pandas，NumPy和Matplotlib。

如果您在设置Python环境时需要帮助，请参阅以下帖子：

*   [如何使用Anaconda设置用于机器学习和深度学习的Python环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 受控序列问题

为了本教程的目的，我们可以设计一个简单的序列问题。

该问题被定义为整数序列。有三个序列，长度在4到1个步骤之间，如下所示：

```py
1, 2, 3, 4
1, 2, 3
1
```

这些可以在Python中定义为列表列表，如下所示（可读性的间距）：

```py
sequences = [
[1, 2, 3, 4],
[1, 2, 3],
[1]
]
```

我们将使用这些序列作为本教程中探索序列填充的基础。

## 序列填充

Keras深度学习库中的 [pad_sequences（）函数](https://keras.io/preprocessing/sequence/)可用于填充可变长度序列。

默认填充值为0.0，适用于大多数应用程序，但可以通过“value”参数指定首选值来更改。例如：

```py
pad_sequences(..., value=99)
```

要应用于序列的开头或结尾的填充（称为前序列或后序列填充）可以通过“填充”参数指定，如下所示。

### 预序列填充

预序列填充是默认值（padding ='pre'）

下面的示例演示了具有0值的预填充3输入序列。

```py
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences)
print(padded)
```

运行该示例将打印3个预先填充零序列的序列。

```py
[[1 2 3 4]
[0 1 2 3]
[0 0 0 1]
```

### 序列后填充

填充也可以应用于序列的末尾，这可能更适合于某些问题域。

可以通过将“padding”参数设置为“post”来指定序列后填充。

```py
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences, padding='post')
print(padded)
```

运行该示例将打印相同的序列，并附加零值。

```py
[[1 2 3 4]
[1 2 3 0]
[1 0 0 0]]
```

### 填充序列到长度

pad_sequences（）函数还可用于将序列填充到可能比任何观察到的序列更长的优选长度。

这可以通过将“maxlen”参数指定为所需长度来完成。然后将对所有序列执行填充以获得所需的长度，如下所述。

```py
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences, maxlen=5)
print(padded)
```

运行示例将每个序列填充到所需的5个步骤的长度，即使观察序列的最大长度仅为4个步骤。

```py
[[0 1 2 3 4]
[0 0 1 2 3]
[0 0 0 0 1]]
```

## 序列截断

序列的长度也可以修剪到所需的长度。

可以使用“maxlen”参数将所需的序列长度指定为多个时间步长。

有两种方法可以截断序列：通过从序列的开头或末尾删除时间步长。

### 序列前截断

默认的截断方法是从序列的开头删除时间步。这称为序列前截断。

下面的示例将序列截断为所需的长度2。

```py
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# truncate sequence
truncated= pad_sequences(sequences, maxlen=2)
print(truncated)
```

运行该示例将从第一个序列中删除前两个时间步，从第二个序列中删除第一个时间步，并填充最终序列。

```py
[[3 4]
[2 3]
[0 1]]
```

### 序列后截断

也可以通过从序列末尾删除时间步长来修剪序列。

对于某些问题域，这种方法可能更合适。

可以通过将“截断”参数从默认的“pre”更改为“post”来配置序列后截断，如下所示：

```py
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# truncate sequence
truncated= pad_sequences(sequences, maxlen=2, truncating='post')
print(truncated)
```

运行该示例将从第一个序列中删除最后两个时间步，从第二个序列中删除最后一个时间步，然后再次填充最终序列。

```py
[[1 2]
[1 2]
[0 1]]
```

## 摘要

在本教程中，您了解了如何准备可变长度序列数据以用于Python中的序列预测问题。

具体来说，你学到了：

*   如何填充具有虚拟值的可变长度序列。
*   如何将可变长度序列填充到新的所需长度。
*   如何将可变长度序列截断为新的所需长度。

您对准备可变长度序列有任何疑问吗？
在评论中提出您的问题，我会尽力回答。