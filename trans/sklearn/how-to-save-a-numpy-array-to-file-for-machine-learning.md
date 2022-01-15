# 如何将 NumPy 数组保存到文件中进行机器学习

> 原文:[https://machinelearning master . com/如何保存数字阵列到文件的机器学习/](https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/)

最后更新于 2020 年 8 月 19 日

用 Python 开发机器学习模型经常需要使用 [NumPy 数组](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)。

NumPy 数组是 Python 中处理数据的高效数据结构，像 scikit-learn 库中的机器学习模型，以及像 Keras 库中的深度学习模型，以 NumPy 数组的格式期待输入数据，以 NumPy 数组的格式进行预测。

因此，通常需要将 NumPy 数组保存到文件中。

例如，您可能会使用缩放等转换来准备数据，并需要将其保存到文件中以备后用。您也可以使用模型进行预测，并需要将预测保存到文件中以备后用。

在本教程中，您将了解如何将 NumPy 数组保存到文件中。

完成本教程后，您将知道:

*   如何将 NumPy 数组保存为 CSV 格式的文件。
*   如何将 NumPy 数组保存到 NPY 格式的文件中。
*   如何将 NumPy 数组保存到压缩的 NPZ 格式文件中。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Save a NumPy Array to File for Machine Learning](img/e4a0bba6e9a2e91b8a2f8207803743bb.png)

如何将 NumPy 数组保存到文件中用于机器学习
图片由[克里斯·科姆](https://www.flickr.com/photos/cosmicherb70/15052283220/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  将 NumPy 数组保存到。CSV 文件(ASCII)
2.  将 NumPy 数组保存到。NPY 文件(二进制)
3.  将 NumPy 数组保存到。NPZ 文件(压缩)

## 1.将 NumPy 数组保存到。CSV 文件(ASCII)

在文件中存储数字数据最常见的文件格式是逗号分隔变量格式，简称 CSV。

您的训练数据和模型的输入数据很可能存储在 CSV 文件中。

将数据保存到 CSV 文件中会很方便，例如模型中的预测。

您可以使用 [savetxt()功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html)将 NumPy 数组保存为 CSV 文件。该函数以文件名和数组作为参数，并将数组保存为 CSV 格式。

您还必须指定分隔符；这是用于分隔文件中每个变量的字符，最常见的是逗号。这可以通过“*分隔符*参数来设置。

### 1.1 将 NumPy 数组保存到 CSV 文件的示例

下面的示例演示了如何将单个 NumPy 数组保存为 CSV 格式。

```py
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('data.csv', data, delimiter=',')
```

运行该示例将定义一个 NumPy 数组，并将其保存到文件“ *data.csv* ”中。

该数组有一行 10 列的数据。我们希望这些数据作为单行数据保存到 CSV 文件中。

运行示例后，我们可以检查“ *data.csv* 的内容。

我们应该看到以下内容:

```py
0.000000000000000000e+00,1.000000000000000000e+00,2.000000000000000000e+00,3.000000000000000000e+00,4.000000000000000000e+00,5.000000000000000000e+00,6.000000000000000000e+00,7.000000000000000000e+00,8.000000000000000000e+00,9.000000000000000000e+00
```

我们可以看到，数据被正确地保存为一行，数组中的浮点数以全精度保存。

### 1.2 从 CSV 文件加载 NumPy 数组的示例

我们可以稍后使用 [loadtext()函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html)将这些数据加载为 NumPy 数组，并指定文件名和相同的逗号分隔符。

下面列出了完整的示例。

```py
# load numpy array from csv file
from numpy import loadtxt
# load array
data = loadtxt('data.csv', delimiter=',')
# print the array
print(data)
```

运行该示例从 CSV 文件加载数据并打印内容，将我们的单行与前面示例中定义的 10 列进行匹配。

```py
[0\. 1\. 2\. 3\. 4\. 5\. 6\. 7\. 8\. 9.]
```

## 2.将 NumPy 数组保存到。NPY 文件(二进制)

有时，我们希望在 NumPy 数组中有效保存大量数据，但我们只需要在另一个 Python 程序中使用这些数据。

因此，我们可以将 NumPy 数组保存为本地二进制格式，这对于保存和加载都是有效的。

这对于已经准备好的输入数据来说是常见的，例如转换后的数据，这些数据将需要用作将来测试一系列机器学习模型或运行许多实验的基础。

那个。npy 文件格式适合这个用例，简称为“ [NumPy 格式](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html#module-numpy.lib.format)”。

这可以通过使用[保存()NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html)并指定要保存的文件名和数组来实现。

### 2.1 将 NumPy 数组保存到 NPY 文件的示例

下面的例子定义了我们的二维 NumPy 数组，并将其保存到一个. npy 文件中。

```py
# save numpy array as npy file
from numpy import asarray
from numpy import save
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
save('data.npy', data)
```

运行该示例后，您将在目录中看到一个名为“ *data.npy* ”的新文件。

您不能用文本编辑器直接检查此文件的内容，因为它是二进制格式。

### 2.2 从 NPY 文件加载 NumPy 数组的示例

您可以稍后使用 [load()函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)将此文件加载为 NumPy 数组。

下面列出了完整的示例。

```py
# load numpy array from npy file
from numpy import load
# load array
data = load('data.npy')
# print the array
print(data)
```

运行该示例将加载文件并打印内容，确认它被正确加载，并且内容与我们期望的相同二维格式相匹配。

```py
[[0 1 2 3 4 5 6 7 8 9]]
```

## 3.将 NumPy 数组保存到。NPZ 文件(压缩)

有时，我们为建模准备数据，这些数据需要在多个实验中重用，但是数据很大。

这可能是预处理的 NumPy 数组，如文本(整数)的语料库或重新缩放的图像数据(像素)的集合。在这种情况下，既希望将数据保存到文件中，也希望以压缩格式保存。

这使得千兆字节的数据减少到数百兆字节，并允许轻松传输到云计算的其他服务器进行长时间的算法运行。

那个。npz 文件格式适合这种情况，并且支持原生 NumPy 文件格式的压缩版本。

[savez_compressed() NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html)允许将多个 NumPy 数组保存为一个压缩数组。npz 文件。

### 3.1 将 NumPy 阵列保存到 NPZ 文件的示例

我们可以使用这个函数将单个 NumPy 数组保存到一个压缩文件中。

下面列出了完整的示例。

```py
# save numpy array as npz file
from numpy import asarray
from numpy import savez_compressed
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
savez_compressed('data.npz', data)
```

运行该示例定义数组，并将其保存到名为“data.npz”的压缩 numpy 格式的文件中。

就像。npy 格式，我们不能用文本编辑器检查保存文件的内容，因为文件格式是二进制的。

### 3.2 从 NPZ 文件加载 NumPy 数组的示例

我们可以稍后使用上一节中相同的 [load()函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)加载这个文件。

在这种情况下，savez_compressed()函数支持将多个数组保存到一个文件中。因此，load()函数可能会加载多个数组。

加载的数组从 dict 中的 load()函数返回，第一个数组的名称为“arr_0”，第二个数组的名称为“arr_1”，依此类推。

下面列出了加载单个阵列的完整示例。

```py
# load numpy array from npz file
from numpy import load
# load dict of arrays
dict_data = load('data.npz')
# extract the first array
data = dict_data['arr_0']
# print the array
print(data)
```

运行该示例加载包含数组字典的压缩 numpy 文件，然后提取我们保存的第一个数组(我们只保存了一个)，然后打印内容，确认数组的值和形状与我们最初保存的相匹配。

```py
[[0 1 2 3 4 5 6 7 8 9]]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 邮件

*   [如何在 Python 中加载机器学习数据](https://machinelearningmastery.com/load-machine-learning-data-python/)
*   [Python 中 NumPy 数组的温和介绍](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)
*   [如何为机器学习对 NumPy 数组进行索引、切片和整形](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)

### 蜜蜂

*   num py . save txt API
*   num py . save API
*   num py . saber API
*   num py . savez _ compressed API
*   num py . load API
*   num py . load txt API

## 摘要

在本教程中，您发现了如何将 NumPy 数组保存到文件中。

具体来说，您了解到:

*   如何将 NumPy 数组保存为 CSV 格式的文件。
*   如何将 NumPy 数组保存到 NPY 格式的文件中。
*   如何将 NumPy 数组保存到压缩的 NPZ 格式文件中。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。