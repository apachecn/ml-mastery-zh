# 处理机器学习的大数据文件的 7 种方法

> 原文： [https://machinelearningmastery.com/large-data-files-machine-learning/](https://machinelearningmastery.com/large-data-files-machine-learning/)

将机器学习算法探索并应用于太大而无法放入内存的数据集非常常见。

这导致了以下问题：

*   如何加载我的多千兆字节数据文件？
*   我尝试运行数据集时算法崩溃;我该怎么办？
*   你能帮我解决内存不足错误吗？

在这篇文章中，我想提供一些您可能想要考虑的常见建议。

![7 Ways to Handle Large Data Files for Machine Learning](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/05/7-Ways-to-Handle-Large-Data-Files-for-Machine-Learning.jpg)

处理机器学习的大数据文件的 7 种方法
[Gareth Thompson](https://www.flickr.com/photos/evo_gt/12267202894/) 的照片，保留一些权利。

## 1.分配更多记忆

某些机器学习工具或库可能受默认内存配置的限制。

检查是否可以重新配置工具或库以分配更多内存。

一个很好的例子是 Weka，你可以[在启动应用程序时将内存作为参数](https://weka.wikispaces.com/OutOfMemoryException)增加。

## 2.使用较小的样本

您确定需要处理所有数据吗？

随机抽取一些数据，例如前 1000 行或 100,000 行。在为所有数据拟合最终模型之前，使用此较小的示例来解决您的问题（使用渐进式数据加载技术）。

我认为这对于机器学习来说是一种很好的做法，可以让您快速抽查算法和结果。

您还可以考虑对用于拟合一种算法的数据量进行灵敏度分析，与模型技能进行比较。也许有一个自然的收益递减点，您可以将其用作较小样本的启发式大小。

## 3.使用具有更多内存的计算机

你必须在你的电脑上工作吗？

也许你可以访问更大的计算机，内存更多。

例如，一个好的选择是在像 Amazon Web Services 这样的云服务上租用计算时间，该服务为机器提供数十 GB 的 RAM，每小时不到一美元。

我发现这种方法在过去非常有用。

看帖子：

*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 4.更改数据格式

您的数据是否以原始 ASCII 文本存储，如 CSV 文件？

也许您可以通过使用其他数据格式来加速数据加载并减少使用内存。一个很好的例子是像 GRIB，NetCDF 或 HDF 这样的二进制格式。

有许多命令行工具可用于将一种数据格式转换为另一种不需要将整个数据集加载到内存中的数据格式。

使用其他格式可以允许您以更紧凑的形式存储数据，以节省内存，例如 2 字节整数或 4 字节浮点数。

## 5.流数据或使用渐进式加载

是否所有数据都需要同时存在于内存中？

也许您可以使用代码或库来根据需要将数据流或逐步加载到内存中进行训练。

这可能需要可以使用诸如随机梯度下降之类的优化技术迭代地学习的算法，而不是需要存储器中的所有数据来执行矩阵运算的算法，诸如线性和逻辑回归的一些实现。

例如，Keras 深度学习库提供此功能以逐步加载图像文件，称为 [flow_from_directory](https://keras.io/preprocessing/image/) 。

另一个例子是 Pandas 库，它可以[以块](http://pandas.pydata.org/pandas-docs/stable/io.html#iterating-through-files-chunk-by-chunk)加载大型 CSV 文件。

## 6.使用关系数据库

关系数据库提供了存储和访问非常大的数据集的标准方法。

在内部，存储在磁盘上的数据可以批量逐步加载，并且可以使用标准查询语言（SQL）进行查询。

可以使用免费的开源数据库工具，如 [MySQL](https://www.mysql.com/) 或 [Postgres](https://www.postgresql.org/) ，大多数（全部？）编程语言和许多机器学习工具可以直接连接到关系数据库。您也可以使用轻量级方法，例如 [SQLite](https://www.sqlite.org/) 。

我发现这种方法在过去对非常大的表格数据集非常有效。

同样，您可能需要使用可以处理迭代学习的算法。

## 7.使用大数据平台

在某些情况下，您可能需要求助于大数据平台。

也就是说，一个专为处理非常大的数据集而设计的平台，允许您在其上使用数据转换和机器学习算法。

两个很好的例子是带有 [Mahout](http://mahout.apache.org/) 机器学习库的 Hadoop 和带有 [MLLib](http://spark.apache.org/mllib/) 库的 Spark。

我相信，如果您已经用尽上述选项，这是最后的选择，只是因为这会给您的机器学习项目带来额外的硬件和软件复杂性。

然而，存在数据非常大且以前的选项不会削减数据的问题。

## 摘要

在这篇文章中，您发现了一些在处理机器学习的大型数据文件时可以使用的策略。

您是否知道或尝试过其他方法？
在下面的评论中分享。

你尝试过这些方法吗？
请在评论中告诉我。