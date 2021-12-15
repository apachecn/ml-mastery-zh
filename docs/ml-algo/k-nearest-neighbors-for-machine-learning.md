# K-Nearest Neighbors for Machine Learning

> 原文： [https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)

在这篇文章中，您将发现用于分类和回归的k-Nearest Neighbors（KNN）算法。阅读这篇文章后你会知道的。

*   KNN使用的模型表示。
*   如何使用KNN学习模型（暗示，不是）。
*   如何使用KNN做出预测
*   KNN的许多名称包括不同的字段如何引用它。
*   如何准备您的数据以充分利用KNN。
*   在哪里可以了解有关KNN算法的更多信息。

这篇文章是为开发人员编写的，并没有统计或数学方面的背景。重点是算法如何工作以及如何将其用于预测建模问题。如果您有任何疑问，请发表评论，我会尽力回答。

让我们开始吧。

![K-Nearest Neighbors for Machine Learning](img/59dba4b611cf5b92ccc9d47703a27bb8.jpg)

K-Nearest Neighbors for Machine Learning
照片由 [Valentin Ottone](https://www.flickr.com/photos/saneboy/3050001226/) 保留，保留一些权利。

## KNN模型表示

KNN的模型表示是整个训练数据集。

它是如此简单。

除了存储整个数据集之外，KNN没有其他模型，因此无需学习。

有效的实现可以使用诸如 [k-d树](https://en.wikipedia.org/wiki/K-d_tree)之类的复杂数据结构来存储数据，以在预测期间有效地查找和匹配新模式。

由于存储了整个训练数据集，因此您可能需要仔细考虑训练数据的一致性。策划它可能是一个好主意，在新数据可用时经常更新并删除错误和异常数据。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 用KNN做出预测

KNN直接使用训练数据集做出预测。

通过搜索K个最相似的实例（邻居）的整个训练集并总结那些K个实例的输出变量，对新实例（x）做出预测。对于回归，这可能是平均输出变量，在分类中，这可能是模式（或最常见）类值。

为了确定训练数据集中的哪个K实例与新输入最相似，使用距离度量。对于实值输入变量，最常用的距离测量是[欧几里德距离](https://en.wikipedia.org/wiki/Euclidean_distance)。

欧几里德距离被计算为跨所有输入属性j的新点（x）和现有点（xi）之间的平方差之和的平方根。

EuclideanDistance（x，xi）= sqrt（sum（（xj - xij）^ 2））

其他流行的距离措施包括：

*   **汉明距离**：计算二进制向量之间的距离（[更多](https://en.wikipedia.org/wiki/Hamming_distance)）。
*   **曼哈顿距离**：使用它们的绝对差值之和计算实际向量之间的距离。也称为城市街区距离（[更多](https://en.wikipedia.org/wiki/Taxicab_geometry)）。
*   **Minkowski距离**：欧几里德和曼哈顿距离的推广（[更多](https://en.wikipedia.org/wiki/Minkowski_distance)）。

可以使用许多其他距离测量，例如Tanimoto， [Jaccard](https://en.wikipedia.org/wiki/Jaccard_index) ， [Mahalanobis](https://en.wikipedia.org/wiki/Mahalanobis_distance) 和[余弦距离](https://en.wikipedia.org/wiki/Cosine_similarity)。您可以根据数据属性选择最佳距离指标。如果您不确定，可以尝试不同的距离指标和不同的K值，并查看哪种混合产生最准确的模型。

如果输入变量在类型上相似（例如，所有测量的宽度和高度），则欧几里德是一种很好的距离测量。如果输入变量在类型上不相似（例如年龄，性别，身高等），曼哈顿距离是一个很好的衡量标准。

可以通过算法调整找到K的值。尝试K的许多不同值（例如1到21的值）并查看哪种值最适合您的问题是一个好主意。

KNN的计算复杂度随着训练数据集的大小而增加。对于非常大的训练集，KNN可以通过从训练数据集中取样来制作随机，从中计算K-最相似的实例。

KNN已经存在了很长时间，并且已经得到很好的研究。因此，不同的学科有不同的名称，例如：

*   **基于实例的学习**：原始训练实例用于做出预测。因此，KNN通常被称为[基于实例的学习](https://en.wikipedia.org/wiki/Instance-based_learning)或基于案例的学习（其中每个训练实例是来自问题域的案例）。
*   **懒惰学习**：不需要学习模型，所有工作都在请求预测时进行。因此，KNN通常被称为[懒惰学习](https://en.wikipedia.org/wiki/Lazy_learning)算法。
*   **非参数**：KNN对正在解决的问题的功能形式没有做出任何假设。因此，KNN被称为[非参数](https://en.wikipedia.org/wiki/Nonparametric_statistics)机器学习算法。

KNN可用于回归和分类问题。

### KNN for Regression

当KNN用于回归问题时，预测基于K-最相似实例的均值或中值。

### KNN for Classification

当KNN用于分类时，输出可以被计算为具有来自K-最相似实例的最高频率的类。每个实例本质上都为他们的班级投票，而得票最多的班级则作为预测。

类概率可以被计算为属于新数据实例的K个最相似实例的集合中的每个类的样本的归一化频率。例如，在二分类问题（类为0或1）中：

p（class = 0）= count（class = 0）/（count（class = 0）+ count（class = 1））

如果您使用K并且您具有偶数个类（例如2个），则最好选择具有奇数的K值以避免出现平局。反之，当你有一个奇数的类时，使用偶数来表示K.

通过将K扩展1并查看训练数据集中下一个最相似实例的类，可以一致地打破关系。

## 维度的诅咒

KNN适用于少量输入变量（p），但在输入数量非常大时会遇到困难。

每个输入变量可以被认为是p维输入空间的维度。例如，如果您有两个输入变量x1和x2，则输入空间将为2维。

随着维数的增加，输入空间的体积以指数速率增加。

在高维度中，可能相似的点可能具有非常大的距离。所有的点都会相互远离，我们对简单的2维和3维空间距离的直觉就会崩溃。这可能一开始感觉不直观，但这个一般性问题被称为“[维度诅咒](https://en.wikipedia.org/wiki/Curse_of_dimensionality)”。

## 为KNN准备最佳数据

*   **重新缩放数据**：如果所有数据具有相同的比例，KNN的表现要好得多。将数据规范化到[0,1]范围是个好主意。如果数据具有高斯分布，则标准化数据也可能是个好主意。
*   **地址缺失数据**：缺少数据意味着无法计算样本之间的距离。可以排除这些样本，也可以估算缺失值。
*   **低维度**：KNN适用于低维数据。您可以在高维数据（数百或数千个输入变量）上尝试它，但要注意它可能不如其他技术那样好。 KNN可以从减少输入特征空间维度的特征选择中受益。

## 进一步阅读

如果您有兴趣从头开始在Python中实现KNN，请查看帖子：

*   [教程从头开始在Python中实现k-最近邻](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

以下是从预测建模角度介绍KNN算法的一些优秀的机器学习文本。

1.  [应用预测建模](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20)，第7章用于回归，第13章用于分类。
2.  [数据挖掘：实用机器学习工具和技术](http://www.amazon.com/dp/0123748569?tag=inspiredalgor-20)，第76和128页
3.  [做数据科学：从前线直接谈话](http://www.amazon.com/dp/1449358659?tag=inspiredalgor-20)，第71页
4.  [机器学习](http://www.amazon.com/dp/0070428077?tag=inspiredalgor-20)，第8章

还可以在维基百科上查看 [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) 。

## 摘要

在这篇文章中，您发现了KNN机器学习算法。你了解到：

*   KNN存储它用作其表示的整个训练数据集。
*   KNN没有学习任何模型。
*   KNN通过计算输入样本和每个训练实例之间的相似性来及时做出预测。
*   有许多距离度量可供选择以匹配输入数据的结构。
*   在使用KNN时，重新调整数据是一个好主意，例如使用规范化。

如果您对此帖子或KNN算法有任何疑问，请在评论中提出，我会尽力回答。