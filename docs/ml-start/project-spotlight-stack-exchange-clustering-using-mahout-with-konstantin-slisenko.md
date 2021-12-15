# 项目焦点：使用 Mahout 和 Konstantin Slisenko 进行栈交换群集

> 原文： [https://machinelearningmastery.com/project-spotlight-stack-exchange-clustering-using-mahout-with-konstantin-slisenko/](https://machinelearningmastery.com/project-spotlight-stack-exchange-clustering-using-mahout-with-konstantin-slisenko/)

这是一个项目焦点，Konstantin Slisenko 是程序员和机器学习爱好者。

## 你能介绍一下自己吗？

我的名字是 Konstantin Slisenko，我来自[白俄罗斯](http://en.wikipedia.org/wiki/Belarus)。我毕业于[白俄罗斯国立信息学和无线电电子学](http://www.bsuir.by/index.jsp?lang=en)。我目前正在修读硕士课程。

[![Konstantin Slisenko](img/f56c3ffce7e1d763cbc059c83b5c3365.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/konstantin-slisenko.png)

Konstantin Slisenko

我是一名 Java 开发人员，在 JazzTeam 公司工作。我喜欢学习新技术。我目前对大数据和机器学习很感兴趣。我喜欢参加会议，结识新的有趣的人。我也喜欢旅行和骑自行车。

## 你的项目叫什么，它做了什么？

我的项目是 [stackoverflow.com](http://stackoverflow.com/) 网站的数据集群。

目标是对 stackoverflow 问题和答案进行分组。分组后，您可以看到栈溢出数据的常见图片以及问题之间的关系。如果您想进行市场调查或撰写有关特定问题的文章（或活动手册），这可能会有所帮助。

[![Stackexchange clustering using Mahout Tags](img/82e009380c84081b07654cb0038a03a1.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/Stackexchange-clustering-using-Mahout-tags.png)

使用 Mahout 标记进行 Stackexchange 聚类

我有改进的想法，例如标记“热门”主题，考虑用户评级等，以便将更多数据添加到公共图片中。我也在考虑训练分类器。当我们获得更新数据并希望将此更新放入系统时，这可能会有所帮助。

## 你是怎么开始的？

首先，我对 [Apache Hadoop](http://hadoop.apache.org/) 产生了兴趣。在我制作了一些 Hadoop 程序之后，我开始研究它的基础架构并了解 [Apache Mahout](https://mahout.apache.org/) 。

我开始深入研究并应用一些示例：准备数据，运行算法，查看输出。有一天，我发现了 [Frank Scholten](https://github.com/frankscholten) 关于 stackoverflow 聚类的资料。你可以[观看他](http://vimeo.com/43903965)的有趣演示。 [Mahout in Action](http://www.amazon.com/dp/1935182684?tag=inspiredalgor-20) 也提到了这个话题。

我现在使用 Frank 的代码作为基础并应用我自己的改进和调整。数据处理包括以下步骤：

1.  Stackexchange 源数据采用 XML 格式。 Hadoop 作业用于提取文本。
2.  然后我使用自定义 Lucene 分析器处理文本数据：删除停用词，应用 Porter Steamer 等。
3.  然后我使用 TF-IDF Mahout 实用程序对文本进行向量化。
4.  对于聚类，我现在使用 Mahout 的 K-Means 算法，但我想在将来尝试其他算法。
5.  在此之后，我将结果存储在面向图形的数据库 Neo4j 中，并使用 HTML 和 JavaScript 对它们进行可视化。

所有可视化都可以在这里找到：[使用 Mahout](http://clustering.slisenko.net:8080/stackexchange-web) 进行 Stackexchange 聚类。

## 你做了哪些有趣的发现？

群集质量取决于您执行数据准备的方式。在此步骤中，您必须非常注意应删除的停用词。

[![Stack Exchange Clustering using Mahout by Konstantin Slisenko](img/bea95becffcd97609771221e9ba400ef.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/Stack-Exchange-Clustering-using-Mahout.png)

使用 Konstantin Slisenko 的 Mahout 栈交换群集

[K-Means 聚类](http://en.wikipedia.org/wiki/K-means_clustering)算法要求您设置聚类 K 的初始数量。我想动态地进行 K 计算。出于这个原因，我打算找到另一种算法。

## 你想在项目上做什么？

*   使用发布日期来确定现在“热门”的主题。
*   尝试其他一些聚类算法，并动态计算簇数。
*   基于集群数据构建分类器。
*   应用更多不同的可视化。
*   应用群集评估来说明哪些群集“好”哪些群集“坏”。
*   对群集数据应用一些索引搜索。
*   我正在考虑 Apache Mahout 贡献 - 提供可视化集群数据的实用程序。

## 学到更多

*   项目：[使用 Mahout 进行 Stackexchange 聚类](http://clustering.slisenko.net:8080/stackexchange-web)
*   [GitHub 上的项目源代码](https://github.com/kslisenko/big-data-research/tree/master/Developments/stackexchange-analyses)
*   [康斯坦丁在 Google+](https://plus.google.com/104628548674452019199) 上分享了机器学习和大数据资源的有趣链接
*   [康斯坦丁的博客](http://www.slisenko.net/)

感谢康斯坦丁。

**你有机器学习方面的项目吗？**

如果你有一个有趣的机器学习方项目并且有兴趣像康斯坦丁一样被描述，请[联系我](http://machinelearningmastery.com/contact/ "Contact")。