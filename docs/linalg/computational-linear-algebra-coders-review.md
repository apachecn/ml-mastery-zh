# 面向程序员的计算线性代数回顾

> 原文： [https://machinelearningmastery.com/computational-linear-algebra-coders-review/](https://machinelearningmastery.com/computational-linear-algebra-coders-review/)

数值线性代数关注在具有实际数据的计算机中实现和执行矩阵运算的实际意义。

这是一个需要先前线性代数经验的领域，并且重点关注操作的表现和精度。 fast.ai 公司发布了一个名为“_ 计算线性代数 _”的免费课程，主题是数字线性代数，包括旧金山大学录制的 Python 笔记本和视频讲座。

在这篇文章中，您将发现计算线性代数的 fast.ai 免费课程。

阅读这篇文章后，你会知道：

*   课程的动机和先决条件。
*   课程中涵盖的主题概述。
*   这门课程究竟适合谁，不适合谁。

让我们开始吧。

![Computational Linear Algebra for Coders Review](img/024fbf58becb9d25db8ac77b15092b68.jpg)

用于编码器审查的计算线性代数
照片由 [Ruocaled](https://www.flickr.com/photos/ruocaled/6330547994/) ，保留一些权利。

## 课程大纲

课程“_ 编码器计算线性代数 _”是由 fast.ai 提供的免费在线课程。他们是一家致力于提供与深度学习相关的免费教育资源的公司。

该课程最初由旧金山大学的 [Rachel Thomas](https://www.linkedin.com/in/rachel-thomas-942a7923/) 于 2017 年授课，作为硕士学位课程的一部分。 Rachel Thomas 是旧金山大学的教授，也是 [fast.ai](http://www.fast.ai/) 的联合创始人，拥有博士学位。在数学方面。

该课程的重点是线性代数的数值方法。这是矩阵代数在计算机上的应用，并解决了实现和使用方法（如表现和精度）的所有问题。

> 本课程的重点是：我们如何以可接受的速度和可接受的准确度进行矩阵计算？

本课程使用 Python 作为使用 NumPy，scikit-learn，numba，pytorch 等的示例。

这些材料使用自上而下的方法进行教学，就像 [MachineLearningMastery](https://machinelearningmastery.com/machine-learning-for-programmers/) 一样，旨在让人们了解如何做事，然后再解释这些方法的工作原理。

> 了解如何实现这些算法将使您能够更好地组合和利用它们，并使您可以根据需要自定义它们。

## 课程先决条件和参考

该课程确实假设熟悉线性代数。

这包括诸如向量，矩阵，诸如矩阵乘法和变换之类的操作之类的主题。

该课程不适用于线性代数领域的新手。

如果您是新的或生锈的线性代数，建议您在参加课程之前查看三个参考文献。他们是：

*   [3Blue 1Brown 线性代数本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)，视频课程
*   [沉浸式线性代数](http://immersivemath.com/ila/)，交互式教科书
*   [深度学习第 2 章](http://www.deeplearningbook.org/contents/linear_algebra.html)，2016 年。

此外，在完成课程的过程中，根据需要提供参考。

预先提出了两个一般参考文本。它们是以下教科书：

*   [数值线性代数](http://amzn.to/2CNOgZp)，1997。
*   [数值方法](http://amzn.to/2CNfSxE)，2012。

## 课程比赛

本节概述了课程的 8（9）部分。他们是：

*   0.课程物流
*   我们为什么来这里？
*   2.使用 NMF 和 SVD 进行主题建模
*   3.使用强大的 PCA 去除背景
*   4.具有鲁棒回归的压缩感知
*   5.使用线性回归预测健康结果
*   6.如何实现线性回归
*   7.具有特征分解的 PageRank
*   8.实现 QR 分解

实际上，课程只有 8 个部分，因为第一部分是参加旧金山大学课程的学生的管理细节。

## 讲座细分

在本节中，我们将逐步介绍课程的 9 个部分，并总结其内容和主题，让您对所期待的内容有所了解，并了解它是否适​​合您。

### 第 0 部分。课程后勤

第一堂课不是课程的一部分。

它介绍了讲师，材料，教学方式以及学生对硕士课程的期望。

> 我将使用自上而下的教学方法，这与大多数数学课程的运作方式不同。通常，在自下而上的方法中，您首先要学习将要使用的所有单独组件，然后逐渐将它们构建为更复杂的结构。这方面的问题是学生经常失去动力，没有“大局”感，也不知道他们需要什么。

本讲座涉及的主题是：

*   讲师背景
*   教学方法
*   技术写作的重要性
*   优秀技术博客列表
*   线性代数评论资源

视频和笔记本：

*   [计算线性代数 1：矩阵数学，精度，记忆，速度和＆amp;并行化](https://www.youtube.com/watch?v=8iGzBMboA0I&index=1&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb)

### 第 1 部分。为什么我们在这里？

本部分介绍了本课程的动机，并介绍了矩阵分解的重要性：这些计算的表现和准确率以及一些示例应用程序的重要性。

> 矩阵无处不在，任何可以放在 Excel 电子表格中的东西都是矩阵，语言和图片也可以表示为矩阵。

本讲中提出的一个重点是，如何将整类矩阵分解方法和一种特定方法（QR 分解）报告为 20 世纪[十大最重要算法之一](http://www.cs.fsu.edu/~lacher/courses/COT4401/notes/cise_v2_i1/index.html)。

> 20 世纪十大科学与工程算法列表包括：线性代数的矩阵分解方法。它还包括 QR 算法

The topics covered in this lecture are:

*   矩阵和张量积
*   矩阵分解
*   准确率
*   内存使用
*   速度
*   并行化＆amp;向量

Videos and Notebook:

*   [计算线性代数 1：矩阵数学，精度，记忆，速度和＆amp;并行化](https://www.youtube.com/watch?v=8iGzBMboA0I&index=1&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb)

### 第 2 部分。使用 NMF 和 SVD 进行主题建模

本部分重点介绍矩阵分解在文本主题建模应用中的应用，特别是奇异值分解方法或 SVD。

在这一部分中有用的是从零开始或与 NumPy 和 scikit-learn 库计算方法的比较。

> 主题建模是开始使用矩阵分解的好方法。

The topics covered in this lecture are:

*   主题频率 - 逆文档频率（TF-IDF）
*   奇异值分解（SVD）
*   非负矩阵分解（NMF）
*   随机梯度下降（SGD）
*   PyTorch 介绍
*   截断 SVD

Videos and Notebook:

*   [计算线性代数 2：主题建模与 SVD＆amp; NMF](https://www.youtube.com/watch?v=kgd40iDT8yY&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=2)
*   [计算线性代数 3：回顾，关于 NMF 的新观点，＆amp;随机 SVD](https://www.youtube.com/watch?v=C8KEtrWjjyo&index=3&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb)

### 第 3 部分。使用强大的 PCA 去除背景

本部分重点介绍使用特征分解和多变量统计的主成分分析方法（PCA）。

重点是在图像数据上使用 PCA，例如将背景与前景分离以隔离变化。这部分还从零开始介绍 LU 分解。

> 在处理高维数据集时，我们经常利用数据具有低内在维度的事实，以减轻维度和规模的诅咒（可能它位于低维子空间或位于低维流形上）。

The topics covered in this lecture are:

*   加载和查看视频数据
*   SVD
*   主成分分析（PCA）
*   L1 Norm 引起稀疏性
*   强大的 PCA
*   LU 分解
*   LU 的稳定性
*   使用 Pivoting 进行 LU 分解
*   高斯消除的历史
*   块矩阵乘法

Videos and Notebook:

*   [计算线性代数 3：回顾，关于 NMF 的新观点，＆amp;随机 SVD](https://www.youtube.com/watch?v=C8KEtrWjjyo&index=3&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [计算线性代数 4：随机 SVD＆amp;强大的 PCA](https://www.youtube.com/watch?v=Ys8R2nUTOAk&index=4&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [计算线性代数 5：强大的 PCA＆amp; LU 分解](https://www.youtube.com/watch?v=O2x5KPJr5ag&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=5)
*   [笔记本](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb)

### 第 4 部分。具有鲁棒回归的压缩感知

这部分介绍了 NumPy 数组（和其他地方）中使用的广播的重要概念以及在机器学习中出现很多的稀疏矩阵。

该部分的应用重点是使用强大的 PCA 在 CT 扫描中去除背景。

> 术语广播描述了在算术运算期间如何处理具有不同形状的数组。 Numpy 首先使用广播一词，但现在用于其他库，如 Tensorflow 和 Matlab;规则因库而异。

The topics covered in this lecture are:

*   广播
*   稀疏矩阵
*   CT 扫描和压缩感知
*   L1 和 L2 回归

Videos and Notebook:

*   [计算线性代数 6：Block Matrix Mult，Broadcasting，＆amp;稀疏存储](https://www.youtube.com/watch?v=YY9_EYNj5TY&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=6)
*   [计算线性代数 7：CT 扫描的压缩感知](https://www.youtube.com/watch?v=ZUGkvIM6ehM&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=7)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression)

### 第 5 部分。使用线性回归预测健康结果

本部分重点介绍用 scikit-learn 演示的线性回归模型的开发。

Numba 库也用于演示如何加速所涉及的矩阵操作。

> 我们想加快速度。我们将使用 Numba，一个直接将代码编译到 C 的 Python 库。

The topics covered in this lecture are:

*   sklearn 中的线性回归
*   多项式特征
*   加速 Numba
*   正规化与噪声

Videos and Notebook:

*   [计算线性代数 8：Numba，多项式特征，如何实现线性回归](https://www.youtube.com/watch?v=SjX55V8zDXI&index=8&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb)

### 第 6 部分。如何实现线性回归

本部分介绍如何使用一套不同的矩阵分解方法求解线性回归的线性最小二乘法。将结果与 scikit-learn 中的实现进行比较。

> 数值分析师推荐通过 QR 进行线性回归作为多年来的标准方法。它自然，优雅，适合“日常使用”。

The topics covered in this lecture are:

*   Scikit Learn 是如何做到的？
*   朴素的解决方案
*   正规方程和 Cholesky 分解
*   QR 分解
*   SVD
*   时间比较
*   调节＆amp;稳定性
*   完全与减少的因子分解
*   矩阵反转是不稳定的

Videos and Notebook:

*   [计算线性代数 8：Numba，多项式特征，如何实现线性回归](https://www.youtube.com/watch?v=SjX55V8zDXI&index=8&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb)

### 第 7 部分。具有特征分解的 PageRank

本部分介绍了特征分解以及 PageRank 算法在 Wikipedia 链接数据集中的实现和应用。

> QR 算法使用称为 QR 分解的东西。两者都很重要，所以不要让他们感到困惑。

The topics covered in this lecture are:

*   SVD
*   DBpedia 数据集
*   动力法
*   QR 算法
*   寻找特征值的两阶段方法
*   Arnoldi Iteration

Videos and Notebook:

*   [计算线性代数 9：具有特征分解的 PageRank](https://www.youtube.com/watch?v=AbB-w77yxD0&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=9)
*   [计算线性代数 10：QR 算法查找特征值，实现 QR 分解](https://www.youtube.com/watch?v=1kw8bpA9QmQ&index=10&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb)

### 第 8 部分。实现 QR 分解

最后一部分介绍了从零开始实现 QR 分解的三种方法，并比较了每种方法的精度和表现。

> 我们在计算特征值时使用 QR 分解并计算最小二乘回归。它是数值线性代数中的重要组成部分。

The topics covered in this lecture are:

*   格拉姆 - 施密特
*   住户
*   稳定性例子

Videos and Notebook:

*   [计算线性代数 10：QR 算法查找特征值，实现 QR 分解](https://www.youtube.com/watch?v=1kw8bpA9QmQ&index=10&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [笔记本](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/8.%20Implementing%20QR%20Factorization.ipynb)

## 评论课程

我觉得这个课程很棒。

一个有趣的数值线性代数步骤，重点是应用程序和可执行代码。

该课程承诺关注矩阵操作的实际问题，如记忆，速度，精度或数值稳定性。本课程首先仔细研究浮点精度和溢出问题。

在整个过程中，经常在方法之间根据执行速度进行比较。

### 如果......不要参加这个课程

本课程不是开发人员对线性代数的介绍，如果这是预期，你可能会落后。

该课程确实假设了线性代数，符号和操作的基础知识的合理流畅性。它并没有预先隐藏这个假设。

如果您对深度学习感兴趣或者更多地了解深度学习方法中使用的线性代数运算，我认为这门课程不是必需的。

### 参加这门课程，如果......

如果您正在自己的工作中实现矩阵代数方法，并且您希望从中获得更多，我强烈推荐这门课程。

如果您通常对矩阵代数的实际意义感兴趣，我也会推荐这门课程。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 课程

*   [新的 fast.ai 课程：计算线性代数](http://www.fast.ai/2017/07/17/num-lin-alg/)
*   [GitHub 上的计算线性代数](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md)
*   [计算线性代数视频讲座](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)
*   [社区论坛](http://forums.fast.ai/c/lin-alg)

### 参考

*   [3Blue 1Brown 线性代数本质，视频课程](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
*   [沉浸式线性代数，交互式教科书](http://immersivemath.com/ila/)
*   [深度学习第 2 章](http://www.deeplearningbook.org/contents/linear_algebra.html)
*   [数值线性代数](http://amzn.to/2CNOgZp)，1997。
*   [数值方法](http://amzn.to/2CNfSxE)，2012。

## 摘要

在这篇文章中，您发现了计算线性代数的 fast.ai 免费课程。

具体来说，你学到了：

*   课程的动机和先决条件。
*   课程中涵盖的主题概述。
*   这门课程究竟适合谁，不适合谁。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。