# 为什么从零开始实现机器学习算法

> 原文： [https://machinelearningmastery.com/why-implement-a-machine-learning-algorithm-from-scratch/](https://machinelearningmastery.com/why-implement-a-machine-learning-algorithm-from-scratch/)

当现有API中提供了如此多的算法时，为什么要从头开始实现机器学习算法？

这是一个很好的问题。在编写第一行代码之前必须考虑的问题。

在这篇文章中，您将发现这个问题的各种有趣甚至发人深省的答案。

这篇文章中的答案总结自Quora问题：“[当有许多高级API如tensorflow可用时，为什么需要手动实现机器学习算法？](https://www.quora.com/Why-is-there-a-need-to-manually-implement-machine-learning-algorithms-when-there-are-many-advanced-APIs-like-tensorflow-available) “。

![Why Implement a Machine Learning Algorithm From Scratch](img/eb1a35be0f1f24f7f32e26c4fcfc220f.jpg)

为什么从头开始实施机器学习算法
照片由 [psyberartist](https://www.flickr.com/photos/psyberartist/3518056742/) ，保留一些权利。

## （重新）实现算法的两个主要原因

我认为所有的答案都可以分解为两个阵营：

1.  **自学**，其中算法被实现为学习练习。
2.  **操作要求**，其中实施了一种算法以满足生产系统的需求。

## 实施自学算法

[Charles Gee](http://qr.ae/RgpWsi) 从自学角度给出了很好的答案。他评论道：

> ...假设我们没有谈论机器学习算法，而是谈论排序算法。当然，许多数据结构都有一个排序函数，几乎不需要编码，但你真的会聘请一个无法做出反对的程序员吗？选择排序？插入排序？归并排序？快速排序？二叉搜索树？

Charles描述了4种不同的用例，从头开始实现机器学习算法是非常可取的：

*   作为机器学习领域的初学者。
*   作为机器学习领域的研究员。
*   作为机器学习领域的老师。
*   作为这些机器学习算法的用户。

## 实现操作要求的算法

[Xavier Amatriain](http://qr.ae/RgpWnT) 在他的回答中关注这个话题。他评论道：

> 首先我要说的是，我确实认为任何团队都应该默认重新使用现有的实现。 ...但是，公司可能决定实施自己版本的ML算法的原因也很多。

Xavier列出了实现机器学习算法的5个理由，如下所示：

*   **表现**。对于特定用例，开源实现可能过于笼统且效率不高。
*   **正确性**。对于特定用例（例如较大规模的数据集），开源实现中可能存在错误或限制。
*   **编程语言**。实现可能仅限于特定的编程语言。
*   **整合**。可能需要将算法实现集成到现有生产系统的基础结构中。
*   **许可**。选择开源许可证可能会受到限制。

## 摘要

在这篇文章中，您发现有两个主要原因可能需要从头开始实现算法。

1.  要了解有关算法如何用于自学的更多信息。
2.  自定义生产系统的算法实现。

## 进一步阅读

我已经多次发布了从头开始实现机器学习算法的好处。

关于这个主题的进一步阅读包括：

*   [从零开始实施机器学习算法的好处](http://machinelearningmastery.com/benefits-of-implementing-machine-learning-algorithms-from-scratch/)
*   [通过从零开始实施它们来理解机器学习算法（以及解决不良代码的策略）](http://machinelearningmastery.com/understand-machine-learning-algorithms-by-implementing-them-from-scratch/)
*   [实现机器学习算法时不要从开源代码开始](http://machinelearningmastery.com/dont-start-with-open-source-code-when-implementing-machine-learning-algorithms/)
*   [如何实现机器学习算法](http://machinelearningmastery.com/how-to-implement-a-machine-learning-algorithm/)