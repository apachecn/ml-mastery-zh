# 停止从零开始编写机器学习算法

> 原文： [https://machinelearningmastery.com/dont-implement-machine-learning-algorithms/](https://machinelearningmastery.com/dont-implement-machine-learning-algorithms/)

### 你不必实现算法
... _如果你是初学者并且刚刚开始。_

停止。

您是否正在实现机器学习算法？

为什么？

从零开始实现算法是我看到初学者犯下的最大错误之一。

在这篇文章中你会发现：

*   初学者陷入的算法实现陷阱。
*   设计世界一流的机器学习算法实现的真正困难。
*   为什么你应该使用现成的实现。

让我们开始吧。

![Don't Implement Machine Learning Algorithms](img/7e0b20781e6e1d8d803f468aebe287bd.jpg)

不要实现机器学习算法
照 [kirandulo](https://www.flickr.com/photos/kirandulo/16555448595/) ，保留一些权利。

## 抓住了实现陷阱

这是我收到的电子邮件的片段：

> ......我真的很挣扎。为什么我必须从零开始实现算法？

似乎很多开发人员都陷入了这一挑战。

他们被告知或暗示：

**算法必须在使用前实现
。**

或者那个：

**你只能通过
实现算法来学习机器学习。**

以下是我偶然发现的一些类似问题：

*   _当_ tensorflow _等许多高级API可用时，为什么需要手动实现机器学习算法？_ （Quora 上的[）](https://www.quora.com/Why-is-there-a-need-to-manually-implement-machine-learning-algorithms-when-there-are-many-advanced-APIs-like-tensorflow-available)
*   _您自己或是否应该使用库，是否有任何实现机器学习算法的价值？_ （Quora 上的[）](https://www.quora.com/Is-there-any-value-implementing-machine-learning-algorithms-by-yourself-or-should-you-use-libraries)
*   _实现机器学习算法有用吗？_ （Quora 上的[）](https://www.quora.com/Is-it-useful-to-implement-machine-learning-algorithms)
*   _我应该使用哪种编程语言来实现机器学习算法？_ （Quora 上的[）](https://www.quora.com/Which-programming-language-should-I-use-to-implement-Machine-Learning-algorithms)
*   _为什么你和其他人有时会从零开始实现机器学习算法？_ （GitHub 上的[）](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/implementing-from-scratch.md)

## 你可能做错了

您不必从零开始实现机器学习算法。

这是传统上用于教授机器学习的自下而上方法的一部分。

1.  学习数学。
2.  学习理论。
3.  从零开始实现算法。
4.  _??? （魔术发生在这里_）。
5.  应用机器学习。

将机器学习算法应用于问题并获得结果比从头实现它们要容易得多。

**好多了！**

学习如何使用算法而不是实现算法不仅更容易，而且是更有价值的技能。您可以开始使用的技能，以便迅速产生真正的影响。

通过应用机器学习，您可以选择许多低调的水果。

## 实现机器学习算法
......真的很难！

用于解决业务问题的算法需要**快速**和**正确**。

### 快速算法

更复杂的非线性方法比线性方法需要更多的数据。

这意味着他们需要做很多工作，这可能需要很长时间。

算法需要快速处理所有这些数据。特别是，规模。

这可能需要以最适合底层库中的特定矩阵运算的方式重新解释作为该方法基础的线性代数。

它可能需要专门的缓存知识才能充分利用您的硬件。

在你得到“ _hello world_ ”实现工作之后，这些并不是特殊的技巧。这些是包含算法实现项目的工程挑战。

### 正确的算法

机器学习算法将为您提供结果，即使它们的实现已经削弱。

你得到一个号码。一个输出。一个预测。

有时预测是正确的，有时则不然。

机器学习算法使用随机性。 [它们是随机算法](http://machinelearningmastery.com/randomness-in-machine-learning/)。

这不仅仅是单元测试的问题，而是要深入了解该技术并设计案例来证明实现符合预期并处理边缘情况。

## 使用现成的实现

你可能是一名优秀的工程师。

但是，与现成的实现相比，你的“ _hello world_ ”算法的实现可能不会削减它。

您的实现可能基于教科书描述，这意味着它将是朴素和缓慢的。您可能有也可能没有专业知识来设计测试以确保实现的正确性。

开源库中的现成实现是为了速度和/或稳健性而构建的。

**你怎么不使用标准的机器学习库？**

它们可以针对旨在尽可能快的非常狭窄的问题类型进行定制。它们也可能用于通用目的，确保它们能够在您考虑的范围之外的各种问题上正确运行。

### 库并非全部创建平等

并非您从Internet下载的所有算法实现都是相同的。

来自GitHub的代码片段可能是研究生“ _hello world_ ”实现，或者它可能是由大型组织的整个研究团队贡献的高度优化的实现。

您需要评估您正在使用的代码的来源。有些来源比其他来源更好或更可靠。

通用目的库通常以某种速度为代价而更加强大。

黑客工程师的照明快速实现经常遭受糟糕的文档记录，并且在满足他们的期望时非常迂腐。

选择实现时请考虑这一点。

### 建议

当被问及时，我通常会推荐以下三种平台之一：

1.  **Weka** 。一个不需要任何代码的图形用户界面。如果您想首先关注机器学习并学习如何解决问题，那就太完美了。
2.  **Python** 。生态系统包括大熊猫和scikit-learn。非常适合将开发中的机器学习问题的解决方案拼接在一起，该解决方案足够强大，可以部署到操作中。
3.  **R** 。更高级的平台虽然拥有深奥的语言，有时还有错误的软件包，但可以访问由学者直接编写的最先进的方法。非常适合一次性项目和研发。

这些只是我的建议，还有更多的机器学习平台可供选择。

## 有时你必须实现

在开始机器学习时，您不必实现机器学习算法。

但是你可以。

这样做有很好的理由。

例如，这里有三大原因：

*   您希望实现以了解算法的工作原理。
*   您无需实现所需的算法。
*   您需要的算法没有合适的（足够快的等）实现。

第一个是我最喜欢的。这可能让你感到困惑。

您可以实现机器学习算法以了解它们的工作原理。我推荐它。开发人员学习这种方式非常有效。

但。

您不必通过实现机器学习算法**启动**。通过在实现机器学习算法之前学习如何使用机器学习算法，您将更快地建立机器学习的信心和技能。

完成实现所需的实现和任何研究将改善您的理解。在下次使用该算法时，可以帮助您获得更好的结果。

## 摘要

在这篇文章中，您发现初学者陷入了从零开始实现机器学习算法的陷阱。

**他们被告知这是唯一的方法。**

您发现机器学习算法的工程快速而强大的实现是一项艰巨的挑战。

您了解到在实现它们之前学习如何使用机器学习算法要容易得多，也更为可取。您还了解到，实现算法是了解更多有关它们如何工作以及从中获取更多信息的好方法，但只有在您知道如何使用它们之后。

**你被困在这个陷阱里吗？**
_在评论中分享您的经验。_

### 进一步阅读

*   [程序员在机器学习中开始犯下的错误](http://machinelearningmastery.com/mistakes-programmers-make-when-starting-in-machine-learning/)
*   [从零开始实现机器学习算法](http://machinelearningmastery.com/understand-machine-learning-algorithms-by-implementing-them-from-scratch/)
*   [从零开始实现机器学习算法的好处](http://machinelearningmastery.com/benefits-of-implementing-machine-learning-algorithms-from-scratch/)