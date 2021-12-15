# Keras 长短期记忆循环神经网络的迷你课程

> 原文： [https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)

长期短期记忆（LSTM）复现神经网络是目前最有趣的深度学习类型之一。

它们被用于展示复杂问题领域的世界级结果，例如语言翻译，自动图像字幕和文本生成。

LSTM 与多层感知机和卷积神经网络的不同之处在于它们专门用于序列预测问题。

在这个迷你课程中，您将了解如何快速将 LSTM 模型用于您自己的序列预测问题。

完成这个迷你课程后，您将知道：

*   LSTM 是什么，如何训练，以及如何准备训练 LSTM 模型的数据。
*   如何开发一套 LSTM 模型，包括堆叠，双向和编解码器模型。
*   如何通过超参数优化，更新和最终模型来充分利用模型。

让我们开始吧。

**注**：这是一个很大的指南;你可能想要为它添加书签。

![Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras](img/296dab1141889e91e78fbb385c103c24.jpg)

具有 Keras 的长期短期记忆循环神经网络的迷你课程
照片由 [Nicholas A. Tonelli](https://www.flickr.com/photos/nicholas_t/14840636880/) ，保留一些权利。

## 这个迷你课程是谁？

在我们开始之前，让我们确保您在正确的位置。

本课程适用于了解一些应用机器学习并且需要快速掌握 LSTM 的开发人员。

也许您想要或需要在项目中开始使用 LSTM。本指南旨在帮助您快速有效地完成此任务。

*   你了解 Python 的方法。
*   你知道你在 SciPy 周围的方式。
*   您知道如何在工作站上安装软件。
*   你知道如何纠缠自己的数据。
*   您知道如何使用机器学习来解决预测性建模问题。
*   你可能知道一点点深度学习。
*   你可能知道一点 Keras。

您知道如何设置工作站以使用 Keras 和 scikit-learn;如果没有，你可以在这里学习如何：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

本指南采用自上而下和结果优先的机器学习风格，您已经习惯了。它将教你如何获得结果，但它不是灵丹妙药。

您将通过本指南开发有用的技能。

完成本课程后，您将：

*   了解 LSTM 的工作原理。
*   知道如何为 LSTM 准备数据。
*   知道如何应用一系列类型的 LSTM。
*   知道如何将 LSTM 调整为问题。
*   知道如何保存 LSTM 模型并使用它来做出预测。

接下来，让我们回顾一下课程。

## 迷你课程概述

这个迷你课程分为 14 节课。

您可以每天完成一节课（推荐）或在一天内完成所有课程（硬核！）。

这取决于你有空的时间和你的热情程度。

以下是 14 个课程，可以帮助您开始使用 Python 中的 LSTM 并提高工作效率。课程分为三个主题：基础，模型和高级。

![Overview of LSTM Mini-Course](img/aa95ec5c3ba105ecfc4181291d0ceea2.jpg)

LSTM 迷你课程概述

### 基金会

这些课程的重点是在使用 LSTM 之前需要了解的事项。

*   **第 01 课**：什么是 LSTM？
*   **第 02 课**：如何训练 LSTM
*   **第 03 课**：如何为 LSTM 准备数据
*   **第 04 课**：如何在 Keras 开发 LSTM

### 楷模

*   **第 05 课**：如何开发香草 LSTMs
*   **第 06 课**：如何开发栈式 LSTM
*   **第 07 课**：如何开发 CNN LSTM
*   **第 08 课**：如何开发编解码器 LSTM
*   **第 09 课**：如何开发双向 LSTM
*   **第 10 课**：如何开发注意力的 LSTM
*   **第 11 课**：如何开发生成 LSTM

### 高级

*   **第 12 课**：如何调整 LSTM 超参数
*   **第 13 课**：如何更新 LSTM 模型
*   **第 14 课**：如何使用 LSTM 做出预测

每节课可能需要 60 秒或 60 分钟。花点时间，按照自己的进度完成课程。提出问题，甚至在下面的评论中发布结果。

课程期望你去学习如何做事。我会给你提示，但每节课的部分内容是强迫你去哪里寻求帮助（提示，我在这个博客上有所有的答案;使用搜索）。

我确实在早期课程中提供了更多帮助，因为我希望你建立一些自信和惯性。

挂在那里;不要放弃！

## 基金会

本节中的课程旨在让您了解 LSTM 的工作原理以及如何使用 Keras 库实现 LSTM 模型。

## 第 1 课：什么是 LSTM？

### **目标**

本课程的目标是充分理解高级 LSTM，以便您可以向同事或经理解释它们是什么以及它们如何工作。

### 问题

*   什么是序列预测？一些例子是什么？
*   传统神经网络对序列预测有哪些局限性？
*   RNN 对序列预测的承诺是什么？
*   什么是 LSTM 及其组成部分是什么？
*   LSTM 有哪些突出的应用？

### 进一步阅读

*   [深度学习的循环神经网络速成课](http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)
*   [循环神经网络序列预测模型的简要介绍](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
*   [循环神经网络对时间序列预测的承诺](http://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)
*   [关于长短期记忆网络对时间序列预测的适用性](http://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/)
*   [专家对长短期记忆网络的简要介绍](http://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
*   [8 深度学习的鼓舞人心的应用](http://machinelearningmastery.com/inspirational-applications-deep-learning/)

## 第 2 课：如何训练 LSTM

### 目标

本课程的目的是了解如何在示例序列上训练 LSTM 模型。

### Questions

*   传统 RNN 的训练有哪些常见问题？
*   LSTM 如何克服这些问题？
*   什么算法用于训练 LSTM？
*   Backpropagation Through Time 如何运作？
*   什么是截断的 BPTT，它提供了什么好处？
*   如何在 Keras 中实现和配置 BPTT？

### Further Reading

*   [沿时间反向传播的温和介绍](http://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
*   [如何准备 Keras 中截断反向传播的序列预测](http://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)

## 第 3 课：如何为 LSTM 准备数据

### Goal

本课程的目标是了解如何准备用于 LSTM 模型的序列预测数据。

### Questions

*   如何准备用于 LSTM 的数字数据？
*   如何准备用于 LSTM 的分类数据？
*   使用 LSTM 时如何处理序列中的缺失值？
*   如何将序列构建为监督学习问题？
*   在使用 LSTM 时，如何处理长序列？
*   你如何处理不同长度的输入序列？
*   如何重塑 Keras 中 LSTM 的输入数据？

### 实验

演示如何将数字输入序列转换为适合训练 LSTM 的形式。

### Further Reading

*   [如何在 Python 中扩展长短期记忆网络的数据](http://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)
*   [如何使用 Python 编写单热编码序列数据](http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)
*   [如何使用 Python 处理序列预测问题中的缺失时间步长](http://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)
*   [如何将时间序列转换为 Python 中的监督学习问题](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [如何处理具有长短期记忆循环神经网络的超长序列](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
*   [如何准备 Keras 中截断反向传播的序列预测](http://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)
*   [可变长度输入序列的数据准备](http://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)

## 第 4 课：如何在 Keras 开发 LSTM

### Goal

本课程的目标是了解如何使用 Python 中的 Keras 深度学习库定义，拟合和评估 LSTM 模型。

### Questions

*   你如何定义 LSTM 模型？
*   你如何编译 LSTM 模型？
*   你如何适应 LSTM 模型？
*   您如何评估 LSTM 模型？
*   如何使用 LSTM 模型做出预测？
*   如何将 LSTM 应用于不同类型的序列预测问题？

### Experiment

准备一个示例，演示 LSTM 模型在序列预测问题上的生命周期。

### Further Reading

*   [Keras 中长期短期记忆模型的 5 步生命周期](http://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)
*   [循环神经网络序列预测模型的简要介绍](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)

## 楷模

本节中的课程旨在教您如何使用 LSTM 模型获得序列预测问题的结果。

## 第 5 课：如何开发香草 LSTM

### Goal

本课程的目标是学习如何开发和评估香草 LSTM 模型。

*   什么是香草 LSTM 架构？
*   什么是香草 LSTM 应用的例子？

### Experiment

设计并执行一个实验，演示序列预测问题的香草 LSTM。

### Further Reading

*   [用 Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) 在 Python 中用 LSTM 循环神经网络进行序列分类
*   [用 Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/) 用 Python 中的 LSTM 循环神经网络进行时间序列预测
*   [Python 中长期短期记忆网络的时间序列预测](http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

## 第 6 课：如何开发栈式 LSTM

### Goal

本课程的目标是学习如何开发和评估堆叠的 LSTM 模型。

### Questions

*   在层次结构的序列问题上使用香草 LSTM 有什么困难？
*   堆叠的 LSTM 是什么？
*   什么是栈式 LSTM 应用于何处的示例？
*   栈式 LSTM 提供哪些好处？
*   如何在 Keras 中实现栈式 LSTM？

### Experiment

设计并执行一个实验，演示具有分层输入结构的序列预测问题的栈式 LSTM。

### Further Reading

*   [用 Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) 在 Python 中用 LSTM 循环神经网络进行序列分类
*   [用 Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/) 用 Python 中的 LSTM 循环神经网络进行时间序列预测

## 第 7 课：如何开发 CNN LSTM

### Goal

本课程的目标是学习如何开发在前端使用卷积神经网络的 LSTM 模型。

### Questions

*   使用具有空间输入数据的香草 LSTM 有什么困难？
*   什么是 CNN LSTM 架构？
*   有哪些 CNN LSTM 的例子？
*   CNN LSTM 提供哪些好处？
*   如何在 Keras 中实现 CNN LSTM 架构？

### Experiment

设计并执行一个实验，演示有空间输入的序列预测问题的 CNN LSTM。

### Further Reading

*   [用 Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) 在 Python 中用 LSTM 循环神经网络进行序列分类

## 第 8 课：如何开发编解码器 LSTM

### Goal

本课程的目标是学习如何开发编解码器 LSTM 模型。

### Questions

*   什么是序列到序列（seq2seq）预测问题？
*   在 seq2seq 问题上使用香草 LSTM 有什么困难？
*   什么是编解码器 LSTM 架构？
*   编解码器 LSTM 有哪些例子？
*   编解码器 LSTM 有什么好处？
*   如何在 Keras 中实现编解码器 LSTM？

### Experiment

设计并执行一个实验，演示序列到序列预测问题的编解码器 LSTM。

### Further Reading

*   [如何在 Python](http://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) 中为长期短期记忆网络使用时间分布层
*   [如何学习使用 seq2seq 循环神经网络相加数字](http://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/)
*   [如何将编解码器 LSTM 用于随机整数的回波序列](http://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/)

## 第 9 课：如何开发双向 LSTM

### Goal

本课程的目标是学习如何开发双向 LSTM 模型。

### Questions

*   什么是双向 LSTM？
*   有哪些使用双向 LSTM 的例子？
*   双向 LSTM 比香草 LSTM 有什么好处？
*   双向架构引发了关于时间步长的问题？
*   如何在 Keras 中实现双向 LSTM？

### Experiment

设计并执行一个实验，在序列预测问题上比较前向，后向和双向 LSTM 模型。

### Further Reading

*   [如何使用 Keras 开发用于 Python 序列分类的双向 LSTM](http://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)

## 第 10 课：如何开发注意力的 LSTM

### Goal

本课的目的是学习如何开发 LSTM 模型。

### Questions

*   具有中性信息的长序列对 LSTM 有何影响？
*   LSTM 型号的注意事项是什么？
*   在 LSTM 中使用注意力的一些例子是什么？
*   注意为序列预测提供了什么好处？
*   如何在 Keras 中实现注意力架构？

### Experiment

设计并执行一项实验，该实验将注意力集中在具有长序列中性信息的序列预测问题上。

### Further Reading

*   [长期短期记忆循环神经网络](http://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项

## 第 11 课：如何开发生成 LSTM

### Goal

本课程的目标是学习如何开发用于生成模型的 LSTM。

*   什么是生成模型？
*   如何将 LSTM 用作生成模型？
*   LSTM 作为生成模型的一些例子是什么？
*   LSTM 作为生成模型有哪些好处？

### Experiment

设计并执行实验以学习文本语料库并生成具有相同语法，语法和样式的新文本样本。

### Further Reading

*   [使用 Keras](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) 在 Python 中使用 LSTM 循环神经网络生成文本

## 高级

本节中的课程旨在教您如何根据自己的序列预测问题从 LSTM 模型中获得最大收益。

## 第 12 课：如何调整 LSTM 超参数

### Goal

本课程的目标是学习如何调整 LSTM 超参数。

### Questions

*   我们如何诊断 LSTM 模型的过度学习或学习不足？
*   调整模型超参数的两种方案是什么？
*   鉴于 LSTM 是随机算法，如何可靠地估计模型技能？
*   列出可以调整的 LSTM 超参数，并提供可以评估的值的示例：
    *   模型初始化和行为。
    *   模型架构和结构。
    *   学习行为。

### Experiment

设计并执行实验以调整 LSTM 的一个超参数并选择最佳配置。

### Further Reading

*   [如何评估深度学习模型的技巧](http://machinelearningmastery.com/evaluate-skill-deep-learning-models/)
*   [如何用 Keras 调整 LSTM 超参数进行时间序列预测](http://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)
*   [如何使用 Keras 网格搜索 Python 中的深度学习模型的超参数](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
*   [如何提高深度学习效能](http://machinelearningmastery.com/improve-deep-learning-performance/)

## 第 13 课：如何更新 LSTM 模型

### Goal

本课程的目标是学习如何在新数据可用后更新 LSTM 模型。

### Questions

*   更新 LSTM 模型以响应新数据有什么好处？
*   使用新数据更新 LSTM 模型有哪些方案？

### Experiment

设计并执行实验以使 LSTM 模型适应序列预测问题，该问题与不同模型更新方案的模型技能的影响形成对比。

### Further Reading

*   [如何在时间序列预测训练期间更新 LSTM 网络](http://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/)

## 第 14 课：如何使用 LSTM 做出预测

### Goal

本课程的目标是学习如何最终确定 LSTM 模型并使用它来预测新数据。

### Questions

*   你如何在 Keras 中保存模型结构和重量？
*   你如何适应最终的 LSTM 模型？
*   如何使用最终模型做出预测？

### Experiment

设计并执行实验以适应最终的 LSTM 模型，将其保存到文件，然后加载它并对保留的验证数据集做出预测。

### Further Reading

*   [保存并加载您的 Keras 深度学习模型](http://machinelearningmastery.com/save-load-keras-deep-learning-models/)
*   [如何训练最终机器学习模型](http://machinelearningmastery.com/train-final-machine-learning-model/)

## 结束！
（_ 看你有多远 _）

你做到了。做得好！

花点时间回顾一下你到底有多远。以下是您学到的知识：

1.  LSTM 是什么以及为什么它们是序列预测的首选深度学习技术。
2.  LSTM 使用 BPTT 算法进行训练，该算法也强加了一种思考序列预测问题的方法。
3.  用于序列预测的数据准备可以涉及掩蔽缺失值以及分割，填充和截断输入序列。
4.  Keras 为 LSTM 模型提供了 5 步生命周期，包括定义，编译，拟合，评估和预测。
5.  香草 LSTM 由输入层，隐藏的 LSTM 层和密集输出层组成。
6.  隐藏的 LSTM 层可以堆叠，但必须从一层到另一层暴露整个序列的输出。
7.  在处理图像和视频数据时，CNN 可用作 LSTM 的输入层。
8.  在预测可变长度输出序列时可以使用编解码器架构。
9.  在双向 LSTM 中向前和向后提供输入序列可以提高某些问题的技能。
10.  该注意力可以为包含中性信息的长输入序列提供优化。
11.  LSTM 可以学习输入数据的结构化关系，进而可以用来生成新的例子。
12.  LSTM 的 LSTM 超参数可以像任何其他随机模型一样进行调整。
13.  当新数据可用时，可以更新适合的 LSTM 模型。
14.  最终的 LSTM 模型可以保存到文件中，然后加载以便对新数据做出预测。

不要轻视这一点;你在很短的时间内走了很长的路。

这只是您与 Keras 的 LSTM 之旅的开始。继续练习和发展你的技能。

## 摘要

**你如何使用迷你课程？**
你喜欢这个迷你课吗？

**你有什么问题吗？有没有任何问题？**
让我知道。在下面发表评论。