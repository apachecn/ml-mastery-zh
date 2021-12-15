# 沿时间反向传播的温和介绍

> 原文： [https://machinelearningmastery.com/gentle-introduction-backpropagation-time/](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)

Backpropagation Through Time 或 BPTT 是用于更新 LSTM 等循环神经网络中权重的训练算法。

为了有效地构建循环神经网络的序列预测问题，您必须对通过时间的反向传播正在做什么以及在训练网络时如何通过时间截断反向传播等可配置变量将影响技能，稳定性和速度有一个强有力的概念性理解。帖子，你会得到一个温和的介绍 Backpropagation 通过时间打算为从业者（没有方程！）。

在这篇文章中，您将获得针对从业者的 Backpropagation Through Time 的温和介绍（无方程！）。

阅读这篇文章后，你会知道：

*   什么反向传播是时间以及它如何与多层感知机网络使用的反向传播训练算法相关。
*   导致需要通过时间截断反向传播的动机，这是用于训练 LSTM 的深度学习中最广泛使用的变体。
*   考虑如何配置截断反向传播时间以及研究和深度学习库中使用的规范配置的符号。

让我们开始吧。

![A Gentle Introduction to Backpropagation Through Time](img/96082bd310ea8de2ee4ab610276dd092.jpg)

时间反向传播的温和介绍
[Jocelyn Kinghorn](https://www.flickr.com/photos/joceykinghorn/11215752936/) 的照片，保留一些权利。

## 反向传播训练算法

反向传播指的是两件事：

*   用于计算导数的数学方法和衍生链规则的应用。
*   用于更新网络权重以最小化错误的训练算法。

我们在这里使用的是后一种对反向传播的理解。

反向传播训练算法的目标是修改神经网络的权重，以便与响应于相应输入的某些预期输出相比最小化网络输出的误差。

它是一种监督学习算法，允许根据所产生的特定错误纠正网络。

一般算法如下：

1.  呈现训练输入模式并通过网络传播以获得输出。
2.  将预测输出与预期输出进行比较并计算误差。
3.  计算相对于网络权重的误差的导数。
4.  调整权重以最小化错误。
5.  重复。

有关 Backpropagation 的更多信息，请参阅帖子：

*   [如何在 Python 中从零开始实现反向传播算法](http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

反向传播训练算法适用于训练固定大小的输入 - 输出对上的前馈神经网络，但是可能在时间上排序的序列数据呢？

## 通过时间反向传播

Backpropagation Through Time，或 BPTT，是将 Backpropagation 训练算法应用于应用于序列数据的循环神经网络，如时间序列。

循环神经网络每步显示一​​个输入并预测一个输出。

从概念上讲，BPTT 的工作方式是展开所有输入时间步长。每个时间步长有一个输入时间步长，一个网络副本和一个输出。然后计算每个时间步长的误差并累积。网络将回滚并更新权重。

在空间上，在给定问题的顺序依赖性的情况下，展开的循环神经网络的每个时间步长可被视为附加层，并且来自先前时间步的内部状态被视为随后时间步的输入。

我们可以总结算法如下：

1.  向网络呈现输入和输出对的一系列时间步长。
2.  展开网络，然后计算并累积每个时间步长的错误。
3.  汇总网络并更新权重。
4.  重复。

随着时间步长的增加，BPTT 在计算上可能是昂贵的。

如果输入序列由数千个时间步长组成，则这将是单个更新权重更新所需的导数的数量。这可能导致重量消失或爆炸（变为零或溢出）并使慢学习和模型技能嘈杂。

## 通过时间截断反向传播

截断反向传播通过时间，或 TBPTT，是用于循环神经网络的 BPTT 训练算法的修改版本，其中序列一次一步地处理并且周期性地（k1 时间步长）BPTT 更新被执行回固定数量的时间步长（ k2 时间步长）。

Ilya Sutskever 在他的论文中明确指出：

> 截断反向传播可以说是训练 RNN 最实用的方法。
> 
> ...
> 
> BPTT 的一个主要问题是单个参数更新的高成本，这使得无法使用大量迭代。
> 
> …
> 
> 使用简单的方法可以降低成本，该方法将 1,000 个长序列分成 50 个序列（比方说），每个序列长度为 20，并将每个长度为 20 的序列视为单独的训练案例。这是一种合理的方法，可以在实践中很好地工作，但它对于跨越 20 多个步骤的时间依赖性是盲目的。
> 
> …
> 
> 截断 BPTT 是一种密切相关的方法。它一次处理一个序列的序列，并且每 k1 个时间步，它运行 BPTT 达 k2 个时间步，因此如果 k2 很小，参数更新可以很便宜。因此，它的隐藏状态已暴露于许多时间步，因此可能包含有关远期过去的有用信息，这些信息将被机会性地利用。

- Ilya Sutskever，[训练循环神经网络](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)，论文，2013 年

We can summarize the algorithm as follows:

1.  向网络呈现输入和输出对的 k1 个时间步长序列。
2.  展开网络然后计算并累积 k2 个时间步长的错误。
3.  汇总网络并更新权重。
4.  重复

TBPTT 算法需要考虑两个参数：

*   **k1** ：更新之间的正向通过时间步数。通常，考虑到重量更新的执行频率，这会影响训练的速度或速度。
*   **k2** ：应用 BPTT 的时间步数。通常，它应该足够大以捕获问题中的时间结构以供网络学习。值太大会导致梯度消失。

为了更清楚：

> ...可以使用有界历史近似值，其中相关信息被保存了固定数量的时间步长，并且忘记了任何早于此的信息。通常，这应该被视为用于简化计算的启发式技术，但是，如下所述，它有时可以充当真实梯度的适当近似值，并且在权重被调整为网络的情况下也可能更合适。运行。让我们称这个算法在时间上被截断反向传播。如果 h 表示保存的先前时间步数，则该算法将表示为 BPTT（h）。
> 
> …
> 
> 注意，在 BPTT（h）中，每次网络运行另外的时间步骤时，重新执行最近的 h 时间步骤的向后传递。为了概括这一点，可以考虑在执行下一个 BPTT 计算之前让网络运行 h0 个额外的时间步骤，其中 h0 &lt;= h。
> 
> …
> 
> 该算法的关键特征是直到时间步 t + h0 才执行下一个后向传递。在中间时间内，网络输入，网络状态和目标值的历史记录保存在历史缓冲区中，但不对该数据进行处理。我们用这个算法表示 BPTT（h; h0）。显然，BPTT（h）与 BPTT（h; 1）相同，BPTT（h; h）是符合消息的 BPTT 算法。

- Ronald J. Williams 和 Jing Peng，[一种基于高效梯度的复发网络轨迹在线训练算法](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf)，1990

我们可以借用 Williams 和 Peng 的符号，并将不同的 TBPTT 配置称为 TBPTT（k1，k2）。

使用这种表示法，我们可以定义一些标准或常用方法：

注意，这里 n 指的是序列中的总时间步数：

*   **TBPTT（n，n）**：在序列的所有时间步长（例如经典 BPTT）的序列末尾进行更新。
*   **TBPTT（1，n）**：一次处理一个时间步，然后进行更新，覆盖到目前为止所见的所有时间步长（例如 Williams 和 Peng 的经典 TBPTT）。
*   **TBPTT（k1,1）**：网络可能没有足够的时间背景来学习，严重依赖内部状态和输入。
*   **TBPTT（k1，k2），其中 k1＆lt; k2＆lt; n** ：每个序列执行多次更新，这可以加速训练。
*   **TBPTT（k1，k2），其中 k1 = k2** ：一种常见配置，其中固定数量的时间步长用于前向和后向时间步（例如 10s 至 100s）。

我们可以看到所有配置都是 TBPTT（n，n）的变体，它基本上试图用可能更快的训练和更稳定的结果来近似相同的效果。

论文中报道的规范 TBPTT 可以被认为是 TBPTT（k1，k2），其中 k1 = k2 = h 并且 h &lt;= n，并且其中所选择的参数小（几十到几百步）。

在像 TensorFlow 和 Keras 这样的库中，事物看起来很相似，并且 h 定义了准备好的数据的时间步长的向量化固定长度。

## 进一步阅读

本节提供了一些进一步阅读的资源。

### 图书

*   [Neural Smithing：前馈人工神经网络中的监督学习](http://www.amazon.com/dp/0262527014?tag=inspiredalgor-20)，1999
*   [深度学习](http://www.amazon.com/dp/0262035618?tag=inspiredalgor-20)，2016 年

### 文件

*   [通过反向传播错误学习表示](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)，1986
*   [通过时间反向传播：它做了什么以及如何做](http://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf)，1990
*   [一种基于高效梯度的复发网络轨迹在线训练算法](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf)，1990
*   [训练循环神经网络](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)，论文，2013 年
*   [基于梯度的递归网络学习算法及其计算复杂度](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_25/Williams%20Zipser95RecNets.pdf)，1995

### 用品

*   维基百科上的 [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
*   [维基百科上的时间反向传播](https://en.wikipedia.org/wiki/Backpropagation_through_time)
*   [截断反向传播的样式](http://r2rt.com/styles-of-truncated-backpropagation.html)
*   [回答问题“RNN：何时应用 BPTT 和/或更新权重？”](https://stats.stackexchange.com/a/220111/8706)在 CrossValidated 上

## 摘要

在这篇文章中，您发现了 Backpropagation Through Time 用于训练复现神经网络。

具体来说，你学到了：

*   什么反向传播是时间以及它如何与多层感知机网络使用的反向传播训练算法相关。
*   导致需要通过时间截断反向传播的动机，这是用于训练 LSTM 的深度学习中最广泛使用的变体。
*   考虑如何配置截断反向传播时间以及研究和深度学习库中使用的规范配置的符号。

您对通过时间的反向传播有任何疑问吗？
在下面的评论中提出您的问题，我会尽力回答。