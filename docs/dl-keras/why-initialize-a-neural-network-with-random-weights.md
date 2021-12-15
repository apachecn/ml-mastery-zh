# 神经网络的权重为什么要随机初始化？

> 原文： [https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)

人工神经网络的权重（weights）必须初始化为小的随机数。

因为用于训练模型的随机优化算法(stochastic optimization algorithm)要求如此，这种算法称为随机梯度下降(stochastic gradient descent)。

要理解这种解决问题的方法，您首先必须了解非确定性和随机算法（nondeterministic and randomized algorithms）的作用，以及为什么随机优化算法在参数搜索过程中需要利用随机数。

在这篇文章中，您将学习为什么神经网络的权重必须随机初始化(randomly initialized)。

阅读这篇文章后，您会知道：

*   处理具有挑战性的问题时我们有时需要非确定性和随机算法。
*   在随机优化算法中，使用随机数来进行初始化和参数的搜索。
*   随机梯度下降是随机优化算法，它需要对神经网络的权重进行随机初始化。

让我们开始吧！

![Why Initialize a Neural Network with Random Weights?](img/d767349f43cecc391f31806440729f12.png)

神经网络的权重为什么要随机初始化？
[lwtt93](https://www.flickr.com/photos/37195641@N03/7086827121/) 的照片，保留一些权利。

## 概要

这篇文章分为以下 4 个部分：

1.  确定性和非确定性算法
2.  随机搜索算法
3.  神经网络中的随机初始化
4.  初始化方法

## 确定性和非确定性算法

经典算法是确定性的（deterministic），例如对列表进行排序的算法。

假设给定一个未排序的列表，排序算法（比如冒泡排序（bubble sort）或快速排序（quick sort））将系统地对列表进行排序，直到有一个有序的结果。"确定性"意味着每次给定相同的列表时，它将以完全相同的方式执行。它将在程序的每个步骤都进行相同的操作。

确定性算法很棒，因为它们可以保证最佳、最差和平均运行时间。问题是，它们并不适合所有问题。

有些问题对计算机来说很难。也许是因为组合的数量，也许是因为数据的大小。它们非常难，因为确定性算法不能有效率地解决它们。该算法可能会运行，但会继续运行直至宇宙因过热而死亡。

另一种解决方案是使用[非确定性算法](https://en.wikipedia.org/wiki/Nondeterministic_algorithm)。这些是在算法执行期间做决策时使用[随机性](https://en.wikipedia.org/wiki/Randomized_algorithm)元素的算法。这意味着当在相同数据上重新运行相同的算法时，将遵循不同的步骤顺序。

他们可以大大加快获得解决方案的过程，但解决方案将是近似的，或者说是“_好的_”, 但往往不是 “_最佳的_”。 不确定性算法往往不能很好地保证运行时间或其解决方案的质量。

不过这不是问题。因为这类算法想要解决的问题通常非常难，所以任何好的解决方案都已经可以使人满意了。

## 随机搜索算法

搜索问题通常非常具有挑战性，这类问题需要使用非确定性算法，而非确定性算法则往往很大程度上依赖随机性。

这类算法本身并不是随机的，而是他们谨慎地使用随机性。它们在一定边界内是随机的，被称为[随机算法](https://en.wikipedia.org/wiki/Stochastic_optimization)。

搜索算法的逐步搜索（incremental or step-wise search）的性质通常意味着搜索过程和搜索算法是一种从初始状态或位置到最终状态或位置的优化（optimization）过程。例如，随机优化问题或随机优化算法，这其中的例子是遗传算法（genetic algorithm），模拟退火（simulated annealing）和随机梯度下降（stochastic gradient descent）。

搜索过程是从可行域中的一个起点到一些足够好的解决方案的一个逐步的过程。

它们在使用随机性方面具有共同特征，例如：

*   在初始化期间使用随机性。
*   在搜索过程中使用随机性。

我们对搜索空间（search space）的结构一无所知。因此，为了消除搜索过程中的偏差，我们从随机选择的一个起点开始。

随着搜索过程的展开，我们有可能陷入搜索空间的不利区域。在搜索过程中使用随机性可能会使我们避免陷入不利区域，并找到更好的候选解决方案。

陷入不利区域并找到次优的解决方案，这种情况被称为陷入局部最优（local optima）。

在搜索过程中使用随机性和在初始化中使用随机性是齐头并进的。

如果我们将搜索找到的任何解决方案视为临时或候选方案，并且搜索过程可以多次执行，那么这两个随机过程就可以更好地协同工作。

这为随机搜索过程提供了多个机会来启动和遍历候选解决方案的空间，以寻找更好的候选解决方案 - 即所谓的全局最优解（global optima）。

我们通常用山脉和山谷景观的类比（例如[适应度景观](https://en.wikipedia.org/wiki/Fitness_landscape)）来描述在候选解决方案空间的搜索过程。如果我们想要寻找最大解，我们可以将景观中的小山丘视为局部的最大解，将最大的山丘视为全局最大解。

这是一个吸引人的研究领域，我在这个领域有一些研究。您可以参见我的书：

*   [聪明的算法：自然启发的编程食谱](http://cleveralgorithms.com/nature-inspired/index.html)

## 神经网络中的随机初始化

人工神经网络是使用随机梯度下降的随机优化算法训练的。

该算法使用随机性，以便为正在学习的数据中的输入到输出的特定映射函数（mapping function）找到足够好的权重集。这意味着每次使用这样的训练算法时，特定训练数据的特定网络将会被训练成具有不同模型技能的不同网络。

这是该算法的一个特征，而不是一个 bug。

我在下面的帖子中更详细地描述了这个问题：

*   [在机器学习中拥抱随机性](https://machinelearningmastery.com/randomness-in-machine-learning/)

如前一节所述，诸如随机梯度下降的随机优化算法在选择搜索的起始点和搜索的过程中使用随机性。

具体而言，随机梯度下降要求将网络的权重初始化为小的随机数（随机，但接近零，例如[0.0, 0.1]）。在每个时期（epoch）之前，训练数据集也使用随机性来进行混合（shuffling），这反过来导致每个批次 (batch) 的梯度估计（gradient estimate）也会不同。

您可以在这篇文章中了解更多关于随机梯度下降的信息：

*   [小批量梯度下降的简要介绍以及如何配置批量大小](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

神经网络在搜索或学习上的进展称为收敛（convergence）。发现次优解或局部最优被称为早熟收敛（premature convergence）。

> 深度学习的训练算法在本质上通常是迭代的（iterative），因此需要用户指定开始迭代的初始点。此外，训练深度模型是一项非常困难的任务，大多数算法都会被不同的初始化而强烈影响。

- 第 301 页，[深度学习](https://amzn.to/2H5wjfg)，2016 年。

如果想要评估一个神经网络架构的表现，最有效方法是多次重复搜索的过程，并总结模型的平均表现。这为网络架构提供了从多个不同初始条件集搜索空间的最佳机会，称为多次重启（multiple restart）或多次重启搜索（multiple-restart search）。

您可以在这篇文章中了解有关神经网络有效评估的更多信息：

*   [如何评估深度学习模型的表现](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

### 为什么不将权重设置为零？

每次训练网络时，我们都可以使用相同的权重集。例如，您可以对所有权重使用 0.0 的值。

在这种情况下，学习算法（learning algorithm）的方程将无法对网络权重进行任何更改，模型将被卡住。重要的是要注意，每个神经元中的偏差权重（bias weight）默认设置为零，而不是一个小的随机值。

具体地，连接到相同输入（input）的隐藏层中并排的神经元（nodes）必须用不同权重，这样学习算法才可以更新权重。

这通常被称为“在训练期间打破对称性的需要”（the need to break symmetry）。

> 关于神经网络的学习算法，也许我们只知道一个完全确定的属性，即初始参数需要在不同神经元之间“打破对称性”。如果具有相同激活函数（activation function）的两个隐藏神经元连接到相同的输入（input），则这些神经元必须具有不同的初始参数。如果它们具有相同的初始参数，那么确定性学习算法（这些算法用于确定性成本和模型）将以相同方式不断更新这两个神经元。

- 第 301 页，[深度学习](https://amzn.to/2H5wjfg)，2016 年。

### 何时初始化为相同权重？

每次训练网络时，我们都可以使用相同的随机数。

在评估网络架构时，这没有用。

但这在生产环境中使用模型的情况下，给定训练数据集，训练相同的最终网络权重集时可能是有帮助的。

您可以在如下这篇文章中了解有关由 Keras 开发的使用固定随机种子（random seed) 来训练神经网络的更多信息：

*   [如何用 Keras 得到可重现的结果]((https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)

## 初始化方法

传统上，神经网络的权重被设置为小的随机数。

神经网络权重的初始化是一个完整的研究领域，因为精心地初始化一个网络可以加速学习过程。

现代深度学习库，例如 Keras，提供了许多网络初始化方法，所有这些都是用小随机数初始化权重的变体。

例如，目前 Keras 为所有类型的网络编写提供了如下方法：

*   **Zeros** ：使张量（tensor）初始化为 0 的初始化器（initializer） 。
*   **Ones** ：使张量初始化为 1 的初始化器。
*   **Constant**：使张量初始化为某个常数的初始化器。
*   **RandomNormal** ：使张量符合正态分布的初始化器。
*   **RandomUniform** ：使张量符合均匀分布的初始化器。
*   **TruncatedNormal** ：使张量符合截断正态分布的初始化器。
*   **VarianceScaling** ：可以根据权重的形状改变其规模的初始化器。
*   **Orthogonal** ：可以生成随机正交矩阵的初始化器。
*   **Identity** ：可以生成单位矩阵的初始化器。
*   **lecun_uniform** ：LeCun 均匀分布初始化器。
*   **glorot_normal** ：Glorot 正态分布初始化器，也称为 Xavier 正态分布初始化器。
*   **glorot_uniform** ：Glorot 均匀初始化器，也叫 Xavier 均匀初始化器。
*   **he_normal** : He 正态分布初始化器。
*   **lecun_normal** ：LeCun 正态分布初始化器。
*   **he_uniform** ：He 均匀方差调节初始化器。

更多信息请参阅[文档](https://keras.io/initializers/)。

出于兴趣，Keras 开发人员为不同类型的神经网络层选择的默认初始值设定如下：

*   **Dense**（例如 MLP）： _glorot_uniform_
*   **LSTM** ： _glorot_uniform_
*   **CNN** ： _glorot_uniform_

您可以在本文中了解更多关于“`glorot_uniform`”的信息，它也被称为“ _Xavier normal_ ”，是以本方法的开发人员 Xavier Glorot 的名字来命名的：

*   [理解深度前馈神经网络训练的难度](http://proceedings.mlr.press/v9/glorot10a.html)，2010。

没有单一的最佳方法来初始化神经网络的权重。

> 现代初始化策略是简单且是启发式的。设计更好的初始化策略是一项艰巨的任务，因为我们还没能清楚理解神经网络优化。 [...]我们对初始点如何影响神经网络的预测能力的理解还特别粗略，几乎还不能为如何选择初始点提供任何指导。

- 第 301 页，[深度学习](https://amzn.to/2H5wjfg)，2016 年。

因此，初始器也是一个超参数。您可以在特定预测性建模问题上探索、测试和试验这个超参数。

您有没有最喜欢的权重初始化方法？
请在下面的评论中告诉我。

## 进一步阅读

如果您希望深入了解此话题，本节将提供更多相关资源。

### 书籍

*   [深度学习](https://amzn.to/2H5wjfg)，2016 年。

### 文章

*   [维基百科上的非确定性算法](https://en.wikipedia.org/wiki/Nondeterministic_algorithm)
*   [维基百科上的随机算法](https://en.wikipedia.org/wiki/Randomized_algorithm)
*   [维基百科上的随机优化](https://en.wikipedia.org/wiki/Stochastic_optimization)
*   [维基百科上的随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
*   [维基百科上的适应度景观](https://en.wikipedia.org/wiki/Fitness_landscape)
*   [神经网络常见问题](ftp://ftp.sas.com/pub/neural/FAQ.html)
*   [Keras 权重初始化](https://keras.io/initializers/)
*   [了解深度前馈神经网络训练的难度](http://proceedings.mlr.press/v9/glorot10a.html)，2010。

### 讨论

*   [stackexchange 论坛：神经网络中什么样的初始权重是好的？](https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network)
*   [为什么神经网络的权重应该初始化为随机数？](https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers)
*   [quora 论坛：神经网络中什么样的初始权重是好的？](https://www.quora.com/What-are-good-initial-weights-in-a-neural-network)

## 摘要

在这篇文章中，您学习了为什么必须随机初始化神经网络的权重。

具体来说，您学到了：

*   为什么具有挑战性的问题需要非确定性和随机算法。
*   在随机优化算法的初始化和搜索过程中使用随机性。
*   随机梯度下降是随机优化算法，需要随机初始化网络权重。

您还有任何问题吗？
请在下面的评论中提问，我会尽力回答。
