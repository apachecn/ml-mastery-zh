# 特征选择简介

> 原文： [https://machinelearningmastery.com/an-introduction-to-feature-selection/](https://machinelearningmastery.com/an-introduction-to-feature-selection/)

您应该使用哪些功能来创建预测模型？

这是一个困难的问题，可能需要深入了解问题领域。

您可以自动选择数据中与您正在处理的问题最有用或最相关的功能。这是一个称为特征选择的过程。

在这篇文章中，您将发现功能选择，可以使用的方法类型以及下次需要为机器学习模型选择功能时可以使用的便捷清单。

[![feature selection](img/b625073b2a067d3091fb55f2b5a1dd59.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/10/feature-selection.jpg)

特征选择简介
照片由 [John Tann](https://www.flickr.com/photos/31031835@N08/6498604953) 拍摄，保留一些权利

## 什么是特征选择

特征选择也称为变量选择或属性选择。

它是自动选择数据中的属性（例如表格数据中的列），这些属性与您正在处理的预测建模问题最相关。

> 特征选择...是选择用于模型构建的相关特征子集的过程

- [特征选择](http://en.wikipedia.org/wiki/Feature_selection)，维基百科条目。

特征选择不同于降维。两种方法都试图减少数据集中的属性数量，但维度降低方法通过创建新的属性组合来实现，其中特征选择方法包括和排除数据中存在的属性而不更改它们。

维数降低方法的示例包括主成分分析，奇异值分解和Sammon映射。

> 功能选择本身很有用，但它主要用作过滤器，可以清除除现有功能之外无用的功能。

- Robert Neuhaus，回答“[您认为特征选择在机器学习中有多重要？](http://www.quora.com/How-valuable-do-you-think-feature-selection-is-in-machine-learning-Which-do-you-think-improves-accuracy-more-feature-selection-or-feature-engineering) “

## 问题特征选择解决

特征选择方法可帮助您创建准确的预测模型。它们可以帮助您选择能够提供更好或更好准确性同时需要更少数据的功能。

特征选择方法可用于从数据中识别和移除不需要的，不相关的和冗余的属性，这些属性对预测模型的准确性没有贡献，或者实际上可能降低模型的准确性。

较少的属性是可取的，因为它降低了模型的复杂性，更简单的模型更易于理解和解释。

> 变量选择的目标有三个方面：提高预测变量的预测表现，提供更快，更具成本效益的预测变量，并更好地理解生成数据的基础过程。

- Guyon和Elisseeff的“[变量和特征选择介绍](http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf)”（PDF）

## 特征选择算法

一般类别的特征选择算法有三种：过滤方法，包装方法和嵌入方法。

### 过滤方法

过滤器功能选择方法应用统计度量来为每个要素指定评分。功能按分数排序，并选择保留或从数据集中删除。这些方法通常是单变量的，并且可以独立地考虑特征，或者考虑因变量。

一些滤波器方法的一些示例包括卡方检验，信息增益和相关系数分数。

### 包装方法

包装方法考虑选择一组特征作为搜索问题，其中准备，评估不同组合并与其他组合进行比较。我们用于评估特征组合并根据模型准确度分配分数的预测模型。

搜索过程可以是有条理的，例如最佳优先搜索，它可以是随机的，例如随机爬山算法，或者它可以使用启发法，例如前向和后向传递来添加和移除特征。

如果包装器方法是递归特征消除算法的示例。

### 嵌入式方法

嵌入式方法了解在创建模型时哪些功能最有助于模型的准确性。最常见的嵌入式特征选择方法是正则化方法。

正则化方法也称为惩罚方法，其将额外约束引入预测算法（例如回归算法）的优化中，该预测算法将模型偏向于较低复杂度（较少系数）。

正则化算法的例子是LASSO，弹性网和岭回归。

## 功能选择教程和秘籍

我们在此博客上看到过许多功能选择示例。

*   **Weka** ：有关如何使用Weka执行特征选择的教程，请参阅“[特征选择以提高准确性并缩短训练时间](http://machinelearningmastery.com/feature-selection-to-improve-accuracy-and-decrease-training-time/ "Feature Selection to Improve Accuracy and Decrease Training Time")”。
*   **Scikit-Learn** ：有关使用scikit-learn在Python中进行递归特征消除的秘籍，请参阅“使用Scikit-Learn 在Python中使用[特征选择”。](http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/ "Feature Selection in Python with Scikit-Learn")
*   **R** ：使用Caret R软件包进行递归特征消除的秘籍，请参阅“使用Caret R软件包”选择“[特征”](http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/ "Feature Selection with the Caret R Package")

## 选择功能时的陷阱

特征选择是应用机器学习过程的另一个关键部分，如模型选择。你不能开火和忘记。

将特征选择视为模型选择过程的一部分非常重要。如果不这样做，可能会无意中将偏差引入模型中，从而导致过度拟合。

> ...应该在不同的数据集上进行特征选择，而不是训练[预测模型] ...不这样做的效果是你会过度训练你的训练数据。

- Ben Allison回答“[是否使用相同的数据进行特征选择和交叉验证是否有偏差？](http://stats.stackexchange.com/questions/40576/is-using-the-same-data-for-feature-selection-and-cross-validation-biased-or-not) “

例如，当您使用精确度估计方法（如交叉验证）时，必须在内循环中包含要素选择。这意味着在训练模型之前，在准备好的折叠上执行特征选择。错误是首先执行特征选择以准备数据，然后对所选特征执行模型选择和训练。

> 如果我们采用适当的程序，并在每个折叠中执行特征选择，则在该折叠中使用的特征的选择中不再存在关于所保持的情况的任何信息。

- Dikran Marsupial在机器学习中执行交叉验证时回答“[最终模型的特征选择”](http://stats.stackexchange.com/questions/2306/feature-selection-for-final-model-when-performing-cross-validation-in-machine)

原因是选择特征的决策是在整个训练集上进行的，而这些决策又被传递到模型上。这可能会导致一种模式，即所选择的特征比其他正在测试的模型增强的模型可以获得看似更好的结果，而实际上它是有偏差的结果。

> 如果对所有数据执行特征选择然后交叉验证，则交叉验证过程的每个折叠中的测试数据也用于选择特征，这是表现分析的偏差。

- Dikran Marsupial回答“[特征选择和交叉验证](http://stats.stackexchange.com/questions/27750/feature-selection-and-cross-validation)”

## 功能选择清单

Isabelle Guyon和Andre Elisseeff是“[变量和特征选择简介](http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf)”（PDF）的作者提供了一个很好的清单，您可以在下次需要为预测建模问题选择数据特征时使用它。

我在这里复制了清单的重要部分：

1.  **你有领域知识吗？** 如果是，请构建一组更好的临时“”功能
2.  **你的功能是否相称？** 如果不是，请考虑将它们标准化。
3.  **你怀疑功能的相互依赖吗？** 如果是，请通过构建联合功能或功能产品来扩展功能集，就像计算机资源允许的那样。
4.  **您是否需要修剪输入变量（例如成本，速度或数据理解原因）？** 如果不是，则构造析取特征或特征的加权和
5.  **您是否需要单独评估功能（例如，了解它们对系统的影响，或者因为它们的数量太大而您需要进行首次过滤）？** 如果是，请使用变量排名方法;否则，无论如何都要获得基线结果。
6.  **你需要预测器吗？** 如果不是，请停止
7.  **你怀疑你的数据是“脏”的（有一些无意义的输入模式和/或嘈杂的输出或错误的类标签）？** 如果是，则使用在步骤5中获得的排名最高的变量检测异常值示例作为表示;检查和/或丢弃它们。
8.  **你知道先尝试一下吗？** 如果不是，请使用线性预测器。使用具有“探测”方法的前向选择方法作为停止标准或使用0范数嵌入方法进行比较，遵循步骤5的排序，使用增加的特征子集构建相同性质的预测变量序列。您可以使用较小的子集匹配或改善表现吗？如果是，请尝试使用该子集的非线性预测器。
9.  **你有新的想法，时间，计算资源和足够的例子吗？** 如果是，请比较几种特征选择方法，包括新想法，相关系数，后向选择和嵌入方法。使用线性和非线性预测变量。选择具有模型选择的最佳方法
10.  **您想要一个稳定的解决方案（以提高表现和/或理解）吗？** 如果是，请对您的数据进行子采样并重新分析几个“bootstrap”。

## 进一步阅读

在特定平台上需要有关功能选择的帮助吗？以下是一些可以帮助您快速入门的教程：

*   [如何在Weka](http://machinelearningmastery.com/perform-feature-selection-machine-learning-data-weka/) 中执行特征选择（无代码）
*   [如何使用scikit-learn](http://machinelearningmastery.com/feature-selection-machine-learning-python/) 在Python中执行功能选择
*   [如何用插入符号](http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/)在R中执行特征选择

要深入了解该主题，您可以选择一本关于该主题的专用书籍，例如以下任何一个：

*   [知识发现和数据挖掘的特征选择](http://www.amazon.com/dp/079238198X?tag=inspiredalgor-20)
*   [特征选择的计算方法](http://www.amazon.com/dp/1584888784?tag=inspiredalgor-20)
*   [计算智能和特征选择：粗糙和模糊方法](http://www.amazon.com/dp/0470229756?tag=inspiredalgor-20)
*   [子空间，潜在结构和特征选择：统计和优化视角研讨会](http://www.amazon.com/dp/3540341374?tag=inspiredalgor-20)
*   [特征提取，构造和选择：数据挖掘视角](http://www.amazon.com/dp/0792381963?tag=inspiredalgor-20)

特征选择是特征工程的子主题。您可能希望深入了解帖子中的特征工程：“

您可能希望在帖子中深入了解功能工程：

*   [发现特征工程，如何设计特征以及如何获得它](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/ "Discover Feature Engineering, How to Engineer Features and How to Get Good at It")