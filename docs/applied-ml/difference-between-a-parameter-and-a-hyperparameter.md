# 参数和超参数之间有什么区别？

> 原文： [https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)

当您开始应用机器学习时，这可能会让您感到困惑。

有许多术语可供使用，许多术语可能无法一致使用。如果您来自另一个可能使用与机器学习相同的术语的研究领域，尤其如此，但它们的使用方式不同。

例如：术语“_模型参数_”和“_模型超参数_”。

对这些术语没有明确的定义是初学者的共同斗争，尤其是那些来自统计学或经济学领域的初学者。

在这篇文章中，我们将仔细研究这些术语。

![What is the Difference Between a Parameter and a Hyperparameter?](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/What-is-the-Difference-Between-a-Parameter-and-a-Hyperparameter.jpg)

参数和超参数之间有什么区别？
摄影： [Bruce Guenter](https://www.flickr.com/photos/10154402@N03/7572253262/) ，保留一些权利。

## 什么是模型参数？

模型参数是模型内部的配置变量，其值可以从数据中估算。

*   在进行预测时，模型需要它们。
*   它们的值定义了模型对您的问题的技能。
*   它们是从数据中估算或学习的。
*   它们通常不是由从业者手动设置的。
*   它们通常被保存为学习模型的一部分。

参数是机器学习算法的关键。它们是从历史训练数据中学习的模型的一部分。

在经典机器学习文献中，我们可以将模型视为假设，将参数视为对特定数据集的假设的定制。

通常使用优化算法来估计模型参数，该优化算法是通过可能的参数值的有效搜索的类型。

*   **统计**：在统计中，您可以假设变量的分布，例如高斯分布。高斯分布的两个参数是平均值（ _mu_ ）和标准偏差（ _sigma_ ）。这适用于机器学习，其中这些参数可以从数据估计并用作预测模型的一部分。
*   **编程**：在编程中，您可以将参数传递给函数。在这种情况下，参数是一个函数参数，可以具有一系列值。在机器学习中，您使用的特定模型是函数，需要参数才能对新数据进行预测。

模型是否具有固定或可变数量的参数确定它是否可被称为“_参数_”或“_非参数_”。

模型参数的一些示例包括：

*   人工神经网络中的权重。
*   支持向量机中的支持向量。
*   线性回归或逻辑回归中的系数。

## 什么是模型超参数？

模型超参数是模型外部的配置，其值无法从数据估计。

*   它们通常用于过程中以帮助估计模型参数。
*   它们通常由从业者指定。
*   它们通常可以使用启发式设置。
*   它们经常针对给定的预测建模问题进行调整。

对于给定问题，我们无法知道模型超参数的最佳值。我们可能会使用经验法则，复制用于其他问题的值，或通过反复试验来搜索最佳值。

当针对特定问题调整机器学习算法时，例如当您使用网格搜索或随机搜索时，您正在调整模型的超参数或命令以发现导致最熟练的模型参数预测。

> 许多模型具有不能从数据直接估计的重要参数。例如，在K-最近邻分类模型中......这种类型的模型参数被称为调整参数，因为没有可用于计算适当值的分析公式。

- 第64-65页， [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，2013

模型超参数通常被称为模型参数，这可能使事情变得混乱。克服这种混乱的一个好的经验法则如下：

**如果你必须手动指定模型参数，那么
它可能是一个模型超参数。**

模型超参数的一些示例包括：

*   训练神经网络的学习率。
*   支持向量机的C和sigma超参数。
*   k-最近邻居中的k。

## 进一步阅读

*   维基百科上的 [Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter)
*   [什么是机器学习中的超参数？ Quora上的](https://www.quora.com/What-are-hyperparameters-in-machine-learning)
*   [模型超参数和模型参数有什么区别？ StackExchange上的](https://datascience.stackexchange.com/questions/14187/what-is-the-difference-between-model-hyperparameters-and-model-parameters)
*   [什么被认为是超参数？](https://www.reddit.com/r/MachineLearning/comments/40tfc4/what_is_considered_a_hyperparameter/) 在Reddit上

## 摘要

在这篇文章中，您发现了清晰的定义以及模型参数和模型超参数之间的区别。

总之，模型参数是根据数据自动估算的，模型超参数是手动设置的，并在过程中用于帮助估计模型参数。

模型超参数通常被称为参数，因为它们是机器学习的必须手动设置和调整的部分。

这篇文章是否帮助您消除了困惑？
请在下面的评论中告诉我。

是否存在您仍不确定的模型参数或超参数？
在评论中发布它们，我会尽力帮助进一步澄清问题。