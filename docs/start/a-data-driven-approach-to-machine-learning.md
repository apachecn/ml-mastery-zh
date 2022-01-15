# 一种选择机器学习算法的数据驱动方法

> 原文： [https://machinelearningmastery.com/a-data-driven-approach-to-machine-learning/](https://machinelearningmastery.com/a-data-driven-approach-to-machine-learning/)

### _ 如果您知道要使用哪种算法或算法配置，
您不需要使用机器学习 _

没有最好的机器学习算法或算法参数。

我想要治愈你这种类型的银弹心态。

我每天都会看到很多这样的问题：

*   _ 哪种机器学习算法最好？_
*   _ 机器学习算法和问题之间的映射是什么？_
*   _ 机器学习算法的最佳参数是什么？_

这些问题有一种模式。

您通常不会也无法事先知道这些问题的答案。你必须通过实证研究发现它。

有一些广泛的笔刷启发式方法可以回答这些问题，但如果您希望从算法或问题中获得最大收益，即使是这些也可能会让您失望。

在这篇文章中，我想鼓励你摆脱这种心态，掌握一种数据驱动的方法，这种方法将改变你接近机器学习的方式。

[![data driven approach to machine learning](img/27fdd33b6b9f0f76050031b085e7dbeb.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/09/data-driven-approach-to-machine-learning.jpg)

数据驱动的机器学习方法
摄影： [James Cullen](https://www.flickr.com/photos/tlgjaymz/467195251) ，保留一些权利

## 最佳机器学习算法

一些算法比其他算法具有更多的“_ 功率 _”。它们是非参数的或高度灵活的和自适应的，或高度自我调整或以上所有。

通常，这种功率的实现成本很高，需要非常大的数据集，有限的可扩展性，或者可能导致过拟合的大量系数。

随着更大的数据集，人们对更简单的扩展和表现良好的方法产生了兴趣。

哪个是最好的算法，你应该总是尝试并花费最多时间学习的算法？

我可以丢掉一些名字，但最明智的答案是“_ 无 _”和“_ 全部 _”。

### 没有最好的机器学习算法

你无法知道 _ 先验 _ 哪种算法最适合你的问题。

再次阅读上述内容。默想吧。

默想吧。

*   您可以应用自己喜欢的算法。
*   您可以应用书籍或纸张中推荐的算法。
*   您可以应用现在赢得最多 Kaggle 比赛的算法。
*   您可以应用最适合您的测试装备，基础架构，数据库或其他任何方法的算法。

这些都是偏见。

他们认为节省时间是捷径。事实上，有些可能是非常有用的捷径，但哪些是捷径？

根据定义，偏差将限制您可以实现的解决方案，您可以实现的准确率，以及最终可能产生的影响。

## 算法映射到问题

存在[一般类别的问题](http://machinelearningmastery.com/practical-machine-learning-problems/ "Practical Machine Learning Problems")，分类和回归等监督问题以及流形学习和聚类等无监督问题。

在计算机视觉，自然语言处理和语音处理等机器学习的子领域中存在这些问题的更具体的实例。我们也可以采用另一种方式，更抽象，并将所有这些问题视为函数逼近和函数优化的实例。

您可以将算法映射到问题类，例如，有一些算法可以处理监督回归问题和监督分类问题，以及两种类型的问题。

您还可以构建[算法目录](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/ "A Tour of Machine Learning Algorithms")，这可能会激发您尝试使用哪种算法。

您可以针对问题竞争算法并报告结果。有时这被称为烘烤，并且在一些会议过程中很受欢迎，用于呈现新算法。

### 算法结果的有限可转移性

一般来说，赛车算法是[反智能](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.73.6198)。它很少科学严谨（苹果到苹果）。

赛车算法的一个关键问题是您无法轻松地将结果从一个问题转移到另一个问题。如果你认为这个陈述是正确的，那么阅读论文和博客中的算法竞赛并不会告诉你哪个算法可以尝试解决你的问题。

如果算法`A`在问题`X`上杀死算法`B`，那么它会告诉你什么算法`A`和`B`问题`Y`？你必须努力解决问题`X`和`Y`。它们是否具有被研究中的算法利用的相同或相似的属性（属性，属性分布，功能形式）？这是一些艰苦的工作。

我们对一台机器学习算法比另一台机器学习算法效果更好有一个细粒度的理解。

## 最佳算法参数

机器学习算法是参数化的，因此您可以根据您的问题定制其行为和结果。

问题在于“_ 如何 _”进行剪裁很少（如果有的话）解释。通常，即使算法开发人员自己也很难理解。

通常，具有随机元素的机器学习算法是复杂系统，因此必须进行研究。第一顺序 - 可以描述参数对复杂系统的影响。如果是这样，您可能有一些关于如何将算法配置为系统的启发式方法。

这是第二个订单，它对你的结果有什么影响，这是不知道的。有时你可以在一般性方面谈论参数对作为一个系统的算法的影响，以及它如何转化为问题类，通常不会。

### 没有最佳算法参数

新的算法配置集本质上是新的算法实例，可以帮助您挑战问题（尽管在他们可以实现的结果中相对受限或类似）。

您无法知道问题的最佳算法参数 _ 先验 _。

*   您可以使用开创性论文中使用的参数。
*   您可以使用书中的参数。
*   你可以使用“_ 我怎么做 _”kaggle post 中列出的参数。

好的经验法则。对？也许，也许不是。

## 数据驱动方法

我们不需要陷入绝望之中。我们成为科学家。

您有偏见可以缩短算法选择和算法参数选择的决策。我们认为，在许多情况下，它们可以很好地为您服务。

我希望您对此提出质疑，考虑放弃启发式和最佳实践，并采用数据驱动的算法选择方法。

而不是选择您喜欢的算法，尝试 10 或 20 算法。

在那些表现出更好的表现，稳健性，速度或任何你最感兴趣的问题的人身上加倍。

网格搜索数十，数百或数千种参数组合，而不是选择通用参数。

成为客观科学家，留下轶事并研究来自您的问题领域的复杂学习系统和数据观察的交集。

### 行动中的数据驱动方法

这是一种强大的方法，需要较少的前期知识，但需要更多的后端计算和实验。

因此，您很可能需要使用较小的数据集样本，以便快速获得结果。您将需要一个完全可以信赖的测试工具。

旁注：您如何完全信任您的测试工具？

您可以通过以数据驱动的方式选择[测试选项](http://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/ "How To Choose The Right Test Options When Evaluating Machine Learning Algorithms")来建立信任，这使您可以确信所选配置是可靠的。估计方法的类型（分裂，增强，k 折交叉验证等）及其配置（k 的大小等）。

### 快速稳健的结果

你会很快得到好结果。

如果随机森林是您最喜欢的算法，您可能需要花费数天或数周的时间徒劳地尝试从算法中获取最大的问题，这可能不适合首先使用该方法。使用数据驱动的方法，您可以尽早折扣（相对）表现不佳的人。你可以快速失败。

它不需要依赖于偏见和最喜欢的算法和配置。获得良好而强大的结果是一项艰苦的工作。

结果是你不再关心算法炒作，它只是包含在[现场检查套件](http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")中的另一种方法。你不再担心你是否错过了不使用算法`X`或`Y`或配置`A`或`B`（ []害怕失去](http://machinelearningmastery.com/the-best-machine-learning-algorithm/ "The Best Machine Learning Algorithm")），你把它们混在一起。

### 利用自动化

数据驱动的方法是搜索问题。您可以利用自动化。

在开始之前，您可以编写可重复使用的脚本来搜索问题的最可靠测试工具。没有更多的临时猜测。

您可以编写可重用的脚本，在各种库和实现中自动尝试 10,20,100 算法。没有更喜欢的算法或库。

不同算法之间的界限消失了。新参数配置是一种新算法。您可以将可重复使用的脚本写入网格或随机搜索每个算法以真正对其功能进行采样。

在前面添加特征工程，以便数据上的每个“_ 视图 _”是一个新的问题，可以对抗算法。最后通过 Bolt-on 集合来组合部分或全部结果（元算法）。

[我一直在这个兔子洞](http://machinelearningmastery.com/the-seductive-trap-of-black-box-machine-learning/ "The Seductive Trap of Black-Box Machine Learning")。这是一种强大的思维方式，它可以获得强大的结果。

## 摘要

在这篇文章中，我们研究了算法和算法参数选择的常见启发式和最佳实践方法。

我们认为这种方法会导致我们的思维局限。当没有这样的东西时，我们渴望获得银弹通用最佳算法和最佳算法配置。

*   没有最好的通用机器学习算法。
*   没有最好的通用机器学习算法参数。
*   算法从一个问题到另一个问题的能力的可转移性是值得怀疑的。

解决方案是成为科学家并研究我们问题的算法。

我们必须采用数据驱动的问题，检查算法，网格搜索算法参数，并快速找到能够可靠，快速地产生良好结果的方法。