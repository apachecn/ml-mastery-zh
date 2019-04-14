# 机器学习算法之旅

> 原文： [https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

在这篇文章中，我们将介绍最流行的机器学习算法。

在现场浏览主要算法以了解可用的方法很有用。

有许多算法可用，当算法名称被抛出时，它可能会感到压倒性的，并且您应该只知道它们是什么以及它们适合的位置。

我想给你两种方法来思考和分类你可能在现场遇到的算法。

*   第一个是**学习风格**的算法分组。
*   第二种是通过**相似性**在形式或功能上的一组算法（如将相似的动物组合在一起）。

这两种方法都很有用，但我们将重点关注通过相似性对算法进行分组，并继续浏览各种不同的算法类型。

阅读本文后，您将更好地了解最受欢迎的监督学习机器学习算法及其相关性。

![Ensemble Learning Method](img/c45920104e4e7e28892d5a52bbdeb900.jpg)

最合适的线条集合的一个很酷的例子。弱成员是灰色的，组合预测是红色的。
维基百科的情节，在公共领域获得许可。

## 学习风格分组的算法

算法可以基于与体验或环境的交互或我们想要调用输入数据的任何内容来对问题进行建模。

在机器学习和人工智能教科书中首先考虑算法可以采用的学习风格。

算法可以有一些主要的学习风格或学习模型，我们将在这里介绍它们适合的算法和问题类型的几个例子。

这种分类法或组织机器学习算法的方法很有用，因为它会强制您考虑输入数据和模型准备过程的角色，并选择最适合您的问题的方法以获得最佳结果。

让我们来看看机器学习算法中的三种不同学习方式：

### **1.监督学习**

![Supervised Learning Algorithms](img/95241c40aaa81e72b7754289befd763c.jpg)输入数据称为训练数据，并且具有已知的标签或结果，例如垃圾邮件/非垃圾邮件或一次的股票价格。

通过训练过程准备模型，其中需要进行预测并在这些预测错误时进行校正。训练过程继续，直到模型在训练数据上达到所需的准确度。

示例问题是分类和回归。

示例算法包括Logistic回归和Back Propagation神经网络。

### **2.无监督学习**

![Unsupervised Learning Algorithms](img/7d8ebc58c92c9df80e23693eb169632b.jpg)输入数据未标记，并且没有已知结果。

通过推导输入数据中存在的结构来准备模型。这可能是提取一般规则。可以通过数学过程系统地减少冗余，或者可以通过相似性来组织数据。

示例问题是聚类，降维和关联规则学习。

示例算法包括：Apriori算法和k-Means。

### **3.半监督学习**

[![Semi-supervised Learning Algorithms](img/f12c7caccaa57cf156e9ab2539c0c74b.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2013/11/Semi-supervised-Learning-Algorithms.png) 输入数据是标记和未标记示例的混合。

存在期望的预测问题，但模型必须学习组织数据以及进行预测的结构。

Example problems are classification and regression.

示例算法是对其他灵活方法的扩展，这些方法对如何对未标记数据建模进行假设。

### 概观

在处理数据以模拟业务决策时，您通常使用有监督和无监督的学习方法。

目前的一个热门话题是在图像分类等领域中的半监督学习方法，其中存在具有极少数标记示例的大型数据集。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 由相似性分组的算法

算法通常根据其功能（它们如何工作）的相似性进行分组。例如，基于树的方法和神经网络启发的方法。

我认为这是分组算法最有用的方法，这是我们将在这里使用的方法。

这是一种有用的分组方法，但并不完美。还有一些算法可以很容易地融入多个类别，例如学习向量量化，它既是神经网络启发方法，也是基于实例的方法。还有一些类别具有描述问题的相同名称和算法类，例如回归和聚类。

我们可以通过两次列出算法或选择主观上“最佳”拟合的组来处理这些情况。我喜欢后一种不重复算法来保持简单的方法。

在本节中，我列出了许多流行的机器学习算法，这些算法按照我认为最直观的方式进行分组。该列表在组或算法中并非详尽无遗，但我认为它具有代表性，对您了解土地的位置非常有用。

**请注意**：对于用于分类和回归的算法存在强烈偏见，这是您将遇到的两种最常见的监督机器学习问题。

如果您知道未列出的算法或一组算法，请将其放在评论中并与我们分享。让我们潜入。

### 回归算法

![Regression Algorithms](img/0b2b976f6a62542ff53a1b233976829a.jpg)回归涉及对变量之间的关系进行建模，这些变量使用模型预测中的误差度量进行迭代求精。

回归方法是统计学的主力，并已被纳入统计机器学习。这可能会令人困惑，因为我们可以使用回归来指代问题类和算法类。实际上，回归是一个过程。

最流行的回归算法是：

*   普通最小二乘回归（OLSR）
*   线性回归
*   Logistic回归
*   逐步回归
*   多元自适应回归样条（MARS）
*   局部估计的散点图平滑（LOESS）

### 基于实例的算法

![Instance-based Algorithms](img/ecbc6f66cdc94fab00ae8f9b81668e1d.jpg)基于实例的学习模型是一个决策问题，其中包含被认为对模型重要或需要的训练数据的实例或示例。

这些方法通常建立示例数据的数据库，并使用相似性度量将新数据与数据库进行比较，以便找到最佳匹配并进行预测。出于这个原因，基于实例的方法也称为赢者通吃方法和基于记忆的学习。重点放在实例之间使用的存储实例和相似性度量的表示上。

最流行的基于实例的算法是：

*   k-最近邻（kNN）
*   学习向量量化（LVQ）
*   自组织地图（SOM）
*   本地加权学习（LWL）

### 正则化算法

![Regularization Algorithms](img/72f51738d210fbe5b403f6c8df5ff0fb.jpg)对另一种方法（通常是回归方法）的扩展，根据其复杂性惩罚模型，有利于更简单的模型，这些模型也更好地推广。

我在这里单独列出了正则化算法，因为它们是流行的，强大的，并且通常对其他方法进行简单的修改。

最流行的正则化算法是：

*   岭回归
*   最小绝对收缩和选择算子（LASSO）
*   弹性网
*   最小角度回归（LARS）

### 决策树算法

![Decision Tree Algorithms](img/b67e7493eea15da38ea6e85ce0419020.jpg)决策树方法构建基于数据中属性的实际值做出的决策模型。

决策在树结构中进行分叉，直到对给定记录做出预测决定。针对分类和回归问题的数据训练决策树。决策树通常快速准确，是机器学习的最佳选择。

最流行的决策树算法是：

*   分类和回归树（CART）
*   迭代Dichotomiser 3（ID3）
*   C4.5和C5.0（强大方法的不同版本）
*   卡方自动交互检测（CHAID）
*   决策树桩
*   M5
*   条件决策树

### 贝叶斯算法

![Bayesian Algorithms](img/97a20e679d50588df31bc4c8c96ea2e4.jpg)贝叶斯方法是那些明确将贝叶斯定理应用于分类和回归等问题的方法。

最流行的贝叶斯算法是：

*   朴素贝叶斯
*   高斯朴素贝叶斯
*   多项朴素贝叶斯
*   平均一依赖估计（AODE）
*   贝叶斯信念网络（BBN）
*   贝叶斯网络（BN）

### 聚类算法

![Clustering Algorithms](img/10f6e2ade3232933045aef9ce8c3982d.jpg)聚类与回归一样，描述了问题的类和方法的类。

聚类方法通常由建模方法组织，例如基于质心和层级。所有方法都涉及使用数据中的固有结构来最好地将数据组织成具有最大共性的组。

最流行的聚类算法是：

*   K-均值
*   K-中位数
*   期望最大化（EM）
*   分层聚类

### 关联规则学习算法

![Assoication Rule Learning Algorithms](img/83ede7dcd3b335f572a21b6b93005192.jpg)关联规则学习方法提取最能解释数据中变量之间观察到的关系的规则。

这些规则可以发现可以被组织利用的大型多维数据集中的重要且商业上有用的关联。

最流行的关联规则学习算法是：

*   Apriori算法
*   Eclat算法

### 人工神经网络算法

![Artificial Neural Network Algorithms](img/69c47f2c9e3fe01e93eac787dd50bd01.jpg)人工神经网络是受生物神经网络的结构和/或功能启发的模型。

它们是一类模式匹配，通常用于回归和分类问题，但实际上是一个庞大的子字段，由各种问题类型的数百种算法和变体组成。

请注意，由于该领域的大量增长和普及，我已将深度学习与神经网络分开。在这里，我们关注更经典的方法。

最流行的人工神经网络算法是：

*   感知
*   反向传播
*   Hopfield网络
*   径向基函数网络（RBFN）

### 深度学习算法

![Deep Learning Algorithms](img/b8696c1f19690778d1be012423c85e62.jpg)深度学习方法是人工神经网络的一种现代更新，利用丰富的廉价计算。

他们关注构建更大更复杂的神经网络，如上所述，许多方法都涉及半监督学习问题，其中大数据集包含非常少的标记数据。

最流行的深度学习算法是：

*   深玻尔兹曼机（DBM）
*   深信仰网络（DBN）
*   卷积神经网络（CNN）
*   堆叠式自动编码器

### 降维算法

![Dimensional Reduction Algorithms](img/0083064b644be69105250e3a9250be36.jpg)与聚类方法一样，维数减少寻求并利用数据中的固有结构，但在这种情况下以无监督的方式或使用较少信息来汇总或描述数据。

这对于可视化维度数据或简化数据是有用的，然后可以在监督学习方法中使用这些数据。许多这些方法可以适用于分类和回归。

*   主成分分析（PCA）
*   主成分回归（PCR）
*   偏最小二乘回归（PLSR）
*   Sammon Mapping
*   多维缩放（MDS）
*   投射追踪
*   线性判别分析（LDA）
*   混合判别分析（MDA）
*   二次判别分析（QDA）
*   灵敏判别分析（FDA）

### 乐团算法

![Ensemble Algorithms](img/a920b74788a68c489876c7b70da1f86b.jpg)集合方法是由多个较弱的模型组成的模型，这些模型是独立训练的，其预测以某种方式组合以进行整体预测。

要将哪些类型的弱学习器结合起来以及将它们结合起来的方式付出了很多努力。这是一种非常强大的技术，因此非常受欢迎。

*   推进
*   自举聚合（套袋）
*   AdaBoost的
*   堆叠泛化（混合）
*   梯度增压机（GBM）
*   梯度升压回归树（GBRT）
*   随机森林

### 其他算法

许多算法都没有涵盖。

例如，支持向量机会进入哪个组？它自己的？

我没有在机器学习过程中涵盖专业任务的算法，例如：

*   特征选择算法
*   算法精度评估
*   绩效衡量标准

我也没有涵盖机器学习专业子领域的算法，例如：

*   计算智能（进化算法等）
*   计算机视觉（CV）
*   自然语言处理（NLP）
*   推荐系统
*   强化学习
*   图形模型
*   和更多…

这些可能会在未来的帖子中出现

## 进一步阅读

这次机器学习算法之旅旨在向您概述那里的内容以及如何将算法相互关联的一些想法。

我收集了一些资源供您继续阅读算法。如果您有具体问题，请发表评论。

### 其他算法列表

如果您有兴趣，还有其他很棒的算法列表。以下是几个手工选择的例子。

*   [机器学习算法列表](http://en.wikipedia.org/wiki/List_of_machine_learning_algorithms)：在维基百科上。虽然很广泛，但我没有发现这个列表或算法的组织特别有用。
*   [机器学习算法类别](http://en.wikipedia.org/wiki/Category:Machine_learning_algorithms)：也在维基百科上，比上面的维基百科上面的列表稍微有用。它按字母顺序组织算法。
*   [CRAN任务视图：机器学习＆amp;统计学习](http://cran.r-project.org/web/views/MachineLearning.html)：R中每个机器学习包支持的所有包和所有算法的列表。让您有一种基础的感觉，即每天的内容以及人们用于分析的内容。
*   [数据挖掘中的十大算法](http://www.cs.uvm.edu/~icdm/algorithms/index.shtml)：[已发表文章](http://link.springer.com/article/10.1007/s10115-007-0114-2)，现在是[书籍](http://www.amazon.com/dp/1420089641?tag=inspiredalgor-20)（Affiliate Link）关于最流行的数据挖掘算法。另一种基础和不那么压倒性的方法，你可以去深入学习。

### 如何研究机器学习算法

算法是机器学习的重要组成部分。这是我热衷的主题，并在这个博客上写了很多。以下是您可能感兴趣的一些手工选择的帖子以供进一步阅读。

*   [如何学习任何机器学习算法](http://machinelearningmastery.com/how-to-learn-a-machine-learning-algorithm/)：一种系统方法，您可以使用“算法描述模板”来研究和理解任何机器学习算法（我用这种方法编写[我的第一本书](http://cleveralgorithms.com/nature-inspired/index.html) ]）。
*   [如何创建机器学习算法的目标列表](http://machinelearningmastery.com/create-lists-of-machine-learning-algorithms/)：如何创建自己的机器学习算法系统列表，以便开始研究下一个机器学习问题。
*   [如何研究机器学习算法](http://machinelearningmastery.com/how-to-research-a-machine-learning-algorithm/)：一种系统方法，可用于研究机器学习算法（与上面列出的模板方法协同工作）。
*   [如何调查机器学习算法行为](http://machinelearningmastery.com/how-to-investigate-machine-learning-algorithm-behavior/)：一种方法，您可以通过创建和执行非常小的研究来了解机器学习算法的工作原理。研究不仅仅适用于学者！
*   [如何实现机器学习算法](http://machinelearningmastery.com/how-to-implement-a-machine-learning-algorithm/)：从头开始实现机器学习算法的过程和提示和技巧。

### 如何运行机器学习算法

有时你只想潜入代码。下面是一些可用于运行机器学习算法的链接，使用标准库对其进行编码或从头开始实现。

*   [如何开始使用R中的机器学习算法](http://machinelearningmastery.com/how-to-get-started-with-machine-learning-algorithms-in-r/)：链接到该站点上的大量代码示例，演示了R中的机器学习算法。
*   [机器学习算法scikit-learn中的秘籍](http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)：Python代码示例的集合，演示了如何使用scikit-learn创建预测模型。
*   [如何在Weka中运行您的第一个分类器](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/)：在Weka中运行您的第一个分类器的教程（**无需代码！**）。

## 最后的话

我希望你发现这次旅行很有用。

如果您对如何改进算法之旅有任何疑问或想法，请发表评论。

**更新＃1** ：继续关于HackerNews 和 [reddit](http://www.reddit.com/r/programming/comments/267zmd/a_tour_of_machine_learning_algorithms/) 的[讨论。](https://news.ycombinator.com/item?id=7783550)

**更新＃2** ：我添加了更多资源和更多算法。我还添加了一个方便的思维导图，您可以下载（见上文）。