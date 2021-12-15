# 应用机器学习的杀器：XGBoost 温和简介

> 原文： [https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

XGBoost 是一种算法库，近年来在应用机器学习和 Kaggle 竞赛中占据统治地位，它专长于处理结构化数据或表格数据。

XGBoost 是为速度和表现而设计的一种梯度提升决策树方法。

在这篇文章中，您将轻松了解 XGBoost 的入门信息，并知晓它究竟是什么，源自何处，以及如何学习它的更多信息。

阅读之后您会学习到：

*   什么是 XGBoost 以及本项目所达成的目标。
*   为什么 XGBoost 必须与您现有的机器学习工具包作区分。
*   在您的下一个机器学习项目中，您可以从哪里获取使用 XGBoost 的更多信息。

让我们开始吧。

![A Gentle Introduction to XGBoost for Applied Machine Learning](img/2bb9abbef6f07700041dc880c95acda8.jpg)

应用机器学习的杀器： XGBoost 简介。照片由
[Sigfrid Lundberg](https://www.flickr.com/photos/sigfridlundberg/14945045482/) 拍摄，保留部分权利。

## 什么是 XGBoost？

XGBoost 的名字源自 e **X** treme **G** radient **B** oosting （极限梯度提升）。

> 其实 xgboost 实际上是在致力于将提升树算法对计算资源的利用推至工程极限。这也是为什么有许多人会使用 xgboost 的原因。

- Tianqi Chen（陈天奇）对 Quora 问题“ [R gbm（梯度提升机）和 xgboost（极限梯度提升）有什么区别？](https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting) “的回答。

它是 [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/) 创建的一种梯度提升机实现，现在有许多开发人员在为这个项目做贡献。它属于分布式机器学习社区（[DMLC](http://dmlc.ml/)） 宽泛范畴中的一种工具。Chen 同时也是流行的 [mxnet 深度学习库](https://github.com/dmlc/mxnet)创建者。

Tianqi Chen 在[ XGBoost 的背后故事与经验](http://homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-the-evolution-of-xgboost.html)中提供了关于 XGBoost 演进的简短而有趣的背景故事。

XGBoost 定义了一个软件库，您可以在您的电脑上下载和安装，有多种接口方式可以调用。具体来说，XGBoost 支持以下主要接口：

*   命令行界面（CLI）。
*   C++（编写 XGBoost 库的语言）。
*   Python 界面以及作为 scikit-learn 的一个模型。
*   R 接口以及作为 caret 包中的模型。
*   Julia。
*   Java 和 JVM 语言，例如 Scala，以及像 Hadoop 这样的平台。

## XGBoost 的特点

XGBoost 库高度专注于计算速度和模型表现，因此几乎没有冗余功能。不过它仍然提供了许多高级功能。

### 模型的特点

XGBoost 模型支持 scikit-learn 和 R 的实现，并且新增了正则化等功能。它支持三种主要的梯度提升形式：

*   **Gradient Boosting** 算法，也称为具有学习率的梯度提升机。
*   对行、列以及分割列进行子采样的**随机梯度提升**。
*   L1 和 L2 正则化的**正则化梯度提升**。

### 系统的特点

XGBoost 库提供了丰富的计算环境，包括而不限于：

*   **在训练期间使用所有 CPU 内核并行化的进行树构建**。
*   **分布式计算**可在一组计算机集群上训练超大型模型。
*   **核外计算（Out-Of-Core）（外扩存储计算）**适用于无法装载入内存的超大型数据集。
*   数据结构的**缓存优化**以及充分利用硬件的算法。

### 算法的特点

算法的实现致力于提高计算时间和内存资源的使用效率。设计目标就是充分利用可用资源来训练模型。一些关键的算法实现特点包括：

*   具有自动处理缺失数据值的**稀疏感知**能力。
*   具有支持树构建并行化的**块结构**。
*   具有**继续训练**能力，以便您可以进一步根据新数据提升已经训练过的模型。

XGBoost 是免费的开源软件，可在 Apache-2 许可范围使用。

## 为什么要使用 XGBoost？

使用 XGBoost 的两个原因也是本项目的两个目标：

1.  执行速度。
2.  模型表现。

### 1\. XGBoost 执行速度

通常，XGBoost 相当快速。与梯度提升的其他实现方法相比，真的很快。

[Szilard Pafka](https://www.linkedin.com/in/szilard) 进行了一些客观的基准测试，比较了 XGBoost 与其它梯度提升实现方法以及 bagged 决策树方法。他在 2015 年 5 月的博客文章“[随机森林方法的基准测试](http://datascience.la/benchmarking-random-forest-implementations/)”中展示了他的结果。

他同时在 [GitHub](https://github.com/szilard/benchm-ml) 上提供了所有代码以及附有更多硬核数字的拓展报告。

![Benchmark Performance of XGBoost](img/6198624aaad562ea913e9dd529a072e5.jpg)

XGBoost 的基准表现，引自[随机森林方法的基准测试](http://datascience.la/benchmarking-random-forest-implementations/)。

他的结果表明 XGBoost 在 R、Python、Spark 和 H2O 的实现几乎总是比其它基准测试的实现方法更快。

在他的实验中，他评论说：

> 我也比较了 xgboost，它是一个可以构建随机森林的流行提升库。它速度快、内存使用效率高，同时具有高精度。

- Szilard Pafka，[可以构建随机森林](http://datascience.la/benchmarking-random-forest-implementations/)。

### 2\. XGBoost 的模型表现

XGBoost 在分类和回归预测性建模问题上对于有着结构化或表格化形式的数据集占据着统治地位。

有证据表明，它是 Kaggle 数据科学平台竞赛获胜者的首选算法。

这里给出一个不完整的第一、第二和第三名竞赛获胜者名单，标题取名为： [XGBoost：机器学习挑战赛获胜解决方案](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions)。

为了使这一点更加具象，下面是来自 Kaggle 比赛获胜者的一些有启示的见解：

> 作为 Kaggle 比赛的赢家，并且仍在增长获胜数字，XGBoost 再次向我们展示了它是一个值得留在您工具箱中的全面算法。

- [Dato 获奖者访谈：第 1 名 Mad Professors](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/)

> 如果感到困惑，不知道作何选择，请使用 xgboost。

- [Avito 获奖者访谈：第 1 名，Owen Zhang](http://blog.kaggle.com/2015/08/26/avito-winners-interview-1st-place-owen-zhang/)

> 我喜欢让单一模特表现的更好，而我最好的单一模特是 XGBoost，它可以自己获得第 10 名。

- [Caterpillar 获奖者访谈：第 1 名](http://blog.kaggle.com/2015/09/22/caterpillar-winners-interview-1st-place-gilberto-josef-leustagos-mario/)

> 我只用过 XGBoost。

- [Liberty Mutual Property Inspection，获奖者访谈：第 1 名，Qingchen Wang](http://blog.kaggle.com/2015/09/28/liberty-mutual-property-inspection-winners-interview-qingchen-wang/)

> 我唯一用过的有监督学习方法是梯度提升，通过优秀的 xgboost 实现。

- [Recruit Coupon Purchase 获奖者访谈：第 2 名，Halla Yang](http://blog.kaggle.com/2015/10/21/recruit-coupon-purchase-winners-interview-2nd-place-halla-yang/)

## XGBoost 使用什么算法？

XGBoost 库执行[梯度提升决策树算法](https://en.wikipedia.org/wiki/Gradient_boosting)。

该算法有许多不同的名称，例如梯度提升（gradient boosting），多重加性回归树（multiple additive regression trees），随机梯度提升（stochastic gradient boosting）或梯度提升机（gradient boosting machines）。

提升是一种调和技术（ensemble technique），它可以添加新模型以纠正现有模型所产生的误差。在这个过程中，模型会被逐步添加，直到不能再进一步改进。一个流行的例子是 [AdaBoost 算法](http://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/)，它对很难预测的数据点进行加权。

梯度提升是一种方法，其中创建新模型以预测先前模型的残差或误差，然后将其加在一起以进行最终预测。它被称为梯度提升，因为它使用梯度下降算法来最小化添加新模型时的损失。

这种方法支持回归和分类预测性建模问题。

关于提升和梯度提升的更多信息，请参阅 Trevor Hastie 关于[梯度提升机器学习](https://www.youtube.com/watch?v=wPqtzj5VZus)的演讲。

&lt;iframe allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/wPqtzj5VZus?feature=oembed" width="500"&gt;&lt;/iframe&gt;

## 官方 XGBoost 资源

关于 XGBoost 的最佳信息来源是项目的[官方 GitHub 仓库。](https://github.com/dmlc/xgboost)

从那里，您可以访问[议题追踪（Issue Tracker）](https://github.com/dmlc/xgboost/issues)以及[用户组（User Group）](https://groups.google.com/forum/#!forum/xgboost-user/)，它们可用于提问和报告 bug。

[Awesome XGBoost 页面](https://github.com/dmlc/xgboost/tree/master/demo)是一个很好的资源库，配有示例代码和帮助信息。

此外还有一个[官方文档页面](https://xgboost.readthedocs.io/en/latest/)，其中包含一系列不同语言的入门指南，教程，操作向导等。

关于 XGBoost 有许多更正式的论文值得阅读，可以从中获取更多关于这个库的使用背景：

*   [Higgs Boson Discovery with Boosted Trees](http://jmlr.org/proceedings/papers/v42/chen14.pdf) ，2014。
*   [XGBoost：A Scalable Tree Boosting System](http://arxiv.org/abs/1603.02754)，2016。

## XGBoost 的演讲

当开始使用像 XGBoost 这样的新工具时，在深入研究代码之前，先回顾一下有关该主题的一些演讲会很有帮助。

### XGBoost：A Scalable Tree Boosting System

XGBoost 库的创建者 Tianqi Chen 2016 年 6 月在洛杉矶 Data Science Group 进行了一次题为“ [XGBoost：A Scalable Tree Boosting System](https://www.youtube.com/watch?v=Vly8xGnNiWs)”的演讲。

&lt;iframe allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/Vly8xGnNiWs?feature=oembed" width="500"&gt;&lt;/iframe&gt;

您可以在此处读到他演讲中的幻灯片：

&lt;iframe allowfullscreen="true" allowtransparency="true" frameborder="0" height="345" id="talk_frame_345261" mozallowfullscreen="true" src="//speakerdeck.com/player/5c6dab45648344208185d2b1ab4fdc95" style="border:0; padding:0; margin:0; background:transparent;" webkitallowfullscreen="true" width="500"&gt;&lt;/iframe&gt;

在 [DataScience LA blog](http://datascience.la/xgboost-workshop-and-meetup-talk-with-tianqi-chen/)可以找到更多信息。

### XGBoost：eXtreme Gradient Boosting

2015 年 12 月一位 XGBoost 的 R 语言接口贡献者在纽约 Data Science Academy 发表了题为“ [XGBoost: eXtreme Gradient Boosting](https://www.youtube.com/watch?v=ufHo8vbk6g4)”的演讲。

&lt;iframe allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/ufHo8vbk6g4?feature=oembed" width="500"&gt;&lt;/iframe&gt;

您也可以在此处读到他演讲中的幻灯片：

&lt;iframe allowfullscreen="" frameborder="0" height="356" marginheight="0" marginwidth="0" scrolling="no" src="https://www.slideshare.net/slideshow/embed_code/key/lhcV8LfZ8RfrG" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" width="427"&gt;&lt;/iframe&gt;

**[Xgboost](https://www.slideshare.net/ShangxuanZhang/xgboost-55872323 "Xgboost")** from **[Vivian Shangxuan Zhang](http://www.slideshare.net/ShangxuanZhang)**

有关此次演讲的更多信息，请访问[NYC Data Science Academy blog](http://blog.nycdatascience.com/faculty/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author-3/)。

## 安装 XGBoost

在[XGBoost 文档网站](http://xgboost.readthedocs.io/en/latest/build.html)上有综合的安装指南。

它涵盖了 Linux，Mac OS X 和 Windows 的安装指南。

它也包括了在 R 和 Python 等平台上的安装向导。

### R 中的 XGBoost

如果您是 R 语言用户，最好的入门位置是 [xgboost 包的 CRAN 页面](https://cran.r-project.org/web/packages/xgboost/index.html)。

在此页面中，您可以访问 [R vignette Package'xgboost'](https://cran.r-project.org/web/packages/xgboost/xgboost.pdf) （pdf）。

此页面还链接了一些优秀的 R 教程，以帮助您入门：

*   [Discover your data]](https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html)
*   [XGBoost Presentation](https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html)
*   [xgboost：eXtreme Gradient Boosting](https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf) (pdf)

还有官方的 [XGBoost R Tutorial](http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html)和[Understand your dataset with XGBoost](http://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html) 。

### Python 中的 XGBoost

安装说明可在 XGBoost 安装指南的 [Python section of the XGBoost installation guide](https://github.com/dmlc/xgboost/blob/master/doc/build.md#python-package-installation)部分找到。

官方 [Python 包简介](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)是在 Python 中使用 XGBoost 最好的起步位置。

若想要快速使用，您可以输入：

```py
sudo pip install xgboost
```

在 [XGBoost Python Feature Walkthrough](https://github.com/tqchen/xgboost/tree/master/demo/guide-python)中，有一个很好的 Python 范例源代码列表。

## 总结

在本文中，您了解了应用机器学习的 XGBoost 方法。

您学到了：

*   XGBoost 是一个用于开发快速和高表现梯度提升树模型的库。
*   XGBoost 目前在一系列困难的机器学习任务中都达到最佳表现。
*   可以在命令行，Python 和 R 中使用这个库，以及如何开始使用。

你用过 XGBoost 吗？请在下面的评论中分享您的经验。

您对 XGBoost 或这篇文章有任何疑问吗？请在下面的评论中提出您的问题，我会尽力回答。
