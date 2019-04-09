# 浅谈机器学习的梯度提升算法

> 原文： [https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

梯度提升是构建预测模型的最强大技术之一。

在这篇文章中，您将发现梯度提升机器学习算法，并轻松介绍它的来源和工作原理。

阅读这篇文章后，你会知道：

*   从学习理论和 AdaBoost 推动的起源。
*   梯度提升的工作原理包括损失函数，弱学习器和加法模型。
*   如何通过各种正则化方案提高基本算法的表现

让我们开始吧。

![A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](img/3dde9db909245469cc477e5f07363bbe.jpg)

机器学习梯度提升算法的温和介绍
[brando.n](https://www.flickr.com/photos/bpprice/12298787813/) 的照片，保留一些权利。

## 提升的起源

提升的想法来自于弱学习器是否可以被修改为变得更好的想法。

Michael Kearns 将目标阐述为“_ 假设推进问题 _”从实际角度阐述了目标：

> ...一种将相对较差的假设转换为非常好的假设的有效算法

- [关于假设提升的思考](https://www.cis.upenn.edu/~mkearns/papers/boostnote.pdf) [PDF]，1988

弱假设或弱学习器被定义为其表现至少略好于随机机会的假设。

这些想法建立在 Leslie Valiant 关于免费分发或[概率近似正确](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning)（PAC）学习的工作之上，这是一个研究机器学习问题复杂性的框架。

假设提升是过滤观察的想法，留下弱学习器可以处理的观察，并专注于开发新的弱学习以处理剩余的困难观察。

> 我们的想法是多次使用弱学习方法来得到一系列假设，每一个假设都重新聚焦于前面发现的困难和错误分类的例子。 ...但是，请注意，如何做到这一点并不明显

- [大概正确：大自然在复杂世界中学习和繁荣的算法](http://www.amazon.com/dp/0465060722?tag=inspiredalgor-20)，第 152 页，2013

### AdaBoost 第一个提升算法

在应用中取得巨大成功的第一个实现提升的是 [Adaptive Boosting 或 AdaBoost](http://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/) 。

> 提升是指通过组合粗略和中度不准确的经验法则来产生非常准确的预测规则的一般问题。

- [在线学习的决策理论推广及其应用](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf) [PDF]，1995

AdaBoost 中的弱学习器是决策树，只有一个分裂，称为决策树桩的短缺。

AdaBoost 通过对观察结果进行加权，更加重视难以分类的实例，而不是那些已经处理好的实例。新的弱学习器按顺序添加，将他们的训练集中在更难的模式上。

> 这意味着难以分类的样本会获得越来越大的权重，直到算法识别出正确分类这些样本的模型

- [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，2013

通过对弱学习器预测的多数投票进行预测，并根据他们的个人准确性进行加权。 AdaBoost 算法最成功的形式是二进制分类问题，称为 AdaBoost.M1。

您可以在帖子中了解有关 AdaBoost 算法的更多信息：

*   [Boosting 和 AdaBoost 机器学习](http://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/)。

### AdaBoost 作为梯度提升的推广

AdaBoost 和相关算法首先由 Breiman 称之为 ARCing 算法的统计框架重铸。

> Arcing 是 Adaptive Reweighting and Combining 的首字母缩写。电弧放电算法中的每个步骤都包括加权最小化，然后重新计算[分类器]和[加权输入]。

- [预测游戏和拱形算法](https://www.stat.berkeley.edu/~breiman/games.pdf) [PDF]，1997

这个框架由 Friedman 进一步开发，称为 Gradient Boosting Machines。后来称为梯度提升或梯度树增强。

统计框架将推进作为一个数值优化问题，其目标是通过使用梯度下降类似过程添加弱学习器来最小化模型的损失。

这类算法被描述为阶段性加法模型。这是因为一次添加一个新的弱学习器，并且模型中现有的弱学习器被冻结并保持不变。

> 请注意，此阶段策略与逐步方法不同，后者在添加新的时重新调整先前输入的术语。

- [贪婪函数逼近：梯度增压机](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) [PDF]，1999

泛化允许使用任意可微分损失函数，将技术扩展到二元分类问题之外，以支持回归，多类分类等。

## 梯度提升的工作原理

梯度提升涉及三个要素：

1.  要优化的损失函数。
2.  做出预测的弱学习器。
3.  添加模型，添加弱学习器以最小化损失函数。

### 1.损失功能

使用的损失函数取决于要解决的问题类型。

它必须是可区分的，但支持许多标准丢失函数，您可以定义自己的函数。

例如，回归可能使用平方误差，分类可能使用对数损失。

梯度提升框架的一个好处是不必为可能想要使用的每个损失函数导出新的增强算法，相反，它是足够通用的框架，可以使用任何可微分损失函数。

### 2.弱小的学习器

决策树被用作梯度提升中的弱学习器。

特别是使用回归树，其输出分裂的实际值并且其输出可以被加在一起，允许添加后续模型输出并“校正”预测中的残差。

树木以贪婪的方式构建，根据基尼等纯度分数选择最佳分割点，或尽量减少损失。

最初，例如在 AdaBoost 的情况下，使用非常短的决策树，其仅具有单个分割，称为决策残余。较大的树木通常可以使用 4 到 8 级。

通常以特定方式约束弱学习器，例如最大层数，节点，分裂或叶节点。

这是为了确保学习器保持弱势，但仍然可以贪婪地构建。

### 3.添加剂模型

树一次添加一个树，模型中的现有树不会更改。

梯度下降程序用于最小化添加树木时的损失。

传统上，梯度下降用于最小化一组参数，例如回归方程中的系数或神经网络中的权重。在计算错误或丢失之后，更新权重以最小化该错误。

我们有弱学习器子模型或更具体的决策树，而不是参数。在计算损失之后，为了执行梯度下降过程，我们必须向模型添加树以减少损失（即遵循梯度）。我们通过参数化树来完成此操作，然后修改树的参数并向右移动（减少剩余损失。

通常，这种方法称为功能梯度下降或具有功能的梯度下降。

> 产生优化[成本]的分类器的加权组合的一种方法是通过函数空间中的梯度下降

- [在函数空间中提升算法作为梯度下降](http://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf) [PDF]，1999

然后将新树的输出添加到现有树序列的输出中，以便纠正或改进模型的最终输出。

一旦损失达到可接受的水平或在外部验证数据集上不再改进，则添加固定数量的树或训练停止。

## 基本梯度提升的改进

梯度提升是一种贪婪算法，可以快速过度训练数据集。

它可以受益于惩罚算法的各个部分的正则化方法，并且通常通过减少过度拟合来改善算法的表现。

在本节中，我们将介绍基本梯度提升的 4 个增强功能：

1.  树约束
2.  收缩
3.  随机抽样
4.  惩罚学习

### 1.树约束

重要的是弱势学习器有技巧但仍然很弱。

树可以通过多种方式进行约束。

一个好的通用启发式方法是，树的创建越多，模型中需要的树就越多，反之，在不受约束的单个树中，所需的树就越少。

以下是可以对决策树构造施加的一些约束：

*   **树木数量**，通常会为模型添加更多树木，过度拟合可能会非常缓慢。建议继续添加树木，直至观察不到进一步的改善。
*   **树深**，更深的树木更复杂的树木和更短的树木是首选。通常，4-8 级可以看到更好的结果。
*   **节点数或叶数**，如深度，这可以约束树的大小，但如果使用其他约束，则不限于对称结构。
*   **每次分割的观察次数**对训练节点的训练数据量施加最小约束，然后才能考虑分割
*   **最小化损失**是对添加到树中的任何拆分的改进的约束。

### 2.加权更新

每棵树的预测按顺序加在一起。

可以对每个树对该总和的贡献进行加权以减慢算法的学习。这种加权称为收缩或学习率。

> 每个更新只是通过“学习速率参数 v”的值进行缩放

— [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) [PDF], 1999

结果是学习速度变慢，反过来需要将更多树木添加到模型中，反过来需要更长的时间来训练，提供树木数量和学习率之间的配置权衡。

> 减小 v [学习率]的值会增加 M [树的数量]的最佳值。

— [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) [PDF], 1999

通常具有 0.1 至 0.3 范围内的小值，以及小于 0.1 的值。

> 与随机优化中的学习速率类似，收缩减少了每棵树的影响，并为将来的树木留出了空间来改进模型。

- [随机梯度提升](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) [PDF]，1999

### 3.随机梯度提升

对装袋合奏和随机森林的深入了解允许从训练数据集的子样本中贪婪地创建树。

可以使用相同的益处来减少梯度提升模型中序列中的树之间的相关性。

这种增强的变化称为随机梯度提升。

> 在每次迭代中，从完整训练数据集中随机（无替换）绘制训练数据的子样本。然后使用随机选择的子样本而不是完整样本来适合基础学习器。

— [Stochastic Gradient Boosting](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) [PDF], 1999

可以使用的一些随机增压变体：

*   在创建每个树之前的子样本行。
*   创建每个树之前的子样本列
*   在考虑每个拆分之前的子采样列。

通常，积极的子采样（例如仅选择 50％的数据）已被证明是有益的。

> 根据用户反馈，使用列子采样可以比传统的行子采样更加防止过度拟合

- [XGBoost：可扩展的树升压系统](https://arxiv.org/abs/1603.02754)，2016

### 4\. Penalized Gradient Boosting

除了它们的结构之外，还可以对参数化树施加附加约束。

像 CART 这样的经典决策树不被用作弱学习器，而是使用称为回归树的修改形式，其在叶节点（也称为终端节点）中具有数值。在一些文献中，树叶中的值可以称为权重。

因此，可以使用流行的正则化函数来规范树的叶权值，例如：

*   L1 权重的正则化。
*   权重的 L2 正则化。

> 额外的正则化项有助于平滑最终学习的权重以避免过度拟合。直观地，正则化目标将倾向于选择采用简单和预测函数的模型。

— [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754), 2016

## 梯度提升资源

梯度提升是一种迷人的算法，我相信你想要更深入。

本节列出了可用于了解梯度提升算法的更多信息。

### 梯度提升视频

*   [Gradient Boosting Machine Learning](https://www.youtube.com/watch?v=wPqtzj5VZus) ，Trevor Hastie，2014
*   [Gradient Boosting](https://www.youtube.com/watch?v=sRktKszFmSk) ，Alexander Ihler，2012
*   [GBM](https://www.youtube.com/watch?v=WZvPUGNJg18) ，John Mount，2015 年
*   [学习：提升](https://www.youtube.com/watch?v=UHBmv7qCey4)，麻省理工学院 6.034 人工智能，2010
*   [xgboost：用于快速准确梯度提升的 R 包](https://www.youtube.com/watch?v=0IhraqUVJ_E)，2016
*   [XGBoost：可扩展的树木升压系统](https://www.youtube.com/watch?v=Vly8xGnNiWs)，陈天琪，2016

### 教科书中的梯度提升

*   第 8.2.3 节 Boosting，第 321 页，[统计学习简介：在 R](http://www.amazon.com/dp/1461471370?tag=inspiredalgor-20) 中的应用。
*   第 8.6 节 Boosting，第 203 页，[应用预测建模](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20)。
*   第 14.5 节“随机梯度提升”，第 390 页，[应用预测建模](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20)。
*   第 16.4 节 Boosting，第 556 页，[机器学习：概率视角](http://www.amazon.com/dp/0262018020?tag=inspiredalgor-20)
*   第 10 章 Boosting 和 Additive 树，第 337 页，[统计学习的要素：数据挖掘，推理和预测](http://www.amazon.com/dp/0387848576?tag=inspiredalgor-20)

### Gradient Boosting Papers

*   [关于假设提升的思考](http://www.cis.upenn.edu/~mkearns/papers/boostnote.pdf) [PDF]，Michael Kearns，1988
*   [在线学习的决策理论推广及其应用](http://cns.bu.edu/~gsc/CN710/FreundSc95.pdf) [PDF]，1995
*   [弧形边缘](http://statistics.berkeley.edu/sites/default/files/tech-reports/486.pdf) [PDF]，1998
*   [随机梯度提升](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) [PDF]，1999
*   [在函数空间中提升算法作为梯度下降](http://maths.dur.ac.uk/~dma6kp/pdf/face_recognition/Boosting/Mason99AnyboostLong.pdf) [PDF]，1999

### 梯度提升幻灯片

*   [Boosted Trees 简介](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)，2014
*   [一个温和的梯度提升介绍](http://www.chengli.io/tutorials/gradient_boosting.pdf)，程力

### 梯度提升网页

*   [提升（机器学习）](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
*   [梯度提升](https://en.wikipedia.org/wiki/Gradient_boosting)
*   [在 scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) 中提升梯度树

## 摘要

在这篇文章中，您发现了用于机器学习中预测建模的梯度提升算法。

具体来说，你学到了：

*   提升学习理论和 AdaBoost 的历史。
*   梯度提升算法如何与损失函数，弱学习器和附加模型一起工作。
*   如何通过正则化提高梯度提升的表现。

您对梯度提升算法或此帖有任何疑问吗？在评论中提出您的问题，我会尽力回答。