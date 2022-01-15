# 机器学习是 Kaggle 比赛

> 原文： [https://machinelearningmastery.com/machine-learning-is-kaggle-competitions/](https://machinelearningmastery.com/machine-learning-is-kaggle-competitions/)

[Julia Evans](https://twitter.com/b0rk) 最近写了一篇题为“[机器学习不是 Kaggle 比赛](http://jvns.ca/blog/2014/06/19/machine-learning-isnt-kaggle-competitions/)”的帖子。

这是一篇有趣的帖子，因为它指出了一个重要的事实。如果你想用机器学习解决业务问题，那么在 [Kaggle](http://www.kaggle.com/) 比赛中表现不错并不是这项技能的良好指标。理由是，在 Kaggle 竞赛中取得好成绩所需的工作只是提供商业利益所需的一部分。

这是一个需要考虑的重点，特别是如果你刚刚开始并发现自己在排行榜上做得很好。在这篇文章中，我们将重点讨论机器学习竞赛与应用机器学习的关系。

[![racing algorithms](img/0e7de320d82cb812b3d7b3ab56bb7b4f.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/racing-algorithms.jpg)

机器学习竞赛
照片由 [tableatny](https://www.flickr.com/photos/53370644@N06/4976494944/in/photostream/) ，保留一些权利

## 比赛与“真实世界”

朱莉娅试图参加一场 Kaggle 比赛并且表现不佳。问题在于她将机器学习作为她在 [Stripe](https://stripe.com) 中的角色的一部分。正是这与她擅长自己的工作以及如何在机器学习竞赛中取得成功所引发的这一点脱节。

范围必须限于能够评估技能。如果你曾经在学校参加考试，你就会知道这一点。

想一想求职面试。您可以让候选人破解生产代码库，或者让他们通过抽象的独立问题来解决问题。这两种方法都有其优点，后者的好处是它足够简单，可以在面试环境中解析和完成。前者可能需要数小时，数天，数周的背景。

您可以纯粹根据他们的考试成绩聘请一名候选人，您可以根据他们在 [Top Coder](http://www.topcoder.com/) 上的排名聘请程序员，您可以根据他们的 Kaggle 分数聘请机器学习工程师，但您必须有信心他们的评估中展示的技能转化为他们在工作中所需的任务。

最后一部分很难。这就是为什么你向候选人提出实时问题以了解他们如何思考的原因。

你可以在飞行中或在工作场所工程师更广泛的期望背景下，在 ML 的比赛中表现出色并且表现糟糕。你也可以在实践中擅长机器学习，并且在 Julia 案例中合理声称的竞争中表现不佳。

## 更广泛的问题解决过程

Julia 论证的关键在于竞争中所需的机器学习只是在实践中提供结果所需的更广泛过程的一部分。

朱莉娅使用预测航班到达时间作为确定这一点的问题背景。她强调了更广泛问题的事实如下：

1.  了解业务问题
2.  选择要优化的指标
3.  确定要使用的数据
4.  清理您的数据
5.  建立一个模型
6.  将模型投入生产
7.  测量模型表现

Julia 指出，Kaggle 比赛只是上面列表中的第 5 点（构建模型）。

这是一个很好的观点，我完全赞同。我想指出，我确实认为我们在 Kaggle 比赛中所做的是机器学习（因此这篇文章的标题），并且更广泛的过程被称为其他东西。也许这就是数据挖掘，也许它是应用机器学习，也许这就是人们抛出数据科学时的意思。随你。

## 机器学习很难

更广泛的过程是至关重要的，我应激 [](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/ "5-Part Process for working through Machine Learning Problems")[所有](http://machinelearningmastery.com/reproducible-machine-learning-results-by-default/ "Reproducible Machine Learning Results By Default") [](http://machinelearningmastery.com/how-to-use-machine-learning-results/ "How to Use Machine Learning Results")[](http://machinelearningmastery.com/small-projects/ "Learn and Practice Applied Machine Learning")[时间](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/ "How to Prepare Data For Machine Learning")。

现在，根据所需的技术技能和经验，考虑流程中的步骤。数据选择，清理和模型构建是一项艰巨的技术任务，需要很高的技能才能做好。在某种程度上，除了构建模型步骤之外，数据分析师甚至业务分析师都可以执行大部分职责。

我可能会站在这里，但也许这就是为什么机器学习被置于如此高的基础上。

建立伟大的模型很难。很难。但是，由机器学习竞赛定义的伟大模型（对损失函数得分）几乎总是与业务所需的伟大模型不同。这种精细调整的模型很脆弱。它们难以投入生产，难以复制，难以理解。

在大多数商业案例中，您需要一个“_ 足够好 _”的模型来挑选域中的结构而不是最好的模型。

Julia 在参考 Netflix 奖中部署获胜模型的[失败时提到了这一点。](http://www.forbes.com/sites/ryanholiday/2012/04/16/what-the-failed-1m-netflix-prize-tells-us-about-business-advice/)

## 比赛很棒

Kaggle 比赛，比如他们面前的比赛，对参赛者来说非常有趣。

传统上，学术界（主要是研究生）使用它们来测试算法，发现和探索特定方法和方法的局限性。算法烘焙在研究论文中很常见，但在实践中几乎没有什么好处。 [这是众所周知的](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.73.6198)。

我相信朱莉娅打算做的关键点和观点是，如果你发现自己在 Kaggle 比赛中努力做得好，就不要绝望。

这很可能是因为竞争环境很艰难，而且你的技能评估不成比例偏向于在实践中做好模型建设所需要的一个方面。