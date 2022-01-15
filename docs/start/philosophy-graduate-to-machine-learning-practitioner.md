# 哲学毕业生到机器学习从业者（Brian Thomas 采访）

> 原文： [https://machinelearningmastery.com/philosophy-graduate-to-machine-learning-practitioner/](https://machinelearningmastery.com/philosophy-graduate-to-machine-learning-practitioner/)

机器学习入门可能令人沮丧。有太多东西值得学习，感觉压倒一切。

因此，许多对机器学习感兴趣的开发人员从未开始。在 ad hoc 数据集上创建模型并进入 Kaggle 竞赛的想法听起来令人兴奋。

哲学硕士毕业生是如何开始机器学习的？

在这篇文章中，我采访了 Brian Thomas。

Brian 在使用理论沉重的在线课程感到沮丧后，采用自上而下的实践应用机器学习的方法开始了机器学习。

发现 Brian 的故事以及他使用的工具和资源。

如果 Brian 能找到开始机器学习的方法，那么你也可以。

![Philosophy Graduate to Machine Learning Practitioner](img/b0dd8eb1b94dfc0bbad7d0544c4f7174.jpg)

哲学毕业于机器学习从业者
摄影： [Andrew E. Larsen](https://www.flickr.com/photos/papalars/422462932/) ，保留一些权利。

## 问：您已经尝试过哪些资源来了解机器学习？

什么有效：

您的 [Jump-Start Scikit-Learn](http://machinelearningmastery.com/jump-start-scikit-learn/) 和 R 中的 Jump-Start 机器学习在早期作为 ML 领域的地图非常有价值，使用这两个工具进入并开始使用不同的机器学习模型。我喜欢将所有不同的算法分解并按照地图进行布局，这些地图组织了我付出访问的努力。

从那里我继续使用 R 进行 Brett Lantz 的[机器学习，我发现它特别好。](http://www.amazon.com/dp/1782162143?tag=inspiredalgor-20)

目前我正在通过 Stephen Marsland 的[机器学习：算法视角](http://www.amazon.com/dp/1466583282?tag=inspiredalgor-20)工作。这是非常好的，我发现它比我大约一年前第一次拿起它时更容易通过。

[![Amazon Image](img/bbd25613cb12b55b69497c7479119553.jpg)](http://www.amazon.com/dp/1466583282?tag=inspiredalgor-20)

总的来说，看起来效果最好的是进入那里并开始使用不同的数据集和不同的模型。我不得不说特别是 scikit-learn 确实帮助我解决了这个话题。我还要向 IPython 倾诉，呃，我应该说 Jupyter 笔记本。对我来说，能够加载一些数据，从 scikit-learn 中尝试不同的模型，然后添加一些用我自己的话来解释模型和结果的降价单元格是非常有益的。

最近我也经历了一些在线机器学习教程，特别是 [Jake VanderPlas](http://www.astro.washington.edu/users/vanderplas/) 和 [Olivier Grisel](http://ogrisel.com/) 关于 scikit-learn 的教程。能够克隆他们的 git repos 并使用代码和他们的演示文稿也是最有启发性的。

*   [Jake VanderPlas 关于 Scikit-Learn](https://github.com/jakevdp/sklearn_tutorial) 的 IPython 笔记本
*   [Oliver Grisel 关于 Scikit-Learn](https://github.com/ogrisel/notebooks) 的 IPython 笔记本

什么行不通：

几乎是我试图通过的 2 或 3 个在线课程，包括 [Andrew Ng 的 CS229 ML 课程](http://cs229.stanford.edu/)和 [Nando de Freitas 来自 UBC](http://www.cs.ubc.ca/~nando/540-2013/) 的在线 ML 课程。

并不是说它们是坏的或者其他什么，我只是没有找到尝试坐下来观看关于随机梯度下降的 50 分钟的讲座非常有帮助，特别是没有数学背景。我开始更好地理解 SGD 将 Marsland 在他的 ML 书中提供的代码粘贴到 Jupyter 笔记本中并玩弄它。

当然我没有正式报名参加这些课程或其他什么，我只是下载了所有的讲座，笔记和作业，并试图通过他们的方式工作。最后它似乎在理论上陷入困境。我认为这说明了很多这个问题：人们在没有很多数学背景的情况下（例如我自己）进入这个问题并看到所有这些数学理论而逃避尖叫。

首先是代码，然后让理论上的理解发展。这似乎是正确的方法，我知道[你肯定同情](http://machinelearningmastery.com/machine-learning-for-programmers/)。

## 问：你能分享一下你的背景和工作吗？

我于 1995 年毕业于大学，获得哲学学士学位。

令人惊讶的是，我无意中打开了进入 IT 就业市场的大门，因为我在这个地方进行了一次行政工作，我正在那里从事合同工作。从那份工作开始，我最终学到了很多关于数据库和编程的知识。

来自哲学的背景，我总是能够把事情分解成他们的组成部分，看看他们如何互相玩耍（这可能解释了我相当不错的解决问题的能力）。然而，来自那个背景，我的数学技能是不存在的！我在高中时代的第二年代数停止了，从来没有超越过那个。

在过去的 7 年里，我一直在一家大型保险公司戴着许多不同的帽子，我的日常职责包括服务器和软件测试管理，其中包括开发大量的 PowerShell（现在是 Python）应用程序以协助实现这一目标。

## 问：您玩过的算法和数据集的具体示例是什么？

我真正开始研究的第一本 ML 书是 Brett Lantz 的[机器学习与 R](http://www.amazon.com/dp/1782162143?tag=inspiredalgor-20) ，我浏览了那里的所有数据集和算法以及诸如 Iris 数据集的“经典命中”。对于初学者恕我直言，这是一本好书。

与此同时，我通过 Lantz 的书，我也在教自己 Python（通过 Charles Dierbach 的[计算机科学导论使用 Python：计算问题 - 解决焦点](http://www.amazon.com/dp/0470555157?tag=inspiredalgor-20)等书籍，专注于 Python 编程） se，而不是 ML）。

在 Lantz 书之后不久，我发现自己在日常生活中越来越倾向于 Python。我使用的唯一的 Python 机器学习书是 Stephen Marsland 的[机器学习：算法视角](http://www.amazon.com/dp/1466583282?tag=inspiredalgor-20)。

最近我还玩了[泰坦尼克号数据集](https://www.kaggle.com/c/titanic)并练习清理数据，选择合适的功能，然后尝试了各种各样的算法，例如 NaïveBayes，k-nearest neighbor， AdaBoost 和随机森林分类器。

我也开始探索使用 GPU 的 Python 软件包（我最近购买了一台带有 NVIDIA GeForce 950M GPU 的[华硕笔记本电脑，并在其上运行了一个很好的](http://amzn.to/1iFqXXj) [CUDA](http://www.nvidia.com/object/cuda_home_new.html) 环境），特别是 Theano。

## 问：我注意到你已经尝试过 Python 和 R 用于机器学习，你对这两者有什么印象？

实际上我用[而不是 Python]深入研究`DOING`机器学习，所以这可能会使我对这两者的观点产生偏差。

然而，它`DID`似乎学习机器学习 R 更简单。

> 这是 Lantz 的书吗？
> 
> 是因为 R 是一种统计编程语言，因此在编码时你必须正确使用各种数学概念吗？

我绝对认为后者可能与这种印象有关。

然而，现在，我在 Python 领域非常坚定，主要是因为 [pandas](http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/) 和 [Theano](http://deeplearning.net/software/theano/) （我此时的两个最爱）等软件包。我特别感兴趣，并一直在玩 Theano。

我喜欢你可以声明变量，它们的类型，然后构建表达式，然后用这些可以自动编译以供 GPU 使用的函数。

那太酷了！

## 问：您在深入机器学习方面的目标或抱负是什么？

回到我自己的哲学背景，哲学的好奇心是让我进入它的原因。

> 机... .learn？怎么样？！？

你不得不承认，这个领域的整个历史和实践都是令人着迷的，触及了各种最终具有哲学性质的问题。此外，随着深度学习等近期发展，整个领域变得越来越有趣。

与深度学习相关的是随着大规模并行 GPU 编程的出现而发生的范式转换。

似乎没有这个，深度学习的最新进展是不可能发生的，告诉 Theano 利用我的 GPU 然后通过深度学习教程算法和 LISA 实验室[上的](http://deeplearning.net/tutorial/) [MNIST](http://yann.lecun.com/exdb/mnist/) 数据流失是很酷的。 ]深度学习网站。

## 最后的话

感谢 Brian 分享他的故事和经历。

Brian 已经开始了，他已经掌握了解决 R 和 Python 问题的技能，现在正在开展更复杂的深度学习主题。

即使仍然接近开始，他的机器学习之旅也是一个良好的开端。他实际上可以练习应用机器学习。

我认为如果你想开始机器学习，Brian 的故事会鼓舞人心。

你在等什么？