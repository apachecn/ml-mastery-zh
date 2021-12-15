# 什么是深度学习？

> 原文： [https://machinelearningmastery.com/what-is-deep-learning/](https://machinelearningmastery.com/what-is-deep-learning/)

> 校对：[linmeishang](https://github.com/linmeishang)

深度学习（Deep Learning）是机器学习的一个子领域，涉及受大脑结构和功能激发的算法，这些算法被称为人工神经网络 (Artificial Neural Networks)。

如果你刚刚接触深度学习，或者你有过一些有关神经网络的经验，你可能会感到困惑。我记得我刚开始学习时也很困惑，许多同事和朋友在 90 年代和 21 世纪初学习和使用神经网络时也是如此。

该领域的领军人物和专家对深度学习有些了解，这些具体而细微的观点会为你了解什么是深度学习提供很多启发。

在这篇文章中，你将看到一系列该领域的专家和领军人物的观点，从而来了解什么是深度学习。

让我们开始吧！

![What is Deep Learning?](img/17428dfdb9702ab6ce021befb3d7b812.png)

什么是深度学习？
[Kiran Foster](https://www.flickr.com/photos/rueful/7885846128/) 的图片。保留部分权利。

## 深度学习是大型神经网络

[来自百度研究院 Coursera 和首席科学家的 Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) 正式成立了谷歌大脑 ([Google Brain](https://en.wikipedia.org/wiki/Google_Brain)) ，最终促成了大量谷歌服务中深度学习技术的产品化。

他谈论过也写过很多关于深度学习的内容，这些都是了解深度学习很好的起点。

在深度学习的早期讨论中，Andrew 描述了传统人工神经网络背景下的深度学习。在 2013 年题为“[深度学习，自学习和非监督特征学习](https://www.youtube.com/watch?v=n1ViNeWhC24)”的演讲中，他将深度学习的理念描述为：

> 使用大脑模拟，希望：
> 
> - 使学习算法更好，更容易使用。
> 
> - 在机器学习和人工智能方面取得革命性进展。
> 
> 我相信这是我们迈向真正人工智能的最好机会。

后来他的评论变得更加细致入微了。

根据 Andrew 的观点，深度学习的核心是我们现在拥有足够快的计算机和足够多的数据来真正训练大型神经网络。关于为什么深度学习是在现在开始迅猛发展，他在 2015 年的 ExtractConf 上一个标题为“[为什么数据科学家需要了解深度学习](https://www.youtube.com/watch?v=O0VN0pGgBZM)”的演讲中评论道：

> 我们现在可以开发非常大的神经网络......而且我们可以访问大量的数据。

他还强调了规模（scale）的重要性。当我们构建更大的神经网络，并用越来越多的数据训练它们时，它们的表现（performance）会不断提高。这通常与其他机器学习方法不同, 因为那些机器学习方法的表现会在某个时候趋于稳定。

> 大多数老一代学习算法的表现在某个时候会趋于稳定。深度学习是可扩展的第一类算法。提供更多数据会使模型的表现不断提高。

他在幻灯片中提供了一个很好的图片：

![Why Deep Learning?](img/7e971b08f1a9d6b073c9659bc13010e5.png)

为什么是深度学习？
[Andrew Ng](http://www.slideshare.net/ExtractConf)的幻灯片。保留所有权利。

最后，他清楚地指出，我们在实践中看到的深度学习的好处来自监督学习（supervised learning）。在 2015年 的 ExtractConf 演讲中，他评论道：

> 如今深度学习的所有价值几乎都是通过监督学习或从已标记的数据中学习而得到的。

早在 2014 年，他在斯坦福大学的一次题为“[深度学习](https://www.youtube.com/watch?v=W15K9PegQt0)”的演讲中作了类似的评论：

> 深度学习突飞猛进的一个原因是它在监督学习中的表现十分惊人。

Andrew 经常说，我们应该，也将会看到更多的来自深度学习中非监督学习的好处，因为该领域在处理大量未标记的数据方面正在渐渐成熟。

[Jeff Dean](https://en.wikipedia.org/wiki/Jeff_Dean_(computer_scientist)) 是谷歌的系统和基础设施组的向导和高级研究员。他参与并可能部分负责谷歌内部深度学习的扩展和应用。Jeff 参与了谷歌大脑（Google Brain）项目以及大型深度学习软件 DistBelief 和后来的 TensorFlow 的开发。

在 2016 年题为“[深度学习: 用于建立智能计算机系统](https://www.youtube.com/watch?v=QSaZGT4-6EY)”的演讲中，他也评论说，深度学习实际上就是大型神经网络。

> 当你听到深度学习这个词的时候，就想想一个大的深度神经网络(deep neural net)。深度（deep）通常指神经网络的层数，书籍出版时就采用这种流行的术语。我通常就把它们想成深度神经网络。

他曾多次做过这个演讲，在[同一演讲的修正版的幻灯片中](http://static.googleusercontent.com/media/research.google.com/en//people/jeff/BayLearn2015.pdf)，他强调了神经网络的可扩展性，表明更多数据和更大的模型会使模型的结果变得更好，这反过来也需要更多的运算能力去训练模型。

![Results Get Better With More Data, Larger Models, More Compute](img/f12c1bb681a84a2372eacda56b1aeadc.png)

更多数据、更大模型、更多运算，带来更好的结果，
[Jeff Dean](http://static.googleusercontent.com/media/research.google.com/en//people/jeff/BayLearn2015.pdf)的幻灯片。保留所有权利。

## 深度学习是层次特征学习 （Hierarchical Feature Learning）

除了可扩展性之外，深度学习模型的另一个经常被引用的好处是，它们能够从原始数据中自动提取特征（feature），也称为[特征学习](https://en.wikipedia.org/wiki/Feature_learning) (feature learning)。

[Yoshua Bengio](https://en.wikipedia.org/wiki/Yoshua_Bengio) 是深度学习领域的另一个领军人物，尽管他刚开始时感兴趣的领域是自动特征学习（automatic feature learning），这种学习是通过大型神经网络实现的。

他从深度学习算法利用特征学习从数据中发现并学习表征（representation）的能力方面描述了深度学习。在 2012 年题为“[非监督和转移学习之表征深度学习](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf)”的论文中，他评论道：

> 深度学习算法试图开发输入（input）数据分布中的未知结构，以便发现良好的表征。良好的表征（representation）通常是在多个级别上的，即用较低级别的特征（feature）定义更高级别的特征。

他在 2009 年的技术报告“[为 AI 学习深度结构](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239) ”中详细阐述了深度学习，其中强调了特征学习中层次（hierarchy）的重要性。

> 深度学习方法旨在学习特征层次（feature hierarchies），它们是由来自较高级别的层次（higher levels of the hierarchy）中的特征组成的，而这些较高级别的层次则是由较低级别的特征（lower level features）所组成的。自动学习在多个抽象级别的特征允许一个系统能够学习到数据背后复杂的函数，这些函数直接从数据中将输入映射到输出，而不完全依赖于人工设计的特征（feature）。

在即将出版的与 Ian Goodfellow 和 Aaron Courville 合著的名为“[深度学习](http://www.deeplearningbook.org)”的书中，他们根据模型的架构（architecture ）深度来定义深度学习。

> 概念的层次结构(the hierarchy of concepts)允许计算机去学习复杂的概念（concepts）。这些复杂的概念是由计算机从简单的概念中建立起来的。如果我们可以画一个图来表示这些概念是如何在各自的基础上互相构建的，那么这个图会很深(deep)，有很多层（layers）。出于这个原因，我们将这种方法称为 AI 深度学习。

这是一本重要的书，有可能在一段时间内成为该领域的权威资源。本书继续描述深度学习领域使用的算法 —— 多层感知器(multilayer perceptrons, MLP)，因为深度学习已经被归入人工神经网络这一更大的领域中。

> 深度学习模型的一个典型例子就是前馈深度网络（feedforward deep network）或多层感知器（multilayer perceptrons，MLP）。

[Peter Norvig](https://en.wikipedia.org/wiki/Peter_Norvig) 是谷歌研究部主任，因其题为“[人工智能：现代方法](http://www.amazon.com/dp/0136042597?tag=inspiredalgor-20)”的人工智能教科书而闻名。

在 2016 年的一次题为“[深度学习及其可理解性 VS 软件工程和验证](https://www.youtube.com/watch?v=X769cyzBNVw)”的演讲中，他以和 Yoshua 非常相似的方式定义了深度学习，重点强调更深层网络结构所激发的抽象能力。

> 深度学习通过由很多抽象层次所形成的表征来学习，而不是从直接输入和输出中学习。

## 为什么称它为“深度学习”？
## 为什么不只是“人工神经网络”？

[Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) 是人工神经网络领域的先驱，并与他人合著了第一篇用于训练多层感知器网络的[反向传播](https://en.wikipedia.org/wiki/Backpropagation)算法的论文。

他可能是第一个用“`deep`”（深度）这个词来描述大型人工神经网络发展的人。

他在 2006 年与他人合著了一篇题为“[深度信念网的快速学习算法](http://www.mitpressjournals.org/doi/pdf/10.1162/neco.2006.18.7.1527)”的论文，其中描述了一种训练“深度”（如在多层网络中）受限玻尔兹曼机（Restricted Boltzmann Machine) 的方法。

> 使用互补先验，我们推导出一种快速、贪婪（greedy）的算法，它可以一次学习一层深层有向信念网络（Directed Belief Networks），前提是前两层形成一个无向联想记忆。

这篇文章和 Geoff 的另一篇与他人合著的关于无向深度网络（Undirected Deep Network）的题为“ [深度玻尔兹曼机](http://www.jmlr.org/proceedings/papers/v5/salakhutdinov09a/salakhutdinov09a.pdf) ”的论文得到了社区的好评（现已引用数百次），因为它们是贪婪按层训练网络的成功例子。在前馈网络中，这种方法允许更多的层。

在科学杂志上的一篇题为“[用神经网络降低数据维度](https://www.cs.toronto.edu/~hinton/science.pdf)”的合著的文章中，同样按照之前对“深度”的定义，他们描述了他们开发网络的方法，这些网络的层数比以前典型网络的层数更多。

> 我们阐述了一种有效的初始化权重的方法，它允许深度自编码器（Auto-encoder）网络学习低维度代码（low-dimensional codes）。作为减少数据维度的工具，这些代码比主成分分析（Principal Components Analysis）更好。

在同一篇文章中，他们还有一个有趣的评论。这个评论与 Andrew Ng 关于最近计算能力的提高以及对大型数据集的访问的评论不谋而合。他们都认为大规模使用这些数据集可以激发神经网络的潜能。

> 自 20 世纪 80 年代以来，显而易见的是，深度自编码器的反向传播（backpropagation）对于非线性降维是非常有效的，只要计算机足够快，数据集足够大，并且初始权重足够接近良好解就可以。如今，这三个条件都满足了。

在 2016 年皇家学会题为“[深度学习](https://www.youtube.com/watch?v=VhmE_UXDOGs)”的演讲中，Geoff 评论说深度信念网络是 2006 年深度学习的开始，这一新的深度学习浪潮的首次成功应用是 2009 年的语音识别，这个文章的标题为“[基于深度信念网络的声学建模](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)”，他们达到了最先进的水平。

这个结果使语音识别和神经网络社区注意到，可能是因为使用“深度”来与先前的神经网络进行区分而造成了这个领域名称的改变。

在皇家学会的谈话中，大概正如您所期望的那样，他们对深度学习的描述非常专注于反向传播。有趣的是，他提出了为什么反向传播（应读作“深度学习”）在上世纪 90 年代没有起飞的 4 个原因。前两点与 Andrew Ng 的评论相符，即数据集太小而且计算机太慢。

![What Was Actually Wrong With Backpropagation in 1986?](img/fba7c01fd78761e808c4f093eff4ccd1.png)

1986 年的反向传播到底错在哪里？
 [Geoff Hinton](https://www.youtube.com/watch?v=VhmE_UXDOGs) 的幻灯片。保留所有权利。

## 深度学习作为跨领域的可扩展学习

深度学习在相似输入（甚至输出）的问题上表现优异。意思是，它们不是少量的表格数据，而是像素数据的图像、文本文档或音频数据。

[Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) 是脸书研究部（Facebook Research）的主管，也是[卷积神经网络（CNN）](http://machinelearningmastery.com/crash-course-convolutional-neural-networks/) 架构之父，这种网络擅长图像数据中的对象（object）识别。这种技术非常成功，因为像多层感知器（MLP）前馈神经网络一样，该技术可以根据数据和模型大小进行扩展，并且可以通过反向传播进行训练。

这使他对深度学习的定义更专注于大型 CNN ，这些 CNN 在照片中的对象识别方面取得了巨大成功。

在劳伦斯利弗莫尔国家实验室 （Lawrence Livermore National Laboratory） 2016 年的一次题为“[加速理解：深度学习，智能应用和 GPU](https://www.youtube.com/watch?v=Qk4SqF9FT-M) ”的演讲中，他将深度学习描述为层次表征（hierarchical representations）的学习，并将其定义为一种为了构建对象识别系统的可扩展的学习方法：

> 深度学习[是] ...所有可训练模块组成的流水线（pipeline）。 ......“深”是因为识别对象的过程有多个阶段，所有这些阶段都是训练的一部分。

![Deep Learning = Learning Hierarchical Representations](img/662da48b47adc3aaeccc61fa6245b2a0.png)

深度学习=层次表征的学习
[Yann LeCun](https://www.youtube.com/watch?v=Qk4SqF9FT-M)的幻灯片。保留所有权利。

[Jurgen Schmidhuber](https://en.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber) 是另一种流行的深度学习算法之父。这种算法像 MLP 和 CNN 一样也可以根据模型大小和数据集大小进行扩展，并且可以通过反向传播进行训练，但是它是为序列数据（sequence data）量身定制的，称为 [长短期记忆网络（LSTM）](http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)，是一种递归神经网络（Recurrent Neural Network）。

我们确实看到了“深度学习”这个命名引起的混淆。在 2014 年题为“[神经网络中的深度学习：概述](http://arxiv.org/pdf/1404.7828v4.pdf)”的论文中，他对该领域命名的问题以及区分深度学习与浅层学习（shallow learning）进行了评论。他还有趣地解释了“深度”是问题复杂程度的深度，而不是用于解决问题的模型的深度。

> 深度学习在何种问题复杂程度上真正开始优于浅层学习？深度学习专家的讨论尚未给这一问题作出结论性的回应。 [...]，让我来下个定义，不过这个定义只是为了让你们有个大致的总览：问题深度大于 10 就需要非常深度的学习。

[Demis Hassabis](https://en.wikipedia.org/wiki/Demis_Hassabis) 是 [DeepMind](https://deepmind.com/) 的创始人，后来被谷歌收购。 DeepMind 在深度学习与强化学习相结合的方面取得了突破，这种学习方法可以处理复杂的学习问题，如玩游戏。在此方面，十分出名的例证是 Atari 游戏和游戏 Go 中的 Alpha Go 。

为了与深度学习命名保持一致，他们将他们的新技术称为深度 Q 网络（Deep Q-Network），将深度学习与 Q-Learning 结合起来。他们还将这个广泛的研究领域命名为“深层强化学习（Deep Reinforcement Learning）”。

在他们 2015 年发表在自然（Nature）杂志上的题为“[通过深度强化学习实现人类控制](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)”的论文中，他们评论了深度神经网络在这些突破中的重要作用，并强调了层次抽象（hierarchical abstraction）的必要性。

> 为了实现这一目标，我们开发了一种新型的代理（agent），一种深度 Q 网络（Deep Q-Network，DQN），它能够将强化学习与一类称为深度神经网络的人工神经网络相结合。值得注意的是，深度神经网络的最新进展使得人工神经网络可以直接从原始传感数据中学习诸如对象类别之类的概念。这些深度神经网路使用若干层节点来逐渐建立数据的抽象表征。

最后，这篇可能是被认为最初定义该领域的论文，即 Yann LeCun，Yoshua Bengio 和 Geoffrey Hinton 在自然（Nature）杂志上发表的题为“[深度学习](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html)”的论文。他们在文中对深度学习作了清晰的定义，这个定义强调多层方法（multi-layered approach）。

> 深度学习允许由多个神经网络层组成的模型通过多层抽象来学习数据的表征。

后来，他们从表征学习（representation learning）和抽象（abstraction）的角度描述了多层方法（multi-layered approach）。

> 深度学习方法是具有多个级别表征的表征学习方法。这些表征可以通过简单但非线性的模块而形成，每个模块将一个级别（从原始输入开始）上的表征转换为更高级别上的表征，即稍微更抽象一点的级别。 [...]深度学习的关键方面是这些特征层次（layers of features）不是由人类工程师设计的：它们是使用学习程序从数据中学习的。

这个定义很好而且很通用，可以很简单地描述大多数的人工神经网络算法。这也可以作为本文的结尾。

## 总结

在这篇文章中，你学习到了深度学习其实就是用于更大数据的大型神经网络，因此需要更强的计算能力。

虽然 Hinton 和其合作者发表的早期的方法侧重于贪婪的分层训练（greedy layerwise training）和无监督方法，如自编码器，但现代最先进的深度学习主要集中在使用反向传播算法训练深层（多层）神经网络模型。最流行的技术有：

*   多层感知器网络（MLP）。
*   卷积神经网络（CNN)。
*   长短期记忆递归神经网络(LSTM)。

我希望本文清晰地解释了什么是深层学习，以及深度学习的各种主流定义是如何融会贯通的。

如果你对深度学习或本文有任何疑问，请在下面的评论中提问，我会尽力回答。
