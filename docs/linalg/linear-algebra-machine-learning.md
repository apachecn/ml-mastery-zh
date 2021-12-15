# 面向机器学习的线性代数

> 原文： [https://machinelearningmastery.com/linear-algebra-machine-learning/](https://machinelearningmastery.com/linear-algebra-machine-learning/)

在开始机器学习之前，您不需要学习线性代数，但在某些时候您可能希望深入学习。

事实上，如果有一个数学领域，我会建议在其他领域之前进行改进，那就是线性代数。它将为您提供工具，帮助您理解和建立更好的机器学习算法直觉所需的其他数学领域。

在这篇文章中，我们仔细研究线性代数，以及为什么你应该花时间提高线性代数的技能和知识，如果你想从机器学习中获得更多。

如果您已经了解了特征向量和 SVD 分解的方法，那么这篇文章可能不适合您。

[![Linear Algebra For Machine Learning](img/c958e2f0633503a0882090f8b667b4bb.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/12/Linear-Algebra-For-Machine-Learning.jpg)

用于机器学习的线性代数
照片由 [Sarah](http://www.flickr.com/photos/dichohecho/4456332671) 拍摄，保留一些权利。

## 什么是线性代数

线性代数是数学的一个分支，它可以让您简明地描述更高维度的平面的坐标和相互作用，并对它们执行操作。

将其视为代数（处理未知数）到任意数量维度的扩展。线性代数是关于线性方程组的工作（线性回归就是一个例子：y = Ax）。我们不是使用标量，而是开始使用矩阵和向量（向量实际上只是一种特殊类型的矩阵）。

> 广义地说，在线性代数中，数据以线性方程的形式表示。这些线性方程又以矩阵和向量的形式表示。

- Vignesh Natarajan 回答问题“[机器学习中如何使用线性代数？](https://www.quora.com/How-is-Linear-Algebra-used-in-Machine-Learning) “

作为一个领域，它对您有用，因为您可以使用线性代数中的符号和形式来描述（甚至使用正确的库执行）机器学习中使用的复杂操作。

> 线性代数得到广泛应用，因为它通常非常好地并行化。此外，大多数线性代数操作可以在没有消息传递的情况下实现，这使得它们适合 MapReduce 实现。

- Raphael Cendrillon 回答“[为什么线性代数是现代科学/计算研究背后的先决条件？](https://www.quora.com/Why-is-Linear-Algebra-a-prerequisite-behind-modern-scientific-computational-research) “

有关维基百科的线性代数的更多信息：

*   [维基百科上的线性代数](http://en.wikipedia.org/wiki/Linear_algebra)
*   [维基百科上的线性代数类别](http://en.wikipedia.org/wiki/Category:Linear_algebra)
*   [线性代数维基百科上的主题列表](http://en.wikipedia.org/wiki/List_of_linear_algebra_topics)

## 机器学习的最小线性代数

线性代数是一个基础领域。我的意思是，其他数学分支使用符号和形式来表达与机器学习相关的概念。

例如，当您想要在优化损失函数时讨论函数导数时，需要使用矩阵和向量进行微积分。当你想谈论统计推断时，它们也被用于概率。

> ......它在数学中无处不在，所以你会发现它在任何使用数学的地方使用......

- 大卫乔伊斯，回答问题“[线性代数有什么意义？](http://www.quora.com/What-is-the-point-of-linear-algebra) “

如果我要说服你学习最少的线性代数来提高你的机器学习能力，那将是以下 3 个主题：

*   **符号**：知道符号将让您阅读论文，书籍和网站中的算法描述，以了解正在发生的事情。即使你使用 for 循环而不是矩阵运算，至少你可以将事物拼凑在一起。
*   **操作**：在向量和矩阵的下一级抽象中工作可以使事情更清晰。这可以应用于描述，编码甚至思考。学习如何进行或应用简单的操作，如添加，乘法，反转，转置等矩阵和向量。
*   **矩阵分解**：如果有一个更深的区域，我建议潜入任何其他区域，它将是矩阵分解，特别是基质沉积方法，如 SVD 和 QR。计算机的数值精度是有限的，使用分解矩阵可以避免可能导致的大量上溢/下溢疯狂。此外，使用库快速 LU，SVD 或 QR 分解将为您的回归问题提供一个普通的最小二乘法。机器学习和统计的床岩。

如果你知道一些线性代数并且不同意我的最小列表，请发表评论。我很想听听你的 3 分钟主题。

> 如果你想深入了解这一切的理论，你需要知道线性代数。如果您想阅读白皮书并考虑最先进的新算法和系统，您需要了解大量的数学知识。

- Jesse Reiss 回答问题“[线性代数在计算机科学中有多重要？](https://www.quora.com/How-important-is-linear-algebra-in-computer-science) “

## 提高线性代数的 5 个理由

当然，我不希望你停下来。我希望你更深入一点。

如果你需要了解更多并且变得更好并不能激励你走上正轨，那么有五个理由可以帮助你实现这一目标。

1.  **Building Block** ：让我再说一遍。线性代数绝对是理解机器学习中所需的微积分和统计数据的关键。更好的线性代数将全面提升你的游戏。认真。
2.  **更深层次的直觉**：如果您能够理解向量和矩阵层面的机器学习方法，那么您将提高对工作方式和时间的直觉。
3.  **从算法中获取更多信息**：对算法及其约束的更深入理解将允许您自定义其应用程序并更好地理解调整参数对结果的影响。
4.  **从零开始实施算法**：您需要了解线性代数，从零开始实现机器学习算法。至少要阅读算法描述，最多有效地使用提供向量和矩阵运算的库。
5.  **设计新算法**：线性代数的符号和工具可以直接在 Octave 和 MATLAB 等环境中使用，使您可以非常快速地对现有算法和全新方法进行原型修改。

无论您喜欢与否，线性代数都会在您的机器学习过程中占据重要位置。

## 3 个视频课程学习线性代数

如果您希望增强线性代数，可以从三个选项开始。

这些是我发现并最近经历的视频课程和讲座，为这篇文章做准备。我发现每个人都适合不同的观众。

我会在两倍的时间内观看所有视频，并以所有这些来源肆意推荐它。另外，做笔记。

### 1.线性代数复习

这是您应该熟悉的线性代数主题的快速提示。这适用于那些在拼贴中使用线性代数并且正在寻找提醒而不是教育的人。

&lt;iframe allow="autoplay; encrypted-media" allowfullscreen="" frameborder="0" height="375" src="https://www.youtube.com/embed/ZumgfOei0Ak?feature=oembed" width="500"&gt;&lt;/iframe&gt;

该视频的标题为“[线性代数用于机器学习](https://www.youtube.com/watch?v=ZumgfOei0Ak)”，由 Patrick van der Smagt 使用伦敦大学拼贴画的幻灯片创建。

### 2.线性代数速成课程

第二个选项是线性代数速成课程，作为课程机器学习课程 [第 1 周的可选模块。](https://class.coursera.org/ml-005/lecture/preview)

这适用于可能较少或根本不熟悉线性代数并正在寻找主题的第一个引导程序的工程师或程序员。

它包含 6 个短视频，您可以在此处访问名为“[机器学习 - 03.线性代数评论](https://www.youtube.com/playlist?list=PLnnr1O8OWc6boN4WHeuisJWmeQHH9D_Vg)”的 YouTube 播放列表。

&lt;iframe allow="autoplay; encrypted-media" allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/videoseries?list=PLnnr1O8OWc6boN4WHeuisJWmeQHH9D_Vg" width="500"&gt;&lt;/iframe&gt;

涵盖的主题包括：

1.  矩阵和向量
2.  加法和标量乘法
3.  矩阵向量乘法
4.  矩阵矩阵乘法
5.  矩阵乘法属性
6.  反转和转置

### 3.线性代数课程

第三种选择是完成线性代数的完整入门课程。缓慢的磨砺让整个领域成为你的头脑。

我推荐可汗学院上的[线性代数流。](https://www.khanacademy.org/math/linear-algebra)

太奇妙了。不仅宽度令人印象深刻，而且它提供了现场检查问题，但 Sal 是一个很好的沟通者，直接切入材料的应用方面。比我参加的任何大学课程都要好得多。

Sal 的课程分为 3 个主要模块：

*   向量空间
*   矩阵变换
*   替代坐标系（基地）

每个模块包含 5-7 个子模块，每个子模块包含 2-7 个视频或问题集，范围为 5-25 分钟（双倍时间更快！）。

这是伟大的材料和低烧伤，我建议做所有这些，也许在周末狂欢。

### 更多资源学习线性代数

如果您正在寻找更一般的建议，请查看问题的答案“[我如何自学线性代数？](http://www.quora.com/How-can-I-self-study-Linear-Algebra) “。这里有一些真正的宝石。

## 编程线性代数

作为程序员或工程师，您可能会做得最好。我知道我这样做。

因此，您可能希望获取编程环境或库，并开始使用测试数据编码矩阵乘法，SVD 和 QR 分解。

以下是您可能想要考虑的一些选项。

*   [Octave](https://www.gnu.org/software/octave/) ：Octave 是 MATLAB 的开源版本，对于大多数操作来说，它们是等效的。这些平台是为线性代数而构建的。这就是他们所做的，他们做得非常好。他们很高兴使用。
*   [R](http://www.statmethods.net/advstats/matrix.html) ：它可以做 t，但它不如 Octave 漂亮。看看这个方便的报告：“ [R](http://bendixcarstensen.com/APC/linalg-notes-BxC.pdf) 线性代数简介”（PDF）
*   [SciPy numpy.linalg](http://docs.scipy.org/doc/numpy/reference/routines.linalg.html) ：如果您是一名具有干净语法并可访问所需操作的 Python 程序员，那么简单而有趣。
*   [BLAS](http://www.netlib.org/blas/) ：基本线性代数子程序，如乘法，逆等。以大多数编程语言移植或提供。
*   [LAPACK](http://www.netlib.org/lapack/) ：线性代数库， [LINPACK](http://www.netlib.org/linpack/) 的后继者。各种矩阵因子分解等的地方。像 BLAS 一样，移植或可用于大多数编程语言。

还有一个新的 Coursera 课程名为“[编码矩阵：线性代数通过计算机科学应用程序](https://www.coursera.org/course/matrix)”作者 Philip Klein 也有一本同名的书“[编码矩阵：线性代数通过应用程序到计算机科学](http://www.amazon.com/dp/0615880991?tag=inspiredalgor-20)“。如果您是一名 Python 程序员并希望增强线性代数，那么这可能值得一看。

## 线性代数书籍

我从应用的例子中学到了很多，但我也读了很多。如果你像我一样，你会想要一个好的教科书，以防万一。

本节列出了一些适合初学者的线性代数顶级教科书。

### 基金会

这是一本初学教科书，涵盖了线性代数的基础。对于参加可汗学院的课程，要么是一个很好的恭维。

*   [线性代数](http://www.amazon.com/dp/0387962050?tag=inspiredalgor-20)简介 Serge Lang。
*   [Gilbert Strang 介绍线性代数](http://www.amazon.com/dp/0980232716?tag=inspiredalgor-20)

### 应用的

这些书更倾向于线性代数的应用。

*   [Lloyd Trefethen 的数值线性代数](http://www.amazon.com/dp/0898713617?tag=inspiredalgor-20)。
*   [线性代数及其应用](http://www.amazon.com/dp/0030105676?tag=inspiredalgor-20)，Gilbert Strang。
*   [Matrix Computations](http://www.amazon.com/dp/1421407949?tag=inspiredalgor-20) 由 Gene Golub 和 Charles Van Loan 撰写

我非常喜欢后一本书“Matrix Computation”，因为它为您提供了理论和算法伪代码的片段。非常酷的数学家和我的编程人。如果你想从零开始自己实现这些程序（而不是使用库），这可能就是你的书。

有关线性代数的优秀初学者书籍的更多建议，请查看：[学习线性代数的最佳书籍是什么？](http://www.quora.com/What-is-the-best-book-for-learning-Linear-Algebra)

## 摘要

在这篇文章中，您已经了解了线性代数及其在机器学习中的重要作用（以及更广泛的数学）。您还注意到要查看的最小线性代数。

我们提到了三个可以用来学习线性代数，复习，速成课程或更深入视频课程的选项，现在都可以免费使用。如果您想深入了解，我们还会查看有关该主题的顶级教科书。

我希望这引起了你对线性代数变得更好的重要性和力量的兴趣。选择一个资源并阅读/观察完成。采取下一步措施，提高您对机器学习的理解。

**更新**：这篇文章的 [Reddit 讨论](http://www.reddit.com/r/MachineLearning/comments/2q7152/linear_algebra_for_machine_learning/)中提到的另外两个高质量资源是[线性代数完成权](http://www.amazon.com/dp/0387982582?tag=inspiredalgor-20) Axler 和[上的麻省理工学院开放课件课程]线性代数](http://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/index.htm)由 Gilbert Strang 教授（上面提到的一些书的作者）。