# 机器学习的最佳编程语言

> 原文： [https://machinelearningmastery.com/best-programming-language-for-machine-learning/](https://machinelearningmastery.com/best-programming-language-for-machine-learning/)

我被问到的一个问题是：

> 什么是机器学习的最佳编程语言？

我已多次回答这个问题，现在是时候在博客文章中进一步探讨这个问题了。

最终，您用于机器学习的编程语言应该考虑您自己的要求和偏好。没有人可以为你有意义地解决这些问题。

没有人可以为你有意义地解决这些问题。

## 使用什么语言

在我发表您的观点之前，最好先了解哪些语言和平台在自选数据分析和机器学习专业人员社区中很受欢迎。

KDnuggets 永远进行了语言民意调查。最近的一项民意调查标题为“ [2013 年用于分析/数据挖掘/数据科学工作的编程/统计语言](http://www.kdnuggets.com/polls/2013/languages-analytics-data-mining-data-science.html)”。趋势几乎与上一年相同。结果表明大量使用 R 和 Python 以及 SQL 进行数据访问。 SAS 和 MATLAB 的排名高于我的预期。我希望 SAS 可以用于更大的企业（财富 500 强）数据分析和用于工程，研究和学生使用的 MATLAB。

[![kdnuggets popular programming languages](img/5d8ea1d54012cbc4978cbc76115ffa98.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/05/kdnuggets-popular-programming-languages.png)

最受欢迎的机器学习平台，取自 [KDnuggets 2013 民意调查](http://www.kdnuggets.com/polls/2013/languages-analytics-data-mining-data-science.html)。

Kaggle 提供机器学习竞赛，并对参赛者在比赛中使用的工具和编程语言进行了调查。他们在 2011 年发布了名为 [Kagglers 的最爱工具](http://blog.kaggle.com/2011/11/27/kagglers-favorite-tools/)的结果（另见[论坛讨论](https://www.kaggle.com/forums/t/1099/data-analysis-tools-and-methods)）。结果表明 R 的使用量很大。结果也表明 MATLAB 和 SAS 的使用效果要低得多。我可以证明我更喜欢 R 而不是 Python 来进行竞争工作。只是感觉它在数据分析和算法选择方面有更多的优势。

[![kaggle most popular tools](img/2440bc8410824bcbee2a7cb05b68dcb2.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/05/kaggle-most-popular-tools.png)

[最受欢迎的工具，用于机器学习竞赛网站 Kaggle](http://blog.kaggle.com/2011/11/27/kagglers-favorite-tools/) 。

上述 Kaggle 博客上的博客文章作者 Ben Hamner 和博客文章的作者在一篇题为“[人们通常使用什么工具来解决问题的论坛帖子中详细介绍了机器学习编程语言的选项](https://www.kaggle.com/forums/t/3642/what-tools-do-people-generally-use-to-solve-problems/19618#post19618)“。

Ben 评论说 MATLAB / Octave 是一种很好的矩阵运算语言，在使用定义明确的特征矩阵时可能会很好。 Python 是全面的，并且可能非常慢，除非你进入 C.他不喜欢使用定义好的特征矩阵并使用 Pandas 和 NLTK。本评论说：“作为一般规则，如果它被发现对统计学家来说很有意思，那么它已经在 R 中实现了”（很好地说）。他还抱怨语言本身是丑陋和痛苦的。最后，Ben 评论朱莉娅，这对库的方式并不多，但却是他最喜欢的语言。他评论说它具有 MATLAB 和 Python 等语言的简洁性和 C 的速度。

Kaggle 首席执行官 Anthony Goldbloom 在 2011 年向 Bay Area R 用户组发表演讲，介绍 R 在 Kaggle 比赛中的受欢迎程度，题为[预测建模竞赛：使数据科学成为一项运动](http://www.meetup.com/R-Users/events/16946398/)（参见 [powerpoint）幻灯片](http://files.meetup.com/1225993/Goldbloom%20-%20Predictive%20modeling%20competitions%20-%20April%202011.ppt)）。演示幻灯片提供了有关编程语言使用的更多细节，并建议了一个与 R 的使用一样大的其他类别。收集原始数据会很好（为什么不是把它发布到他们自己的数据社区，认真！？）。

[![popular languages on kaggle](img/4546506115fb99b38b25b437e06f56d4.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/05/popular-languages-on-kaggle.png)

Kaggle 上流行的编程语言，取自 [Kaggle 演示文稿](http://www.meetup.com/R-Users/events/16946398/)。

John Langford 在他的博客 Hunch 上有一篇关于编程语言属性的优秀文章，在使用名为“[机器学习实现的编程语言](http://hunch.net/?p=230)”的机器学习算法时要考虑。他将属性分为对速度的关注和可编程性的关注（编程简易性）。他指出了强大的行业标准算法实现，所有这些都在 C 和评论中表示他没有使用 R 或 MATLAB（这篇文章是在 8 年前写的）。花一些时间阅读学术界和行业专家的一些评论。这是一个深刻而微妙的问题，实际上归结为您正在解决的问题的具体细节以及解决问题的环境。

## 机器学习语言

我想在我想要执行的机器学习活动的上下文中编程语言。

### MATLAB /八度

我认为 MATLAB 非常适合表示和使用矩阵。因此，我认为在攀入给定方法的线性代数时，它是一种优秀的语言或平台。我认为，当你试图找出问题或深入研究方法时，它很适合在第一次和非常深入地学习算法。例如，它在初学者的大学课程中很受欢迎，例如 [Andrew Ng 的 Coursera 机器学习课程](https://www.coursera.org/course/ml)。

### [R

R 是统计分析和扩展机器学习的主力军。很多人都在谈论学习曲线，我没有真正看到问题。它是使用统计方法和图表来理解和探索数据的平台。它拥有大量的机器学习算法，以及由算法开发人员编写的高级实现。

我认为你可以用 R 来探索，建模和原型。我认为它适合一次性项目，其中包含一系列预测，报告或研究论文。例如，它是[最受欢迎的机器学习竞争对手平台，如 Kaggle](http://blog.kaggle.com/2011/11/27/kagglers-favorite-tools/) 。

### 蟒蛇

Python 如果是一种流行的科学语言和机器学习的后起之秀。如果可以从 R 中获取数据分析，我会感到惊讶，但 NumPy 中的矩阵处理可能会挑战 MATLAB，而 [IPython](http://machinelearningmastery.com/ipython-from-the-shell-to-a-book-with-a-single-tool-with-fernando-perez/ "IPython from the shell to a book with a single tool with Fernando Perez") 等通信工具非常具有吸引力，是未来再现性的一步。

我认为用于机器学习和数据分析的 SciPy 栈可用于一次性项目（如论文），而 [scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library") 等框架已经足够成熟，可用于生产系统。

### Java 的家庭/ C 家族

实现使用机器学习的系统是一项与其他任何工程一样的工程挑战。您需要良好的设计和开发的要求。机器学习是算法，而不是魔术。在严格的生产实现中，您需要一个健壮的库，或者根据需要自定义算法的实现。

有强大的库，例如，Java 有 [Weka](http://machinelearningmastery.com/what-is-the-weka-machine-learning-workbench/ "What is the Weka Machine Learning Workbench") 和 Mahout。此外，请注意，回归（ [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) ）和 SVM（ [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) ）等核心算法的更深层实现是用 C 语言编写的，并由 Python 和其他工具包利用。我认为你很认真，你可以用 R 或 Python 原型，但是你会因为执行速度和系统可靠性等原因而用更重的语言实现。例如， [BigML 的后端在 Clojure](http://blog.bigml.com/2013/06/21/clojure-based-machine-learning/) 中实现。

### 其他问题

*   **不是程序员**：如果你[不是程序员](http://machinelearningmastery.com/what-if-im-not-a-good-programmer/ "What if I’m Not a Good Programmer")（或者不是一个自信的程序员），我建议通过 GUI 界面来玩机器学习，如 [Weka](http://machinelearningmastery.com/what-is-the-weka-machine-learning-workbench/ "What is the Weka Machine Learning Workbench") 。
*   **研究和行动的一种语言**：您可能希望使用相同的语言进行原型设计和生产，以降低无法有效转移结果的风险。
*   **宠物语言**：你可能有一种最喜欢的语言的宠物语言，并希望坚持这一点。您可以自己实现算法或利用库。大多数语言都有某种形式的机器学习包，无论多么原始。

机器学习编程语言的问题在博客和问答网站上很流行。一些选择讨论包括：

*   [机器学习和编程语言](http://suhasmathur.com/2012/02/machine-learning-and-programming-languages/)，2012
*   [哪种编程语言拥有最好的机器学习库存储库？](http://www.quora.com/Which-programming-language-has-the-best-repository-of-machine-learning-libraries) 在 Quora 上，2012 年
*   [哪种编程语言拥有最好的机器学习库存储库？ 2010 年 MetaOptimize 上的](http://metaoptimize.com/qa/questions/1645/which-programming-language-has-the-best-repository-of-machine-learning-libraries)
*   [您建议使用哪种编程语言来构建机器学习问题？](http://stats.stackexchange.com/questions/19889/what-programming-language-do-you-recommend-to-prototype-a-machine-learning-probl) ，CrossValidated，2011

您使用什么编程语言进行机器学习和数据分析，为什么推荐它？

我很想听听你的想法，发表评论。