# 5 程序员在机器学习中开始犯错误

> 原文： [https://machinelearningmastery.com/mistakes-programmers-make-when-starting-in-machine-learning/](https://machinelearningmastery.com/mistakes-programmers-make-when-starting-in-machine-learning/)

没有正确的方法进入机器学习。我们都学习略有不同的方式，并且对我们想要做的事情或机器学习有不同的目标。

一个共同的目标是快速提高机器学习效率。如果这是你的目标那么这篇文章强调了程序员在快速成为高效机器学习从业者的道路上所犯的五个常见错误。

[![Mistakes Programmers Make when Starting in Machine Learning](img/378094d7ee7ea8fb10a8ba5a9930ca7e.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/01/Mistakes-Programmers-Make-when-Starting-in-Machine-Learning.jpg)

程序员在机器学习中开始犯错误
照片归 [aarontait](http://www.flickr.com/photos/aarontait/3661306617/sizes/l/) 所有，保留一些权利。

## 1.将机器学习放在基座上

机器学习[只是另一种技术](http://machinelearningmastery.com/applied-machine-learning-is-a-meritocracy/ "Applied Machine Learning is a Meritocracy")，您可以使用它来创建复杂问题的解决方案。

因为它是一个新兴的领域，机器学习通常在研究生的学术出版物和教科书中传达。这使它看起来精英而且难以穿透。

心理转变需要在机器学习中有效，从技术到过程，从精确到“足够好”，但对于程序员有兴趣采用的其他复杂方法也是如此。

## 2.编写机器学习代码

通过编写代码从机器学习开始可能会使事情变得困难，因为这意味着您至少解决了两个问题而不是一个：技术如何工作以便您可以实现它以及如何将技术应用于给定问题。

一次处理一个问题并利用机器学习和统计环境以及算法库来学习如何将技术应用于问题要容易得多。这使您可以相对快速地检查和调整各种算法，并调整看起来很有希望的一两个算法，而不是花费大量时间来解释含有算法描述的含糊不清的研究论文。

实现算法可以被视为一个单独的项目，以便稍后完成，例如学习练习或原型系统需要我投入运营。一次学习一件事，我推荐[从基于 GUI 的机器学习框架](http://machinelearningmastery.com/what-if-im-not-a-good-programmer/ "What if I’m Not a Good Programmer")开始，无论你是否是程序员。

## 3.手动做事

一个过程围绕应用的机器学习，包括问题定义，数据准备和结果的呈现，以及其他任务。这些过程以及算法的测试和调整可以而且应该是自动化的。

自动化是构建，测试和部署的现代软件开发的重要组成部分。脚本数据准备，算法测试和调优以及结果的准备有很大的优势，以获得严格和改进速度的好处。记住并重用专业软件开发中学到的经验教训。

无法从自动化开始（例如 Makefiles 或类似的构建系统）可能是由于许多程序员从书本和课程中学习机器学习而不太关注该领域的应用性质。实际上，为应用机器学习提供自动化是程序员的一个巨大机会。

## 4.重塑常见问题的解决方案

成千上万的人可能已经实现了您正在实现的算法，或者解决了与您正在解决的问题类似的问题类型，利用他们的经验教训。

有很多关于解决应用机器学习的知识。其中大部分内容可能与书籍和研究出版物有关，但您可以访问它。做好功课，搜索谷歌，谷歌图书，谷歌学术搜索，并联系机器学习社区。

如果要实现算法：

*   你必须实现它吗？您可以在库或工具中重用现有的开源算法实现吗？
*   你必须从零开始实现吗？您可以编写审查，学习或移植现有的开源实现吗？
*   您是否必须解释规范算法描述？您可以查看和学习其他书籍，论文，论文或博客文章中的算法描述吗？

[![Reinvent Solutions to Common Problems](img/5b8e562971591f84858b54cef1484eca.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/01/Reinvent-Solutions-to-Common-Problems.jpg)

照片归功于 [justgrimes](http://www.flickr.com/photos/notbrucelee/6884703709/sizes/l/) ，保留一些权利

如果您要解决问题：

*   你必须测试所有关于问题的算法吗？您是否可以利用相同通用类型的此类或类似问题实例进行研究，以提出表现良好的算法和算法类？
*   你需要收集自己的数据吗？他们可以直接使用或作为问题代理的公开数据集或 API 是否可以快速了解哪些方法可能表现良好？
*   你必须优化算法的参数吗？您可以使用启发式算法来配置论文或算法研究中提供的算法吗？

如果您遇到编程库或特定类型的数据结构问题，您的策略是什么？在机器学习领域使用相同的策略。联系社区并询问您可能利用的资源，以加快您的项目学习和进度。首先考虑论坛和 Q＆amp; A 网站，然后联系学者和专家，作为下一步。

## 5.忽略数学

你[不需要数学理论](http://machinelearningmastery.com/what-if-im-not-good-at-mathematics/ "What if I’m Not Good at Mathematics")来开始，但数学是机器学习的重要组成部分。这样做的原因是它提供了描述问题和系统行为的最有效和最明确的方法。

忽略算法的数学处理可能导致诸如对方法的理解有限或采用算法的有限解释之类的问题。例如，许多机器学习算法在其核心处具有逐步更新的优化。了解正在解决的优化的性质（函数是凸）允许您使用利用这些知识的有效优化算法。

内化算法的数学处理是缓慢的，并且掌握在掌握之中。特别是如果您从零开始实现高级算法（包括内部优化算法），请花时间从数学角度学习算法。

## 摘要

在这篇文章中，您了解了程序员在开始机器学习时所犯的 5 个常见错误。五节课是：

*   不要把机器学习放在基座上
*   不要写机器学习代码
*   不要手动做事
*   不要重新解决常见问题的解决方案
*   不要忽视数学

**UPDATE** ：继续 [HackerNews](https://news.ycombinator.com/item?id=7140090) 和 [DataTau](http://www.datatau.com/item?id=1410) 的对话。