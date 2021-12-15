# 如何研究机器学习算法行为

> 原文： [https://machinelearningmastery.com/how-to-investigate-machine-learning-algorithm-behavior/](https://machinelearningmastery.com/how-to-investigate-machine-learning-algorithm-behavior/)

机器学习算法是需要学习才能理解的复杂系统。

机器学习算法的静态描述是一个很好的起点，但不足以了解算法的行为方式。您需要查看算法的实际效果。

通过对运行中的机器学习算法进行试验，您可以根据可以在不同类别的问题上实现的结果，建立对算法参数的因果关系的直觉。

在这篇文章中，您将了解如何研究机器学习算法。您将学习一个简单的5步过程，您可以使用它来设计和完成您的第一个机器学习算法实验。

您会发现，机器学习实验不仅适用于学术，也可以实现，并且在掌握的过程中需要进行实验，因为您将获得的经验因果知识在其他任何地方都无法获得。

[![Investigate Machine Learning Algorithm Behavior](img/1ec819a6ee908b974563dd5606c0a672.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/10/Investigate-Machine-Learning-Algorithm-Behavior.jpg)

调查机器学习算法行为
照片由[美国陆军RDECOM](http://www.flickr.com/photos/rdecom/7222825892) ，保留一些权利

## 什么是调查机器学习算法

在研究机器学习算法时，您的目标是找到能够在问题和问题类别中推广的良好结果的行为。

您通过对算法行为进行系统研究来研究机器学习算法。这是通过设计和执行受控实验来完成的。

完成实验后，您可以解释并显示结果。结果可让您了解算法更改之间的因果关系，行为以及您可以实现的结果。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的60多种算法的方便思维导图。

下载，打印并使用它。

## 如何研究机器学习算法

在本节中，我们将介绍一个简单的5步过程，您可以使用它来研究机器学习算法。

### 1.选择算法

选择您有疑问的算法。

这可能是您在问题上使用的算法，或者您认为在将来可能要使用的其他上下文中表现良好的算法。

出于实验目的，采用现成的算法实现是有用的。这为您提供了一个基线，如果有任何错误，很可能很少。

[自己实现算法](http://machinelearningmastery.com/how-to-implement-a-machine-learning-algorithm/ "How to Implement a Machine Learning Algorithm")可以是[了解算法程序](http://machinelearningmastery.com/benefits-of-implementing-machine-learning-algorithms-from-scratch/ "Benefits of Implementing Machine Learning Algorithms From Scratch")的好方法，但也可以在实验中引入其他变量，如错误和必须的无数微决策为每个算法实现。

### 2.确定一个问题

您必须有一个您正在寻求回答的研究问题。问题越具体，答案就越有用。

一些示例问题包括：

*   将kNN中的k增加为训练数据集大小的一部分有什么影响？
*   在SVM中选择不同内核对二元分类问题的影响是什么？
*   不同属性缩放对二元分类问题的逻辑回归有什么影响？
*   在随机森林中将随机属性添加到训练数据集对分类准确率的影响是什么？

设计您想要回答的有关算法的问题。考虑列出问题的五个变体并深入研究最具体的问题。

### 3.设计实验

选择将构成实验的问题中的元素。

例如，从上面提出以下问题：“_不同属性缩放对二元分类问题的逻辑回归有什么影响？_ “

您可以从实验设计中选择的元素包括：

*   **属性缩放方法**。您可以包括规范化，标准化，将属性提升为幂，取对数等方法。
*   **Logistic回归**。您要使用哪种逻辑回归实现。
*   **二元分类问题**。具有数字属性的不同标准二元分类问题。将需要多个问题，一些具有相同比例的属性（如[电离层](https://archive.ics.uci.edu/ml/datasets/Ionosphere)）和其他具有各种尺度属性的问题（如[糖尿病](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)）。
*   **表现**。需要模型表现分数，例如分类准确率。

花点时间仔细选择问题的元素，以便最好地回答您的问题。

### 4.执行实验和报告结果

完成你的实验。

如果算法是随机的，您可能需要多次重复实验运行并采用均值和标准差。

如果您正在寻找实验运行（例如不同参数）之间的结果差异，您可能需要使用统计工具来指示差异是否具有统计显着性（例如学生t检验）。

像 [R](http://machinelearningmastery.com/what-is-r/ "What is R") 和 [scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/ "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library") / SciPy这样的工具有完成这些类型实验的工具，但您需要将它们组合在一起并编写实验脚本。其他工具如 [Weka](http://machinelearningmastery.com/what-is-the-weka-machine-learning-workbench/ "What is the Weka Machine Learning Workbench") 将工具内置到图形用户界面中（请参阅此[教程，了解如何在Weka](http://machinelearningmastery.com/design-and-run-your-first-experiment-in-weka/ "Design and Run your First Experiment in Weka") 中运行您的第一个实验）。您使用的工具比实验设计的问题和严谨性更重要。

总结您的实验结果。您可能想要使用表格和图形。单独呈现结果是不够的。他们只是数字。您必须将数字与问题联系起来，并通过实验设计过滤其含义。

结果表明您的研究问题是什么？

戴上你持怀疑态度的帽子。您可以对结果有什么漏洞或限制。不要回避这一部分。了解这些局限性与了解实验结果同样重要。

### 5.重复

重复这个过程。

继续调查您选择的算法。您甚至可能希望使用不同的参数或不同的测试数据集重复相同的实验。您可能希望解决实验中的限制。

不要停止一个实验，开始建立知识库和算法的直觉。

通过一些简单的工具，一些好的问题以及严谨和怀疑的良好表现，您可以非常快速地开始对算法的行为提出世界级的理解。

## 调查算法不仅适用于学者

您可以研究机器学习算法的行为。

你不需要更高的学位，你不需要接受研究方法的训练，你不需要成为一名学者。

机器学习算法的仔细系统调查对任何拥有计算机和兴趣的人都是开放的。事实上，如果你想掌握机器学习，你必须熟悉机器学习算法的系统研究。知识根本就不在那里，你必须自己出去并凭经验收集它。

在谈论您的发现的适用性时，您需要持怀疑态度并要小心。

您不需要有任何独特的问题。通过调查标准问题，您将获得很多收益，例如在一些标准数据集中推广的一个参数的影响。您可能会发现常见的最佳实践启发式的局限性或对立点。

## 行动步骤

在这篇文章中，您发现了通过受控实验研究机器学习算法行为的重要性。您发现了一个简单的5步过程，您可以使用它设计并在机器学习算法上执行第一个实验。

采取行动。使用您在本博文中学到的过程，完成第一次机器学习实验。一旦你完成了一个，即使是非常小的一个，你将拥有完成第二个甚至更多的信心，工具和能力。

我很想知道你的第一个实验。发表评论并分享您的结果或您学到的知识。