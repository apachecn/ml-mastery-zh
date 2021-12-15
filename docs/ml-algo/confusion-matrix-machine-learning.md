# 什么是机器学习中的混淆矩阵

> 原文： [https://machinelearningmastery.com/confusion-matrix-machine-learning/](https://machinelearningmastery.com/confusion-matrix-machine-learning/)

### 使混淆矩阵更容易混淆。

混淆矩阵是用于总结分类算法的表现的技术。

如果每个类中的观察数量不等，或者数据集中有两个以上的类，则单独的分类准确率可能会产生误导。

计算混淆矩阵可以让您更好地了解分类模型的正确性以及它所犯的错误类型。

在这篇文章中，您将发现用于机器学习的混淆矩阵。

阅读这篇文章后你会知道：

*   混淆矩阵是什么以及为什么需要使用它。
*   如何从头开始计算2类分类问题的混淆矩阵。
*   如何在Weka，Python和R中创建混淆矩阵

让我们开始吧。

*   **2017年10月更新**：修复了工作示例中的一个小错误（感谢Raktim）。
*   **2017年12月更新**：修正了准确度计算中的一个小错误（感谢Robson Pastor Alexandre）

![What is a Confusion Matrix in Machine Learning](img/302dba7f39692ffff7955a0cd5b7d149.jpg)

什么是机器学习中的混淆矩阵
摄影： [Maximiliano](https://www.flickr.com/photos/agent1994/5062439082/) Kolus，保留一些权利

## 分类准确率及其局限性

分类准确率是正确预测与总预测的比率。

```py
classification accuracy = correct predictions / total predictions
```

它通常通过将结果乘以100来表示为百分比。

```py
classification accuracy = correct predictions / total predictions * 100
```

通过反转该值，分类精度也可以很容易地转换为错误分类率或错误率，例如：

```py
error rate = (1 - (correct predictions / total predictions)) * 100
```

分类准确率是一个很好的起点，但在实践中经常会遇到问题。

分类准确率的主要问题在于它隐藏了您需要的细节，以便更好地理解分类模型的表现。有两个例子，您最有可能遇到此问题：

1.  当你的数据有2个以上的类。对于3个或更多类，您可以获得80％的分类准确度，但是您不知道是否因为所有类都被预测得同样好，或者模型是否忽略了一个或两个类。
2.  当您的数据没有偶数个类时。您可以达到90％或更高的准确度，但如果每100个记录中有90个记录属于一个类别，则这不是一个好分数，您可以通过始终预测最常见的类值来达到此分数。

分类准确率可以隐藏诊断模型表现所需的详细信息。但幸运的是，我们可以通过混淆矩阵来区分这些细节。

## 什么是混淆矩阵？

混淆矩阵是对分类问题的预测结果的总结。

使用计数值汇总正确和不正确预测的数量，并按每个类进行细分。这是混淆矩阵的关键。

**混淆矩阵显示了分类模型
在做出预测时的混淆方式。**

它不仅可以让您了解分类器所犯的错误，更重要的是可以了解正在进行的错误类型。

正是这种分解克服了仅使用分类精度的限制。

### 如何计算混淆矩阵

以下是计算混淆矩阵的过程。

1.  您需要具有预期结果值的测试数据集或验证数据集。
2.  对测试数据集中的每一行做出预测。
3.  从预期结果和预测计数：
    1.  每个班级的正确预测数量。
    2.  每个类的错误预测数，由预测的类组织。

然后将这些数字组织成表格或矩阵，如下所示：

*   **预期在一边**：矩阵的每一行对应一个预测的类。
*   **在顶部预测**：矩阵的每一列对应一个实际的类。

然后将正确和不正确分类的计数填入表中。

类的正确预测总数将进入该类值的预期行和该类值的预测列。

同样，类的错误预测总数将进入该类值的预期行和该类值的预测列。

> 在实践中，诸如此类的二元分类器可以产生两种类型的错误：它可能错误地分配默认为无默认类别的个人，或者它可能错误地将未默认的个人分配给默认类别。通常有兴趣确定正在进行这两种类型的错误中的哪一种。混淆矩阵[...]是显示此信息的便捷方式。

- 第145页，[统计学习导论：应用于R](http://www.amazon.com/dp/1461471370?tag=inspiredalgor-20) ，2014年

该矩阵可用于易于理解的2类问题，但通过向混淆矩阵添加更多行和列，可轻松应用于具有3个或更多类值的问题。

让我们通过一个例子来解释创建混淆矩阵。

## 2级混淆矩阵案例研究

让我们假设我们有一个两级分类问题，即预测照片是否包含男人或女人。

我们有一个包含10个记录的测试数据集，其中包含预期结果和一组来自我们的分类算法的预测。

```py
Expected, 	Predicted
man,		woman
man, 		man
woman,		woman
man,		man
woman,		man
woman, 		woman
woman, 		woman
man, 		man
man, 		woman
woman, 		woman
```

让我们开始并计算这组预测的分类准确率。

该算法使10个预测中的7个正确，准确度为70％。

```py
accuracy = total correct predictions / total predictions made * 100
accuracy = 7 / 10 * 100
```

但是犯了什么类型的错误？

让我们把结果变成混淆矩阵。

首先，我们必须计算每个类的正确预测数。

```py
men classified as men: 3
women classified as women: 4
```

现在，我们可以计算每个类的错误预测数量，按预测值组织。

```py
men classified as women: 2
woman classified as men: 1
```

我们现在可以将这些值安排到2级混淆矩阵中：

```py
		men	women
men		3	1
women	2	4
```

我们可以从这张桌子上学到很多东西。

*   数据集中的实际总人数是men列（3 + 2）上的值的总和
*   数据集中的实际女性总数是女性专栏中的值总和（1 + 4）。
*   正确的值组织在矩阵的左上角到右下角的对角线中（3 + 4）。
*   通过预测男性为女性比预测女性为男性更多的错误。

### 两类问题很特殊

在一个两类问题中，我们经常寻求从正常观察中区分具有特定结果的观察结果。

如疾病状态或事件，无疾病状态或无事件。

这样，我们可以将事件行分配为“_正_”，将无事件行分配为“_负_”。然后我们可以将预测的事件列分配为“`true`”，将无事件分配为“`false`”。

这给了我们：

*   “**真阳性**”用于正确预测的事件值。
*   “**误报**”表示错误预测的事件值。
*   “**真阴性**”用于正确预测的无事件值。
*   “**误报**”表示错误预测的无事件值。

我们可以在混淆矩阵中总结如下：

```py
  			event			no-event
event		true positive		false positive
no-event	false negative		true negative
```

这有助于计算更高级的分类指标，例如分类器的精确度，召回率，特异性和灵敏度。

例如，分类准确度计算为真阳性+真阴性。

> 考虑有两个类的情况。 [...]表格的第一行对应于预测为事件的样本。有些是正确预测的（真正的阳性或TP），而其他的则被错误地分类（假阳性或FP）。类似地，第二行包含具有真阴性（TN）和假阴性（FN）的预测阴性。

- 第256页， [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，2013

现在我们已经完成了一个简单的2级混淆矩阵案例研究，让我们看看如何在现代机器学习工具中计算混淆矩阵。

## 混淆矩阵的代码示例

本节提供了使用顶级机器学习平台的混淆矩阵的一些示例。

这些示例将为您提供有关混淆矩阵的内容的背景信息，以便您在实践中使用实际数据和工具。

### Weka中的混淆矩阵示例

在Explorer界面中估计模型的技能时，Weka机器学习工作台将自动显示混淆矩阵。

下面是在Pima Indians Diabetes数据集上训练k-最近邻算法后，来自Weka Explorer界面的屏幕截图。

混淆矩阵列在底部，您可以看到还提供了大量的分类统计数据。

混淆矩阵将字母a和b分配给类值，并为行提供预期的类值，并为每列提供预测的类值（“分类为”）。

![Weka Confusion Matrix and Classification Statistics](img/dbfc8aead6f4b4b654b52d9bc9a47e20.jpg)

Weka混淆矩阵和分类统计

您可以在此处了解有关 [Weka机器学习工作台的更多信息](http://machinelearningmastery.com/applied-machine-learning-weka-mini-course/)。

### 使用scikit-learn在Python中的示例混淆矩阵

Python中用于机器学习的scikit-learn库可以计算混淆矩阵。

给定一个预期值的数组或列表以及机器学习模型中的预测列表，confusion_matrix（）函数将计算混淆矩阵并将结果作为数组返回。然后，您可以打印此数组并解释结果。

```py
# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix

expected = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
results = confusion_matrix(expected, predicted)
print(results)
```

运行此示例打印混淆矩阵数组，总结设计的2类问题的结果。

```py
[[4 2]
[1 3]]
```

在scikit-learn API文档中了解有关 [confusion_matrix（）函数的更多信息。](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### R中带有插入符号的示例混淆矩阵

R中用于机器学习的插入符号库可以计算混淆矩阵。

给定预期值列表和机器学习模型中的预测列表，confusionMatrix（）函数将计算混淆矩阵并将结果作为详细报告返回。然后，您可以打印此报告并解释结果。

```py
# example of a confusion matrix in R
library(caret)

expected <- factor(c(1, 1, 0, 1, 0, 0, 1, 0, 0, 0))
predicted <- factor(c(1, 0, 0, 1, 0, 0, 1, 1, 1, 0))
results <- confusionMatrix(data=predicted, reference=expected)
print(results)
```

运行此示例计算混淆矩阵报告和相关统计信息并打印结果。

```py
Confusion Matrix and Statistics

          Reference
Prediction 0 1
         0 4 1
         1 2 3

               Accuracy : 0.7
                 95% CI : (0.3475, 0.9333)
    No Information Rate : 0.6
    P-Value [Acc > NIR] : 0.3823

                  Kappa : 0.4
 Mcnemar's Test P-Value : 1.0000

            Sensitivity : 0.6667
            Specificity : 0.7500
         Pos Pred Value : 0.8000
         Neg Pred Value : 0.6000
             Prevalence : 0.6000
         Detection Rate : 0.4000
   Detection Prevalence : 0.5000
      Balanced Accuracy : 0.7083

       'Positive' Class : 0
```

本报告中有大量信息，尤其是混淆矩阵本身。

了解更多关于插入符号API文档 [PDF]中 [confusionMatrix（）函数的信息。](ftp://cran.r-project.org/pub/R/web/packages/caret/caret.pdf)

## 进一步阅读

关于混淆矩阵的内容并不多，但本节列出了一些您可能有兴趣阅读的其他资源。

*   维基百科上的[混淆矩阵](https://en.wikipedia.org/wiki/Confusion_matrix)
*   [混淆矩阵术语的简单指南](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
*   [混淆矩阵在线计算器](http://www.marcovanetti.com/pages/cfmatrix/)

## 摘要

在这篇文章中，您发现了机器学习的混淆矩阵。

具体来说，您了解到：

*   分类准确率的局限性以及何时可以隐藏重要细节。
*   混淆矩阵以及如何从头开始计算并解释结果。
*   如何使用Weka，Python scikit-learn和R插入库来计算混淆矩阵。

**你有什么问题吗？**
在下面的评论中提出您的问题，我会尽力回答。