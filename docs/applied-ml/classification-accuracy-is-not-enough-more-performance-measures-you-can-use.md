# 分类准确率不够：可以使用更多表现测量

> 原文： [https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)

当您为分类问题构建模型时，您几乎总是希望将该模型的准确率视为所有预测所做的正确预测的数量。

这是分类准确率。

在之前的文章中，我们研究了[评估模型](http://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/ "How To Choose The Right Test Options When Evaluating Machine Learning Algorithms")的稳健性，使用交叉验证和多重交叉验证来预测未见数据，其中我们使用了分类准确度和平均分类准确度。

一旦你拥有一个你认为可以做出强有力预测的模型，你需要确定它是否足以解决你的问题。单独的分类准确率通常不足以做出此决定。

[![Classification Accuracy](img/7a329bed5d17347b0e270df35266f1fe.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/classification-accuracy.jpg)

分类准确度
摄影：Nina Matthews摄影，保留一些权利

在这篇文章中，我们将介绍Precision和Recall表现度量，您可以使用它们来评估模型的二分类问题。

## 乳腺癌复发

[乳腺癌数据集](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer)是标准的机器学习数据集。它包含9个属性，描述了286名患有乳腺癌并且在乳腺癌中存活并且在5年内是否复发的女性。

这是一个二分类问题。在286名女性中，201名患者未复发乳腺癌，剩下的85名女性患乳腺癌。

我认为对于这个问题，假阴性可能比误报更糟糕。你同意吗？更详细的筛查可以清除误报，但是假阴性被送回家并丢失以进行后续评估。

## 分类准确率

[分类精度](http://en.wikipedia.org/wiki/Accuracy_and_precision)是我们的出发点。它是正确预测的数量除以预测的总数，乘以100将其变为百分比。

### 一切都没有复发

仅预测不会复发乳腺癌的模型将达到（201/286）* 100或70.28％的准确度。我们称之为“所有不复发”。这是一个高精度，但一个可怕的模型。如果它被单独用于决策支持以告知医生（不可能，但一起玩），它会将85名妇女误认为他们的乳腺癌不会再发生（高假阴性）。

### 所有复发

仅预测乳腺癌复发的模型将达到（85/286）* 100或29.72％的准确度。我们称之为“所有复发”。这个模型具有可怕的准确率，并且会让201名女性认为乳腺癌复发，但实际上没有（高假阳性）。

### 大车

CART或[分类和回归树](http://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees)是一种功能强大但简单的决策树算法。在这个问题上，CART可以达到69.23％的准确率。这低于我们的“All No Recurrence”模型，但这个模型更有价值吗？

我们可以看出，单独的分类准确率不足以为此问题选择模型。

## 混乱矩阵

用于呈现分类器的预测结果的清晰且明确的方式是使用[混淆矩阵](http://en.wikipedia.org/wiki/Table_of_confusion#Table_of_confusion)（也称为[列联表](http://en.wikipedia.org/wiki/Contingency_table)）。

对于二分类问题，该表有2行2列。顶部是观察到的类标签，而旁边是预测的类标签。每个单元格包含落入该单元格的分类器所做的预测数量。

[![Truth Table Confusion Matrix](img/77dddc6647d29c6f5b500f602293112c.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/truth_table.png)

真相表混淆矩阵

在这种情况下，一个完美的分类器将正确预测201没有复发和85复发，这将进入左上角的细胞没有复发/没有复发（真阴性）和右下角细胞复发/复发（真阳性）。

不正确的预测显然会分解为另外两个单元格。假阴性是分类器标记为不再发生的重复。我们没有这些。假阳性不是分类器标记为重复的重复。

这是一个有用的表，它提供了数据中的类分布和分类器预测的类分布以及错误类型的细分。

### 所有无复发混淆矩阵

混淆矩阵突出显示大量假阴性（85）。

[![All No Recurrence Confusion Matrix](img/68da7b948431f37954013a9140484934.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/no_recurrence_confusion_matrix.png)

所有无复发混淆矩阵

### 所有递归混淆矩阵

混淆矩阵突出了大量（201）的误报。

[![All Recurrence Confusion Matrix](img/1e9aa8d22b023c09637f4ba6b747c4f0.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/recurrence_confusion_matrix.png)

所有递归混淆矩阵

### CART混淆矩阵

这看起来像一个更有价值的分类器，因为它正确地预测了10个重复事件以及188个没有重复事件。该模型还显示了适度数量的假阴性（75）和假阳性（13）。

[![CART Confusion Matrix](img/a0eba0e35ba562edceef57a31926c486.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/cart_confusion_matrix.png)

CART混淆矩阵

## 准确率悖论

正如我们在这个例子中所看到的，准确率可能会产生误导。有时可能需要选择精度较低的模型，因为它对问题具有更强的预测能力。

例如，在存在大类不平衡的问题中，模型可以预测所有预测的多数类的值并实现高分类准确率，问题在于该模型在问题域中没有用。正如我们在乳腺癌中看到的那样。

这被称为[准确率悖论](http://en.wikipedia.org/wiki/Accuracy_paradox)。对于类似的问题，需要这些额外的措施来评估分类器。

## 精确

[精度](http://en.wikipedia.org/wiki/Information_retrieval#Precision)是真阳性的数量除以真阳性和假阳性的数量。换句话说，它是正预测的数量除以预测的正类值的总数。它也被称为[阳性预测值](http://en.wikipedia.org/wiki/Positive_predictive_value)（PPV）。

精度可以被认为是分类器精确度的度量。低精度也可以表示大量的误报。

*   All No Reurrence模型的精度为0 /（0 + 0）或不是数字，或0。
*   All Recurrence模型的精度为85 /（85 + 201）或0.30。
*   CART模型的精度为10 /（10 + 13）或0.43。

精度表明CART是一个更好的模型，即使它具有较低的准确度，所有重复发生比全无重复模型更有用。 All Reurrence模型和CART之间的精确度差异可以通过All Recurrence模型预测的大量误报来解释。

## 召回

[召回](http://en.wikipedia.org/wiki/Information_retrieval#Recall)是真阳性的数量除以真阳性的数量和假阴性的数量。换句话说，它是正预测的数量除以测试数据中的正类值的数量。它也称为灵敏度或真阳性率。

召回可以被认为是分类器完整性的度量。低召回率表示许多假阴性。

*   All No Recurrence模型的召回是0 /（0 + 85）或0。
*   召回全复发模型为85 /（85 + 0）或1。
*   召回CART为10 /（10 + 75）或0.12。

正如您所料，All Reurrence模型具有完美的回忆，因为它预测所有实例的“重复”。 CART的召回率低于All Recurrence模型的召回率。这可以通过CART模型预测的大量（75）假阴性来解释。

## F1得分

[F1分数](http://en.wikipedia.org/wiki/F1_score)是2 *（（精确*召回）/（精确+召回））。它也被称为F分数或F量度。换句话说，F1分数表达了精确度和召回之间的平衡。

*   全无复发模型的F1为2 *（（0 * 0）/ 0 + 0）或0。
*   全复发模型的F1为2 *（（0.3 * 1）/0.3+1）或0.46。
*   CART模型的F1为2 *（（0.43 * 0.12）/0.43+0.12）或0.19。

如果我们希望基于精确度和召回之间的平衡来选择模型，F1测量表明所有重复模型都是最佳模型，并且CART模型还没有足够的竞争力。

## 摘要

在这篇文章中，您了解了准确率悖论以及类别不平衡的问题，因为单独的分类准确率无法被信任以选择表现良好的模型。

通过示例，您了解了混淆矩阵，以此来描述未见数据集的预测中的错误细分。您了解了总结模型准确率（准确率）和召回（完整性）的措施，以及F1分数中两者之间平衡的描述。