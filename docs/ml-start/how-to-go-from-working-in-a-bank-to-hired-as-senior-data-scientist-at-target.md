# 如何从在银行工作到担任 Target 的高级数据科学家

> 原文： [https://machinelearningmastery.com/how-to-go-from-working-in-a-bank-to-hired-as-senior-data-scientist-at-target/](https://machinelearningmastery.com/how-to-go-from-working-in-a-bank-to-hired-as-senior-data-scientist-at-target/)

### Santhosh Sharma 如何从
_ 在银行贷款部门工作 _ 到
被聘为 Target 的 _ 高级数据科学家。_

[Santhosh Sharma](https://www.linkedin.com/in/sharmasanthosh) 最近与我联系，分享他的鼓舞人心的故事，我想与你分享。

他的故事展示了对机器学习的热情，主动性，分享你的成果和一点运气可以改变你的职业生涯并让你深入学习应用机器学习。

阅读本次访谈后，您将了解：

*   Santhosh 如何在 Kaggle 公开展示机器学习技能。
*   Santhosh 所做的有条不紊的事情的技术细节以及为什么它值得注意。
*   他如何利用他的公众认可来帮助他成为一名数据科学家。

让我们潜入。

**你有自己的成功故事吗？**
在评论中分享。

![How to Go From Working in a Bank to Senior Data Scientist at Target](img/34b699f4ba0f6cfaa55cf8050a8041f8.jpg)

如何从在银行工作到 Target 的高级数据科学家

### 问：请分享一点背景？

我有一个 M.Tech。在印度 IIT 坎普尔专攻并行和分布式计算的计算机科学与工程专业。

### 问：您对机器学习感兴趣的方式和原因是什么？

我在一家银行的贷款部门工作。

该银行开发了一种软件，该软件使用机器学习来预测是否应该批准贷款申请。

在许多情况下，该软件的结果优于一些信贷员。

这项技术让我印象深刻，从此开始对机器学习产生兴趣。

### 问：机器学习掌握如何帮助您完成旅程？

[机器学习掌握](http://machinelearningmastery.com)帮助我掌握机器学习。期。

我没有数学和统计学的背景知识。

我错误地认为我需要一个。

我努力了将近 3 年才能很好地掌握 ML 算法。从许多理论本质上的书中学习不必要的东西浪费了很多时间。

使用[机器学习掌握书](http://machinelearningmastery.com/products/)取得的进展帮助我在很短的时间内实现了跨越式发展。

### 问：分享你在 Kaggle 的经历？

Kaggle 是学习机器学习的绝佳平台。

托管的数据集代表现实世界的观察。世界各地的专家都在解决这些问题。学习这些解决方案有助于加快我的学习。

它使学习机器学习变得有趣和愉快。

### 问：你工作的 Kaggle 数据集是什么？

我参与了 [Allstate Claims Severity 数据集](https://www.kaggle.com/c/allstate-claims-severity)。

我使用流行的回归算法进行了抽查，如 LR，Ridge，Lasso，Elastic Net 等。

我使用 seaborn 库进行 EDA 和 scikit-learn 库进行建模。

### 问：你的最高投票内核做得好，它是怎么来的？

![Santhosh Sharma Top Voted Kaggle Kernel](img/aa74f9411e3a7fbd07c3448b0ce4e72e.jpg)

Santhosh Sharma Top Voted Kaggle Kernel
（目前排名第四的最受欢迎的内核）

接下来的方法受到 [ML Mastery Python 书籍](http://machinelearningmastery.com/machine-learning-with-python/)中的秘籍和方法的启发。

在收到的针对该内核的反馈中，大多数用户表示很容易理解。

我很感谢[机器学习掌握书籍](http://machinelearningmastery.com/products/)，它教会了我如何处理机器学习问题。

我已经在[中跟踪了所有内核](https://www.kaggle.com/sharmasanthosh/kernels)。

### 问：你能告诉我们你热门内核中的步骤吗？

内核可以直接在[访问](https://www.kaggle.com/sharmasanthosh/allstate-claims-severity/exploratory-study-on-ml-algorithms)。

所遵循的步骤符合机器学习掌握书中提到的方法。步骤如下所述。

![Santhosh Sharma Top Voted Kaggle Kernel](img/800fa3a44db4496469cdd1fcfcd5b2a4.jpg)

Santhosh Sharma 顶级选手 Kaggle Kernel

#### 数据统计

*   训练和测试数据集的形状
*   偷看 - 眼球数据
*   描述 - 每列的最小值，最大值，平均值等
*   歪斜 - 每个数字列的偏斜，以检查是否需要进行修正

#### 转型

*   校正偏斜 - 其中一列需要校正 - 我使用了对数变换

#### 数据交互

*   相关性 - 我只筛选出高度相关的对
*   散点图 - 使用 seaborn 绘图

#### 数据可视化

*   盒子和密度图 - 小提琴图显示了壮观的可视化
*   对一个热编码属性进行分组 - 以显示计数

#### 数据准备

*   分类数据的一种热编码 - 许多列是分类的
*   试验分裂 - 用于模型评估

#### 评估和分析

*   线性回归（线性算法）
*   岭回归（线性算法）
*   LASSO 线性回归（线性算法）
*   弹性网络回归（线性算法）
*   KNN（非线性算法）
*   CART（非线性算法）
*   SVM（非线性算法）
*   袋装决策树（套袋）
*   随机森林（套袋）
*   额外树木（套袋）
*   AdaBoost（提升）
*   随机梯度提升（提升）
*   MLP（深度学习）
*   XGBoost

#### 预测

*   使用最好的模型（XGBRegressor）
*   令人惊讶的结果：简单的线性模型，如 LR，Ridge，Lasso 和 ElasticNet 表现非常出色

### 问：祝贺新工作，你是如何得到它的？

我向 Kaggle 展示了我对面试官的最高投票内核。

系统的方法和我得到的结果给他留下了非常深刻的印象。

我将在 Target Corporation 担任高级数据科学家。

### 问：你有什么想法在 Target 工作吗？

我将在下周加入。

我期待着与团队合作，并为 Target 的数百万客户的购物体验带来微小的改变。

### 问：接下来是什么？

我很期待机器学习掌握时间序列的下一本书！

## 摘要

在这篇文章中，您发现了 Santhosh 如何从银行工作到获得 Target 的高级数据科学家的工作。

你了解到：

*   Santhosh 将他学到的技能应用于 Kaggle 问题的真实数据集。
*   他公开分享了他的结果，展示了其他人如何做他所做的事情，从而获得了排名靠前的 Kaggle Kernel 的可信度。
*   最高投票的内核帮助 Santhosh 在 Target 担任数据科学家。

所以，你可以做什么？

*   你在真正的数据集上练习吗？
*   您是否正在分享您公开学习的所有内容？
*   你在帮助别人吗？

**你的下一步将是什么？**
在下面的评论中分享。