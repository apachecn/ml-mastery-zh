# 超越预测

> 原文： [https://machinelearningmastery.com/going-beyond-predictions/](https://machinelearningmastery.com/going-beyond-predictions/)

您使用预测模型进行的预测并不重要，而是使用那些重要的预测。

[Jeremy Howard](https://www.linkedin.com/in/howardjeremy) 是机器学习竞赛平台 [Kaggle](http://www.kaggle.com/) 的总裁兼首席科学家。 2012年，他在 [O'reilly Strata会议](http://strataconf.com/)上发表了他所谓的动力传动系统方法，用于构建超越预测的“_数据产品_”。

在这篇文章中，您将发现Howard的动力传动系统方法以及如何使用它来构建系统的开发而不是做出预测。

[![The Drivetrain Approach](img/7d35b5c7e6992228a41104f4f68c9ddf.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/08/drivetrain-approach.png)

动力传动系统方法
图片来自 [O'Reilly](http://radar.oreilly.com/2012/03/drivetrain-approach-data-products.html) ，保留所有权利

## 激励方法

在投资和加入公司之前，Jeremy Howard是一位顶级的Kaggle参与者。在像数据科学运动的形成谈话中，您可以深入了解霍华德在深入挖掘数据和建立有效模型方面的敏锐能力。

在2012年的Strata演讲中，霍华德曾在Kaggle工作了一两年，并且看过很多比赛和很多竞争数据科学家。你不能不认为他更加全面的方法论是由于他对专注于预测及其准确率的沮丧而产生的。

预测是可访问的部分，它们是竞争的焦点是有道理的。我看到了他的动力传动系统方法，因为他放下了手套，并挑战社区以争取更多。

## 动力传动系统方法

霍华德在Strata 2012上发表了一个35分钟的演讲，名为“ [_从预测建模到优化：下一个前沿_](https://www.youtube.com/watch?v=vYrWTDxoeGg) ”。

&lt;iframe allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/vYrWTDxoeGg?feature=oembed" width="500"&gt;&lt;/iframe&gt;

该方法还在博客文章“ [_设计出色的数据产品：动力传动系统方法：构建数据产品的四步流程_](http://radar.oreilly.com/2012/03/drivetrain-approach-data-products.html) ”中进行了描述，该工艺也可作为[使用]独立的免费电子书](http://shop.oreilly.com/product/0636920026082.do)（完全相同的内容，我可以告诉）。

在演讲中，他介绍了他的动力传动系统方法的四个步骤：

1.  **定义目标**：我想要实现什么结果？
2.  **杠杆**：我们可以控制哪些输入？
3.  **数据**：我们可以收集哪些数据？
4.  **模型**：杠杆如何影响目标？

他描述了收集数据，因为他真正指的是对因果关系数据的需求，大多数组织都没有收集这些数据。必须通过执行大量随机实验来收集此数据。

这是关键。它超越了测试新页面标题的粉状A / B，它涉及对无偏行为的评估，例如对随机选择的建议的响应。

建模的第四步是包含以下子流程的管道：

*   **目标**：我想要达到什么样的结果。
*   **原始数据**：无偏见的因果数据
*   **建模器**：数据中因果关系的统计模型。
*   **模拟器**：插入临时输入（移动控制杆）并评估对目标的影响的能力。
*   **优化器**：使用模拟器搜索输入（离开值）以最大化（或最小化）所需结果。
*   **切实可行的结果**：用结果实现目标

## 实例探究

这种方法有点抽象，需要用一些例子来澄清。

在演示中，霍华德以谷歌搜索为例：

*   **目标**：您想阅读哪个网站？
*   **杠杆**：您可以在 [SERP](http://en.wikipedia.org/wiki/Search_engine_results_page) 上访问的网站的排序。
*   **数据**：页面之间的链接网络。
*   **模型**：未讨论，但人们会假设正在进行的实验和页面权限指标的改进。

扩展此示例，Google很可能通过注入其他结果并了解用户的行为来在SERP中执行随机体验。这将允许基于点击的可能性，用户点击的模拟以及针对给定用户的SERP中的最可点击条目的优化来构建预测模型。现在，我希望谷歌的广告可以使用像这样的方法，这可能是一个更清晰的例子。

霍华德还将营销作为建议的改进领域。他评论说，目标是CLTV的最大化。杠杆包括产品，优惠，折扣和客户服务电话的推荐。可以作为原始数据收集的因果关系将是概率或购买以及喜欢产品的概率，但不了解产品。

他还给出了最佳决策小组之前启动的例子，以最大化保险利润。他还将谷歌自动驾驶汽车作为另一个例子，而不是像现在的GPS显示那样进行粉状路线搜索。

我觉得有更多机会详细阐述这些想法。我认为，如果通过一步一步的例子以更清晰的方式介绍方法，那么对这些想法的反应就会更大。

## 摘要

超越预测的概念需要经常重复。很容易陷入一个特定的问题。我们讨论了很多关于预先定义问题的尝试，以减少这种影响。

霍华德的动力传动系统方法是一种工具，您可以使用它来设计一个系统来解决使用机器学习的复杂问题，而不是使用机器学习来做出预测并将其称为一天。

这些想法与[响应面法（RSM）](http://machinelearningmastery.com/clever-application-of-a-predictive-model/ "Clever Application Of A Predictive Model")有很多重叠。虽然没有明确说明，但Irfan Ahmad在他的[预测模型分类](http://blog.kaggle.com/2012/03/05/irfans-taxonomy-of-predictive-modelling/)中同时暗示了相关帖子中的链接，需要澄清霍华德的一些术语。