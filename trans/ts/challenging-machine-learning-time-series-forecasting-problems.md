# 10 个具有挑战性的机器学习时间序列预测问题

> 原文:[https://machinelearning master . com/challenge-machine-learning-时序-预测-问题/](https://machinelearningmastery.com/challenging-machine-learning-time-series-forecasting-problems/)

最后更新于 2019 年 8 月 21 日

机器学习方法可以为时间序列预测问题提供很多帮助。

困难在于大多数方法都是在简单的单变量时间序列预测问题上演示的。

在这篇文章中，你将发现一系列具有挑战性的时间序列预测问题。在这些问题中，经典的线性统计方法是不够的，需要更先进的机器学习方法。

如果您正在寻找具有挑战性的时间序列数据集来练习机器学习技术，那么您就来对地方了。

**用我的新书[用 Python 进行时间序列预测](https://machinelearningmastery.com/introduction-to-time-series-forecasting-with-python/)来启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

让我们开始吧。

![Challenging Machine Learning Time Series Forecasting Problems](img/4b40042628884bd307da9afa2ca9f47a.png)

挑战机器学习时间序列预测问题
图片由 [Joao Trindade](https://www.flickr.com/photos/joao_trindade/6907788972/) 提供，保留部分权利。

## 概观

我们将从竞争数据科学网站[Kaggle.com](https://www.kaggle.com/)上仔细查看 10 个具有挑战性的时间序列数据集。

不是所有的数据集都是严格的时间序列预测问题；我在定义上比较宽松，也包括了在混淆之前是时间序列的问题，或者有明确的时间成分的问题。

它们是:

*   雨下了多少？一和二
*   在线产品销售
*   罗斯曼商店销售
*   沃尔玛招聘-商店销售预测
*   获得有价值的购物者挑战
*   墨尔本大学 AES/MathWorks/NIH 癫痫发作预测
*   AMS 2013-2014 太阳能预测大赛
*   2012 年全球能源预测竞赛-风力预测
*   EMC 数据科学全球黑客马拉松(空气质量预测)
*   Grupo Bimbo 库存需求

这并不是 Kaggle 上托管的所有时间序列数据集。
我错过了一个好的吗？请在下面的评论中告诉我。

## 雨下了多少？一和二

给定来自极化雷达的观测值和衍生测量值，问题是预测雨量计每小时总量的概率分布。

作为混淆数据的一部分，时间结构(例如小时到小时)被移除，这将使它成为一个有趣的时间序列问题。

比赛在同一年用不同的数据集进行了两次:

*   [下了多少雨？](https://www.kaggle.com/c/how-much-did-it-rain)
*   [下了多少雨？二](https://www.kaggle.com/c/how-much-did-it-rain-ii)

第二场比赛的获胜者是艾伦·西姆，他使用了一种非常大的递归神经网络算法。

[在这里可以看到采访比赛获胜者的博文](http://blog.kaggle.com/tag/how-much-did-it-rain/)。

## 在线产品销售

鉴于产品和产品发布的细节，问题是预测未来 12 个月的销售数字。

这是一个多步预测或序列预测，没有可从中推断的销售历史。

我找不到任何优秀的解决方案。你能吗？

[在竞赛页面](https://www.kaggle.com/c/online-sales)了解更多信息。

## 罗斯曼商店销售

给定一千多家商店的历史日销售额，问题是预测每家商店 6 周的日销售额。

这既提供了探索商店多步预测的机会，也提供了利用跨商店模式的能力。

通过精心的特征工程和梯度增强的使用，取得了最好的结果。

[在这里可以看到采访比赛获胜者的博文](http://blog.kaggle.com/tag/rossmann-store-sales/)。

[在竞赛页面](https://www.kaggle.com/c/rossmann-store-sales)了解更多信息。

## 沃尔玛招聘-商店销售预测

给定多个商店中多个部门的历史每周销售数据，以及促销的详细信息，问题是预测商店部门的销售数字。

这既提供了探索部门甚至商店预测的机会，也提供了利用跨部门和跨商店模式的能力。

表现最好的人大量使用 ARIMA 模型，并谨慎处理公共假期。

请参见此处的[获胜解和此处](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/forums/t/8125/first-place-entry)的[第二名解。](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/forums/t/8023/thank-you-and-2-rank-model)

[在竞赛页面](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)了解更多信息。

## 获得有价值的购物者挑战

给定历史购物行为，问题是预测哪些客户在接受折扣后可能会重复购买(被收购)。

大量的交易使这成为一个大数据下载，近 3gb。

这个问题提供了一个机会来模拟特定或聚集客户的时间序列，并预测客户转换的概率。

我找不到任何优秀的解决方案。你能吗？

[在竞赛页面](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)了解更多信息。

## 墨尔本大学 AES/MathWorks/NIH 癫痫发作预测

给定用颅内脑电图观察数月或数年的人脑活动痕迹，问题是预测 10 分钟片段是否表明癫痫发作的概率。

描述了一种利用统计特征工程和梯度增强的第四位解决方案。

[在竞赛页面](https://www.kaggle.com/c/melbourne-university-seizure-prediction)了解更多信息。

**更新**:数据集已经被取下来。

## AMS 2013-2014 太阳能预测大赛

给定多个站点的历史气象预报，问题是预测每个站点一年的日总太阳能。

数据集提供了按站点和跨站点建模空间和时间序列并对每个站点进行多步预测的机会。

[获胜的方法使用了一组梯度增强模型](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/6321/our-approach?forumMessageId=33783#post33783)。

[在竞赛页面](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest)了解更多信息。

## 2012 年全球能源预测竞赛-风力预测

给定多个站点的历史风力预测和发电量，问题是预测未来 48 小时的每小时发电量。

该数据集提供了为单个站点和跨站点每小时时间序列建模的机会。

我找不到任何优秀的解决方案。你能吗？

[在竞赛页面](https://www.kaggle.com/c/GEF2012-wind-forecasting)了解更多信息。

## EMC 数据科学全球黑客马拉松(空气质量预测)

给定八天的空气污染物小时测量，问题是预测未来三天特定时间的污染物。

数据集提供了对多元时间序列建模和执行多步预测的机会。

对表现最好的解决方案的很好的描述描述了对滞后变量进行训练的随机森林模型集合的使用。

[在竞赛页面](https://www.kaggle.com/c/dsg-hackathon)了解更多信息。

## 摘要

在这篇文章中，你发现了一系列具有挑战性的时间序列预测问题。

这些问题为现场[Kaggle.com](https://www.kaggle.com/)的竞技机器学习提供了基础。因此，每个问题也提供了很好的讨论来源和现有的世界级解决方案，可以作为灵感和起点。

如果你有兴趣更好地理解机器学习在时间序列预测中的作用，我建议选择其中的一个或多个问题作为起点。

你研究过这些问题中的一个或多个吗？
在下面的评论中分享你的经历。

关于 Kaggle.com 有没有一个时间序列的问题是这篇文章没有提到的？
在下面的评论中告诉我。