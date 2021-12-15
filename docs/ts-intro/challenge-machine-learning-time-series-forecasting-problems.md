# 10 挑战机器学习时间序列预测问题

> 原文： [https://machinelearningmastery.com/challenge-machine-learning-time-series-forecasting-problems/](https://machinelearningmastery.com/challenging-machine-learning-time-series-forecasting-problems/)

机器学习方法可以为时间序列预测问题提供很多东西。

困难在于大多数方法都是在简单的单变量时间序列预测问题上得到证明的。

在这篇文章中，您将发现一系列具有挑战性的时间序列预测问题。这些是传统线性统计方法不足以及需要更先进的机器学习方法的问题。

如果您正在寻找具有挑战性的时间序列数据集来练习机器学习技术，那么您来对地方了。

让我们潜入。

![Challenging Machine Learning Time Series Forecasting Problems](img/756585d66459845a6c28b3a4cb9f7854.jpg)

具有挑战性的机器学习时间序列预测问题
照片由 [Joao Trindade](https://www.flickr.com/photos/joao_trindade/6907788972/) ，保留一些权利。

## 概观

我们将仔细研究来自竞争数据科学网站 [Kaggle.com](https://www.kaggle.com/) 的 10 个具有挑战性的时间序列数据集。

并非所有数据集都是严格的时间序列预测问题;我在定义中已经松散，并且还包括在混淆之前的时间序列或具有明确的时间分量的问题。

他们是：

*   下雨了多少？我和我
*   在线产品销售
*   罗斯曼商店销售
*   沃尔玛招聘 - 商店销售预测
*   收购超值购物者挑战赛
*   墨尔本大学 AES / MathWorks / NIH 癫痫发作预测
*   AMS 2013-2014 太阳能预测大赛
*   2012 年全球能源预测竞赛 - 风力预测
*   EMC 数据科学全球黑客马拉松（空气质量预测）
*   Grupo Bimbo 库存需求

这不是 Kaggle 上托管的所有时间序列数据集。
我错过了一个好人吗？请在下面的评论中告诉我。

## 下雨了多少？我和我

鉴于极化雷达的观测和推导测量，问题是预测雨量计中每小时总量的概率分布。

时间结构（例如小时到小时）被删除，作为混淆数据的一部分，这将使其成为一个有趣的时间序列问题。

比赛在同一年进行了两次，使用不同的数据集：

*   [下雨了多少？](https://www.kaggle.com/c/how-much-did-it-rain)
*   [下雨了多少？ II](https://www.kaggle.com/c/how-much-did-it-rain-ii)

第二场比赛由 Aaron Sim 赢得，他使用了一个非常大的递归神经网络算法。

[博客帖子采访比赛获胜者可以在这里访问](http://blog.kaggle.com/tag/how-much-did-it-rain/)。

## 在线产品销售

鉴于产品和产品发布的细节，问题是预测未来 12 个月的销售数据。

这是一个多步预测或序列预测，没有可以推断的销售历史。

我找不到任何表现最佳的解决方案。你能？

[在竞赛页面](https://www.kaggle.com/c/online-sales)上了解更多信息。

## 罗斯曼商店销售

鉴于超过一千家商店的历史每日销售额，问题是预测每家商店的每周销售数据为 6 周。

这既提供了探索存储多步预测的机会，也提供了利用跨商店模式的能力。

通过仔细的特征工程和梯度增强的使用实现了最佳结果。

[博客帖子采访比赛获胜者可以在这里访问](http://blog.kaggle.com/tag/rossmann-store-sales/)。

[在竞赛页面](https://www.kaggle.com/c/rossmann-store-sales)上了解更多信息。

## 沃尔玛招聘 - 商店销售预测

根据多个商店中多个部门的历史每周销售数据以及促销细节，问题是预测商店部门的销售数据。

这既提供了探索部门明智甚至商店预测的机会，也提供了利用跨部门和跨店模式的能力。

表现最佳者大量使用 ARIMA 模型并谨慎处理公众假期。

请看这里[获胜解决方案](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/forums/t/8125/first-place-entry)和[第二位解决方案](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/forums/t/8023/thank-you-and-2-rank-model)。

[在竞赛页面](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)上了解更多信息。

## 收购超值购物者挑战赛

鉴于历史购物行为，问题是预测哪些客户在购买折扣优惠后可能会重复购买（获得）。

大量的事务使这个数据下载量接近 3 千兆字节。

该问题提供了对特定或聚合客户的时间序列进行建模并预测客户转换概率的机会。

我找不到任何表现最佳的解决方案。你能？

[在竞赛页面](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)上了解更多信息。

## 墨尔本大学 AES / MathWorks / NIH 癫痫发作预测

鉴于颅内脑电图观察到数月或数年的人脑活动痕迹，问题是预测 10 分钟段是否表明癫痫发作的可能性。

描述[第四位解决方案](https://www.kaggle.com/c/melbourne-university-seizure-prediction/forums/t/26098/solution-4th-place)，其利用统计特征工程和梯度增强。

[在竞赛页面](https://www.kaggle.com/c/melbourne-university-seizure-prediction)上了解更多信息。

**更新**：数据集已被删除。

## AMS 2013-2014 太阳能预测大赛

鉴于多个地点的历史气象预报，问题是预测每个地点的每日太阳能总量为一年。

数据集提供了按站点和跨站点对空间和时间时间序列建模的机会，并为每个站点进行多步预测。

[获胜方法使用了梯度增强模型](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/6321/our-approach?forumMessageId=33783#post33783)的集合。

[在竞赛页面](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest)上了解更多信息。

## 2012 年全球能源预测竞赛 - 风力预测

鉴于多个地点的历史风力预报和发电，问题是预测未来 48 小时的每小时发电量。

该数据集提供了对各个站点以及跨站点的每小时时间序列进行建模的机会。

我找不到任何表现最佳的解决方案。你能？

[在竞赛页面](https://www.kaggle.com/c/GEF2012-wind-forecasting)上了解更多信息。

## EMC 数据科学全球黑客马拉松（空气质量预测）

考虑到每天 8 小时的空气污染物测量，问题是在接下来的三天内在特定时间预测污染物。

数据集提供了对多变量时间序列建模并执行多步预测的机会。

最佳表现解决方案的[良好记录描述了使用在滞后变量上训练的随机森林模型的集合。](http://blog.kaggle.com/2012/05/01/chucking-everything-into-a-random-forest-ben-hamner-on-winning-the-air-quality-prediction-hackathon/)

[在竞赛页面](https://www.kaggle.com/c/dsg-hackathon)上了解更多信息。

## 摘要

在这篇文章中，您发现了一系列具有挑战性的时间序列预测问题。

这些问题为网站上机器学习竞赛奠定了基础 [Kaggle.com](https://www.kaggle.com/) 。因此，每个问题也提供了很好的讨论来源和现有的世界级解决方案，可以作为灵感和起点。

如果您有兴趣更好地理解机器学习在时间序列预测中的作用，我建议您选择其中一个或多个问题作为起点。

你有过这些问题中的一个或多个吗？
在下面的评论中分享您的经历。

在这篇文章中没有提到 Kaggle.com 上是否存在时间序列问题？
请在下面的评论中告诉我。