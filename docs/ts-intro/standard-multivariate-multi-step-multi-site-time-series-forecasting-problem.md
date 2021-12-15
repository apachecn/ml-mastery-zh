# 标准多变量，多步骤和多站点时间序列预测问题

> 原文： [https://machinelearningmastery.com/standard-multivariate-multi-step-multi-site-time-series-forecasting-problem/](https://machinelearningmastery.com/standard-multivariate-multi-step-multi-site-time-series-forecasting-problem/)

实时世界时间序列预测具有挑战性，其原因不仅限于问题特征，例如具有多个输入变量，需要预测多个时间步骤，以及需要对多个物理站点执行相同类型的预测。

在这篇文章中，您将发现具有这些属性的标准化但复杂的时间序列预测问题，但是它很小且充分理解，可用于探索和更好地理解在具有挑战性的数据集上开发预测模型的方法。

阅读这篇文章后，你会知道：

*   解决空气质量数据集的竞争和动机。
*   概述定义的预测问题及其涵盖的数据挑战。
*   可以下载并立即开始使用的免费数据文件的说明。

让我们开始吧。

![A Standard Multivariate, Multi-Step, and Multi-Site Time Series Forecasting Problem](img/8c668fe76c196b8ed940205890106c8e.jpg)

标准多变量，多步骤和多站点时间序列预测问题
照[某人](https://www.flickr.com/photos/136665246@N05/32844473304/)，保留一些权利。

## EMC 数据科学全球黑客马拉松

该数据集被用作 Kaggle 比赛的中心。

具体而言，由数据科学伦敦和数据科学全球主办的 24 小时黑客马拉松作为大数据周活动的一部分，两个组织现在似乎不存在，6 年后。

比赛涉及数千美元的现金奖励，数据集由伊利诺伊州库克县当地政府提供，建议数据集中提到的所有位置都在该地区。

挑战的动机是开发一个更好的预测空气质量的模型，取自[竞赛描述](https://www.kaggle.com/c/dsg-hackathon)：

> EPA 的空气质量指数每天被患有哮喘和其他呼吸系统疾病的人使用，以避免可能引发攻击的危险水平的室外空气污染物。据世界卫生组织统计，目前估计有 2.35 亿人患有哮喘。在全球范围内，它现在是儿童中最常见的慢性疾病，自 1980 年以来美国的发病率翻了一番。

竞赛描述表明，获胜模型可以作为新的空气质量预测系统的基础，尽管尚不清楚是否为此目的转换了任何模型。

比赛是由一名 Kaggle 员工 [Ben Hamner](https://www.linkedin.com/in/ben-hamner-98759712/) 赢得的，根据利益冲突，他可能没有收到奖金。 Ben 在博客文章中描述了他的获胜方法，题为“[将所有东西放入随机森林：Ben Hamner 赢得空气质量预测黑客马拉松](http://blog.kaggle.com/2012/05/01/chucking-everything-into-a-random-forest-ben-hamner-on-winning-the-air-quality-prediction-hackathon/)”并在 GitHub 上提供了他的[代码。](https://github.com/benhamner/Air-Quality-Prediction-Hackathon-Winning-Model)

在这个论坛帖子中有一个很好的讨论解决方案和相关代码，标题为“[分区模型的一般方法？](https://www.kaggle.com/c/dsg-hackathon/discussion/1821) “。

## 预测建模问题

该数据描述了跨多个站点或物理位置的多变量时间序列的多步预测问题。

随着时间的推移进行多次天气测量，预测在多个物理位置的特定未来时间间隔内的一系列空气质量测量。

这是一个具有挑战性的时间序列预测问题，具有很多现实世界预测的质量：

*   **数据不完整**。并非所有天气和空气质量措施都适用于所有地点。
*   **缺少数据**。并非所有可用的措施都有完整的历史。
*   **多变量输入**：每个预测的模型输入由多个天气观测组成。
*   **多步输出**：模型输出是一系列不连续的预测空气质量测量。
*   **多站点输出**：该模式必须为多个物理站点输出多步预测。

## 下载数据集文件

该数据集可从 Kaggle 网站免费获得。

您必须先创建一个帐户并使用 Kaggle 登录，然后才能获得下载数据集的权限。

数据集可以从这里下载：

*   [竞赛数据](https://www.kaggle.com/c/dsg-hackathon/data)

## 数据集文件的说明

您必须单独下载 4 个感兴趣的文件;他们是：

### 文件：SiteLocations.csv

此文件包含由唯一标识符标记的站点位置列表，以及它们在地球上按经度和纬度测量的精确位置。

所有坐标在西北半球似乎相对较近，例如美国。

以下是该文件的示例。

```py
"SITE_ID","LATITUDE","LONGITUDE"
1,41.6709918952829,-87.7324568962847
32,41.755832412403,-87.545349670582
50,41.7075695897648,-87.5685738570845
57,41.9128621248178,-87.7227234452095
64,41.7907868783739,-87.6016464917605
...
```

### 文件：SiteLocations_with_more_sites.csv

此文件具有与`SiteLocations.csv`相同的格式，并且似乎列出与该文件相同的所有位置以及一些其他位置。

正如文件名所示，它只是网站列表的更新版本。

以下是该文件的示例。

```py
"SITE_ID","LATITUDE","LONGITUDE"
1,41.6709918952829,-87.7324568962847
14,41.834243,-87.6238
22,41.6871654376343,-87.5393154841479
32,41.755832412403,-87.545349670582
50,41.7075695897648,-87.5685738570845
...
```

### 文件：TrainingData.csv

该文件包含用于建模的训练数据。

数据以非标准化的方式呈现。每行数据包含一组跨越多个位置的一小时的气象测量值以及该小时的每个位置的目标或结果。

措施包括：

*   时间信息，包括时间块，连续时间块内的索引，平均月份，星期几和一天中的小时。
*   风测量，如方向和速度。
*   温度测量，例如最小和最大环境温度。
*   压力测量，如最小和最大气压。

目标变量是不同物理位置的不同空气质量或污染测量的集合。

并非所有地点都有全天候测量，并非所有地点都与所有目标措施有关。此外，对于那些记录的变量，存在标记为 NA 的缺失值。

以下是该文件的示例。

```py
"rowID","chunkID","position_within_chunk","month_most_common","weekday","hour","Solar.radiation_64","WindDirection..Resultant_1","WindDirection..Resultant_1018","WindSpeed..Resultant_1","WindSpeed..Resultant_1018","Ambient.Max.Temperature_14","Ambient.Max.Temperature_22","Ambient.Max.Temperature_50","Ambient.Max.Temperature_52","Ambient.Max.Temperature_57","Ambient.Max.Temperature_76","Ambient.Max.Temperature_2001","Ambient.Max.Temperature_3301","Ambient.Max.Temperature_6005","Ambient.Min.Temperature_14","Ambient.Min.Temperature_22","Ambient.Min.Temperature_50","Ambient.Min.Temperature_52","Ambient.Min.Temperature_57","Ambient.Min.Temperature_76","Ambient.Min.Temperature_2001","Ambient.Min.Temperature_3301","Ambient.Min.Temperature_6005","Sample.Baro.Pressure_14","Sample.Baro.Pressure_22","Sample.Baro.Pressure_50","Sample.Baro.Pressure_52","Sample.Baro.Pressure_57","Sample.Baro.Pressure_76","Sample.Baro.Pressure_2001","Sample.Baro.Pressure_3301","Sample.Baro.Pressure_6005","Sample.Max.Baro.Pressure_14","Sample.Max.Baro.Pressure_22","Sample.Max.Baro.Pressure_50","Sample.Max.Baro.Pressure_52","Sample.Max.Baro.Pressure_57","Sample.Max.Baro.Pressure_76","Sample.Max.Baro.Pressure_2001","Sample.Max.Baro.Pressure_3301","Sample.Max.Baro.Pressure_6005","Sample.Min.Baro.Pressure_14","Sample.Min.Baro.Pressure_22","Sample.Min.Baro.Pressure_50","Sample.Min.Baro.Pressure_52","Sample.Min.Baro.Pressure_57","Sample.Min.Baro.Pressure_76","Sample.Min.Baro.Pressure_2001","Sample.Min.Baro.Pressure_3301","Sample.Min.Baro.Pressure_6005","target_1_57","target_10_4002","target_10_8003","target_11_1","target_11_32","target_11_50","target_11_64","target_11_1003","target_11_1601","target_11_4002","target_11_8003","target_14_4002","target_14_8003","target_15_57","target_2_57","target_3_1","target_3_50","target_3_57","target_3_1601","target_3_4002","target_3_6006","target_4_1","target_4_50","target_4_57","target_4_1018","target_4_1601","target_4_2001","target_4_4002","target_4_4101","target_4_6006","target_4_8003","target_5_6006","target_7_57","target_8_57","target_8_4002","target_8_6004","target_8_8003","target_9_4002","target_9_8003"
1,1,1,10,"Saturday",21,0.01,117,187,0.3,0.3,NA,NA,NA,14.9,NA,NA,NA,NA,NA,NA,NA,NA,5.8,NA,NA,NA,NA,NA,NA,NA,NA,747,NA,NA,NA,NA,NA,NA,NA,NA,750,NA,NA,NA,NA,NA,NA,NA,NA,743,NA,NA,NA,NA,NA,2.67923294292042,6.1816228132982,NA,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,NA,2.38965627997991,NA,5.56815355612325,0.690015329704154,NA,NA,NA,NA,NA,NA,2.84349016287551,0.0920223353681394,1.69321097077376,0.368089341472558,0.184044670736279,0.368089341472558,0.276067006104418,0.892616653070952,1.74842437199465,NA,NA,5.1306307034019,1.34160578423204,2.13879182993514,3.01375212399952,NA,5.67928016629218,NA
2,1,2,10,"Saturday",22,0.01,231,202,0.5,0.6,NA,NA,NA,14.9,NA,NA,NA,NA,NA,NA,NA,NA,5.8,NA,NA,NA,NA,NA,NA,NA,NA,747,NA,NA,NA,NA,NA,NA,NA,NA,750,NA,NA,NA,NA,NA,NA,NA,NA,743,NA,NA,NA,NA,NA,2.67923294292042,8.47583334194495,NA,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,NA,1.99138023331659,NA,5.56815355612325,0.923259948195698,NA,NA,NA,NA,NA,NA,3.1011527019063,0.0920223353681394,1.94167127626774,0.368089341472558,0.184044670736279,0.368089341472558,0.368089341472558,1.73922213845783,2.14412041407765,NA,NA,5.1306307034019,1.19577906855465,2.72209869264472,3.88871241806389,NA,7.42675098668978,NA
3,1,3,10,"Saturday",23,0.01,247,227,0.5,1.5,NA,NA,NA,14.9,NA,NA,NA,NA,NA,NA,NA,NA,5.8,NA,NA,NA,NA,NA,NA,NA,NA,747,NA,NA,NA,NA,NA,NA,NA,NA,750,NA,NA,NA,NA,NA,NA,NA,NA,743,NA,NA,NA,NA,NA,2.67923294292042,8.92192983362627,NA,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,NA,1.7524146053186,NA,5.56815355612325,0.680296803933673,NA,NA,NA,NA,NA,NA,3.06434376775904,0.0920223353681394,2.52141198908702,0.460111676840697,0.184044670736279,0.368089341472558,0.368089341472558,1.7852333061419,1.93246904273093,NA,NA,5.13639545700122,1.40965825154816,3.11096993445111,3.88871241806389,NA,7.68373198968942,NA
4,1,4,10,"Sunday",0,0.01,219,218,0.2,1.2,NA,NA,NA,14,NA,NA,NA,NA,NA,NA,NA,NA,4.8,NA,NA,NA,NA,NA,NA,NA,NA,751,NA,NA,NA,NA,NA,NA,NA,NA,754,NA,NA,NA,NA,NA,NA,NA,NA,748,NA,NA,NA,NA,NA,2.67923294292042,5.09824561921501,NA,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,NA,2.38965627997991,NA,5.6776192223642,0.612267123540305,NA,NA,NA,NA,NA,NA,3.21157950434806,0.184044670736279,2.374176252498,0.460111676840697,0.184044670736279,0.368089341472558,0.276067006104418,1.86805340797323,2.08890701285676,NA,NA,5.21710200739181,1.47771071886428,2.04157401948354,3.20818774490271,NA,4.83124285639335,NA
5,1,5,10,"Sunday",1,0.01,2,216,0.2,0.3,NA,NA,NA,14,NA,NA,NA,NA,NA,NA,NA,NA,4.8,NA,NA,NA,NA,NA,NA,NA,NA,751,NA,NA,NA,NA,NA,NA,NA,NA,754,NA,NA,NA,NA,NA,NA,NA,NA,748,NA,NA,NA,NA,NA,2.67923294292042,4.87519737337435,NA,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,0.114975168664303,NA,2.31000107064725,NA,5.6776192223642,0.694874592589394,NA,NA,NA,NA,NA,NA,3.67169118118876,0.184044670736279,2.46619858786614,0.460111676840697,0.184044670736279,0.368089341472558,0.276067006104418,1.70241320431058,2.60423209091834,NA,NA,5.21710200739181,1.45826715677396,2.13879182993514,3.4998411762575,NA,4.62565805399363,NA
...
```

### 文件：SubmissionZerosExceptNAs.csv

此文件包含预测问题的提交样本。

每行指定在一段连续时间内针对给定小时的所有目标位置的每个目标度量的预测。

以下是该文件的示例。

```py
"rowID","chunkID","position_within_chunk","hour","month_most_common","target_1_57","target_10_4002","target_10_8003","target_11_1","target_11_32","target_11_50","target_11_64","target_11_1003","target_11_1601","target_11_4002","target_11_8003","target_14_4002","target_14_8003","target_15_57","target_2_57","target_3_1","target_3_50","target_3_57","target_3_1601","target_3_4002","target_3_6006","target_4_1","target_4_50","target_4_57","target_4_1018","target_4_1601","target_4_2001","target_4_4002","target_4_4101","target_4_6006","target_4_8003","target_5_6006","target_7_57","target_8_57","target_8_4002","target_8_6004","target_8_8003","target_9_4002","target_9_8003"
193,1,193,21,10,0,0,-1e+06,0,0,0,0,0,0,0,-1e+06,0,-1e+06,0,0,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,0,0,0,0,0,0,0,0,0,-1e+06,-1e+06,0,0,0,0,-1e+06,0,-1e+06
194,1,194,22,10,0,0,-1e+06,0,0,0,0,0,0,0,-1e+06,0,-1e+06,0,0,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,0,0,0,0,0,0,0,0,0,-1e+06,-1e+06,0,0,0,0,-1e+06,0,-1e+06
195,1,195,23,10,0,0,-1e+06,0,0,0,0,0,0,0,-1e+06,0,-1e+06,0,0,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,0,0,0,0,0,0,0,0,0,-1e+06,-1e+06,0,0,0,0,-1e+06,0,-1e+06
196,1,196,0,10,0,0,-1e+06,0,0,0,0,0,0,0,-1e+06,0,-1e+06,0,0,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,0,0,0,0,0,0,0,0,0,-1e+06,-1e+06,0,0,0,0,-1e+06,0,-1e+06
197,1,197,1,10,0,0,-1e+06,0,0,0,0,0,0,0,-1e+06,0,-1e+06,0,0,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,-1e+06,0,0,0,0,0,0,0,0,0,-1e+06,-1e+06,0,0,0,0,-1e+06,0,-1e+06
...
```

## 构建预测问题

这个预测问题的很大一部分挑战是可以为建模设置问题的大量方法。

这是具有挑战性的，因为不清楚哪个框架可能是这个特定建模问题的最佳框架。

例如，下面是一些问题，可以引发关于如何构建问题的思考。

*   是否更好地归咎或忽略遗漏的观察结果？
*   以时间序列的天气观测或仅观察当前时间的观测结果是否更好？
*   是否更好地使用来自一个或多个源位置的天气观测来做出预测？
*   为每个位置设置一个模型或为所有位置设置一个模式更好吗？
*   每个预测时间有一个模型或者所有预测时间都有一个模型更好吗？

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [EMC 数据科学全球黑客马拉松（空气质量预测）](https://www.kaggle.com/c/dsg-hackathon)
*   [下载数据集](https://www.kaggle.com/c/dsg-hackathon/data)
*   [将所有东西放入随机森林：Ben Hamner 赢得空气质量预测黑客马拉松](http://blog.kaggle.com/2012/05/01/chucking-everything-into-a-random-forest-ben-hamner-on-winning-the-air-quality-prediction-hackathon/)
*   [EMC 数据科学全球黑客马拉松（空气质量预测）的获奖代码](https://github.com/benhamner/Air-Quality-Prediction-Hackathon-Winning-Model)
*   [分区模型的一般方法？](https://www.kaggle.com/c/dsg-hackathon/discussion/1821)

## 摘要

在这篇文章中，您发现了 Kaggle 空气质量数据集，该数据集为复杂的时间序列预测提供了标准数据集。

具体来说，你学到了：

*   解决空气质量数据集的竞争和动机。
*   概述定义的预测问题及其涵盖的数据挑战。
*   可以下载并立即开始使用的免费数据文件的说明。

你有没有研究过这个数据集，或者你打算做什么？
在下面的评论中分享您的经历。