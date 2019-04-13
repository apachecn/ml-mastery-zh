# 项目焦点：Shashank Singh 的人脸识别

> 原文： [https://machinelearningmastery.com/project-spotlight-face-recognition-with-shashank-singh/](https://machinelearningmastery.com/project-spotlight-face-recognition-with-shashank-singh/)

Shashank Singh 是程序员和机器学习爱好者，这是一个项目焦点。

## 你能介绍一下自己吗？

我做过计算机科学技术学士学位。我在 23 岁的时候共同创立了一家创业公司，在 26 岁生日时大获成功。在那之后，我感觉特别低，很长一段时间都没有灵感。

[![Shashank Singh](img/5c3a9a14d80e94f1a0d9a8425aedf533.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/04/Shashank-Singh.jpg)

Shashank Singh

我搬到印度孟买加入[田园软件](http://www.idyllic-software.com/)，我接触了那些对生活有如此不同观点的神奇人物，并创建了一个名为“喝咖啡休息”的问题解决者的小型非正式聚会。

当我看到两个孩子在我经常去的酒吧外面乞讨食物时，对我来说生活改变的时刻就在眼前。我知道我想以任何方式帮助这些孩子。这启动了一个思考过程，导致了我的项目 _Helping Faceless_ 。

## 你的项目叫什么，它做了什么？

[Helping Faceless](http://www.helpingfaceless.com/) 项目（和 Android 应用程序）正试图通过使用最先进的人脸识别和数据分析来打击贩卖儿童的行为。

[![Child Beggars](img/aa27e439b1022c484c6a7b8fbdce34ea.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/04/child-beggers.jpg)

儿童乞丐

## 你是怎么开始的？

我们从一个简单的 Ruby on Rails API 服务器开始，接受来自应用程序和其他来源的信息。我们一直在缓慢但稳定地增加这个简单服务器的复杂性，以创建更多功能。

为了保持日益复杂的检查，我们使用面向服务的体系结构，整个系统被分解为更小的模块化应用程序，在线上相互连接。所以最后我们使用最适合手头任务的语言或框架。

[![Helping Faceless App](img/416db4700e98f262ccea58d47a8fd149.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/04/helping-faceless-app.jpg)

帮助匿名应用程序

我们目前的技术堆栈如下：

*   服务器端：Ruby on Rails
*   客户端：适用于 Android 的 Java，适用于 IOS 的 Objective C，适用于 NGO 的 Web 前端
*   分析：Python（Scipy / Pandas / Numpy / scipy.stats FTW !!）。我们正在整合 [Apache Storm](http://storm.incubator.apache.org/) 和 [Apache Mahout](https://mahout.apache.org/) 进行分析和后续报告生成。

我们使用 [Heroku](https://www.heroku.com/) ， [Linode](https://www.linode.com) 作为 VPS。 [Airbrake](https://airbrake.io/) 的家伙们很棒，他们帮助我们提供了一个更强大的免费帐户来捕捉错误和错误。此外，我们还使用 Heap Analytics 来根据流量来确定服务使用情况。

对于面部识别需求，我们使用密歇根大学的名为 [OpenBR](http://openbiometrics.org/) （Open Biometrics）的库。它的模块化设计使其更容易放入我们的管道中（参见 2013 年论文[开源生物识别](http://openbiometrics.org/publications/klontz2013open.pdf)）。这种模块化设计使其具有优于 [OpenCV](http://opencv.org/) 的独特优势，同时使实验变得非常简单。

如果你想帮助我们我们的代码是[可以在 Github](https://github.com/shashanksingh/face_rec_server) 上找到，只需分叉并开始编码

## 你做了哪些有趣的发现？

人脸识别在电视节目中几乎听起来很神奇，但实际上除非你像 Facebook 这样的科技巨头，否则它几乎很糟糕。

我们通过建立类似于油井制造工艺的流程来规避这种高错误率。到达我们系统的每一段智能都经过验证，然后转化为可理解的块或组。

照片进入一个单独的管道，彼此匹配，以创建一个巨大的相似性矩阵。然后我们采用前 20％的相似性得分图像并通过我们的众包部分运行它们以供人们验证我们的假设，这消除了误报并为我们提供了更原始的数据点，然后通过更好的第三方人脸识别算法进一步筛选。

此外，我们正在使用 [Apache Mahout](https://mahout.apache.org/) 建立关于此数据的高级报告和情报系统。

## 你想在项目上做什么？

**错误**

*   面部识别阶段的错误检测率较低，我们正在关注你的 Facebook（参见 Facebook 出版物 [DeepFace：在面部验证中关闭人类绩效](https://www.facebook.com/publications/546316888800776/)）[
    ](https://www.facebook.com/publications/546316888800776/)
*   更好的性别和年龄检测。

**特色**

理想的愿望清单如此之大，我们不得不修剪它以适应现实的时间表，但这些是我很想拥有的东西。

*   承诺的游戏化和基于贡献的频率。
*   App 侧面识别。
*   儿童失踪时的实时警报。

**愿景**

*   把它带到全国乃至菲律宾等东南亚国家。
*   人口贩运：目前我们用于人脸识别的模型仅在 10-20 岁的时候进行了训练，我们希望通过增加训练数据来扩展它。
*   为非政府组织和政府组织建立一个安全共享数据的平台。

我们的幻灯片让我们更好地了解了我们的愿景和目标：[帮助无脸滑板](http://www.haikudeck.com/helping-faceless-education-presentation-98L1RQtn2X)

## 学到更多

*   App： [http://bit.ly/HelpingFacelessMInterview](http://bit.ly/HelpingFacelessMInterview)
*   Facebook： [https://www.facebook.com/helpingfaceless](https://www.facebook.com/helpingfaceless)
*   Twitter： [http://www.twitter.com/helpingfaceless](http://www.twitter.com/helpingfaceless)
*   网站： [http://www.helpingfaceless.com](http://www.helpingfaceless.com)

**你有机器学习方面的项目吗？**

如果你有一个有趣的机器学习方面的项目，并有兴趣像 Shashank 一样被描述，请[与我联系](http://machinelearningmastery.com/contact/ "Contact")。