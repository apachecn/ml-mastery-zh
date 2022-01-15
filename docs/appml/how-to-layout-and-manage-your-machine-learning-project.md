# 如何布局和管理您的机器学习项目

> 原文： [https://machinelearningmastery.com/how-to-layout-and-manage-your-machine-learning-project/](https://machinelearningmastery.com/how-to-layout-and-manage-your-machine-learning-project/)

项目布局对于机器学习项目至关重要，就像软件开发项目一样。我认为它像语言。项目布局组织思想并为您提供想法的背景，就像知道事物的名称为您提供思考的基础一样。

在这篇文章中，我想强调一下在机器学习项目的布局和管理中的一些注意事项。这与[项目和科学再现性](http://machinelearningmastery.com/reproducible-machine-learning-results-by-default/ "Reproducible Machine Learning Results By Default")的目标密切相关。没有“最佳”方式，您将需要选择并采用最能满足您的偏好和项目要求的实践。

## 工作流程激励问题

Jeromy Anglim 在 2010 年墨尔本 R 用户小组的演讲中介绍了 R 的项目布局状况。视频有点摇摇欲坠，但对该主题进行了很好的讨论。

&lt;iframe allowfullscreen="" frameborder="0" height="375" src="https://www.youtube.com/embed/bbaPSJechgY?feature=oembed" width="500"&gt;&lt;/iframe&gt;

我非常喜欢 Jeromy 演讲中的动机问题：

*   将项目划分为文件和文件夹？
*   将 R 分析纳入报告？
*   将默认 R 输出转换为发布质量表，数字和文本？
*   建立最终产品？
*   对分析进行排序？
*   将代码划分为函数？

您可以在 Jeromy 的博客， [PDF 演示幻灯片](https://github.com/jeromyanglim/RMeetup_Workflow/raw/master/backup/Rmeetup_Workflow_handout.pdf)和演示文稿的 [YouTube 视频上查看演示文稿的](https://www.youtube.com/watch?v=bbaPSJechgY)[摘要。](http://jeromyanglim.blogspot.com.au/2010/12/r-workflow-slides-from-talk-at.html)

## 项目工作流程的目标

David Smith 在标题为 [R](http://blog.revolutionanalytics.com/2010/10/a-workflow-for-r.html) 的工作流程的帖子中提供了他认为的良好项目工作流程目标的摘要。我认为这些非常好，在设计自己的项目布局时应该牢记这一点。

*   **透明度**：项目的逻辑和清晰布局，使读者更直观。
*   **可维护性**：使用文件和目录的标准名称轻松修改项目。
*   **模块化**：离散任务分为单独的脚本，只有一个责任。
*   **可移植性**：轻松将项目移动到另一个系统（相对路径和已知依赖项）
*   **再现性**：在未来或其他人中轻松运行并创建相同的人工制品。
*   **效率**：很少考虑元项目细节，比如工具，更多关于你正在解决的问题。

## ProjectTemplate

John Myles White 有一个名为 [ProjectTemplate](http://projecttemplate.net/) 的 R 项目，旨在为统计分析项目自动创建一个定义良好的布局。它提供了自动加载和重叠数据的约定和实用程序。

ProjectTemplate 的徽标，用于布置 R 统计分析项目的项目。
项目布局比我想要的要大，但可以深入了解组织项目的高度结构化方式。

*   **cache** ：每次执行分析时不需要重新生成的预处理数据集。
*   **config** ：项目的配置设置
*   **数据**：原始数据文件。
*   **munge** ：预处理数据 munging 代码，其输出放在缓存中。
*   **src** ：统计分析脚本。
*   **diagnostics** ：用于诊断数据集是否存在损坏或异常值的脚本。
*   **doc** ：关于分析的文档。
*   **图**：从分析中创建的图表。
*   **lib** ：Helper 库函数但不是核心统计分析。
*   **logs** ：脚本输出和任何自动记录。
*   **分析**：用于对代码时序进行基准测试的脚本。
*   **报告**：输出报告和可能会进入报告（如表格）的内容。
*   **测试**：代码的单元测试和回归套件。
*   **README** ：指出任何新人参与项目的注释。
*   **TODO** ：您计划进行的未来改进和错误修复列表。

您可以在 [ProjectTemplate 主页](http://projecttemplate.net/)，John 的网站上的[博客文章](http://www.johnmyleswhite.com/notebook/2010/08/26/projecttemplate/) [GitHub 页面](https://github.com/johnmyleswhite/ProjectTemplate)进行开发以及 [CRAN 页面](http://cran.r-project.org/web/packages/ProjectTemplate/)进行分发了解更多信息。

## 数据管理

Software Carpentry 提供了一个标题为“数据管理”的简短演示文稿。数据管理方法的灵感来自 William Stafford Noble 题为[组织计算生物学项目快速指南](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1000424)的文章。

&lt;iframe allowfullscreen="" frameborder="0" height="375" src="https://www.youtube.com/embed/3MEJ38BO6Mo?feature=oembed" width="500"&gt;&lt;/iframe&gt;

该演示文稿描述了在磁盘或版本控制中维护多个版本数据的问题。它评论了数据存档的主要要求，并提出了一种日期目录名称和数据文件元数据文件的方法，这些文件本身是在版本控制中管理的。这是一个有趣的方法。

您可以在此处查看[视频和幻灯片以进行演示](http://software-carpentry.org/v4/data/mgmt.html)。

## 最佳实践

关于问答网站上的数据分析项目的项目布局和代码组织的最佳实践有很多讨论。例如，一些流行的例子包括：

*   [你如何管理你的文件＆amp;您的项目目录？](https://www.biostars.org/p/821/#825)
*   [统计分析和报告编写的工作流程](http://stackoverflow.com/questions/1429907/workflow-for-statistical-analysis-and-report-writing)
*   [项目组织与 R](http://stackoverflow.com/questions/13036472/project-organization-with-r)
*   [组织 R 代码和输出的有效方法是什么？](http://stats.stackexchange.com/questions/10987/what-are-efficient-ways-to-organize-r-code-and-output)

一个很好的例子是问题[如何有效地管理统计分析项目？](http://stats.stackexchange.com/questions/2910/how-to-efficiently-manage-a-statistical-analysis-project) 变成了一个描述最佳实践的社区维基。总之，这些做法分为以下几个部分：

*   **数据管理**：使用目录结构，永远不要直接修改原始数据，检查数据一致性，使用 GNU make。
*   **编码**：将代码组织成功能单元，将所有内容，自定义函数记录在专用文件中。
*   **分析**：记录随机种子，将参数分成配置文件，使用多变量图
*   **版本控制**：使用版本控制，备份所有内容，使用问题跟踪器。
*   **编辑/报告**：组合代码和报告并使用正式的报告生成器。

## 更多实践

每个项目我都试图改进我的项目布局。这很难，因为项目因数据和目标而异，语言和工具也是如此。我已经尝试了所有已编译的代码和所有脚本语言版本。我发现的一些好建议包括：

*   坚持 [POSIX 文件系统布局](http://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)（var，etc，bin，lib 等）。
*   将所有命令放在脚本中。
*   从 GNU make 目标调用所有脚本。
*   制作创建环境和下载公共数据集的目标。
*   创建秘籍并让基础结构检查并创建每次运行的任何缺少的输出产品。

最后一点是游戏改变者。它允许您管理工作流程并定义秘籍，并放弃数据分析，预处理，模型配置，功能选择等任务。框架知道如何执行秘籍并创建结果供您查看。 [我在](http://machinelearningmastery.com/the-seductive-trap-of-black-box-machine-learning/ "The Seductive Trap of Black-Box Machine Learning")之前谈过这种方法。

您如何布局和组织机器学习项目？发表评论。