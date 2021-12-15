# 项目聚焦：使用 Artem Yankov 在 Python 中推荐事件

> 原文： [https://machinelearningmastery.com/project-spotlight-with-artem-yankov/](https://machinelearningmastery.com/project-spotlight-with-artem-yankov/)

这是 Artem Yankov 的项目聚焦。

## 你能介绍一下自己吗？

我叫 Artem Yankov，过去 3 年我曾在 [Badgeville](http://badgeville.com/) 担任软件工程师。我在那里使用 Ruby 和 Scala，虽然我之前的背景包括使用各种语言，如：汇编，C / C ++，Python，Clojure 和 JS。

我喜欢黑客攻击小项目并探索不同的领域，例如我看过的两个几乎随机的领域是机器人和恶意软件分析。我不能说我成了专家，但我确实有很多乐趣。我制造的一个小型机器人看起来非常难看，但它可以通过 [MS Kinect](http://www.microsoft.com/en-us/kinectforwindows/) “看到”它来反映我的手臂动作。

直到去年我完成 [Andrew Ng 关于 Coursera](https://www.coursera.org/course/ml) 的课程并且我真的很喜欢它时，我根本没有做任何机器学习。

## 你的项目叫什么，它做了什么？

该项目名为 [hapsradar.com](http://hapsradar.com) ，它是一个活动推荐网站，专注于现在或不久的将来发生的事情。

我是一个可怕的周末计划者，我常常想知道如果我突然决定在我的家/互联网之外做一些事情该怎么办。我发现正在发生的事情的典型算法是访问 meetup.com 和 eventbrite 等网站，浏览大量类别，点击大量按钮并阅读当前事件列表。

因此，当我完成机器学习课程并开始寻找项目来练习我的技能时，我认为我可以通过从这些网站获取事件列表然后根据我喜欢的方式构建推荐来真正自动化此事件搜索过程。

[![HapsRadar by Artem Yankov](img/e715ac239501fe3fea7f5fda20667d1d.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/03/HapsRadar.png)

HapsRadar by Artem Yankov

该网站非常简约，目前只提供来自两个网站的活动： [meetup.com](http://www.meetup.com/) 和 [eventbrite.com](http://www.eventbrite.com/) 。用户需要在推荐引擎启动之前评估至少 100 个事件。然后它每晚运行并使用用户喜欢/不喜欢进行训练，然后尝试预测用户可能喜欢的事件。

## 你是怎么开始的？

我的开始只是因为我想练习我的机器学习技巧，并让我更有趣，我选择解决一个真正的问题。经过一番评估后，我决定使用 python 作为我的推荐人。这是我使用的工具：

*   [熊猫](http://pandas.pydata.org/)
*   [Scikit-Learn](http://scikit-learn.org/stable/)
*   [PostgreSQL](http://www.postgresql.org/)
*   Scala / [PlayFramework](http://www.playframework.com/) （用于事件提取器和网站）

使用 meetup.com 和 eventbrite.com 提供的标准 API 获取事件并存储在 postgresql 中。在我开始使用爬虫之前，我通过电子邮件仔细检查了我是否可以做这样的事情，特别是因为我想每天运行这些爬虫来保持我的数据库更新所有事件。

这些人非常好，eventbrite 甚至没有任何问题地提高了我的 API 速率限制。 meetup.com 有一个很好的流媒体 API，允许您订阅所有发生的变化。我想抓住 [yelp.com](http://www.yelp.com) ，因为他们有事件列表，但是他们完全禁止这样做。

在我第一次删除数据后，我构建了一个简单的站点，在给定的邮政编码的某个范围内显示事件（我目前只为美国提取事件）。

现在是推荐部分。构建我的功能的主要材料是事件标题和事件描述。我决定事件发生的时间，或者离家多远的事情都不会增加很多价值，因为我只想简单回答一下问题：这个事件是否与我的兴趣有关？

### **想法＃1。预测主题**

一些获取的事件具有标签或类别，其中一些没有。

最初我认为我应该尝试使用标记事件来预测未标记事件的标记，然后将它们用作训练功能。花了一些时间后，我认为这可能不是一个好主意。大多数标记事件只有 1-3 个标记，它们通常非常不准确甚至完全随机。

我认为 eventbrite 允许客户输入任何标签作为标签，而人们不太善于提出好话。此外，每个事件的标签数量通常较低，即使您使用人工智能，也不足以判断事件

当然，有可能找到已经准确分类的文本并将其用于预测主题，但这又提出了许多其他问题：在何处获取机密文本？与我的活动描述有多相关？我应该使用多少个标签？所以我决定找到另一个想法。

### 想法＃2。 LDA 主题建模

经过一些研究，我发现了一个名为 [gensim](http://radimrehurek.com/gensim/) 的真棒 python 库，它实现了 LDA（ [Latent Dirichlet Allocation](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) ）用于主题建模。

值得注意的是，这里使用主题并不意味着用英语定义的主题，如“体育”，“音乐”或“编程”。 LDA 中的主题是对单词的概率分布。粗略地说，它发现了具有一定概率的单词集合。每个这样的集群都是一个“主题”。然后，您可以为模型提供新文档，并为其推断主题。

使用 LDA 非常简单。首先，我通过删除停止英语单词，逗号，html 标签等来清理文档（在我的案例文档中是事件的描述和标题）。然后我根据所有事件描述构建字典：

来自 gensim import corpora 的`，模型
dct = corpora.Dictionary（clean_documents）`

然后我过滤非常罕见的单词

`dct.filter_extremes(no_below=2)`

要训​​练模型，所有文件都需要转换成文字袋：

`corpus = [dct.doc2bow(doc) for doc in clean_documents]`

然后像这样创建模型

`lda = ldamodel.LdaModel(corpus=corpus, id2word=dct, num_topics=num_topics)`

其中 num_topics 是需要在文档上建模的许多主题。在我的情况下它是 100.然后将任何文字袋形式的文件转换为稀疏矩阵形式的主题表示：

`x = lda[doc_bow]`

所以现在我可以为任何给定的事件获得一个特征矩阵，我可以很容易地为用户评分的事件获得一个训练矩阵：

`docs_bow = [dct.doc2bow（doc）for doc in rated_events]
X_train = [lda [doc_bow] for doc_bow in docs_bow]`

这似乎或多或少是不错的解决方案，使用 SVM（[支持向量机](http://en.wikipedia.org/wiki/Support_vector_machine)）分类器我得到了大约 85％的准确度，当我查看预测事件对我来说它确实看起来非常准确。

注意：并非所有分类器都支持稀疏矩阵，有时您需要将其转换为完整矩阵。 Gensim 有办法做到这一点。

`gensim.matutils.sparse2full(sparse_matrix, num_topics)`

### 想法＃3。 TF-IDF 向量化器

我想尝试构建特征的另一个想法是 [TF-IDF 向量化器](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)。

[Scikit-learn](http://scikit-learn.org/stable/) 支持开箱即用，它正在做的是根据文档中该单词的频率除以文档中单词的频率为文档中的每个单词分配权重。文件的语料库。因此，如果您经常看到这个词的重量会很低，并且可以滤除噪音。要从所有文档中构建向量化器：

来自 sklearn.feature_extraction.text 的`导入 TfidfVectorizer
vectorizer = TfidfVectorizer（max_df = 0.5，sublinear_tf = True，stop_words ='english'）
vectorizer.fit（all_events）`

然后将给定的文档集转换为 TF-IDF 表示：

`X_train = vectorizer.transform(rated_events)`

现在，当我尝试将其提供给真正花费很长时间的分类器时，结果很糟糕。这实际上并不令人意外，因为在这种情况下，几乎每个单词都是一个特征。所以我开始寻找一种方法来选择表现最佳的功能。

Scikit-learn 提供方法 [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) ，你可以传递评分函数和许多功能来选择它，它为你执行魔术。对于得分，我使用 chi2（[卡方检验](http://en.wikipedia.org/wiki/卡方_test)），我不会说你到底为什么。我只是凭经验发现它在我的情况下表现得更好，并在我的 todo 桶中“研究了 chi2 背后的理论”。

`来自 sklearn.feature_selection import SelectKBest，chi2
num_features = 100
ch2 = SelectKBest（chi2，k = num_features）
X_train = ch2.fit_transform（X_train，y_train）.toarray（）`

就是这样。 X_train 是我的训练集。

### 训练分类器

我不高兴承认这一点，但我选择的分类器并没有多少科学参与。我只是试了一堆，然后选择表现最好的那一个。就我而言，它是 SVM。至于我用[网格搜索](http://scikit-learn.org/stable/modules/grid_search.html)选择最好的参数，所有 scikit-learn 提供开箱即用的参数。在代码中它看起来像这样：

`clf = svm.SVC（）
params = dict（gamma = [0.001,0.01,0.1,0.2,1,10,100]，C = [1,10,100,1000]，kernel = [“linear “，”rb“]）
clf = grid_search.GridSearchCV（clf，param_grid = params，cv = 5，scoring ='f1'）`

我选择 [f1-score](http://en.wikipedia.org/wiki/F1_score) 作为评分方法只是因为它是我或多或少了解的那个。网格搜索将尝试上述参数的所有组合，执行交叉验证并找到表现最佳的参数。

我尝试将这个分类器提供给 X_train，主题是用 LDA 和 TF-IDF + Chi2 建模的。两者表现相似，但主观上看起来像 TF-IDF + Chi2 解决方案产生了更好的预测。我对 v1 的结果非常满意，并花费了其余的时间来修复网站的 UI。

## 你做了哪些有趣的发现？

我学到的一件事是，如果你正在建立一个推荐系统，并期望你的用户一次来评价一堆东西，那么它可以工作 - 你错了。

我在我的朋友们上试过这个网站，虽然评分过程对我来说似乎非常简单和快速，但是很难让他们花几分钟点击“喜欢”按钮。虽然没关系，因为我的主要目标是练习技能并为自己构建一个工具，我想如果我想从中做出更大的东西，我需要弄清楚如何使评级过程更简单。

我学到的另一件事是，为了提高效率，我需要更多地理解算法。当您了解自己在做什么时，调整参数会更有趣。

## 你想在项目上做什么？

我目前的主要问题是 UI。我想保持简约，但我需要弄清楚如何使评级过程更有趣和方便。事件浏览也可能更好。

完成这部分后，我正在考虑寻找新的事件来源：会议，音乐会等。也许我会为此添加一个移动应用程序。

## 学到更多

*   项目： [hapsradar.com](http://hapsradar.com)

由于阿尔乔姆。

**你有机器学习方面的项目吗？**

如果你有一个使用机器学习的侧面项目，并希望像 Artem 一样被推荐，请[与我联系](http://machinelearningmastery.com/contact/ "Contact")。