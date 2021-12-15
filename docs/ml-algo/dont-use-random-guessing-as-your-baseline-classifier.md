# 不要使用随机猜测作为基线分类器

> 原文： [https://machinelearningmastery.com/dont-use-random-guessing-as-your-baseline-classifier/](https://machinelearningmastery.com/dont-use-random-guessing-as-your-baseline-classifier/)

我最近通过电子邮件收到了以下问题：

> 嗨，杰森，快速提问。一类失衡：90例竖起大拇指向下10例。在这种情况下，我们如何计算随机猜测的准确率？

我们可以使用一些基本概率回答这个问题（我打开了excel并输入了一些数字）。

![Don't Use Random Guessing As Your Baseline Classifier](img/785ff78c162b78fabe10fb6476919d35.jpg)

不要使用随机猜测作为您的基线分类器
照片由 [cbgrfx123](https://www.flickr.com/photos/72005145@N00/5600978712) ，保留一些权利。

假设0级和1级的分割是90％-10％。我们也说你会用相同的比例随机猜测。

随机猜测两分类问题的理论精度是：

```py
= P(class is 0) * P(you guess 0) + P(class is 1) * P(you guess 1)
```

我们可以在我们的示例90％-10％分割中测试这个：

```py
= (0.9 * 0.9) + (0.1 * 0.1)
= 0.82
= 0.82 * 100 or 82%
```

要检查数学，您可以插入50％-50％的数据分割，它符合您的直觉：

```py
= (0.5 * 0.5) + (0.5 * 0.5)
= 0.5
= 0.5 * 100 or 50%
```

如果我们查看Google，我们会在Cross Validated上找到类似的问题“[不平衡分类问题的机会级准确度是多少？](http://stats.stackexchange.com/questions/148149/what-is-the-chance-level-accuracy-in-unbalanced-classification-problems) “答案几乎相同。再次，一个很好的确认。

有趣的是，所有这一切都有一个重要的要点。

## 不要使用随机猜测作为基线

如果您正在寻找用作基线准确度的分类器，请不要使用随机猜测。

有一个名为Zero Rule的分类器（或简称为0R或ZeroR）。这是您可以在分类问题上使用的最简单的规则，它只是预测数据集中的多数类（例如[模式](https://en.wikipedia.org/wiki/Mode_(statistics))）。

在上面的例子中，0级和1级的90％-10％，它将为每个预测预测0级，并达到90％的准确率。这比使用随机猜测的理论最大值好8％。

使用零规则方法作为基线。

此外，在这种不平衡的分类问题中，您应该使用除精度之外的度量，例如Kappa或ROC曲线下的面积。

有关分类问题的替代表现度量的更多信息，请参阅帖子：

*   [分类准确度不够：可以使用的更多表现测量](http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)

有关处理不平衡分类问题的更多信息，请参阅帖子：

*   [打击机器学习数据集中不平衡类别的8种策略](http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

你对这篇文章有任何疑问吗？在评论中提问。