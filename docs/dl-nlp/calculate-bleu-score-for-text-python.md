# 在 Python 中计算文本 BLEU 分数的温和介绍

> 原文： [https://machinelearningmastery.com/calculate-bleu-score-for-text-python/](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)

BLEU 或双语评估 Understudy 是用于将文本的候选翻译与一个或多个参考翻译进行比较的分数。

虽然是为翻译而开发的，但它可用于评估为一系列自然语言处理任务生成的文本。

在本教程中，您将发现使用 Python 中的 NLTK 库评估和评分候选文本的 BLEU 分数。

完成本教程后，您将了解：

*   轻轻地介绍 BLEU 分数和对计算内容的直觉。
*   如何使用 NLTK 库为句子和文档计算 Python 中的 BLEU 分数。
*   如何使用一套小例子来确定候选人和参考文本之间的差异如何影响最终的 BLEU 分数。

让我们开始吧。

![A Gentle Introduction to Calculating the BLEU Score for Text in Python](img/114a80a1936806a1189371f26560ca8e.jpg)

在 Python 中计算文本 BLEU 分数的温和介绍
照片由 [Bernard Spragg 撰写。 NZ](https://www.flickr.com/photos/volvob12b/15624500507/) ，保留一些权利。

## 教程概述

本教程分为 4 个部分;他们是：

1.  双语评估 Understudy 得分
2.  计算 BLEU 分数
3.  累积和个人 BLEU 分数
4.  工作的例子

## 双语评估 Understudy 得分

双语评估 Understudy 分数，或简称 BLEU，是用于评估生成的句子到参考句子的度量。

完美匹配得分为 1.0，而完美匹配得分为 0.0。

该评分是为评估自动机器翻译系统的预测而开发的。它并不完美，但确实提供了 5 个引人注目的好处：

*   计算速度快，成本低廉。
*   这很容易理解。
*   它与语言无关。
*   它与人类评价高度相关。
*   它已被广泛采用。

BLEU 评分由 Kishore Papineni 等人提出。在 2002 年的论文“ [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)”。

该方法通过将候选翻译中的匹配 n-gram 计数到参考文本中的 n-gram 来进行工作，其中 1-gram 或 unigram 将是每个标记，并且 bigram 比较将是每个单词对。无论字顺序如何，都进行比较。

> BLEU 实现者的主要编程任务是将候选者的 n-gram 与参考翻译的 n-gram 进行比较并计算匹配数。这些匹配与位置无关。匹配越多，候选翻译就越好。

- [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)，2002。

修改匹配的 n-gram 的计数以确保它考虑参考文本中的单词的出现，而不是奖励产生大量合理单词的候选翻译。这在本文中称为修正的 n-gram 精度。

> 不幸的是，MT 系统可以过度生成“合理”的单词，导致不可能但高精度的翻译[...]直观地，问题很明显：在识别出匹配的候选词之后，应该认为参考词已经用尽。我们将这种直觉形式化为修改后的单字组精度。

- [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)，2002。

该分数用于比较句子，但是还提出了通过其出现来标准化 n-gram 的修改版本以用于更好的多个句子的评分块。

> 我们首先逐句计算 n-gram 匹配。接下来，我们为所有候选句子添加剪切的 n-gram 计数，并除以测试语料库中的候选 n-gram 的数量，以计算整个测试语料库的修改的精确度分数 pn。

- [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)，2002。

在实践中不可能获得满分，因为翻译必须与参考完全匹配。人类翻译甚至无法做到这一点。用于计算 BLEU 分数的参考文献的数量和质量意味着比较数据集之间的分数可能很麻烦。

> BLEU 度量范围从 0 到 1.少数翻译将获得 1 分，除非它们与参考翻译相同。出于这个原因，即使是一个人类翻译也不一定会在大约 500 个句子（40 个一般新闻报道）的测试语料中得分 1.一个人类翻译对四个参考文献得分为 0.3468，对两个参考文献得分为 0.2571。

- [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)，2002。

除了翻译，我们还可以通过深度学习方法将 BLEU 评分用于其他语言生成问题，例如：

*   语言生成。
*   图像标题生成。
*   文字摘要。
*   语音识别。

以及更多。

## 计算 BLEU 分数

Python Natural Language Toolkit 库（即 NLTK）提供了 BLEU 分数的实现，您可以使用它来根据引用评估生成的文本。

### 句子 BLEU 分数

NLTK 提供 [sentence_bleu（）](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu)函数，用于针对一个或多个参考句子评估候选句子。

引用句子必须作为句子列表提供，其中每个引用是一个令牌列表。候选句子作为令牌列表提供。例如：

```
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)
```

运行此示例会打印出一个完美的分数，因为候选者会精确匹配其中一个引用。

```
1.0
```

### 语料库 BLEU 分数

NLTK 还提供称为 [corpus_bleu（）](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.corpus_bleu)的函数，用于计算多个句子（例如段落或文档）的 BLEU 分数。

必须将引用指定为文档列表，其中每个文档是引用列表，并且每个备选引用是令牌列表，例如，令牌列表列表。必须将候选文档指定为列表，其中每个文档是令牌列表，例如，令牌列表列表。

这有点令人困惑;这是一个文档的两个引用的示例。

```
# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score)
```

运行该示例将像以前一样打印出完美的分数。

```
1.0
```

## 累积和个人 BLEU 分数

NLTK 中的 BLEU 分数计算允许您在计算 BLEU 分数时指定不同 n-gram 的权重。

这使您可以灵活地计算不同类型的 BLEU 分数，例如个人和累积的 n-gram 分数。

让我们来看看。

### 个人 N-Gram 分数

单独的 N-gram 分数是仅匹配特定顺序的克数的评估，例如单个单词（1-gram）或单词对（2-gram 或 bigram）。

权重被指定为元组，其中每个索引引用克顺序。要仅为 1-gram 匹配计算 BLEU 分数，您可以为 1-gram 指定权重 1，为 2,3 和 4 指定权重（1,0,0,0）。例如：

```
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)
```

运行此示例会打印 0.5 分。

```
0.75
```

我们可以针对 1 到 4 的单个 n-gram 重复此示例，如下所示：

```
# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
```

运行该示例将给出以下结果。

```
Individual 1-gram: 1.000000
Individual 2-gram: 1.000000
Individual 3-gram: 1.000000
Individual 4-gram: 1.000000
```

虽然我们可以计算单个 BLEU 分数，但这不是该方法的用途，并且分数没有很多意义，或者似乎可以解释。

### 累积 N-Gram 分数

累积分数指的是从 1 到 n 的所有阶数的单个 n-gram 分数的计算，并通过计算加权几何平均值对它们进行加权。

默认情况下， _sentence_bleu（）_ 和 _corpus_bleu（）_ 分数计算累积的 4 克 BLEU 分数，也称为 BLEU-4。

对于 1 克，2 克，3 克和 4 克的分数，BLEU-4 的重量分别为 1/4（25％）或 0.25。例如：

```
# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)
```

运行此示例将打印以下分数：

```
0.707106781187
```

累积和单个 1 克 BLEU 使用相同的权重，例如（1,0,0,0）。 2 克重量为 1 克和 2 克各分配 50％，3 克重量为 1,2 克和 3 克分数各 33％。

让我们通过计算 BLEU-1，BLEU-2，BLEU-3 和 BLEU-4 的累积分数来具体化：

```
# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
```

运行该示例将打印以下分数。他们是完全不同的，更具表现力

它们与独立的单个 n-gram 分数完全不同且更具表现力。

```
Cumulative 1-gram: 0.750000
Cumulative 2-gram: 0.500000
Cumulative 3-gram: 0.632878
Cumulative 4-gram: 0.707107
```

在描述文本生成系统的技能时，通常会报告累积的 BLEU-1 到 BLEU-4 分数。

## 工作的例子

在本节中，我们尝试通过一些例子为 BLEU 评分进一步发展直觉。

我们使用以下单个参考句在句子级别工作：

> 快速的棕色狐狸跳过懒狗

首先，让我们看看一个完美的分数。

```
# prefect match
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)
```

运行该示例打印完美匹配。

```
1.0
```

接下来，让我们改变一个词，'_ 快速 _'改为'_ 快 _'。

```
# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)
```

这个结果是得分略有下降。

```
0.7506238537503395
```

尝试更改两个单词，'_ 快速 _'到'_ 快速 _'和'_ 懒惰 _'到'_ 困 _'。

```
# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)
```

运行该示例，我们可以看到技能的线性下降。

```
0.4854917717073234
```

如果候选人的所有单词都不同怎么办？

```
# all words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
score = sentence_bleu(reference, candidate)
print(score)
```

我们得分可能更差。

```
0.0
```

现在，让我们尝试一个比参考词少的候选词（例如删掉最后两个词），但这些词都是正确的。

```
# shorter candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
score = sentence_bleu(reference, candidate)
print(score)
```

当两个单词出错时，得分很像得分。

```
0.7514772930752859
```

如果我们让候选人的两个单词长于参考文件怎么样？

```
# longer candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space']
score = sentence_bleu(reference, candidate)
print(score)
```

再次，我们可以看到我们的直觉成立并且得分类似于“_ 两个单词错 _”。

```
0.7860753021519787
```

最后，让我们比较一个太短的候选人：长度只有两个单词。

```
# very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
print(score)
```

首先运行此示例将打印一条警告消息，指示无法执行评估的 3 克及以上部分（最多 4 克）。这是公平的，因为我们只有 2 克与候选人一起工作。

```
UserWarning:
Corpus/Sentence contains 0 counts of 3-gram overlaps.
BLEU scores might be undesirable; use SmoothingFunction().
  warnings.warn(_msg)
```

接下来，我们的分数确实非常低。

```
0.0301973834223185
```

我鼓励你继续玩实例。

数学很简单，我也鼓励你阅读论文并探索自己在电子表格中计算句子级别的分数。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   维基百科上的 [BLEU](https://en.wikipedia.org/wiki/BLEU)
*   [BLEU：一种自动评估机器翻译的方法](http://www.aclweb.org/anthology/P02-1040.pdf)，2002。
*   [nltk.translate.bleu_score](http://www.nltk.org/_modules/nltk/translate/bleu_score.html) 的源代码
*   [nltk.translate 包 API 文档](http://www.nltk.org/api/nltk.translate.html)

## 摘要

在本教程中，您发现了用于评估和评分候选文本以在机器翻译和其他语言生成任务中引用文本的 BLEU 分数。

具体来说，你学到了：

*   轻轻地介绍 BLEU 分数和对计算内容的直觉。
*   如何使用 NLTK 库为句子和文档计算 Python 中的 BLEU 分数。
*   如何使用一套小例子来确定候选人和参考文本之间的差异如何影响最终的 BLEU 分数的直觉。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。