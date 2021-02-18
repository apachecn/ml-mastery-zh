# 如何使用 scikit-learn 为机器学习准备文本数据

> 原文： [https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)

文本数据需要特殊准备才能开始使用它进行预测建模。

必须解析文本以删除称为分词的单词。然后，需要将单词编码为整数或浮点值，以用作机器学习算法的输入，称为特征提取（或向量化）。

scikit-learn 库提供易于使用的工具，可以执行文本数据的分词和特征提取。

在本教程中，您将了解如何使用 scikit-learn 在 Python 中为预测建模准备文本数据。

完成本教程后，您将了解：

*   如何使用 CountVectorizer 将文本转换为字数统计向量。
*   如何使用 TfidfVectorizer 将文本转换为字频向量。
*   如何使用 HashingVectorizer 将文本转换为唯一的整数。

让我们开始吧。

![How to Prepare Text Data for Machine Learning with scikit-learn](img/37154d4c77b0854b4cf91ea1592fd802.jpg)

如何使用 scikit-learn
照片由 [Martin Kelly](https://www.flickr.com/photos/martkelly/34474622542/) 为机器学习准备文本数据，保留一些权利。

## 词袋模型

使用机器学习算法时，我们无法直接使用文本。

相反，我们需要将文本转换为数字。

我们可能想要执行文档分类，因此每个文档都是“_ 输入 _”，类标签是我们的预测算法的“_ 输出 _”。算法将数字向量作为输入，因此我们需要将文档转换为固定长度的数字向量。

在机器学习中思考文本文档的简单而有效的模型被称为 Bag-of-Words 模型，或 BoW。

该模型很简单，因为它抛弃了单词中的所有订单信息，并关注文档中单词的出现。

这可以通过为每个单词分配唯一编号来完成。然后，我们看到的任何文档都可以编码为具有已知单词词汇长度的固定长度向量。向量中每个位置的值可以用编码文档中每个单词的计数或频率填充。

这是单词模型的包，我们只关注编码方案，这些编码方案表示存在的单词或它们在编码文档中存在的程度，而没有关于顺序的任何信息。

有很多方法可以扩展这个简单的方法，既可以更好地阐明“_ 字 _”是什么，也可以定义对向量中每个字进行编码的内容。

scikit-learn 库提供了我们可以使用的 3 种不同方案，我们将简要介绍每种方案。

## Word 计数与 CountVectorizer

[CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 提供了一种简单的方法，既可以标记文本文档的集合，也可以构建已知单词的词汇表，还可以使用该词汇表对新文档进行编码。

您可以按如下方式使用它：

1.  创建`CountVectorizer`类的实例。
2.  调用`fit()`函数以便从一个或多个文档中学习词汇。
3.  根据需要在一个或多个文档上调用`transform()`函数，将每个文档编码为向量。

返回的编码向量具有整个词汇表的长度，并且返回每个单词出现在文档中的次数的整数计数。

因为这些向量将包含许多零，所以我们将它们称为稀疏。 Python 提供了一种在 [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) 包中处理稀疏向量的有效方法。

从对 transform（）的调用返回的向量将是稀疏向量，您可以将它们转换回 numpy 数组，以通过调用 toarray（）函数来查看并更好地了解正在发生的事情。

下面是使用 CountVectorizer 进行标记，构建词汇表，然后对文档进行编码的示例。

```py
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
```

在上面，您可以看到我们访问词汇表以查看通过调用标记的确切内容：

```py
print(vectorizer.vocabulary_)
```

我们可以看到默认情况下所有单词都是小写的，并且忽略了标点符号。可以配置分词的这些和其他方面，我建议您查看 [API 文档](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)中的所有选项。

首先运行该示例打印词汇表，然后打印编码文档的形状。我们可以看到词汇中有 8 个单词，因此编码的向量长度为​​8。

然后我们可以看到编码的向量是稀疏矩阵。最后，我们可以看到编码向量的数组版本，显示除了出现 2 的（index 和 id 7）之外的每个单词的出现次数为 1。

```py
{'dog': 1, 'fox': 2, 'over': 5, 'brown': 0, 'quick': 6, 'the': 7, 'lazy': 4, 'jumped': 3}
(1, 8)
<class 'scipy.sparse.csr.csr_matrix'>
[[1 1 1 1 1 1 1 2]]
```

重要的是，相同的向量化程序可用于包含词汇表中未包含的单词的文档。忽略这些单词，并且在结果向量中不给出计数。

例如，下面是使用上面的向量化器来编码文档中的一个单词和一个单词不是的单词的示例。

```py
# encode another document
text2 = ["the puppy"]
vector = vectorizer.transform(text2)
print(vector.toarray())
```

运行此示例将打印编码稀疏向量的数组版本，显示词汇中一个单词出现一次，而词汇中未出现的另一个单词完全被忽略。

```py
[[0 0 0 0 0 0 0 1]]
```

然后，编码的向量可以直接与机器学习算法一起使用。

## 使用 TfidfVectorizer 的单词频率

字数是一个很好的起点，但非常基本。

简单计数的一个问题是，诸如“”之类的单词会出现很多次，并且它们的大数量在编码向量中不会非常有意义。

另一种方法是计算单词频率，到目前为止，最流行的方法称为 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 。这是“_ 术语频率 - 反向文档 _”频率的首字母缩写，它是分配给每个单词的结果分数的组成部分。

*   **术语频率**：总结了给定单词在文档中出现的频率。
*   **反向文档频率**：这缩小了文档中出现的很多单词。

在没有进入数学的情况下，TF-IDF 是单词频率分数，试图突出更有趣的单词，例如，在文档中频繁但不在文档中。

[TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 将对文档进行标记，学习词汇和逆文档频率权重，并允许您对新文档进行编码。或者，如果您已经学习了 CountVectorizer，则可以将其与 [TfidfTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) 一起使用，以计算逆文档频率并开始编码文档。

与 CountVectorizer 一样使用相同的创建，拟合和转换过程。

下面是使用 TfidfVectorizer 学习 3 个小文档中的词汇和逆文档频率然后编码其中一个文档的示例。

```py
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

从文档中学习 8 个单词的词汇表，并且在输出向量中为每个单词分配唯一的整数索引。

针对词汇表中的每个单词计算逆文档频率，将最低得分 1.0 分配给最频繁观察的单词：“ _[指示 7 处的 _”。

最后，第一个文档被编码为 8 个元素的稀疏数组，我们可以查看每个单词的最终评分，其中包含“”，“`fox`”和“ _ 狗 _“来自词汇中的其他词汇。

```py
{'fox': 2, 'lazy': 4, 'dog': 1, 'quick': 6, 'the': 7, 'over': 5, 'brown': 0, 'jumped': 3}
[ 1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718
1.69314718 1\. ]
(1, 8)
[[ 0.36388646 0.27674503 0.27674503 0.36388646 0.36388646 0.36388646
0.36388646 0.42983441]]
```

将得分归一化为 0 到 1 之间的值，然后可以将编码的文档向量直接用于大多数机器学习算法。

## 散列 HashingVectorizer

计数和频率可能非常有用，但这些方法的一个限制是词汇量可能变得非常大。

反过来，这将需要大的向量来编码文档并对内存施加大量要求并减慢算法速度。

一个聪明的解决方法是使用单向哈希值将它们转换为整数。聪明的部分是不需要词汇表，你可以选择任意长的固定长度向量。缺点是散列是单向函数，因此无法将编码转换回单词（这对于许多监督学习任务可能无关紧要）。

[HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) 类实现了这种方法，可用于一致地对单词进行哈希处理，然后根据需要对文档进行标记和编码。

下面的示例演示了用于编码单个文档的 HashingVectorizer。

选择任意固定长度的向量大小 20。这对应于散列函数的范围，其中小值（如 20）可能导致散列冲突。记得回到 compsci 类，我相信有一些启发式方法可以用来根据估计的词汇量来选择哈希长度和碰撞概率。

请注意，此向量化程序不需要调用以适应训练数据文档。相反，在实例化之后，它可以直接用于开始编码文档。

```py
from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

运行该示例将示例文档编码为 20 个元素的稀疏数组。

编码文档的值默认对应于标准化字数，范围为-1 到 1，但可以通过更改默认配置来使其成为简单的整数计数。

```py
(1, 20)
[[ 0\.          0\.          0\.          0\.          0\.          0.33333333
   0\.         -0.33333333  0.33333333  0\.          0\.          0.33333333
   0\.          0\.          0\.         -0.33333333  0\.          0.
  -0.66666667  0\.        ]]
```

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 自然语言处理

*   [维基百科上的词袋模型](https://en.wikipedia.org/wiki/Bag-of-words_model)
*   维基百科上的[标记](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
*   维基百科上的 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

### scikit 学习

*   [第 4.2 节。特征提取](http://scikit-learn.org/stable/modules/feature_extraction.html)，scikit-learn 用户指南
*   [scikit-learn 特征提取 API](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction)
*   [使用文本数据](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)，scikit-learn Tutorial

### 类 API

*   [CountVectorizer scikit-learn API](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
*   [TfidfVectorizer scikit-learn API](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
*   [TfidfTransformer scikit-learn API](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
*   [HashingVectorizer scikit-learn API](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)

## 摘要

在本教程中，您了解了如何使用 scikit-learn 为机器学习准备文本文档。

我们在这些示例中只是略微表面，我想强调这些类有许多配置细节来影响值得探索的文档的分词。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。