# 如何实现自然语言处理的波束搜索解码器

> 原文： [https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/)

自然语言处理任务，例如字幕生成和机器翻译，涉及生成单词序列。

针对这些问题开发的模型通常通过在输出词的词汇表中生成概率分布来操作，并且由解码算法来对概率分布进行采样以生成最可能的词序列。

在本教程中，您将发现可用于文本生成问题的贪婪搜索和波束搜索解码算法。

完成本教程后，您将了解：

*   解码文本生成问题的问题。
*   贪婪的搜索解码器算法以及如何在 Python 中实现它。
*   光束搜索解码器算法以及如何在 Python 中实现它。

让我们开始吧。

![How to Implement Beam Search Decoder for Natural Language Processing](img/1420681abf00887764bb6e8e4d7b981b.jpg)

如何为自然语言处理实现波束搜索解码器
照片由[参见 1，Do1，Teach1](https://www.flickr.com/photos/mpaulmd/14695285810/) ，保留一些权利。

## 用于文本生成的解码器

在自然语言处理任务（例如字幕生成，文本摘要和机器翻译）中，所需的预测是一系列单词。

为这些类型的问题开发的模型通常在输出序列中的每个单词的词汇表中的每个单词上输出概率分布。然后将其留给解码器处理以将概率转换为最终的单词序列。

在自然语言处理任务中使用循环神经网络时，您可能会遇到这种情况，其中文本生成为输出。神经网络模型中的最后一层具有输出词汇表中每个单词的一个神经元，并且 softmax 激活函数用于输出词汇表中的每个单词作为序列中的下一个单词的可能性。

解码最可能的输出序列涉及基于其可能性搜索所有可能的输出序列。词汇量的大小通常是数十或数十万个单词，甚至数百万个单词。因此，搜索问题在输出序列的长度上是指数级的，并且难以完全搜索（NP-complete）。

在实践中，启发式搜索方法用于为给定预测返回一个或多个近似或“足够好”的解码输出序列。

> 由于搜索图的大小在源句长度中是指数的，我们必须使用近似来有效地找到解。

- 第 272 页，[自然语言处理与机器翻译手册](http://amzn.to/2xQzTnt)，2011。

候选词的序列根据其可能性进行评分。通常使用贪婪搜索或波束搜索来定位候选文本序列。我们将在这篇文章中介绍这两种解码算法。

> 每个单独的预测都有一个相关的分数（或概率），我们感兴趣的是具有最大分数（或最大概率）的输出序列[...]一种流行的近似技术是使用贪婪预测，在每个阶段获得最高得分项目。虽然这种方法通常很有效，但它显然不是最佳的。实际上，使用光束搜索作为近似搜索通常比贪婪方法好得多。

- 第 227 页，[自然语言处理中的神经网络方法](http://amzn.to/2fC1sH1)，2017。

## 贪婪的搜索解码器

一个简单的近似是使用贪婪搜索，在输出序列的每一步选择最可能的单词。

这种方法的好处是速度非常快，但最终输出序列的质量可能远非最佳。

我们可以用 Python 中的一个小设计示例来演示解码的贪婪搜索方法。

我们可以从涉及 10 个单词序列的预测问题开始。每个单词被预测为 5 个单词的词汇表上的概率分布。

```py
# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
```

我们假设单词已经整数编码，这样列索引可用于查找词汇表中的相关单词。因此，解码任务成为从概率分布中选择整数序列的任务。

[argmax（）](https://en.wikipedia.org/wiki/Arg_max)数学函数可用于选择具有最大值的数组的索引。我们可以使用此函数来选择序列中每个步骤最有可能的单词索引。该功能直接在 [numpy](https://docs.scipy.org/doc/numpy-1.9.3/reference/generated/numpy.argmax.html) 中提供。

下面的 _greedy_decoder（）_ 函数使用 argmax 函数实现此解码器策略。

```py
# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [argmax(s) for s in data]
```

综上所述，下面列出了演示贪婪解码器的完整示例。

```py
from numpy import array
from numpy import argmax

# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [argmax(s) for s in data]

# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# decode sequence
result = greedy_decoder(data)
print(result)
```

运行该示例会输出一个整数序列，然后可以将这些整数映射回词汇表中的单词。

```py
[4, 0, 4, 0, 4, 0, 4, 0, 4, 0]
```

## 光束搜索解码器

另一种流行的启发式算法是波束搜索，它扩展了贪婪搜索并返回最可能的输出序列列表。

在构建序列时，不是贪婪地选择最可能的下一步，波束搜索扩展了所有可能的后续步骤并且最有可能保持 _k_ ，其中 _k_ 是用户指定的参数并通过概率序列控制光束数量或平行搜索。

> 局部波束搜索算法跟踪 k 个状态而不仅仅是一个状态。它以 k 个随机生成的状态开始。在每个步骤中，生成所有 k 个状态的所有后继者。如果任何一个是目标，则算法停止。否则，它从完整列表中选择 k 个最佳后继者并重复。

- 第 125-126 页，[人工智能：现代方法（第 3 版）](http://amzn.to/2x7ynhW)，2009。

我们不需要从随机状态开始;相反，我们从 _k_ 开始，最可能是单词作为序列的第一步。

对于贪婪搜索，公共波束宽度值为 1，对于机器翻译中的常见基准问题，公共波束宽度值为 5 或 10。较大的波束宽度导致模型的更好表现，因为多个候选序列增加了更好地匹配目标序列的可能性。这种增加的表现导致解码速度降低。

> 在 NMT 中，通过简单的波束搜索解码器转换新的句子，该解码器找到近似最大化训练的 NMT 模型的条件概率的平移。波束搜索策略从左到右逐字地生成翻译，同时在每个时间步长保持固定数量（波束）的活动候选。通过增加光束尺寸，转换表现可以以显着降低解码器速度为代价而增加。

- [神经机器翻译的光束搜索策略](https://arxiv.org/abs/1702.01806)，2017。

搜索过程可以通过达到最大长度，到达序列结束标记或达到阈值可能性来分别停止每个候选。

让我们以一个例子来具体化。

我们可以定义一个函数来对给定的概率序列和波束宽度参数 _k_ 进行波束搜索。在每个步骤中，使用所有可能的后续步骤扩展每个候选序列。通过将概率相乘来对每个候选步骤进行评分。选择具有最可能概率的 _k_ 序列，并修剪所有其他候选者。然后重复该过程直到序列结束。

概率是小数字，将小数字相乘可以产生非常小的数字。为了避免使浮点数下溢，概率的自然对数相乘，这使得数字更大且易于管理。此外，通常通过最小化分数来执行搜索也是常见的，因此，概率的负对数相乘。最后的调整意味着我们可以按照他们的分数按升序对所有候选序列进行排序，并选择第一个 k 作为最可能的候选序列。

下面的 _beam_search_decoder（）_ 函数实现了波束搜索解码器。

```py
# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * -log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences
```

我们可以将它与前一节中的样本数据联系起来，这次返回 3 个最可能的序列。

```py
from math import log
from numpy import array
from numpy import argmax

# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * -log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
	print(seq)
```

运行该示例将打印整数序列及其对数似然。

尝试不同的 k 值。

```py
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0], 0.025600863289563108]
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 1], 0.03384250043584397]
[[4, 0, 4, 0, 4, 0, 4, 0, 3, 0], 0.03384250043584397]
```

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [维基百科上的 Argmax](https://en.wikipedia.org/wiki/Arg_max)
*   [Numpy argmax API](https://docs.scipy.org/doc/numpy-1.9.3/reference/generated/numpy.argmax.html)
*   [维基百科上的光束搜索](https://en.wikipedia.org/wiki/Beam_search)
*   [神经机器翻译的光束搜索策略](https://arxiv.org/abs/1702.01806)，2017。
*   [人工智能：现代方法（第 3 版）](http://amzn.to/2x7ynhW)，2009。
*   [自然语言处理中的神经网络方法](http://amzn.to/2fC1sH1)，2017。
*   [自然语言处理和机器翻译手册](http://amzn.to/2xQzTnt)，2011。
*   [法老：用于基于短语的统计机器翻译模型的光束搜索解码器](https://link.springer.com/chapter/10.1007%2F978-3-540-30194-3_13?LI=true)，2004。

## 摘要

在本教程中，您发现了可用于文本生成问题的贪婪搜索和波束搜索解码算法。

具体来说，你学到了：

*   解码文本生成问题的问题。
*   贪婪的搜索解码器算法以及如何在 Python 中实现它。
*   光束搜索解码器算法以及如何在 Python 中实现它。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。