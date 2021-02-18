# 如何开始深度学习的时间序列预测（7 天迷你课程）

> 原文： [https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/](https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/)

### 时间序列预测速成课程的深度学习。

#### 在 7 天内为您的时间序列项目带来深度学习方法。

时间序列预测具有挑战性，尤其是在处理长序列，噪声数据，多步预测和多个输入和输出变量时。

深度学习方法为时间序列预测提供了许多希望，例如时间依赖的自动学习和趋势和季节性等时间结构的自动处理。

在本速成课程中，您将了解如何开始并自信地开发深度学习模型，以便在 7 天内使用 Python 进行时间序列预测问题。

这是一个重要且重要的帖子。您可能想要将其加入书签。

让我们开始吧。

[![How to Get Started with Deep Learning for Time Series Forecasting (7-Day Mini-Course)](img/ddabd4eb61af3e12ea5e27191572ab52.jpg)](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/11/How-to-Get-Started-with-Deep-Learning-for-Time-Series-Forecasting-7-Day-Mini-Course.jpg)

如何开始深度学习时间序列预测（7 天迷你课程）
摄影： [Brian Richardson](https://www.flickr.com/photos/seriousbri/3736154699/) ，保留一些权利。

## 谁是这个崩溃课程？

在我们开始之前，让我们确保您在正确的位置。

以下列表提供了有关本课程设计对象的一般指导原则。

你得知道：

*   您需要了解时间序列预测的基础知识。
*   你需要了解基本的 Python，NumPy 和 Keras 的深度学习方法。

你不需要知道：

*   你不需要成为一个数学家！
*   你不需要成为一名深度学习专家！
*   你不需要成为时间序列专家！

这个速成课程将带您从了解一点机器学习的开发人员到可以为您自己的时间序列预测项目带来深度学习方法的开发人员。

**注意**：这个速成课程假设你有一个有效的 Python 2 或 3 SciPy 环境，至少安装了 NumPy 和 Keras 2。如果您需要有关环境的帮助，可以按照此处的分步教程进行操作：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 速成课程概述

这个速成课程分为 7 节课。

您可以每天完成一节课（推荐）或在一天内完成所有课程（硬核）。这取决于你有空的时间和你的热情程度。

以下 7 个课程将通过深入学习 Python 中的时间序列预测来帮助您开始并提高工作效率：

*   **第 01 课**：深度学习的承诺
*   **第 02 课**：如何转换时间序列数据
*   **第 03 课**：时间序列预测的 MLP
*   **第 04 课**：时间序列预测的 CNN
*   **第 05 课**：时间序列预测的 LSTM
*   **第 06 课：** CNN-LSTM 用于时间序列预测
*   **第 07 课**：编码器 - 解码器 LSTM 多步预测

每节课可能需要 60 秒或 30 分钟。花点时间，按照自己的进度完成课程。在下面的评论中提出问题甚至发布结果。

课程期望你去学习如何做事。我将给你提示，但每节课的部分内容是强迫你学习去哪里寻求帮助，以及深入学习，时间序列预测和 Python 中最好的工具（提示， _ 我直接在这个博客上找到了所有答案，使用搜索框 _）。

我确实以相关帖子的链接形式提供了更多帮助，因为我希望你建立一些信心和惯性。

在评论中发布您的结果，我会为你欢呼！

挂在那里，不要放弃。

**注**：这只是一个速成课程。有关更多详细信息和 25 个充实教程，请参阅我的书，主题为“[深度学习时间序列预测](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)”。

## 第一课：深度学习的承诺

在本课程中，您将发现时间序列预测的深度学习方法的前景。

通常，像 Multilayer Perceptrons 或 MLP 这样的神经网络提供的功能很少，例如：

*   **强健噪音**。神经网络对输入数据和映射函数中的噪声具有鲁棒性，甚至可以在存在缺失值的情况下支持学习和预测。
*   **非线性**。神经网络不会对映射函数做出强有力的假设，并且很容易学习线性和非线性关系。
*   **多变量输入**。可以指定任意数量的输入要素，为多变量预测提供直接支持。
*   **多步骤预测**。可以指定任意数量的输出值，为多步骤甚至多变量预测提供
    直接支持。

仅就这些功能而言，前馈神经网络可用于时间序列预测。

### 你的任务

在本课程中，您必须提出卷积神经网络和循环神经网络的一种功能，这些功能可能有助于建模时间序列预测问题。

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [循环神经网络对时间序列预测的承诺](https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)

在下一课中，您将了解如何转换时间序列预测的时间序列数据。

## 课程 02：如何转换时间序列的数据

在本课程中，您将了解如何将时间序列数据转换为监督学习格式。

大多数实际机器学习使用监督学习。

监督学习是输入变量（X）和输出变量（y）的地方，您可以使用算法来学习从输入到输出的映射函数。目标是近似真实的底层映射，以便在有新输入数据时，可以预测该数据的输出变量。

时间序列数据可以表达为监督学习。

给定时间序列数据集的数字序列，我们可以将数据重组为看起来像监督学习问题。我们可以使用前面的时间步长作为输入变量，并使用下一个时间步作为输出变量。

例如，系列：

```py
1, 2, 3, 4, 5, ...
```

可以转换为具有输入和输出组件的样本，这些组件可以用作训练集的一部分，以训练监督学习模型，如深度学习神经网络。

```py
X,				y
[1, 2, 3]		4
[2, 3, 4]		5
...
```

这称为滑动窗口转换，因为它就像在先前观察中滑动窗口一样，用作模型的输入以预测序列中的下一个值。在这种情况下，窗口宽度是 3 个时间步长。

### 你的任务

在本课程中，您必须开发 Python 代码，将每日女性分娩数据集转换为具有一定数量输入和一个输出的监督学习格式。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [时间序列预测作为监督学习](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
*   [如何将时间序列转换为 Python 中的监督学习问题](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
*   [如何为长期短期记忆网络准备单变量时间序列数据](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

在下一课中，您将了解如何开发用于预测单变量时间序列的多层感知器深度学习模型。

## 第 03 课：时间序列预测的 MLP

在本课程中，您将了解如何为单变量时间序列预测开发多层感知器模型或 MLP。

我们可以将一个简单的单变量问题定义为整数序列，使模型适合该序列，并让模型预测序列中的下一个值。我们将问题框架为 3 输入和 1 输出，例如：[10,20,30]作为输入，[40]作为输出。

首先，我们可以定义模型。我们将通过第一个隐藏层上的`input_dim`参数将输入时间步数定义为 3。在这种情况下，我们将使用随机梯度下降的有效 Adam 版本并优化均方误差（'`mse`'）损失函数。

一旦定义了模型，它就可以适合训练数据，并且拟合模型可以用于进行预测。

下面列出了完整的示例。

```py
# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例将使模型适合数据，然后预测下一个样本外的值。

给定[50,60,70]作为输入，模型正确地预测 80 作为序列中的下一个值。

### 你的任务

在本课程中，您必须下载每日女性分娩数据集，将其分为训练集和测试集，并开发一个可以对测试集进行合理准确预测的模型。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [多层感知器神经网络速成课程](https://machinelearningmastery.com/neural-networks-crash-course/)
*   [Keras 中深度学习的时间序列预测](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)
*   [用于时间序列预测的多层感知器网络的探索性配置](https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/)

在下一课中，您将了解如何开发用于预测单变量时间序列的卷积神经网络模型。

## 第 04 课：CNN 进行时间序列预测

在本课程中，您将了解如何开发用于单变量时间序列预测的卷积神经网络模型或 CNN。

我们可以将一个简单的单变量问题定义为整数序列，使模型适合该序列，并让模型预测序列中的下一个值。我们将问题框架为 3 输入和 1 输出，例如：[10,20,30]作为输入，[40]作为输出。

与 MLP 模型的一个重要区别是 CNN 模型需要具有[_ 样本，时间步长，特征 _]形状的三维输入。我们将以[_ 样本，时间步长 _]的形式定义数据并相应地重新整形。

我们将通过第一个隐藏层上的`input_shape`参数将输入时间步数定义为 3，将要素数定义为 1。

我们将使用一个卷积隐藏层，后跟最大池池。然后，在由 Dense 层解释并输出预测之前，将滤镜图展平。该模型使用随机梯度下降的有效 Adam 模型，并优化均方误差（'`mse`'）损失函数。

一旦定义了模型，它就可以适合训练数据，并且拟合模型可以用于进行预测。

下面列出了完整的示例。

```py
# univariate cnn example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例将使模型适合数据，然后预测下一个样本外的值。

给定[50,60,70]作为输入，模型正确地预测 80 作为序列中的下一个值。

### 你的任务

在本课程中，您必须下载每日女性分娩数据集，将其分为训练集和测试集，并开发一个可以对测试集进行合理准确预测的模型。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [用于机器学习的卷积神经网络的速成课程](https://machinelearningmastery.com/crash-course-convolutional-neural-networks/)

在下一课中，您将了解如何开发长短期记忆网络模型以预测单变量时间序列。

## 课 05：时间序列预测的 LSTM

在本课程中，您将了解如何开发长期短期记忆神经网络模型或 LSTM 以进行单变量时间序列预测。

我们可以将一个简单的单变量问题定义为整数序列，使模型适合该序列，并让模型预测序列中的下一个值。我们将问题框架为 3 输入和 1 输出，例如：[10,20,30]作为输入，[40]作为输出。

与 MLP 模型的重要区别在于，与 CNN 模型一样，LSTM 模型需要具有形状[_ 样本，时间步长，特征 _]的三维输入。我们将以[_ 样本，时间步长 _]的形式定义数据并相应地重新整形。

我们将通过第一个隐藏层上的`input_shape`参数将输入时间步数定义为 3，将要素数定义为 1。

我们将使用一个 LSTM 层来处理 3 个时间步的每个输入子序列，然后使用 Dense 层来解释输入序列的摘要。该模型使用随机梯度下降的有效 Adam 模型，并优化均方误差（'`mse`'）损失函数。

一旦定义了模型，它就可以适合训练数据，并且拟合模型可以用于进行预测。

下面列出了完整的示例。

```py
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例将使模型适合数据，然后预测下一个样本外的值。

给定[50,60,70]作为输入，模型正确地预测 80 作为序列中的下一个值。

### 你的任务

在本课程中，您必须下载每日女性分娩数据集，将其分为训练集和测试集，并开发一个可以对测试集进行合理准确预测的模型。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [专家对长短期记忆网络的简要介绍](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
*   [深度学习的循环神经网络崩溃课程](https://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)

在下一课中，您将了解如何针对单变量时间序列预测问题开发混合 CNN-LSTM 模型。

## 第 06 课：CNN-LSTM 用于时间序列预测

在本课程中，您将了解如何开发用于单变量时间序列预测的混合 CNN-LSTM 模型。

该模型的好处是该模型可以支持非常长的输入序列，可以通过 CNN 模型作为块或子序列读取，然后由 LSTM 模型拼凑在一起。

我们可以将一个简单的单变量问题定义为整数序列，使模型适合该序列，并让模型预测序列中的下一个值。我们将问题框架为 4 输入和 1 输出，例如：[10,20,30,40]作为输入，[50]作为输出。

当使用混合 CNN-LSTM 模型时，我们将进一步将每个样本分成更多的子序列。 CNN 模型将解释每个子序列，并且 LSTM 将来自子序列的解释拼凑在一起。因此，我们将每个样本分成 2 个子序列，每个子序列 2 次。

CNN 将被定义为每个子序列有一个特征需要 2 个时间步长。然后将整个 CNN 模型包装在 TimeDistributed 包装层中，以便可以将其应用于样本中的每个子序列。然后在模型输出预测之前由 LSTM 层解释结果。

该模型使用随机梯度下降的有效 Adam 模型，并优化均方误差（'mse'）损失函数。

一旦定义了模型，它就可以适合训练数据，并且拟合模型可以用于进行预测。

下面列出了完整的示例。

```py
# univariate cnn-lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# define dataset
X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60, 70, 80])
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], 2, 2, 1))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 2, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70, 80])
x_input = x_input.reshape((1, 2, 2, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例将使模型适合数据，然后预测下一个样本外的值。

给定[50,60,70,80]作为输入，模型正确地预测 90 作为序列中的下一个值。

### 你的任务

在本课程中，您必须下载每日女性分娩数据集，将其分为训练集和测试集，并开发一个可以对测试集进行合理准确预测的模型。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [CNN 长短期记忆网络](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)
*   [如何在 Python](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) 中为长期短期内存网络使用时间分布层

在下一课中，您将了解如何开发用于多步时间序列预测的编码器 - 解码器 LSTM 网络模型。

## 课程 07：编码器 - 解码器 LSTM 多步预测

在本课程中，您将了解如何为多步时间序列预测开发编码器 - 解码器 LSTM 网络模型。

我们可以将一个简单的单变量问题定义为整数序列，使模型适合该序列，并让模型预测序列中的下两个值。我们将问题框架为 3 输入和 2 输出，例如：[10,20,30]作为输入，[40,50]作为输出。

LSTM 模型需要具有[_ 样本，时间步长，特征 _]形状的三维输入。我们将以[_ 样本，时间步长 _]的形式定义数据并相应地重新整形。使用编码器 - 解码器模型时，输出也必须以这种方式成形。

我们将通过第一个隐藏层上的`input_shape`参数将输入时间步数定义为 3，将要素数定义为 1。

我们将定义一个 LSTM 编码器来读取和编码 3 个时间步的输入序列。对于使用 RepeatVector 层的模型所需的两个输出时间步长，模型将重复编码序列 2 次。在使用包含在 TimeDistributed 层中的 Dense 输出层之前，这些将被馈送到解码器 LSTM 层，该层将为输出序列中的每个步骤产生一个输出。

该模型使用随机梯度下降的有效 Adam 模型，并优化均方误差（'`mse`'）损失函数。

一旦定义了模型，它就可以适合训练数据，并且拟合模型可以用于进行预测。

下面列出了完整的示例。

```py
# multi-step encoder-decoder lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([[40,50],[50,60],[60,70],[70,80]])
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3, 1)))
model.add(RepeatVector(2))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

运行该示例将使模型适合数据，然后预测接下来的两个样本外值。

给定[50,60,70]作为输入，模型正确地预测[80,90]作为序列中的下两个值。

### 你的任务

在本课程中，您必须下载每日女性分娩数据集，将其分为训练集和测试集，并开发一个可以对测试集进行合理准确预测的模型。

你可以从这里下载数据集： [daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

在下面的评论中发表您的答案。我很乐意看到你发现了什么。

### 更多信息

*   [编码器 - 解码器长短期存储器网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
*   [多步时间序列预测的 4 种策略](https://machinelearningmastery.com/multi-step-time-series-forecasting/)
*   [Python 中长期短期记忆网络的多步时间序列预测](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/)

## 结束！
（_ 看你有多远 _）

你做到了。做得好！

花点时间回顾一下你到底有多远。

你发现：

*   深度学习神经网络对时间序列预测问题的承诺。
*   如何将时间序列数据集转换为监督学习问题。
*   如何为单变量时间序列预测问题开发多层感知器模型。
*   如何建立一个单变量时间序列预测问题的卷积神经网络模型。
*   如何为单变量时间序列预测问题开发长短期记忆网络模型。
*   如何为单变量时间序列预测问题开发混合 CNN-LSTM 模型。
*   如何为多步时间序列预测问题开发编码器 - 解码器 LSTM 模型。

这只是您深入学习时间序列预测的旅程的开始。继续练习和发展你的技能。

下一步，查看我的书[深度学习时间序列](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)。

## 摘要

**你是如何使用迷你课程的？**
你喜欢这个速成班吗？

**你有什么问题吗？有没有任何问题？**
让我知道。在下面发表评论。