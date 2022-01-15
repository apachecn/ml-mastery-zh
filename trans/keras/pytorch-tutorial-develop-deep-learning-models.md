# PyTorch 教程：如何用 Python 开发深度学习模型

> 原文：<https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/>

最后更新于 2020 年 8 月 27 日

具有深度学习的预测建模是现代开发人员需要了解的技能。

PyTorch 是脸书开发和维护的第一个开源深度学习框架。

从本质上来说，PyTorch 是一个数学库，允许您对基于图形的模型进行高效计算和自动微分。直接实现这一点是具有挑战性的，尽管谢天谢地，现代的 PyTorch API 提供了类和习惯用法，允许您轻松开发一套深度学习模型。

在本教程中，您将发现一个在 PyTorch 中开发深度学习模型的分步指南。

完成本教程后，您将知道:

*   Torch 和 PyTorch 的区别以及如何安装和确认 PyTorch 工作正常。
*   PyTorch 模型的五步生命周期以及如何定义、拟合和评估模型。
*   如何为回归、分类和预测建模任务开发 PyTorch 深度学习模型？

我们开始吧。

![PyTorch Tutorial - How to Develop Deep Learning Models](img/13ff658685e88a714278b5d716b7a04a.png)

PyTorch 教程-如何开发深度学习模型
图片由[迪米特里·B](https://flickr.com/photos/ru_boff/14863560864/)提供。，保留部分权利。

## PyTorch 教程概述

本教程的重点是将 PyTorch API 用于常见的深度学习模型开发任务；我们不会深入学习数学和理论。为此，我推荐[从这本优秀的书](https://amzn.to/2Y8JuBv)开始。

在 python 中学习深度学习的最好方法是做。一头扎进去。你可以回头再找更多的理论。

我已经将每个代码示例设计成使用最佳实践和独立的，这样您就可以将它直接复制并粘贴到您的项目中，并根据您的特定需求进行调整。这将让你有一个很好的开端来尝试从官方文档中找出应用编程接口。

这是一个大型教程，因此，它分为三个部分；它们是:

1.  如何安装 PyTorch
    1.  什么是火炬和 PyTorch？
    2.  如何安装 PyTorch
    3.  如何确认 PyTorch 已安装
2.  PyTorch 深度学习模型生命周期
    1.  第一步:准备数据
    2.  步骤 2:定义模型
    3.  第三步:训练模型
    4.  步骤 4:评估模型
    5.  第五步:做预测
3.  如何开发 PyTorch 深度学习模型
    1.  如何开发二元分类的 MLP
    2.  如何开发多类分类的 MLP
    3.  如何开发回归 MLP
    4.  如何开发一个用于图像分类的有线电视网络

### 你可以用 Python 做深度学习！

完成本教程。最多需要 60 分钟！

**你不需要什么都懂(至少现在不需要)**。你的目标是把教程从头到尾看一遍，并得到一个结果。你不需要第一遍就明白所有的事情。边走边列出你的问题。大量使用 API 文档来了解您正在使用的所有功能。

**你不需要Prophet道数学**。数学是描述算法如何工作的简洁方式，特别是来自线性代数、概率和微积分的工具。这些不是你可以用来学习算法如何工作的唯一工具。您还可以使用代码并探索具有不同输入和输出的算法行为。知道数学不会告诉你选择什么算法或者如何最好地配置它。你只能通过精心控制的实验来发现这一点。

**你不需要知道算法是如何工作的**。了解这些限制以及如何配置深度学习算法非常重要。但是关于算法的学习可以在以后进行。你需要长时间慢慢积累这些算法知识。今天，从适应平台开始。

**不需要做 Python 程序员**。如果你不熟悉 Python 语言，它的语法会很直观。就像其他语言一样，关注函数调用(例如 function())和赋值(例如 a =“b”)。这会让你大受鼓舞。你是一个开发者；你知道如何快速掌握一门语言的基础知识。开始吧，稍后再深入细节。

**不需要成为深度学习专家**。你可以在后面学习各种算法的好处和局限性，还有很多教程可以阅读，以便在深度学习项目的步骤上复习。

## 1.如何安装 PyTorch

在本节中，您将发现什么是 PyTorch，如何安装它，以及如何确认它安装正确。

### 1.1.什么是火炬和 PyTorch？

[PyTorch](https://github.com/pytorch/pytorch) 是脸书开发维护的一个用于深度学习的开源 Python 库。

该项目始于 2016 年，很快成为开发人员和研究人员的流行框架。

[Torch](https://github.com/torch/torch7) ( *Torch7* )是用 C 语言编写的深度学习开源项目，一般通过 Lua 接口使用。这是 PyTorch 的前身项目，现在已不再积极开发。PyTorch 在名称中包含“ *Torch* ”，以“ *Py* ”前缀确认之前的 Torch 库，表示新项目的 Python 重点。

PyTorch API 简单而灵活，使其成为学者和研究人员在开发新的深度学习模型和应用程序时的最爱。广泛的使用导致了许多特定应用的扩展(例如文本、计算机视觉和音频数据)，并且可能预先训练了可以直接使用的模型。因此，它可能是最受学者欢迎的图书馆。

与像 [Keras](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/) 这样更简单的界面相比，PyTorch 的灵活性是以易用性为代价的，尤其是对于初学者来说。选择使用 PyTorch 而不是 Keras，放弃了一些易用性、略陡的学习曲线、更多的代码以获得更大的灵活性，或许还有更有活力的学术社区。

### 1.2.如何安装 PyTorch

在安装 PyTorch 之前，请确保您安装了 Python，例如 Python 3.6 或更高版本。

如果没有安装 Python，可以使用 Anaconda 安装。本教程将向您展示如何:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

安装 PyTorch 开源深度学习库的方法有很多。

在您的工作站上安装 PyTorch 最常见，也可能是最简单的方法是使用 pip。

例如，在命令行上，您可以键入:

```py
sudo pip install torch
```

深度学习最受欢迎的应用可能是[计算机视觉](https://machinelearningmastery.com/what-is-computer-vision/)，PyTorch 计算机视觉包叫做“ [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) ”

强烈建议安装 torchvision，安装方式如下:

```py
sudo pip install torchvision
```

如果您更喜欢使用特定于您的平台或软件包管理器的安装方法，您可以在此查看完整的安装说明列表:

*   pytorch 安装指南

现在没有必要设置 GPU。

本教程中的所有示例在现代中央处理器上都可以正常工作。如果您想为您的 GPU 配置 PyTorch，您可以在完成本教程后进行配置。不要分心！

### 1.3.如何确认 PyTorch 已安装

一旦安装了 PyTorch，确认库安装成功并且您可以开始使用它是很重要的。

不要跳过这一步。

如果 PyTorch 安装不正确或在此步骤中出现错误，您将无法在以后运行这些示例。

创建一个名为 *versions.py* 的新文件，并将以下代码复制粘贴到文件中。

```py
# check pytorch version
import torch
print(torch.__version__)
```

保存文件，然后打开命令行，将目录更改为保存文件的位置。

然后键入:

```py
python versions.py
```

然后，您应该会看到如下输出:

```py
1.3.1
```

这确认了 PyTorch 安装正确，并且我们都使用相同的版本。

这也向您展示了如何从命令行运行 Python 脚本。我建议以这种方式从命令行运行所有代码，而不是从笔记本或 IDE 运行。

## 2.PyTorch 深度学习模型生命周期

在这一节中，您将发现深度学习模型的生命周期以及可以用来定义模型的 PyTorch API。

模型有一个生命周期，这个非常简单的知识为建模数据集和理解 PyTorch API 提供了基础。

生命周期中的五个步骤如下:

*   1.准备数据。
*   2.定义模型。
*   3.训练模型。
*   4.评估模型。
*   5.做预测。

让我们依次仔细看看每一步。

**注意**:使用 PyTorch API 实现这些步骤的方法有很多，虽然我的目的是向您展示最简单的，或者最常见的，或者最惯用的。

如果你发现了更好的方法，请在下面的评论中告诉我。

### 第一步:准备数据

第一步是加载和准备数据。

神经网络模型需要数值输入数据和数值输出数据。

您可以使用标准 Python 库来加载和准备表格数据，如 CSV 文件。例如，Pandas 可以用来加载你的 CSV 文件，scikit-learn 的工具可以用来编码分类数据，比如类标签。

PyTorch 提供了[数据集类](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)，您可以扩展和定制该类来加载数据集。

例如，数据集对象的构造函数可以加载数据文件(例如 CSV 文件)。然后，您可以覆盖可用于获取数据集长度(行数或样本数)的 *__len__()* 函数，以及可用于通过索引获取特定样本的 *__getitem__()* 函数。

加载数据集时，还可以执行任何所需的转换，如缩放或编码。

自定义*数据集*类的框架如下。

```py
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X = ...
        self.y = ...

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
```

加载后，PyTorch 提供[数据加载器类](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)在模型的训练和评估过程中导航*数据集*实例。

可以为训练数据集、测试数据集甚至验证数据集创建*数据加载器*实例。

[random_split()函数](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)可用于将数据集分割成训练集和测试集。分割后，可以将从*数据集*中选择的行提供给数据加载器，以及批次大小和数据是否应该在每个时期进行混洗。

例如，我们可以通过传入数据集中选定的行样本来定义*数据加载器*。

```py
...
# create the dataset
dataset = CSVDataset(...)
# select rows from the dataset
train, test = random_split(dataset, [[...], [...]])
# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)
```

一旦定义，就可以枚举*数据加载器*，每次迭代产生一批样本。

```py
...
# train the model
for i, (inputs, targets) in enumerate(train_dl):
	...
```

### 步骤 2:定义模型

下一步是定义一个模型。

在 PyTorch 中定义模型的习惯用法包括定义一个扩展[模块类](https://pytorch.org/docs/stable/nn.html#module)的类。

您的类的构造函数定义了模型的层，forward()函数是定义如何通过模型的定义层向前传播输入的覆盖。

有很多层可用，例如[线性](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear)用于全连接层， [Conv2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d) 用于卷积层， [MaxPool2d](https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d) 用于汇集层。

激活函数也可以定义为图层，如 [ReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU) 、 [Softmax](https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax) 、 [Sigmoid](https://pytorch.org/docs/stable/nn.html#torch.nn.Sigmoid) 。

下面是一个简单的一层 MLP 模型的例子。

```py
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 1)
        self.activation = Sigmoid()

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
```

在构造函数中定义层之后，也可以初始化给定层的权重。

常见的例子包括[泽维尔](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_)和 [He 权重](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_)初始化方案。例如:

```py
...
xavier_uniform_(self.layer.weight)
```

### 第三步:训练模型

训练过程需要定义损失函数和优化算法。

常见的损失函数包括:

*   [BCELoss](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss) :二元分类的二元交叉熵损失。
*   [交叉熵](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss):多类分类的分类交叉熵损失。
*   [均方损失](https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss):回归的均方损失。

有关损失函数的更多信息，请参见教程:

*   [用于训练深度学习神经网络的损失和损失函数](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

随机梯度下降用于优化，标准算法由 [SGD 类](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)提供，不过也有其他版本的算法，如[亚当](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)。

```py
# define the optimization
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

训练模型包括为训练数据集枚举*数据加载器*。

首先，训练时期的数量需要一个循环。然后对于随机梯度下降的小批量需要一个内环。

```py
...
# enumerate epochs
for epoch in range(100):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
    	...
```

模型的每次更新都涉及相同的一般模式，包括:

*   清除最后一个误差梯度。
*   输入通过模型的前向传递。
*   计算模型输出的损失。
*   通过模型反向传播误差。
*   更新模型以减少损失。

例如:

```py
...
# clear the gradients
optimizer.zero_grad()
# compute the model output
yhat = model(inputs)
# calculate loss
loss = criterion(yhat, targets)
# credit assignment
loss.backward()
# update model weights
optimizer.step()
```

### 第四步:评估模型

一旦模型合适，就可以在测试数据集上对其进行评估。

这可以通过对测试数据集使用*数据加载器*并收集测试集的预测，然后将预测与测试集的期望值进行比较并计算性能指标来实现。

```py
...
for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat = model(inputs)
    ...
```

### 第五步:做预测

拟合模型可用于对新数据进行预测。

例如，您可能只有一张图像或一行数据，并且想要进行预测。

这要求您将数据包装在 [PyTorch Tensor](https://pytorch.org/docs/stable/tensors.html) 数据结构中。

张量只是保存数据的 NumPy 数组的 PyTorch 版本。它还允许您在模型图中执行自动微分任务，比如在训练模型时向后调用*(*)。

预测也将是张量，尽管您可以通过从自动微分图中分离张量并调用 NumPy 函数来检索 NumPy 数组。

```py
...
# convert row to data
row = Variable(Tensor([row]).float())
# make prediction
yhat = model(row)
# retrieve numpy array
yhat = yhat.detach().numpy()
```

现在我们已经在高级别上熟悉了 PyTorch API 和模型生命周期，让我们看看如何从头开始开发一些标准的深度学习模型。

## 3.如何开发 PyTorch 深度学习模型

在本节中，您将发现如何使用标准深度学习模型开发、评估和进行预测，包括多层感知器(MLP)和卷积神经网络(CNN)。

多层感知器模型，简称 MLP，是一个标准的全连接神经网络模型。

它由节点层组成，其中每个节点连接到上一层的所有输出，每个节点的输出连接到下一层节点的所有输入。

MLP 是具有一个或多个完全连接的层的模型。这种模型适用于表格数据，即每个变量有一列，每个变量有一行的表格或电子表格中的数据。有三个预测建模问题，你可能想探索与 MLP；它们是二元分类、多类分类和回归。

让我们在真实数据集上为每一种情况拟合一个模型。

**注**:本节模型有效，但未优化。看看能不能提高他们的成绩。在下面的评论中发表你的发现。

### 3.1.如何开发二元分类的 MLP

我们将使用电离层二进制(两类)分类数据集来演示二进制分类的 MLP。

这个数据集包括预测在给定雷达回波的情况下，大气中是否存在结构。

数据集将使用 Pandas 自动下载，但您可以在这里了解更多信息。

*   [电离层数据集(csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv) 。
*   [电离层数据集描述](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)。

我们将使用[标签编码器](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)将字符串标签编码为整数值 0 和 1。该模型将适用于 67%的数据，剩余的 33%将用于评估，使用 [train_test_split()功能](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)进行分割。

用“ *He Uniform* 权重初始化来使用“ *relu* 激活是一个很好的做法。这种组合对于克服训练深度神经网络模型时[消失梯度](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)的问题大有帮助。有关 ReLU 的更多信息，请参见教程:

*   [整流线性单元的温和介绍](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

该模型预测 1 类概率，并使用 sigmoid 激活函数。该模型使用随机梯度下降进行优化，并寻求最小化[二元交叉熵损失](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)。

下面列出了完整的示例。

```py
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
```

运行该示例首先报告训练和测试数据集的形状，然后拟合模型并在测试数据集上对其进行评估。最后，对单行数据进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

**你得到了什么结果？**
**能不能换个模式做得更好？**
将你的发现发布到下面的评论中。

在这种情况下，我们可以看到模型达到了大约 94%的分类精确率，然后预测一行数据属于类别 1 的概率为 0.99。

```py
235 116
Accuracy: 0.948
Predicted: 0.998 (class=1)
```

### 3.2.如何开发多类分类的 MLP

我们将使用鸢尾花多类分类数据集来演示多类分类的 MLP。

这个问题涉及到预测鸢尾花的种类给定的花的措施。

数据集将使用 Pandas 自动下载，但您可以在这里了解更多信息。

*   [虹膜数据集(csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv) 。
*   [虹膜数据集描述](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names)。

鉴于它是一个多类分类，模型必须在输出层为每个类有一个节点，并使用 softmax 激活函数。损失函数是交叉熵，它适用于整数编码的类标签(例如，一个类为 0，下一个类为 1，等等)。).

下面列出了在鸢尾花数据集上拟合和评估 MLP 的完整示例。

```py
# pytorch mlp for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(4)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction
row = [5.1,3.5,1.4,0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
```

运行该示例首先报告训练和测试数据集的形状，然后拟合模型并在测试数据集上对其进行评估。最后，对单行数据进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

**你得到了什么结果？
能不能换个型号做得更好？**
将你的发现发布到下面的评论中。

在这种情况下，我们可以看到该模型实现了大约 98%的分类准确率，然后预测了一行数据属于每个类的概率，尽管类 0 的概率最高。

```py
100 50
Accuracy: 0.980
Predicted: [[9.5524162e-01 4.4516966e-02 2.4138369e-04]] (class=0)
```

### 3.3.如何开发回归 MLP

我们将使用波士顿住房回归数据集来演示用于回归预测建模的 MLP。

这个问题涉及到根据房子和邻居的属性来预测房子的价值。

数据集将使用 Pandas 自动下载，但您可以在这里了解更多信息。

*   [波士顿住房数据集(csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv) 。
*   [波士顿住房数据集描述](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)。

这是一个涉及预测单个数值的回归问题。因此，输出层只有一个节点，并使用默认或线性激活函数(无激活函数)。拟合模型时，均方误差(mse)损失最小。

回想一下，这是回归，不是分类；因此，我们无法计算分类精确率。有关这方面的更多信息，请参见教程:

*   [机器学习中分类和回归的区别](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

下面列出了在波士顿住房数据集上拟合和评估 MLP 的完整示例。

```py
# pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(13)
# train the model
train_model(train_dl, model)
# evaluate the model
mse = evaluate_model(test_dl, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
# make a single prediction (expect class=1)
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = predict(row, model)
print('Predicted: %.3f' % yhat)
```

运行该示例首先报告训练和测试数据集的形状，然后拟合模型并在测试数据集上对其进行评估。最后，对单行数据进行预测。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

**你得到了什么结果？
能不能换个型号做得更好？**
将你的发现发布到下面的评论中。

在这种情况下，我们可以看到模型实现了大约 82 的 MSE，这是大约 9 的 RMSE(单位是千美元)。然后为单个示例预测值 21。

```py
339 167
MSE: 82.576, RMSE: 9.087
Predicted: 21.909
```

### 3.4.如何开发一个用于图像分类的有线电视网络

卷积神经网络，简称 CNNs，是一种为图像输入而设计的网络。

它们由带有提取特征(称为特征图)的[卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)和提取特征到最显著元素的[汇集层](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)的模型组成。

中枢神经系统最适合图像分类任务，尽管它们可以用于以图像为输入的各种任务。

一个流行的图像分类任务是 [MNIST 手写数字分类](https://en.wikipedia.org/wiki/MNIST_database)。它涉及成千上万的手写数字，这些数字必须被归类为 0 到 9 之间的数字。

torchvision API 提供了一个方便的函数来直接下载和加载这个数据集。

以下示例加载数据集并绘制前几幅图像。

```py
# load mnist dataset in pytorch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from matplotlib import pyplot
# define location to save or load the dataset
path = '~/.torch/datasets/mnist'
# define the transforms to apply to the data
trans = Compose([ToTensor()])
# download and define the datasets
train = MNIST(path, train=True, download=True, transform=trans)
test = MNIST(path, train=False, download=True, transform=trans)
# define how to enumerate the datasets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=True)
# get one batch of images
i, (inputs, targets) = next(enumerate(train_dl))
# plot some images
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, i+1)
	# plot raw pixel data
	pyplot.imshow(inputs[i][0], cmap='gray')
# show the figure
pyplot.show()
```

运行该示例会加载 MNIST 数据集，然后汇总默认的训练和测试数据集。

```py
Train: X=(60000, 28, 28), y=(60000,)
Test: X=(10000, 28, 28), y=(10000,)
```

然后创建一个图，显示训练数据集中手写图像示例的网格。

![Plot of Handwritten Digits From the MNIST dataset](img/8879fffacbd0941cfe4a29c2533f7d01.png)

来自 MNIST 数据集的手写数字图

我们可以训练一个 CNN 模型来对 MNIST 数据集中的图像进行分类。

请注意，图像是灰度像素数据的数组，因此，我们必须在数据中添加通道维度，然后才能将图像用作模型的输入。

最好将像素值从默认范围 0-255 缩放到均值为零、标准差为 1。有关缩放像素值的更多信息，请参见教程:

*   [如何手动缩放图像像素数据进行深度学习](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)

下面列出了在 MNIST 数据集上拟合和评估 CNN 模型的完整示例。

```py
# pytorch cnn for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# model definition
class CNN(Module):
    # define model elements
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        # second hidden layer
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        # fully connected layer
        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        X = X.view(-1, 4*4*50)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X

# prepare the dataset
def prepare_data(path):
    # define standardization
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # load dataset
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(10):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# prepare the data
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = CNN(1)
# # train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
```

运行该示例首先报告训练和测试数据集的形状，然后拟合模型并在测试数据集上对其进行评估。

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

**你得到了什么结果？
能不能换个型号做得更好？**
将你的发现发布到下面的评论中。

在这种情况下，我们可以看到该模型在测试数据集上实现了大约 98%的分类准确率。然后我们可以看到，模型为训练集中的第一幅图像预测了类别 5。

```py
60000 10000
Accuracy: 0.985
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [深度学习](https://amzn.to/2Y8JuBv)，2016 年。
*   [为深度学习编程 PyTorch:创建和部署深度学习应用程序](https://amzn.to/2LA71Gq)，2018。
*   [用 PyTorch](https://amzn.to/2Yw2s5q) 深度学习，2020。
*   [带 fastai 和 PyTorch 的程序员深度学习:没有博士学位的 ai 应用](https://amzn.to/2P0MQDM)，2020。

### PyTorch 项目

*   [PyTorch 主页](https://pytorch.org/)。
*   [PyTorch 文件](https://pytorch.org/docs/stable/index.html)
*   pytorch 安装指南
*   [问号，维基百科](https://en.wikipedia.org/wiki/PyTorch)。
*   在 GitHub 上的 pytorch。

### 蜜蜂

*   [torch.utils.data API](https://pytorch.org/docs/stable/data.html) 。
*   [torch.nn API](https://pytorch.org/docs/stable/nn.html) 。
*   [torch.nn.init API](https://pytorch.org/docs/stable/nn.init.html) 。
*   [torch.optim API](https://pytorch.org/docs/stable/optim.html) 。
*   [火炬。张量原料药](https://pytorch.org/docs/stable/tensors.html)

## 摘要

在本教程中，您发现了一个在 PyTorch 中开发深度学习模型的分步指南。

具体来说，您了解到:

*   Torch 和 PyTorch 的区别以及如何安装和确认 PyTorch 工作正常。
*   PyTorch 模型的五步生命周期以及如何定义、拟合和评估模型。
*   如何为回归、分类和预测建模任务开发 PyTorch 深度学习模型？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。