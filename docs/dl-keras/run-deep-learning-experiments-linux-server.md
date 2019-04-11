# 如何在 Linux 服务器上运行深度学习实验

> 原文： [https://machinelearningmastery.com/run-deep-learning-experiments-linux-server/](https://machinelearningmastery.com/run-deep-learning-experiments-linux-server/)

编写代码后，必须在具有大量 RAM，CPU 和 GPU 资源的大型计算机上运行深度学习实验，通常是云中的 Linux 服务器。

最近，我被问到这个问题：

> “你如何进行深度学习实验？”

这是一个我喜欢回答的好问题。

在这篇文章中，您将发现我用于在 Linux 上运行深度学习实验的方法，命令和脚本。

阅读这篇文章后，你会知道：

*   如何设计建模实验以将模型保存到文件。
*   如何运行单个 Python 实验脚本。
*   如何从 shell 脚本顺序运行多个 Python 实验。

让我们开始吧。

![How to Run Deep Learning Experiments on a Linux Server](img/5b3d95a2ac65cd8206b3a11433959297.png)

如何在 Linux 服务器上运行深度学习实验
[Patrik Nygren](https://www.flickr.com/photos/lattefarsan/10538489333/) 的照片，保留一些权利。

## 1\. Linux 服务器

我在工作站上编写了所有建模代码，并在远程 Linux 服务器上运行所有代码。

目前，我的偏好是在 EC2 上使用[亚马逊深度学习 AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB) 。有关为自己的实验设置此服务器的帮助，请参阅帖子：

*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 2.建模代码

我编写代码，以便每个 python 文件有一个实验。

大多数情况下，我正在处理大型数据，如图像字幕，文本摘要和机器翻译。

每个实验都适合模型，并将整个模型或权重保存到 [HDF5 文件](http://www.h5py.org/)，以便以后重复使用。

有关将模型保存到文件的更多信息，请参阅以下帖子：

*   [保存并加载您的 Keras 深度学习模型](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
*   [如何在 Keras 检查深度学习模型](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

我尝试准备一套实验（通常是 10 个或更多）以便在一个批次中运行。我还尝试将数据准备步骤分离为首先运行的脚本，并创建可随时加载和使用的训练数据集的 pickle 版本。

## 3.运行实验

每个实验可能会在训练期间输出一些诊断信息，因此，每个脚本的输出都会重定向到特定于实验的日志文件。如果事情失败，我也会重定向标准错误。

在运行时，Python 解释器可能不会经常刷新输出，尤其是在系统负载不足的情况下。我们可以使用 Python 解释器上的 _-u_ 标志强制将输出刷新到日志中。

运行单个脚本（ _myscript.py_ ）如下所示：

```py
python -u myscript.py >myscript.py.log 2>&1
```

我可以创建一个“_ 模型”_ 和一个“_ 结果”_ 目录，并更新要保存到这些目录的模型文件和日志文件，以保持代码目录清晰。

## 4.运行批量实验

每个 Python 脚本都按顺序运行。

创建一个 shell 脚本，按顺序列出多个实验。例如：

```py
#!/bin/sh

# run experiments
python -u myscript1.py >myscript1.py.log 2>&1
python -u myscript2.py >myscript2.py.log 2>&1
python -u myscript3.py >myscript3.py.log 2>&1
python -u myscript4.py >myscript4.py.log 2>&1
python -u myscript5.py >myscript5.py.log 2>&1
```

该文件将保存为“ _run.sh”_，与代码文件放在同一目录中并在服务器上运行。

例如，如果所有代码和 run.sh 脚本都位于“ _ec2-user_ ”主目录的“ _experiments_ ”目录中，则脚本将按如下方式运行：

```py
nohup /home/ec2-user/experiments/run.sh > /home/ec2-user/experiments/run.sh.log </dev/null 2>&1 &
```

该脚本作为后台进程运行，无法轻易中断。我还捕获了这个脚本的结果，以防万一。

您可以在本文中了解有关在 Linux 上运行脚本的更多信息：

*   [10 个亚马逊网络服务深度学习命令行方案](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/)

就是这样。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [源代码深度学习 AMI（CUDA 8，亚马逊 Linux）](https://aws.amazon.com/marketplace/pp/B01M0AXXQB)
*   [保存并加载您的 Keras 深度学习模型](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
*   [如何在 Keras 检查深度学习模型](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
*   [10 个亚马逊网络服务深度学习命令行方案](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/)
*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 摘要

在这篇文章中，您发现了我用于在 Linux 上运行深度学习实验的方法，命令和脚本。

具体来说，你学到了：

*   如何设计建模实验以将模型保存到文件。
*   如何运行单个 Python 实验脚本。
*   如何从 shell 脚本顺序运行多个 Python 实验。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。