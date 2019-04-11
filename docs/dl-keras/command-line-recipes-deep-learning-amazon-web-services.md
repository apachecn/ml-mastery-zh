# 10 个用于 Amazon Web Services 深度学习的命令行秘籍

> 原文： [https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/)

在 Amazon Web Services EC2 上运行大型深度学习流程是学习和开发模型的一种廉价而有效的方法。

只需几美元，您就可以访问数十 GB 的 RAM，数十个 CPU 内核和多个 GPU。我强烈推荐它。

如果您不熟悉 EC2 或 Linux 命令行，那么在云中运行深度学习脚本时，您会发现一组命令非常宝贵。

在本教程中，您将发现我每次使用 EC2 来适应大型深度学习模型时使用的 10 个命令的私有列表。

阅读这篇文章后，你会知道：

*   如何将数据复制到 EC2 实例和从 EC2 实例复制数据。
*   如何将脚本设置为安全地运行数天，数周或数月。
*   如何监控进程，系统和 GPU 表现。

让我们开始吧。

**注意**：从工作站执行的所有命令都假定您运行的是 Linux 类型的环境（例如 Linux，OS X 或 cygwin）。

**你有在 EC2 上运行模型的任何其他提示，技巧或喜欢的命令吗？**
请在下面的评论中告诉我。

![10 Command Line Recipes for Deep Learning on Amazon Web Services](img/2ff0956a0537ec5bd319693e3ac5b79a.png)

10 亚马逊网络服务深度学习命令行方案
[chascar](https://www.flickr.com/photos/chascar/6480093119/) 的照片，保留一些权利。

## 概观

本文中提供的命令假定您的 AWS EC2 实例已在运行。

为保持一致性，还做了一些其他假设：

*   您的服务器 IP 地址是 _54.218.86.47_ ;将其更改为服务器实例的 IP 地址。
*   您的用户名是 _ec2-user_ ;将其更改为您实例上的用户名。
*   您的 SSH 密钥位于 _〜/ .ssh /_ 中，文件名为 _aws-keypair.pem_ ;将其更改为 SSH 密钥位置和文件名。
*   您正在使用 Python 脚本。

如果您需要帮助来设置和运行基于 GPU 的 AWS EC2 实例以进行深度学习，请参阅教程：

*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 1.从您的工作站登录到服务器

您必须先登录服务器才能执行任何有用的操作。

您可以使用 SSH 安全 shell 轻松登录。

我建议将 SSH 密钥存储在 _〜/ .ssh /_ 目录中，并使用有用的名称。我使用名称 _aws-keypair.pem_ 。请记住：文件必须具有权限 600。

以下命令将使您登录到服务器实例。请记住将用户名和 IP 地址更改为相关的用户名和服务器实例 IP 地址。

```py
ssh -i ~/.ssh/aws-keypair.pem ec2-user@54.218.86.47
```

## 2.将文件从工作站复制到服务器

使用安全副本（scp）将文件从工作站复制到服务器实例。

以下示例在您的工作站上运行，将工作站本地目录中的 _script.py_ Python 脚本复制到您的服务器实例。

```py
scp -i ~/.ssh/aws-keypair.pem script.py ec2-user@54.218.86.47:~/
```

## 3.在服务器上运行脚本作为后台进程

您可以将 Python 脚本作为后台进程运行。

此外，您可以以这样的方式运行它，它将忽略来自其他进程的信号，忽略任何标准输入（stdin），并将所有输出和错误转发到日志文件。

根据我的经验，所有这些都是长期运行脚本以适应大型深度学习模型所必需的。

```py
nohup python /home/ec2-user/script.py >/home/ec2-user/script.py.log </dev/null 2>&1 &
```

这假设您正在运行位于 _/ home / ec2-user /_ 目录中的 _script.py_ Python 脚本，并且您希望将此脚本的输出转发到文件 _script.py.log_ 位于同一目录中。

调整你的需求。

如果这是你第一次体验 nohup，你可以在这里了解更多：

*   维基百科上的 [nohup](https://en.wikipedia.org/wiki/Nohup)

如果这是您第一次重定向标准输入（stdin），标准输出（标准输出）和标准错误（sterr），您可以在此处了解更多信息：

*   维基百科上的[重定向](https://en.wikipedia.org/wiki/Redirection_(computing))

## 4.在服务器上的特定 GPU 上运行脚本

如果您的 AWS EC2 实例可以针对您的问题处理它，我建议您同时运行多个脚本。

例如，您选择的 EC2 实例可能有 4 个 GPU，您可以选择在每个实例上运行一个脚本。

使用 CUDA，您可以指定要与环境变量 _CUDA_VISIBLE_DEVICES_ 一起使用的 GPU 设备。

我们可以使用上面相同的命令来运行脚本并指定要使用的特定 GPU 设备，如下所示：

```py
CUDA_VISIBLE_DEVICES=0 nohup python /home/ec2-user/script.py >/home/ec2-user/script.py.log </dev/null 2>&1 &
```

如果您的实例上有 4 个 GPU 设备，则可以将 _CUDA_VISIBLE_DEVICES = 0_ 指定为 _CUDA_VISIBLE_DEVICES = 3。_

我希望这可以用于 Theano 后端，但我只测试了用于 Keras 的 TensorFlow 后端。

您可以在帖子中了解有关 _CUDA_VISIBLE_DEVICES_ 的更多信息：

*   [CUDA Pro 提示：使用 CUDA_VISIBLE_DEVICES 控制 GPU 可见性](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)

## 5.监视服务器上的脚本输出

您可以在脚本运行时监视脚本的输出。

如果您在每个时期或每个算法运行后输出分数，这可能很有用。

此示例将列出脚本日志文件的最后几行，并在脚本中添加新行时更新输出。

```py
tail -f script.py.log
```

如果屏幕暂时没有获得新输出，亚马逊可能会积极关闭您的终端。

另一种方法是使用 watch 命令。我发现亚马逊将保持这个终端开放：

```py
watch "tail script.py.log"
```

我发现 python 脚本的标准输出（粗壮）似乎没有经常更新。

我不知道这是 EC2 还是 Python 的东西。这意味着您可能无法经常更新日志中的输出。当缓冲区达到固定大小或运行结束时，它似乎被缓冲并输出。

你对此有更多了解吗？
请在下面的评论中告诉我。

## 6.监视服务器上的系统和进程表现

监控 EC2 系统表现是个好主意。特别是你正在使用和剩下的 RAM 量。

您可以使用将每隔几秒更新一次的 top 命令来执行此操作。

```py
top -M
```

如果您知道其进程标识符（PID），还可以监视系统和进程。

```py
top -p PID -M
```

## 7.监控服务器上的 GPU 表现

密切关注 GPU 表现是一个好主意。

如果您计划并行运行多个脚本并使用 GPU RAM，请再次关注运行 GPU 的 GPU 利用率。

您可以使用 _nvidia-smi_ 命令来关注 GPU 的使用情况。我喜欢使用 _watch_ 命令来保持终端打开并清除每个新结果的屏幕。

```py
watch "nvidia-smi"
```

## 8.检查服务器上仍在运行哪些脚本

密切关注哪些脚本仍在运行也很重要。

您可以使用 _ps_ 命令执行此操作。

同样，我喜欢使用 watch 命令来保持终端打开。

```py
watch "ps -ef | grep python"
```

## 9.编辑服务器上的文件

我建议不要在服务器上编辑文件，除非你真的需要。

不过，您可以使用 _vi_ 编辑器编辑文件。

下面的示例将在 vi 中打开您的脚本。

```py
vi ~/script.py
```

当然，您可以使用自己喜欢的命令行编辑器，如 emacs;如果您是 Unix 命令行的新手，本说明非常适合您。

如果这是您第一次接触 vi，您可以在此处了解更多信息：

*   维基百科上的 [vi](https://en.wikipedia.org/wiki/Vi)

## 10.从您的工作站下载服务器中的文件

我建议将模型以及任何结果和图表明确保存到新脚本和单独文件中作为脚本的一部分。

您可以使用安全副本（scp）将这些文件从服务器实例下载到工作站。

以下示例从您的工作站运行，并将所有 PNG 文件从您的主目录复制到您的工作站。

```py
scp -i ~/.ssh/aws-keypair.pem ec2-user@54.218.86.47:~/*.png .
```

## 其他提示和技巧

本节列出了在 AWS EC2 上进行大量工作时的一些其他提示。

*   **一次运行多个脚本**。我建议选择具有多个 GPU 并一次运行多个脚本的硬件以充分利用该平台。
*   **仅在工作站上编写和编辑脚本**。将 EC2 视为伪生产环境，并且只在那里复制脚本和数据才能运行。在您的工作站上进行所有开发并编写代码的小测试，以确保它能按预期工作。
*   **将脚本输出显式保存到文件**。将结果，图形和模型保存到文件中，以后可以将这些文件下载到工作站进行分析和应用。
*   **使用 watch 命令**。亚马逊积极地杀死没有活动的终端会话。您可以使用 watch 命令密切关注事物，该命令可以足够频繁地发送数据以保持终端打开。
*   **从工作站**运行命令。打算在服务器上运行的上述任何命令也可以通过在命令前添加“ _ssh -_ i _〜/ .ssh /_ aws _ 来运行工作站-keypair。_ pem _ec2-user@54.218.86.47_ “并引用您要运行的命令。这对于全天检查流程非常有用。

## 摘要

在本教程中，您发现了我每次使用 GPU 在 AWS EC2 实例上训练大型深度学习模型时使用的 10 个命令。

具体来说，你学到了：

*   如何将数据复制到 EC2 实例和从 EC2 实例复制数据。
*   如何将脚本设置为安全地运行数天，数周或数月。
*   如何监控进程，系统和 GPU 表现。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。