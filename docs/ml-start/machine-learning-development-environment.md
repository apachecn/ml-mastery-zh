# 机器学习开发环境

> 原文： [https://machinelearningmastery.com/machine-learning-development-environment/](https://machinelearningmastery.com/machine-learning-development-environment/)

用于机器学习的开发环境可能与用于解决预测建模问题的机器学习方法一样重要。

一周几次，我得到一个问题，如：

> 您的机器学习开发环境是什么？

在这篇文章中，您将发现我使用的开发环境，并建议开发人员应用机器学习。

阅读这篇文章后，你会知道：

*   工作站和服务器硬件在机器学习中的作用之间的重要区别。
*   如何确保以可重复的方式安装和更新机器学习依赖项。
*   如何开发机器学习代码并以不会引入新问题的安全方式运行它。

让我们开始吧。

![Machine Learning Development Environment](img/18708d8a29f74d9357942024558dcd95.jpg)

机器学习开发环境
摄影： [Mohamed Aymen Bettaieb](https://www.flickr.com/photos/130799750@N03/16169265087/) ，保留一些权利。

您的机器学习开发环境是什么样的？
请在下面的评论中告诉我。

## 机器学习硬件

无论您是在学习机器学习还是正在开发大型操作模型，您的工作站硬件都无关紧要。

原因如下：

> 我不建议您在工作站上安装大型模型。

机器学习开发涉及许多小测试，以找出问题的初步答案，例如：

*   使用什么数据。
*   如何准备数据。
*   使用什么型号。
*   使用什么配置。

最终，您在工作站上的目标是确定要运行的实验。我称之为初步实验。对于初步实验，请使用较少的数据：一个适合您硬件功能的小样本。

较大的实验需要几分钟，几小时甚至几天才能完成。它们应该在除工作站之外的大型硬件上运行。

这可能是服务器环境，如果您使用深度学习方法，可能使用 GPU 硬件。此硬件可能由您的雇主提供，或者您可以在云中廉价租用，例如 AWS。

确实，您的工作站速度越快（CPU），工作站的容量（RAM）越多，您可以运行的初步小型实验越多或越大，您可以从更大的实验中获得的实验越多。因此，尽可能获得最好的硬件，但总的来说，使用你所拥有的东西。

我自己就像大型 Linux 机箱一样，拥有大量内存和大量内核，可用于严肃的 R＆amp; D.对于日常工作，我喜欢 iMac，同样拥有尽可能多的内核和尽可能多的 RAM。

综上所述：

*   **工作站**。处理一小部分数据，找出要运行的大型实验。
*   **服务器**。运行需要数小时或数天的大型实验，并帮助您确定在操作中使用的模型。

## 安装机器学习依赖项

您必须安装用于机器学习开发的库依赖项。

这主要是您正在使用的库。

在 Python 中，这可能是熊猫，scikit-learn，Keras 等等。在 R 中，这是所有的包，也许是插入符号。

除了安装依赖项之外，您还应该有一个可重复的过程，以便您可以在几秒钟内再次设置开发环境，例如在新工作站和新服务器上。

我建议使用包管理器和脚本（如 shell 脚本）来安装所有内容。

在我的 iMac 上，我使用 macport 来管理已安装的软件包。我认为有两个脚本：一个用于在新的 mac 上安装我需要的所有软件包（例如在工作站或笔记本电脑升级之后），另一个脚本专门用于更新已安装的软件包。

库总是随着错误修复而更新，因此更新特定安装库（及其依赖项）的第二个脚本是关键。

这些是 shell 脚本，我可以随时运行，并且随着我需要安装新库而不断更新。

如果您需要有关设置环境的帮助，其中一个教程可能有所帮助：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)
*   [如何在 Mac OS X 上安装 Python 3 环境以进行机器学习和深度学习](https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/)
*   [如何使用 Python 3](https://machinelearningmastery.com/linux-virtual-machine-machine-learning-development-python-3/) 为机器学习开发创建 Linux 虚拟机

您可能希望在具有可重复环境方面将事情提升到新的水平，例如使用 [Docker](https://www.docker.com/) 等容器或维护您自己的虚拟化实例。

In summary:

*   **安装脚本**。维护一个脚本，您可以使用该脚本重新安装开发环境所需的所有内容。
*   **更新脚本**。维护脚本以更新机器学习开发的所有关键依赖关系并定期运行。

## 机器学习编辑器

我推荐一个非常简单的编辑环境。

机器学习开发的艰苦工作不是编写代码;相反，它处理已经提到的未知数。未知数如：

*   使用什么数据。
*   如何准备数据。
*   使用什么算法。
*   使用什么配置。

编写代码很容易，特别是因为您很可能使用现代机器学习库中的现有算法实现。

因此，您不需要花哨的 IDE;它无法帮助您获得这些问题的答案。

相反，我建议使用一个非常简单的文本编辑器，它提供基本的代码突出显示。

就个人而言，我使用并推荐 [Sublime Text](https://www.sublimetext.com/) ，但任何类似的文本编辑器都可以正常工作。

![Example of a Machine Learning Text Editor](img/d7e67eb116b49974d54495cce7ef951e.jpg)

机器学习文本编辑器的示例

一些开发人员喜欢使用笔记本，例如 [Jupyter](http://jupyter.org/index.html) 。我没有使用或推荐它们，因为我发现这些环境对开发具有挑战性;他们可以隐藏错误并为开发引入依赖性陌生感。

为了研究机器学习和机器学习开发，我建议编写可以直接从命令行或 shell 脚本运行的脚本或代码。

例如，可以使用相应的解释器直接运行 R 脚本和 Python 脚本。

![Example of Running a Machine Learning Model](img/edc337627c242fffba81b84ec2545d4d.jpg)

运行机器学习模型的示例

有关如何从命令行运行实验的更多建议，请参阅帖子：

*   [如何在 Linux 服务器上运行深度学习实验](https://machinelearningmastery.com/run-deep-learning-experiments-linux-server/)

获得最终模型（或预测集）后，可以使用项目的标准开发工具将其集成到应用程序中。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [机器学习用计算机硬件](https://machinelearningmastery.com/computer-hardware-for-machine-learning/)
*   [如何使用亚马逊网络服务上的 Keras 开发和评估大型深度学习模型](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)
*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)
*   [如何在 Mac OS X 上安装 Python 3 环境以进行机器学习和深度学习](https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/)
*   [如何使用 Python 3](https://machinelearningmastery.com/linux-virtual-machine-machine-learning-development-python-3/) 为机器学习开发创建 Linux 虚拟机
*   [如何在 Linux 服务器上运行深度学习实验](https://machinelearningmastery.com/run-deep-learning-experiments-linux-server/)

## 摘要

在这篇文章中，您发现了用于机器学习开发的硬件，依赖项和编辑器。

具体来说，你学到了：

*   工作站和服务器硬件在机器学习中的作用之间的重要区别。
*   如何确保以可重复的方式安装和更新机器学习依赖项。
*   如何开发机器学习代码并以不会引入新问题的安全方式运行它。

What does your machine learning development environment look like?
Let me know in the comments below.

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。