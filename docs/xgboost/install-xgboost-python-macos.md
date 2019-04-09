# 如何在 macOS 上为 Python 安装 XGBoost

> 原文： [https://machinelearningmastery.com/install-xgboost-python-macos/](https://machinelearningmastery.com/install-xgboost-python-macos/)

XGBoost 是一个用于开发非常快速和准确的梯度提升模型的库。

它是 Kaggle 数据科学竞赛中许多获奖解决方案中心的图书馆。

在本教程中，您将了解如何在 macOS 上安装 Python 的 XGBoost 库。

让我们开始吧。

![How to Install XGBoost for Python on macOS](img/6b9d2f1049c556cbd5210724ef317015.jpg)

如何在 macOS 上安装 XGBoost for Python
照片来自 [auntjojo](https://www.flickr.com/photos/7682623@N02/7944342576/) ，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  安装 MacPorts
2.  构建 XGBoost
3.  安装 XGBoost

**注意**：我已经在一系列不同的 macOS 版本上使用了这个程序多年，并且没有改变。本教程是在 macOS High Sierra（10.13.1）上编写和测试的。

## 1.安装 MacPorts

您需要安装 GCC 和 Python 环境才能构建和安装 XGBoost for Python。

我推荐使用 GCC 7 和 Python 3.6，我建议使用 [MacPorts](https://www.macports.org/) 安装这些先决条件。

*   1.有关逐步安装 MacPorts 和 Python 环境的帮助，请参阅本教程：

[＆gt;＆gt;如何在 Mac OS X 上安装 Python 3 环境以进行机器学习和深度学习](https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/)

*   2.安装 MacPorts 和可用的 Python 环境后，您可以按如下方式安装和选择 GCC 7：

```py
sudo port install gcc7
sudo port select --set gcc mp-gcc7
```

*   3.确认您的 GCC 安装成功，如下所示：

```py
gcc -v
```

你应该看到印刷版的 GCC;例如：

```py
..
gcc version 7.2.0 (MacPorts gcc7 7.2.0_0)
```

你看到什么版本？
请在下面的评论中告诉我。

## 2.构建 XGBoost

下一步是为您的系统下载并编译 XGBoost。

*   1.首先，从 GitHub 查看代码库：

```py
git clone --recursive https://github.com/dmlc/xgboost
```

*   2.转到 xgboost 目录。

```py
cd xgboost/
```

*   3.复制我们打算用来将 XGBoost 编译到位的配置。

```py
cp make/config.mk ./config.mk
```

*   4.编译 XGBoost;这要求您指定系统上的核心数（例如，8，根据需要进行更改）。

```py
make -j8
```

构建过程可能需要一分钟，不应产生任何错误消息，尽管您可能会看到一些可以安全忽略的警告。

例如，编译的最后一个片段可能如下所示：

```py
...
a - build/learner.o
a - build/logging.o
a - build/c_api/c_api.o
a - build/c_api/c_api_error.o
a - build/common/common.o
a - build/common/hist_util.o
a - build/data/data.o
a - build/data/simple_csr_source.o
a - build/data/simple_dmatrix.o
a - build/data/sparse_page_dmatrix.o
a - build/data/sparse_page_raw_format.o
a - build/data/sparse_page_source.o
a - build/data/sparse_page_writer.o
a - build/gbm/gblinear.o
a - build/gbm/gbm.o
a - build/gbm/gbtree.o
a - build/metric/elementwise_metric.o
a - build/metric/metric.o
a - build/metric/multiclass_metric.o
a - build/metric/rank_metric.o
a - build/objective/multiclass_obj.o
a - build/objective/objective.o
a - build/objective/rank_obj.o
a - build/objective/regression_obj.o
a - build/predictor/cpu_predictor.o
a - build/predictor/predictor.o
a - build/tree/tree_model.o
a - build/tree/tree_updater.o
a - build/tree/updater_colmaker.o
a - build/tree/updater_fast_hist.o
a - build/tree/updater_histmaker.o
a - build/tree/updater_prune.o
a - build/tree/updater_refresh.o
a - build/tree/updater_skmaker.o
a - build/tree/updater_sync.o
c++ -std=c++11 -Wall -Wno-unknown-pragmas -Iinclude -Idmlc-core/include -Irabit/include -I/include -O3 -funroll-loops -msse2 -fPIC -fopenmp -o xgboost build/cli_main.o build/learner.o build/logging.o build/c_api/c_api.o build/c_api/c_api_error.o build/common/common.o build/common/hist_util.o build/data/data.o build/data/simple_csr_source.o build/data/simple_dmatrix.o build/data/sparse_page_dmatrix.o build/data/sparse_page_raw_format.o build/data/sparse_page_source.o build/data/sparse_page_writer.o build/gbm/gblinear.o build/gbm/gbm.o build/gbm/gbtree.o build/metric/elementwise_metric.o build/metric/metric.o build/metric/multiclass_metric.o build/metric/rank_metric.o build/objective/multiclass_obj.o build/objective/objective.o build/objective/rank_obj.o build/objective/regression_obj.o build/predictor/cpu_predictor.o build/predictor/predictor.o build/tree/tree_model.o build/tree/tree_updater.o build/tree/updater_colmaker.o build/tree/updater_fast_hist.o build/tree/updater_histmaker.o build/tree/updater_prune.o build/tree/updater_refresh.o build/tree/updater_skmaker.o build/tree/updater_sync.o dmlc-core/libdmlc.a rabit/lib/librabit.a -pthread -lm -fopenmp
```

这一步对你有用吗？
请在下面的评论中告诉我。

## 3.安装 XGBoost

您现在可以在系统上安装 XGBoost 了。

*   1.将目录更改为 xgboost 项目的 Python 包。

```py
cd python-package
```

*   2.安装 Python XGBoost 包。

```py
sudo python setup.py install
```

安装非常快。

例如，在安装结束时，您可能会看到如下消息：

```py
...
Installed /opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/xgboost-0.6-py3.6.egg
Processing dependencies for xgboost==0.6
Searching for scipy==1.0.0
Best match: scipy 1.0.0
Adding scipy 1.0.0 to easy-install.pth file

Using /opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
Searching for numpy==1.13.3
Best match: numpy 1.13.3
Adding numpy 1.13.3 to easy-install.pth file

Using /opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
Finished processing dependencies for xgboost==0.6
```

*   3.通过打印 xgboost 版本确认安装是否成功，这需要加载库。

将以下代码保存到名为 _version.py 的文件中。_

```py
import xgboost
print("xgboost", xgboost.__version__)
```

从命令行运行脚本：

```py
python version.py
```

您应该看到 XGBoost 版本打印到屏幕：

```py
xgboost 0.6
```

你是怎么做的？
在以下评论中发布您的结果。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [如何在 Mac OS X 上安装 Python 3 环境以进行机器学习和深度学习](https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/)
*   [MacPorts 安装指南](https://www.macports.org/install.php)
*   [XGBoost 安装指南](http://xgboost.readthedocs.io/en/latest/build.html)

## 摘要

在本教程中，您了解了如何在 macOS 上逐步安装 XGBoost for Python。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。