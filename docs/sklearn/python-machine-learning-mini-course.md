# Python 机器学习迷你课程

> 原文： [https://machinelearningmastery.com/python-machine-learning-mini-course/](https://machinelearningmastery.com/python-machine-learning-mini-course/)

### _14 天内从开发人员到机器学习从业者 _

Python 是应用机器学习增长最快的平台之一。

在这个迷你课程中，您将了解如何入门，构建准确的模型，并在 14 天内使用 Python 自信地完成预测建模机器学习项目。

这是一个重要且重要的帖子。您可能想要将其加入书签。

让我们开始吧。

*   **2016 年 10 月更新**：更新了 sklearn v0.18 的示例。
*   **2018 年 2 月更新**：更新 Python 和库版本。
*   **更新 March / 2018** ：添加了备用链接以下载某些数据集，因为原件似乎已被删除。

![Python Machine Learning Mini-Course](img/88976f0a20963c78c165e917b60c8435.jpg)

Python 机器学习迷你课程
摄影： [Dave Young](https://www.flickr.com/photos/dcysurfer/7056436373/) ，保留一些权利。

## 这个迷你课程是谁？

在我们开始之前，让我们确保您在正确的位置。

以下列表提供了有关本课程设计对象的一般指导原则。

如果你没有完全匹配这些点，请不要惊慌，你可能只需要在一个或另一个区域刷新以跟上。

*   **开发人员知道如何编写一些代码**。这意味着一旦您了解基本语法，就可以获得像 Python 这样的新编程语言。这并不意味着你是一个向导编码器，只是你可以毫不费力地遵循基本的 C 语言。
*   **知道一点机器学习的开发人员**。这意味着您了解机器学习的基础知识，如交叉验证，一些算法和[偏方差权衡](http://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)。这并不意味着你是一个机器学习博士，只是你知道地标或知道在哪里查找它们。

这个迷你课程既不是 Python 的教科书，也不是机器学习的教科书。

它将把你从一个知道一点机器学习的开发人员带到一个开发人员，他可以使用 Python 生态系统获得结果，这是一个不断上升的专业机器学习平台。

## 迷你课程概述

这个迷你课程分为 14 节课。

您可以每天完成一节课（推荐）或在一天内完成所有课程（硬核！）。这取决于你有空的时间和你的热情程度。

以下是 14 个课程，通过 Python 中的机器学习，可以帮助您开始并提高工作效率：

*   **第 1 课**：下载并安装 Python 和 SciPy 生态系统。
*   **第 2 课**：在 Python，NumPy，Matplotlib 和 Pandas 中徘徊。
*   **第 3 课**：从 CSV 加载数据。
*   **第 4 课**：通过描述性统计理解数据。
*   **第 5 课**：用可视化理解数据。
*   **第 6 课**：通过预处理数据准备建模。
*   **第 7 课**：采用重采样方法的算法评估。
*   **第 8 课**：算法评估指标。
*   **第 9 课**：抽样检查算法。
*   **第 10 课**：模型比较和选择。
*   **第 11 课**：通过算法调整提高准确度。
*   **第 12 课**：通过集合预测提高准确度。
*   **第 13 课**：完成并保存你的模型。
*   **第 14 课**：Hello World 端到端项目。

每节课可能需要 60 秒或 30 分钟。花点时间，按照自己的进度完成课程。在下面的评论中提出问题甚至发布结果。

课程期望你去学习如何做事。我会给你提示，但每节课的部分内容是强迫你学习去哪里寻求 Python 平台的帮助（提示，我直接在这个博客上有所有的答案，使用搜索特征）。

我确实在早期课程中提供了更多帮助，因为我希望你建立一些自信和惯性。

**挂在那里，不要放弃！**

## 第 1 课：下载并安装 Python 和 SciPy

在您访问平台之前，您无法开始使用 Python 进行机器学习。

今天的课程很简单，您必须在计算机上下载并安装 Python 3.6 平台。

访问 [Python 主页](https://www.python.org/)并下载适用于您的操作系统（Linux，OS X 或 Windows）的 Python。在您的计算机上安装 Python。您可能需要使用特定于平台的软件包管理器，例如 OS X 上的 macport 或 RedHat Linux 上的 yum。

您还需要安装 [SciPy 平台](https://www.python.org/)和 scikit-learn 库。我建议使用与安装 Python 相同的方法。

您可以使用 Anaconda 一次安装（更容易）[。推荐给初学者。](https://www.continuum.io/downloads)

通过在命令行键入“python”，首次启动 Python。

使用以下代码检查您需要的所有版本：

```
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
```

如果有任何错误，请停止。现在是时候解决它们了。

需要帮忙？看本教程：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 第 2 课：在 Python，NumPy，Matplotlib 和 Pandas 中解决。

您需要能够读取和编写基本的 Python 脚本。

作为开发人员，您可以非常快速地学习新的编程语言。 Python 区分大小写，使用散列（＃）进行注释，并使用空格来表示代码块（空白很重要）。

今天的任务是在 Python 交互式环境中练习 Python 编程语言的基本语法和重要的 SciPy 数据结构。

*   练习分配，使用 Python 中的列表和流控制。
*   练习使用 NumPy 数组。
*   练习在 Matplotlib 中创建简单的图。
*   练习使用 Pandas Series 和 DataFrames。

例如，下面是创建 Pandas **DataFrame** 的简单示例。

```
# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
```

## 第 3 课：从 CSV 加载数据

机器学习算法需要数据。您可以从 CSV 文件加载自己的数据，但是当您开始使用 Python 进行机器学习时，您应该在标准机器学习数据集上练习。

今天课程的任务是将数据加载到 Python 中以及查找和加载标准机器学习数据集。

有许多优秀的 CSV 格式标准机器学习数据集，您可以在 [UCI 机器学习库](http://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)上下载和练习。

*   使用标准库中的 [CSV.reader（）](https://docs.python.org/2/library/csv.html)练习将 CSV 文件加载到 Python 中。
*   练习使用 NumPy 和 [numpy.loadtxt（）](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html)函数加载 CSV 文件。
*   练习使用 Pandas 和 [pandas.read_csv（）](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)函数加载 CSV 文件。

为了帮助您入门，下面是一个片段，它将直接从 UCI 机器学习库使用 Pandas 加载 Pima 印第安人糖尿病数据集。

```
# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)
```

做得好到目前为止！在那里挂。

**到目前为止有任何问题吗？在评论中提问。**

## 第 4 课：使用描述性统计数据理解数据

将数据加载到 Python 后，您需要能够理解它。

您可以越好地理解数据，您可以构建的模型越好，越准确。理解数据的第一步是使用描述性统计。

今天，您的课程是学习如何使用描述性统计数据来理解您的数据。我建议使用 Pandas DataFrame 上提供的辅助函数。

*   使用 **head（）**功能了解您的数据，查看前几行。
*   使用 **shape** 属性查看数据的尺寸。
*   使用 **dtypes** 属性查看每个属性的数据类型。
*   使用 **describe（）**功能查看数据分布。
*   使用 **corr（）**函数计算变量之间的成对相关性。

以下示例加载 Pima 印第安人糖尿病数据集的开始并总结每个属性的分布。

```
# Statistical Summary
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
description = data.describe()
print(description)
```

**试一试！**

## 第 5 课：使用可视化理解数据

从昨天的课程开始，您必须花时间更好地了解您的数据。

提高对数据理解的第二种方法是使用数据可视化技术（例如绘图）。

今天，您的课程是学习如何在 Python 中使用绘图来理解单独的属性及其交互。同样，我建议使用 Pandas DataFrame 上提供的辅助函数。

*   使用 **hist（）**功能创建每个属性的直方图。
*   使用**图（kind ='box'）**功能创建每个属性的盒须图。
*   使用 **pandas.scatter_matrix（）**函数创建所有属性的成对散点图。

例如，下面的代码片段将加载糖尿病数据集并创建数据集的散点图矩阵。

```
# Scatter Plot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()
```

![Sample Scatter Plot Matrix](img/9f1a9b82f44e5cb7b01a368f464abfd8.jpg)

样本散点图矩阵

## 第 6 课：通过预处理数据准备建模

您的原始数据可能未设置为建模的最佳形状。

有时您需要预处理数据，以便最好地将数据中问题的固有结构呈现给建模算法。在今天的课程中，您将使用 scikit-learn 提供的预处理功能。

scikit-learn 库提供了两种用于转换数据的标准习语。每种变换在不同情况下都很有用：拟合和多变换以及组合拟合和变换。

您可以使用许多技术来准备建模数据。例如，尝试以下某些操作

*   使用比例和中心选项标准化数值数据（例如，平均值为 0，标准差为 1）。
*   使用范围选项标准化数值数据（例如，范围为 0-1）。
*   探索更高级的功能工程，例如二值化。

例如，下面的代码片段加载 Pima Indians 糖尿病数据集，计算标准化数据所需的参数，然后创建输入数据的标准化副本。

```
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```

## 第 7 课：使用重采样方法进行算法评估

用于训练机器学习算法的数据集称为训练数据集。用于训练算法的数据集不能用于为您提供有关新数据模型准确率的可靠估计。这是一个很大的问题，因为创建模型的整个想法是对新数据做出预测。

您可以使用称为重采样方法的统计方法将训练数据集拆分为子集，一些用于训练模型，另一些则用于估计模型对未见数据的准确率。

今天课程的目标是练习使用 scikit-learn 中提供的不同重采样方法，例如：

*   将数据集拆分为训练和测试集。
*   使用 k 折交叉验证估算算法的准确率。
*   使用留一交叉验证估算算法的准确率。

下面的片段使用 scikit-learn 使用 10 倍交叉验证来估计 Pima Indians 糖尿病数据集开始时 逻辑回归算法的准确率。

```
# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```

你得到了什么准确度？请在评论中告诉我。

**您是否意识到这是中途点？做得好！**

## 第 8 课：算法评估指标

您可以使用许多不同的度量标准来评估数据集上的机器学习算法的技能。

您可以通过 **cross_validation.cross_val_score（）**函数在 scikit-learn 中指定用于测试工具的度量标准，默认值可用于回归和分类问题。今天课程的目标是练习使用 scikit-learn 包中提供的不同算法表现指标。

*   练习在分类问题上使用 Accuracy 和 LogLoss 指标。
*   练习生成混淆矩阵和分类报告。
*   练习在回归问题上使用 RMSE 和 RSquared 指标。

下面的片段演示了计算皮马印第安人糖尿病数据集开始时的 LogLoss 指标。

```
# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
```

你得到了什么日志损失？请在评论中告诉我。

## 第 9 课：抽样检查算法

您不可能事先知道哪种算法在您的数据上表现最佳。

你必须使用反复试验的过程来发现它。我称这种点检算法。 scikit-learn 库提供了许多机器学习算法和工具的接口，用于比较这些算法的估计精度。

在本课程中，您必须练习现场检查不同的机器学习算法。

*   点检数据集上的线性算法（例如线性回归，逻辑回归和线性判别分析）。
*   在数据集上检查一些非线性算法（例如 KNN，SVM 和 CART）。
*   在数据集上对一些复杂的集成算法进行抽查（例如随机森林和随机梯度增强）。

例如，下面的片段在波士顿房价数据集上点检查 K 最近邻 算法。

```
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```

你得到的误差是什么意思？请在评论中告诉我。

## 第 10 课：模型比较和选择

现在您已了解如何在数据集上查看机器学习算法，您需要知道如何比较不同算法的估计表现并选择最佳模型。

在今天的课程中，您将练习比较 Python 中的机器学习算法与 scikit-learn 的准确率。

*   在数据集上比较线性算法。
*   在数据集上比较非线性算法。
*   将相同算法的不同配置相互比较。
*   创建比较算法的结果图。

以下示例将 Pima Indians 糖尿病数据集开始时的 逻辑回归和线性判别分析相互比较。

```
# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

哪种算法效果更好？你能做得更好吗？请在评论中告诉我。

## 第 11 课：通过算法调整提高准确率

一旦找到一个或两个在数据集上表现良好的算法，您可能希望提高这些模型的表现。

提高算法表现的一种方法是将其参数调整为特定数据集。

scikit-learn 库提供了两种搜索机器学习算法参数组合的方法。今天课程的目标是练习每一个。

*   使用您指定的网格搜索调整算法的参数。
*   使用随机搜索调整算法的参数。

下面使用的片段是在皮马印第安人糖尿病数据集开始时使用网格搜索岭回归算法的示例。

```
# Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
```

哪些参数达到了最佳效果？你能做得更好吗？请在评论中告诉我。

## 第 12 课：使用集合预测提高准确率

另一种可以提高模型表现的方法是组合多个模型的预测。

有些型号提供内置的这种功能，例如用于装袋的随机森林和用于增强的随机梯度增强。另一种称为投票的集合可用于将来自多个不同模型的预测组合在一起。

在今天的课程中，您将练习使用整体方法。

*   使用随机森林和额外树木算法练习套袋合奏。
*   使用梯度增强机和 AdaBoost 算法练习增强乐团。
*   通过将多个模型的预测结合在一起来实践投票合奏。

下面的代码片段演示了如何在皮马印第安人糖尿病数据集中使用随机森林算法（袋装决策树集合）。

```
# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

你能设计一个更好的合奏吗？请在评论中告诉我。

## 第 13 课：完成并保存模型

一旦在机器学习问题上找到了表现良好的模型，就需要完成它。

在今天的课程中，您将练习与完成模型相关的任务。

练习使用您的模型对新数据做出预测（在训练和测试期间看不到的数据）。
练习保存训练有素的模型进行归档并重新加载。

例如，下面的代码段显示了如何创建 逻辑回归模型，将其保存到文件，然后稍后加载并对未见数据做出预测。

```
# Save Model Using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
```

## 第 14 课：Hello World 端到端项目

您现在知道如何完成预测建模机器学习问题的每个任务。

在今天的课程中，您需要练习将各个部分组合在一起，并通过端到端的标准机器学习数据集进行操作。

完成[虹膜数据集](https://archive.ics.uci.edu/ml/datasets/Iris)端到端（机器学习的 hello 世界）

这包括以下步骤：

1.  使用描述性统计和可视化了解您的数据。
2.  预处理数据以最好地揭示问题的结构。
3.  使用您自己的测试工具对许多算法进行抽查。
4.  使用算法参数调整改善结果。
5.  使用集合方法改善结果。
6.  最终确定模型以备将来使用。

慢慢来，并记录您的结果。

你用的是什么型号的？你得到了什么结果？请在评论中告诉我。

## 结束！
（_ 看你有多远 _）

你做到了。做得好！

花点时间回顾一下你到底有多远。

*   您开始对机器学习感兴趣，并希望能够使用 Python 练习和应用机器学习。
*   您下载，安装并启动了 Python，这可能是第一次并开始熟悉该语言的语法。
*   在一些课程中，您慢慢地，稳定地学习了预测建模机器学习项目的标准任务如何映射到 Python 平台上。
*   基于常见机器学习任务的秘籍，您使用 Python 端到端地完成了第一次机器学习问题。
*   使用标准模板，您收集的秘籍和经验现在能够自己完成新的和不同的预测建模机器学习问题。

不要轻视这一点，你在很短的时间内走了很长的路。

这只是您使用 Python 进行机器学习之旅的开始。继续练习和发展你的技能。

## 摘要

你是如何使用迷你课程的？
你喜欢这个迷你课吗？

你有任何问题吗？有没有任何问题？
让我知道。在下面发表评论。