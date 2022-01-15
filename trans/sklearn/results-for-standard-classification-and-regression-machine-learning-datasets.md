# 标准机器学习数据集的最佳结果

> 原文：<https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/>

最后更新于 2020 年 8 月 28 日

初学机器学习的人在小的真实数据集上练习是很重要的。

所谓的标准机器学习数据集包含实际的观察结果，适合记忆，并且被很好地研究和理解。因此，初学者可以使用它们来快速测试、探索和实践数据准备和建模技术。

从业者可以确认他们是否具备在标准机器学习数据集上获得良好结果所需的数据技能。好的结果是高于给定数据集技术上可能的第 80 或第 90 百分位结果的结果。

从业者在标准机器学习数据集上开发的技能可以为处理更大、更具挑战性的项目提供基础。

在这篇文章中，你将发现用于分类和回归的标准机器学习数据集，以及你可能期望在每个数据集上获得的基线和良好结果。

看完这篇文章，你会知道:

*   标准机器学习数据集的重要性。
*   如何在标准机器学习数据集上系统地评估模型？
*   用于分类和回归的标准数据集，以及每个数据集的基线和预期良好性能。

**用我的新书[Python 机器学习精通](https://machinelearningmastery.com/machine-learning-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **更新 Jun/2020** :增加了玻璃和马绞痛数据集的改进结果。
*   **更新 2020 年 8 月**:增加了马绞痛、房屋和汽车进口的更好结果(感谢德拉戈斯斯坦)

![Results for Standard Classification and Regression Machine Learning Datasets](img/3d4368591e2bdbe94fbd76149d2ee7be.png)

标准分类和回归机器学习数据集的结果
图片由 [Don Dearing](https://flickr.com/photos/iwaswired/4562196414/) 提供，保留部分权利。

## 概观

本教程分为七个部分；它们是:

1.  小型机器学习数据集的价值
2.  标准机器学习数据集的定义
3.  标准机器学习数据集
4.  标准数据集的良好结果
5.  模型评估方法
6.  分类数据集的结果
    1.  二元分类数据集
        1.  电离层
        2.  皮马印度糖尿病
        3.  声纳
        4.  威斯康星州乳腺癌
        5.  马绞痛
    2.  多类分类数据集
        1.  鸢尾花
        2.  玻璃
        3.  葡萄酒
        4.  小麦种子
7.  回归数据集的结果
    1.  房屋
    2.  汽车保险
    3.  鲍鱼
    4.  汽车进口

## 小型机器学习数据集的价值

有许多用于分类和回归预测建模问题的小型机器学习数据集经常被重用。

有时，数据集被用作演示机器学习或数据准备技术的基础。其他时候，它们被用作比较不同技术的基础。

这些数据集是在应用机器学习的早期收集并公开提供的，当时数据和真实世界的数据集很少。因此，它们已经成为一个标准或经典，仅仅是因为它们的广泛采用和重用，而不是因为对问题的任何内在兴趣。

在其中一个数据集上找到一个好的模型并不意味着你已经“*解决了*”这个一般性的问题。此外，一些数据集可能包含可能被认为有问题或文化不敏感的名称或指标(*，这很可能不是收集数据时的意图*)。因此，它们有时也被称为“T4”玩具“T5”数据集。

这样的数据集对于机器学习算法的比较点并不真正有用，因为大多数经验实验几乎不可能重现。

然而，这样的数据集在今天的应用机器学习领域是有价值的。即使在标准机器学习库、大数据和大量数据的时代。

它们之所以有价值，主要有三个原因；它们是:

1.  数据集是真实的。
2.  数据集很小。
3.  数据集是**理解的**。

**真实数据集**与[虚构数据集](https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/)相比非常有用，因为它们杂乱无章。可能存在测量误差、缺失值、错误标记的示例等等。这些问题中的一些或全部必须被搜索和解决，并且是我们在自己的项目中可能遇到的一些属性。

**小数据集**与可能有数千兆字节大小的大数据集相比非常有用。小数据集可以很容易地放入内存中，并允许轻松快速地测试和探索许多不同的数据可视化、数据准备和建模算法。测试想法和获得反馈的速度对于初学者来说至关重要，而小数据集恰恰促进了这一点。

**与新的或新创建的数据集相比，理解的数据集**非常有用。特征被很好地定义，特征的单位被指定，数据的来源是已知的，并且数据集已经在几十个、几百个，以及在某些情况下几千个研究项目和论文中被很好地研究。这提供了一个可以比较和评估结果的环境，一个在全新领域中不可用的属性。

鉴于这些属性，我强烈建议机器学习初学者(以及对特定技术不熟悉的实践者)从标准机器学习数据集开始。

## 标准机器学习数据集的定义

我想更进一步，定义一个“*标准*”机器学习数据集的一些更具体的属性。

标准机器学习数据集具有以下属性。

*   少于 10，000 行(样本)。
*   少于 100 列(功能)。
*   最后一列是目标变量。
*   以 CSV 格式存储在单个文件中，没有标题行。
*   缺少用问号字符('？'标记的值)
*   有可能取得比天真更好的结果。

现在我们已经对数据集有了一个清晰的定义，让我们来看看一个“*好的*结果意味着什么。

## 标准机器学习数据集

如果数据集经常用于书籍、研究论文、教程、演示文稿等，则它是标准的机器学习数据集。

这些所谓的经典或标准机器学习数据集的最佳存储库是加州大学欧文分校(UCI)的机器学习存储库。该网站按类型对数据集进行分类，并提供关于每个数据集的数据和附加信息的下载以及相关论文的参考。

我为每种问题类型选择了五个或更少的数据集作为起点。

本文中使用的所有标准数据集都可以在 GitHub 上找到:

*   [机器学习掌握数据集](https://github.com/jbrownlee/Datasets)

还为每个数据集和数据集的其他详细信息(所谓的“*”)提供下载链接。名称*“文件”)。

每个代码示例都会自动为您下载给定的数据集。如果这是一个问题，您可以手动下载 CSV 文件，将其放在与代码示例相同的目录中，然后更改代码示例以使用文件名而不是网址。

例如:

```py
...
# load dataset
dataframe = read_csv('ionosphere.csv', header=None)
```

## 标准数据集的良好结果

初学者在使用标准机器学习数据集时面临的一个挑战是什么代表了一个好的结果。

一般来说，如果一个模型能够展示出比简单方法更好的性能，比如预测分类中的多数类或者回归中的平均值，那么这个模型就是有技巧的。这称为基线模型或性能基线，它提供了特定于数据集的相对性能度量。您可以在此了解更多信息:

*   [如何知道你的机器学习模型是否有好的性能](https://machinelearningmastery.com/how-to-know-if-your-machine-learning-model-has-good-performance/)

假设我们现在有了一种方法来确定模型对数据集是否有技巧，初学者仍然对给定数据集的性能上限感兴趣。这是了解你在应用机器学习过程中是否“T0”变好所必需的信息。

好并不意味着完美的预测。所有的模型都会有预测误差，完美的预测是不可能的(易处理？)在真实数据集上。

为数据集定义“*好的*”或“*最好的*”结果具有挑战性，因为它通常取决于模型评估方法，特别是评估中使用的数据集和库的版本。

好意味着在给定可用资源的情况下，*足够好*。通常，这意味着在给定无限的技能、时间和计算资源的情况下，技能得分可能高于数据集的第 80 或第 90 个百分点。

在本教程中，您将发现如何计算基线性能以及每个数据集上可能的“*好的*”(接近最佳)性能。您还将发现如何指定用于实现性能的数据准备和模型。

没有解释如何做到这一点，而是给出了一个简短的 Python 代码示例，您可以使用它来重现基线和良好的结果。

## 模型评估方法

评估方法简单快速，一般推荐在处理小型预测建模问题时使用。

程序评估如下:

*   使用 10 倍交叉验证对模型进行评估。
*   评估程序重复三次。
*   交叉验证拆分的随机种子是重复数(1、2 或 3)。

这产生了模型性能的 30 个估计，从中可以计算出平均值和标准偏差来总结给定模型的性能。

使用重复编号作为每个交叉验证拆分的种子，可确保在数据集上评估的每个算法获得相同的数据拆分，从而确保公平的直接比较。

使用 scikit-learn Python 机器学习库，下面的示例可用于评估给定的模型(或 [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) )。[repeated stratifiedfold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)类定义了用于分类的折叠和重复次数， [cross_val_score()函数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)定义了分数并执行评估，返回一个分数列表，从中可以计算出平均值和标准偏差。

```py
...
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
```

对于回归，我们可以使用 [RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) 类和 MAE 分数。

```py
...
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
```

所报告的“*好*”分数是我个人的一套“*在给定数据集*脚本上快速获得好结果的最好成绩。我认为这些分数代表了在每个数据集上可以达到的好分数，也许在每个数据集可能达到的第 90 或 95 个百分点，如果不是更好的话。

也就是说，我并不是说它们是最好的分数，因为我没有对表现良好的模型进行超参数调整。我把这个留给感兴趣的从业者做练习。如果从业者能够处理给定的数据集，获得最高百分分数足以证明其能力，则不需要最佳分数。

**注**:我会随着自己个人脚本的完善，取得更好的成绩，更新成绩和模型。

**对于一个数据集你能得到更好的分数吗？**
我很想知道。在下面的评论中分享你的模型和分数，我会尽量重现并更新帖子(*并给你满分！*)

让我们开始吧。

## 分类数据集的结果

分类是一个预测建模问题，它预测给定一个或多个输入变量的一个标签。

分类任务的基线模型是预测多数标签的模型。这可以在 scikit-learn 中使用带有“*最频繁*策略的 [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) 类来实现；例如:

```py
...
model = DummyClassifier(strategy='most_frequent')
```

分类模型的标准评估是分类精确率，尽管这对于不平衡和一些多类问题并不理想。尽管如此，不管是好是坏，这个分数将被使用(现在)。

准确性报告为 0 (0%或无技能)和 1 (100%或完美技能)之间的分数。

有两种主要类型的分类任务:二进制和多类分类，根据给定数据集要预测的标签数量分为两个或两个以上。鉴于分类任务在机器学习中的流行，我们将分别处理这两种分类问题。

### 二元分类数据集

在本节中，我们将回顾以下二元分类预测建模数据集的基线和良好性能:

1.  电离层
2.  皮马印度糖尿病
3.  声纳
4.  威斯康星州乳腺癌
5.  马绞痛

#### 电离层

*   下载:[电离层. csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv)
*   详情:[电离层.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Ionosphere
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = SVC(kernel='rbf', gamma='scale', C=10)
steps = [('s',StandardScaler()), ('n',MinMaxScaler()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (351, 34), (351,)
Baseline: 0.641 (0.006)
Good: 0.948 (0.033)
```

#### 皮马印度糖尿病

*   下载:[pima-印度人-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)
*   详情:[皮马-印第安人-糖尿病.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Pima Indian Diabetes
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = LogisticRegression(solver='newton-cg',penalty='l2',C=1)
m_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (768, 8), (768,)
Baseline: 0.651 (0.003)
Good: 0.774 (0.055)
```

#### 声纳

*   下载: [sonar.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv)
*   详情:[声纳.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Sonar
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = KNeighborsClassifier(n_neighbors=2, metric='minkowski', weights='distance')
steps = [('p',PowerTransformer()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (208, 60), (208,)
Baseline: 0.534 (0.012)
Good: 0.882 (0.071)
```

#### 威斯康星州乳腺癌

*   下载:[乳腺癌-威斯康星. csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv)
*   详情:[乳腺癌-威斯康星.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Wisconsin Breast Cancer
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv'
dataframe = read_csv(url, header=None, na_values='?')
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = SVC(kernel='sigmoid', gamma='scale', C=0.1)
steps = [('i',SimpleImputer(strategy='median')), ('p',PowerTransformer()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (699, 9), (699,)
Baseline: 0.655 (0.003)
Good: 0.973 (0.019)
```

#### 马绞痛

*   下载:[马绞痛 csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv)
*   详情:[马绞痛名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例(归功于 Dragos Stan)。

```py
# baseline and good result for Horse Colic
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = XGBClassifier(colsample_bylevel=0.9, colsample_bytree=0.9, importance_type='gain', learning_rate=0.01, max_depth=4, n_estimators=200, reg_alpha=0.1, reg_lambda=0.5, subsample=1.0)
imputer = SimpleImputer(strategy='median')
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (300, 27), (300,)
Baseline: 0.637 (0.010)
Good: 0.893 (0.057)
```

### 多类分类数据集

在本节中，我们将回顾以下多类分类预测建模数据集的基线和良好性能:

1.  鸢尾花
2.  玻璃
3.  葡萄酒
4.  小麦种子

#### 鸢尾花

*   下载: [iris.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv)
*   详细信息:[iris . name](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Iris
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = LinearDiscriminantAnalysis()
steps = [('p',PowerTransformer()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (150, 4), (150,)
Baseline: 0.333 (0.000)
Good: 0.980 (0.039)
```

#### 玻璃

*   下载: [glass.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv)
*   详情:[玻璃名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

注意:测试工具从 10 倍交叉验证更改为 5 倍交叉验证，以确保每一倍都有所有类的示例，并避免警告消息。

```py
# baseline and good result for Glass
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
weights = {0:1.0, 1:1.0, 2:2.0, 3:2.0, 4:2.0, 5:2.0}
model = RandomForestClassifier(n_estimators=1000, class_weight=weights, max_features=2)
m_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (214, 9), (214,)
Baseline: 0.355 (0.009)
Good: 0.815 (0.048)
```

#### 葡萄酒

*   下载: [wine.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv)
*   详情:[葡萄酒.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Wine
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = QuadraticDiscriminantAnalysis()
steps = [('s',StandardScaler()), ('n',MinMaxScaler()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (178, 13), (178,)
Baseline: 0.399 (0.017)
Good: 0.992 (0.020)
```

#### 小麦种子

*   下载:[小麦种子. csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)
*   详情:[小麦种子.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Wine
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# evaluate naive
naive = DummyClassifier(strategy='most_frequent')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = RidgeClassifier(alpha=0.2)
steps = [('s',StandardScaler()), ('m',model)]
pipeline = Pipeline(steps=steps)
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (210, 7), (210,)
Baseline: 0.333 (0.000)
Good: 0.973 (0.036)
```

## 回归数据集的结果

回归是一个预测建模问题，它预测给定一个或多个输入变量的数值。

分类任务的基线模型是预测平均值或中值的模型。这可以在 scikit-learn 中使用[dummymergressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)类使用“*中位数*策略来实现；例如:

```py
...
model = DummyRegressor(strategy='median')
```

回归模型的标准评估是平均绝对误差(MAE)，尽管这对于所有回归问题来说并不理想。尽管如此，不管是好是坏，这个分数将被使用(现在)。

MAE 被报告为 0(完美技能)和非常大的数字或无穷大(无技能)之间的错误分数。

在本节中，我们将回顾以下回归预测建模数据集的基线和良好性能:

1.  房屋
2.  汽车保险
3.  鲍鱼
4.  汽车进口

#### 房屋

*   下载: [housing.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   详情:[房屋名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Housing
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = y.astype('float32')
# evaluate naive
naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = XGBRegressor(colsample_bylevel=0.4, colsample_bynode=0.6, colsample_bytree=1.0, learning_rate=0.06, max_depth=5, n_estimators=700, subsample=0.8)
m_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (506, 13), (506,)
Baseline: 6.544 (0.754)
Good: 1.928 (0.292)
```

#### 汽车保险

*   下载:[车险. csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv)
*   详情:[车险.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Auto Insurance
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = y.astype('float32')
# evaluate naive
naive = DummyRegressor(strategy='median')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(naive, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = HuberRegressor(epsilon=1.0, alpha=0.001)
steps = [('p',PowerTransformer()), ('m',model)]
pipeline = Pipeline(steps=steps)
target = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
m_scores = cross_val_score(target, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (63, 1), (63,)
Baseline: 66.624 (19.303)
Good: 28.358 (9.747)
```

#### 鲍鱼

*   下载:[鲍鱼. csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv)
*   详情:[鲍鱼.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Abalone
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
# minimally prepare dataset
y = y.astype('float32')
# evaluate naive
naive = DummyRegressor(strategy='median')
transform = ColumnTransformer(transformers=[('c', OneHotEncoder(), [0])], remainder='passthrough')
pipeline = Pipeline(steps=[('ColumnTransformer',transform), ('Model',naive)])
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = SVR(kernel='rbf',gamma='scale',C=10)
target = TransformedTargetRegressor(regressor=model, transformer=PowerTransformer(), check_inverse=False)
pipeline = Pipeline(steps=[('ColumnTransformer',transform), ('Model',target)])
m_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (4177, 8), (4177,)
Baseline: 2.363 (0.116)
Good: 1.460 (0.075)
```

#### 汽车进口

*   下载: [auto_imports.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto_imports.csv)
*   详细信息:[auto _ imports . name](https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto_imports.names)

下面列出了在该数据集上实现基线和良好结果的完整代码示例。

```py
# baseline and good result for Auto Imports
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto_imports.csv'
dataframe = read_csv(url, header=None, na_values='?')
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print('Shape: %s, %s' % (X.shape,y.shape))
y = y.astype('float32')
# evaluate naive
naive = DummyRegressor(strategy='median')
cat_ix = [2,3,4,5,6,7,8,14,15,17]
num_ix = [0,1,9,10,11,12,13,16,18,19,20,21,22,23,24]
steps = [('c', Pipeline(steps=[('s',SimpleImputer(strategy='most_frequent')),('oe',OneHotEncoder(handle_unknown='ignore'))]), cat_ix), ('n', SimpleImputer(strategy='median'), num_ix)]
transform = ColumnTransformer(transformers=steps, remainder='passthrough')
pipeline = Pipeline(steps=[('ColumnTransformer',transform), ('Model',naive)])
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores = absolute(n_scores)
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# evaluate model
model = XGBRegressor(colsample_bylevel=0.2, colsample_bytree=0.6, learning_rate=0.05, max_depth=6, n_estimators=200, subsample=0.8)
pipeline = Pipeline(steps=[('ColumnTransformer',transform), ('Model',model)])
m_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
m_scores = absolute(m_scores)
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
```

**注**:考虑到算法或评估程序的随机性，或数值精确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例，您应该会看到以下结果。

```py
Shape: (201, 25), (201,)
Baseline: 5509.486 (1440.942)
Good: 1361.965 (290.236)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何知道你的机器学习模型是否有好的性能](https://machinelearningmastery.com/how-to-know-if-your-machine-learning-model-has-good-performance/)

### 文章

*   [UCI 机器学习资源库](https://archive.ics.uci.edu/ml/index.php)
*   [Statlog 数据集:结果比较](http://www.is.umk.pl/~duch/projects/projects/datasets-stat.html)，wodzisaw Duch。
*   [用于分类的数据集:结果比较](http://www.is.umk.pl/~duch/projects/projects/datasets.html)，wodzisaw Duch。
*   [机器学习，神经和统计分类](https://amzn.to/2lDHgeK)，1994。
*   [机器学习，神经和统计分类，主页](http://www1.maths.leeds.ac.uk/~charles/statlog/)，1994。
*   [数据集加载实用程序，scikit-learn](https://scikit-learn.org/stable/datasets/index.html) 。

## 摘要

在这篇文章中，您发现了用于分类和回归的标准机器学习数据集，以及人们可能期望在每个数据集上实现的基线和良好结果。

具体来说，您了解到:

*   标准机器学习数据集的重要性。
*   如何在标准机器学习数据集上系统地评估模型？
*   用于分类和回归的标准数据集，以及每个数据集的基线和预期良好性能。

**我错过了你最喜欢的数据集吗？**
在评论里告诉我，我会给它算个分，甚至可能加到这个帖子里。

**对于一个数据集你能得到更好的分数吗？**
我很想知道；在下面的评论中分享你的模型和分数，我会试着重现它并更新帖子(并给你完全的信任！)

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。