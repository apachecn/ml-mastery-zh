# Python 中机器学习的特征选择

> 原文： [https://machinelearningmastery.com/feature-selection-machine-learning-python/](https://machinelearningmastery.com/feature-selection-machine-learning-python/)

用于训练机器学习模型的数据功能会对您可以实现的表现产生巨大影响。

不相关或部分相关的功能会对模型表现产生负面影响。

在这篇文章中，您将发现[自动特征选择技术](http://machinelearningmastery.com/an-introduction-to-feature-selection/)，您可以使用 scikit-learn 在 python 中准备机器学习数据。

让我们开始吧。

*   **2016 年 12 月更新**：修正了 RFE 部分中有关所选变量的拼写错误。谢谢安德森。
*   **更新 Mar / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Feature Selection For Machine Learning in Python](img/da26835b5c336c5edfc7bb7c062aa7ba.jpg)

Python 中机器学习的特征选择
[Baptiste Lafontaine](https://www.flickr.com/photos/magn3tik/6022696093/) 的照片，保留一些权利。

## 特征选择

特征选择是一个过程，您可以自动选择数据中对您感兴趣的预测变量或输出贡献最大的那些特征。

在数据中具有不相关的特征会降低许多模型的准确性，尤其是线性和逻辑回归等线性算法。

在建模数据之前执行特征选择的三个好处是：

*   **减少过度拟合**：冗余数据越少意味着根据噪声做出决策的机会就越少。
*   **提高准确度**：误导性较差的数据意味着建模精度提高。
*   **缩短训练时间**：数据越少意味着算法训练越快。

您可以在文章[特征选择](http://scikit-learn.org/stable/modules/feature_selection.html)中了解有关使用 scikit-learn 进行特征选择的更多信息。

## 机器学习的特征选择

本节列出了 4 种用于 Python 机器学习的特征选择秘籍

这篇文章包含特征选择方法的秘籍。

每个秘籍都设计为完整且独立，因此您可以将其直接复制并粘贴到项目中并立即使用。

秘籍使用[皮马印第安人糖尿病数据集](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)来证明特征选择方法（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。这是一个二元分类问题，其中所有属性都是数字。

### 1.单变量选择

统计测试可用于选择与输出变量具有最强关系的那些特征。

scikit-learn 库提供 [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) 类，可以与一系列不同的统计测试一起使用，以选择特定数量的功能。

以下示例使用卡方检（chi ^ 2）统计检验非负特征来从 Pima Indians 糖尿病数据集中选择 4 个最佳特征。

```
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
```

您可以看到每个属性的分数和选择的 4 个属性（分数最高的分数）：`plas`，`test`，`mass`和`age`。

```
[  111.52   1411.887    17.605    53.108  2175.565   127.669     5.393
   181.304]
[[ 148\.     0\.    33.6   50\. ]
 [  85\.     0\.    26.6   31\. ]
 [ 183\.     0\.    23.3   32\. ]
 [  89\.    94\.    28.1   21\. ]
 [ 137\.   168\.    43.1   33\. ]]
```

### 2.递归特征消除

递归特征消除（或 RFE）通过递归地移除属性并在剩余的属性上构建模型来工作。

它使用模型精度来识别哪些属性（和属性组合）对预测目标属性的贡献最大。

您可以在 scikit-learn 文档中了解有关 [RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE) 类的更多信息。

下面的示例使用 RFE 和逻辑回归算法来选择前 3 个特征。算法的选择并不重要，只要它技巧性和一致性。

```
# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
```

你可以看到 RFE 选择前[3]特征为`preg`，_ 质量 _ 和`pedi`。

这些在`support_`数组中标记为 True，并在`ranking_`数组中标记为选项“1”。

```
Num Features: 3
Selected Features: [ True False False False False  True  True False]
Feature Ranking: [1 2 3 5 6 1 1 4]
```

### 3.主成分分析

主成分分析（或 PCA）使用线性代数将数据集转换为压缩形式。

通常，这称为数据简化技术。 PCA 的一个属性是您可以选择转换结果中的维数或主成分数。

在下面的示例中，我们使用 PCA 并选择 3 个主要组件。

通过查看 [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) API，了解有关 scikit-PCA 课程的更多信息。在[主成分分析维基百科文章](https://en.wikipedia.org/wiki/Principal_component_analysis)中深入研究 PCA 背后的数学。

```
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)
```

您可以看到转换的数据集（3 个主要组件）与源数据几乎没有相似之处。

```
Explained Variance: [ 0.88854663  0.06159078  0.02579012]
[[ -2.02176587e-03   9.78115765e-02   1.60930503e-02   6.07566861e-02
    9.93110844e-01   1.40108085e-02   5.37167919e-04  -3.56474430e-03]
 [  2.26488861e-02   9.72210040e-01   1.41909330e-01  -5.78614699e-02
   -9.46266913e-02   4.69729766e-02   8.16804621e-04   1.40168181e-01]
 [ -2.24649003e-02   1.43428710e-01  -9.22467192e-01  -3.07013055e-01
    2.09773019e-02  -1.32444542e-01  -6.39983017e-04  -1.25454310e-01]]
```

### 4.特征重要性

随机森林和额外树木等袋装决策树可用于估计特征的重要性。

在下面的示例中，我们为 Pima 印第安人糖尿病数据集开始构建 ExtraTreesClassifier 分类器。您可以在 scikit-learn API 中了解有关 [ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) 类的更多信息。

```
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
```

您可以看到我们获得了每个属性的重要性分数，其中分数越大，属性越重要。评分表明`plas`，_ 年龄 _ 和 _ 质量 _ 的重要性。

```
[ 0.11070069  0.2213717   0.08824115  0.08068703  0.07281761  0.14548537 0.12654214  0.15415431]
```

## 摘要

在这篇文章中，您发现了使用 scikit-learn 在 Python 中准备机器学习数据的功能选择。

您了解了 4 种不同的自动特征选择技术：

*   单变量选择。
*   递归特征消除。
*   主成分分析。
*   功能重要性。

如果您要查找有关功能选择的更多信息，请参阅以下相关帖子：

*   [使用 Caret R 封装进行特征选择](http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/)
*   [特征选择提高准确性并缩短训练时间](http://machinelearningmastery.com/feature-selection-to-improve-accuracy-and-decrease-training-time/)
*   [特征选择介绍](http://machinelearningmastery.com/an-introduction-to-feature-selection/)
*   [使用 Scikit-Learn 在 Python 中进行特征选择](http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/)

您对功能选择或此帖有任何疑问吗？在评论中提出您的问题，我会尽力回答。