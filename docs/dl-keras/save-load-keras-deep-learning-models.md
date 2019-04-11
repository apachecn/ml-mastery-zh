# 保存并加载您的 Keras 深度学习模型

> 原文： [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)

Keras 是一个简单而强大的 Python 库，用于深度学习。

鉴于深度学习模型可能需要数小时，数天甚至数周才能进行训练，因此了解如何从磁盘保存和加载它们非常重要。

在这篇文章中，您将了解如何将 Keras 模型保存到文件中并再次加载它们以进行预测。

让我们开始吧。

*   **2017 年 3 月更新**：添加了先安装 h5py 的说明。在每个示例中的最终打印语句中添加了缺少括号。
*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。
*   **更新 March / 2018** ：添加了备用链接以下载数据集，因为原始图像已被删除。

![Save and Load Your Keras Deep Learning Models](img/f7c8bf23f866c7d1a641e25ab2c1e6cd.png)

保存并加载您的 Keras 深度学习模型
照片由 [art_inthecity](https://www.flickr.com/photos/art_inthecity/6346545268/) 保留，保留一些权利。

## 教程概述

Keras 将保存模型架构和保存模型权重的问题分开。

模型权重保存为 [HDF5 格式](http://www.h5py.org/)。这是一种网格格式，非常适合存储多维数字数组。

可以使用两种不同的格式描述和保存模型结构：JSON 和 YAML。

在这篇文章中，我们将看两个保存模型并将其加载到文件的示例：

*   将模型保存为 JSON。
*   将模型保存到 YAML。

每个示例还将演示如何将模型权重保存并加载到 HDF5 格式的文件中。

这些例子将使用在 Pima Indians 糖尿病二元分类数据集开始时训练的相同简单网络。这是一个包含所有数字数据的小型数据集，易于使用。您可以[下载此数据集](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)并将其放在您的工作目录中，文件名为“ _pima-indians-diabetes.csv_ ”（更新：[从这里下载](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)）。

确认您安装了最新版本的 Keras（截至 2017 年 3 月的 v1.2.2）。

注意：您可能需要先安装 _h5py_ ：

```py
sudo pip install h5py
```

## 将您的神经网络模型保存为 JSON

JSON 是一种用于分层描述数据的简单文件格式。

Keras 提供了使用带有 _to_json（）_ 函数的 JSON 格式描述任何模型的功能。这可以保存到文件中，然后通过 _model_from_json（）_ 函数加载，该函数将根据 JSON 规范创建新模型。

使用 _save_weights（）_ 函数直接从模型保存权重，然后使用对称 _load_weights（）_ 函数加载。

以下示例训练和评估 Pima Indians 数据集上的简单模型。然后将模型转换为 JSON 格式并写入本地目录中的 model.json。网络权重写入本地目录中的 _model.h5_ 。

从保存的文件加载模型和重量数据，并创建新模型。在加载模型使用之前编译它是很重要的。这样使用该模型进行的预测可以使用 Keras 后端的适当有效计算。

以相同的方式评估模型，打印相同的评估分数。

```py
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

运行此示例提供下面的输出。

```py
acc: 78.78%
Saved model to disk
Loaded model from disk
acc: 78.78%
```

该模型的 JSON 格式如下所示：

```py
{  
   "keras_version":"2.0.2",
   "backend":"theano",
   "config":[  
      {  
         "config":{  
            "dtype":"float32",
            "bias_regularizer":null,
            "activation":"relu",
            "bias_constraint":null,
            "use_bias":true,
            "bias_initializer":{  
               "config":{  

               },
               "class_name":"Zeros"
            },
            "kernel_regularizer":null,
            "activity_regularizer":null,
            "kernel_constraint":null,
            "trainable":true,
            "name":"dense_1",
            "kernel_initializer":{  
               "config":{  
                  "maxval":0.05,
                  "minval":-0.05,
                  "seed":null
               },
               "class_name":"RandomUniform"
            },
            "batch_input_shape":[  
               null,
               8
            ],
            "units":12
         },
         "class_name":"Dense"
      },
      {  
         "config":{  
            "kernel_regularizer":null,
            "bias_regularizer":null,
            "activation":"relu",
            "bias_constraint":null,
            "use_bias":true,
            "bias_initializer":{  
               "config":{  

               },
               "class_name":"Zeros"
            },
            "activity_regularizer":null,
            "kernel_constraint":null,
            "trainable":true,
            "name":"dense_2",
            "kernel_initializer":{  
               "config":{  
                  "maxval":0.05,
                  "minval":-0.05,
                  "seed":null
               },
               "class_name":"RandomUniform"
            },
            "units":8
         },
         "class_name":"Dense"
      },
      {  
         "config":{  
            "kernel_regularizer":null,
            "bias_regularizer":null,
            "activation":"sigmoid",
            "bias_constraint":null,
            "use_bias":true,
            "bias_initializer":{  
               "config":{  

               },
               "class_name":"Zeros"
            },
            "activity_regularizer":null,
            "kernel_constraint":null,
            "trainable":true,
            "name":"dense_3",
            "kernel_initializer":{  
               "config":{  
                  "maxval":0.05,
                  "minval":-0.05,
                  "seed":null
               },
               "class_name":"RandomUniform"
            },
            "units":1
         },
         "class_name":"Dense"
      }
   ],
   "class_name":"Sequential"
}
```

## 将您的神经网络模型保存到 YAML

此示例与上述 JSON 示例大致相同，只是 [YAML](https://en.wikipedia.org/wiki/YAML) 格式用于模型规范。

使用 YAML 描述模型，保存到文件 model.yaml，然后通过 _model_from_yaml（）_ 函数加载到新模型中。权重的处理方式与上面 HDF5 格式相同，如 model.h5。

```py
# MLP for Pima Indians Dataset serialize to YAML and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

运行该示例将显示以下输出：

```py
acc: 78.78%
Saved model to disk
Loaded model from disk
acc: 78.78%
```

以 YAML 格式描述的模型如下所示：

```py
backend: theano
class_name: Sequential
config:
- class_name: Dense
  config:
    activation: relu
    activity_regularizer: null
    batch_input_shape: !!python/tuple [null, 8]
    bias_constraint: null
    bias_initializer:
      class_name: Zeros
      config: {}
    bias_regularizer: null
    dtype: float32
    kernel_constraint: null
    kernel_initializer:
      class_name: RandomUniform
      config: {maxval: 0.05, minval: -0.05, seed: null}
    kernel_regularizer: null
    name: dense_1
    trainable: true
    units: 12
    use_bias: true
- class_name: Dense
  config:
    activation: relu
    activity_regularizer: null
    bias_constraint: null
    bias_initializer:
      class_name: Zeros
      config: {}
    bias_regularizer: null
    kernel_constraint: null
    kernel_initializer:
      class_name: RandomUniform
      config: {maxval: 0.05, minval: -0.05, seed: null}
    kernel_regularizer: null
    name: dense_2
    trainable: true
    units: 8
    use_bias: true
- class_name: Dense
  config:
    activation: sigmoid
    activity_regularizer: null
    bias_constraint: null
    bias_initializer:
      class_name: Zeros
      config: {}
    bias_regularizer: null
    kernel_constraint: null
    kernel_initializer:
      class_name: RandomUniform
      config: {maxval: 0.05, minval: -0.05, seed: null}
    kernel_regularizer: null
    name: dense_3
    trainable: true
    units: 1
    use_bias: true
keras_version: 2.0.2
```

## 进一步阅读

*   [如何保存 Keras 型号？ Keras 文档中的](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。
*   [关于 Keras 文档中的 Keras 型号](https://keras.io/models/about-keras-models/)。

## 摘要

在这篇文章中，您了解了如何序列化您的 Keras 深度学习模型。

您学习了如何将训练过的模型保存到文件中，然后加载它们并使用它们进行预测。

您还了解到，使用 HDF5 格式可以轻松存储模型权重，并且网络结构可以以 JSON 或 YAML 格式保存。

您对保存深度学习模型或此帖子有任何疑问吗？在评论中提出您的问题，我会尽力回答。