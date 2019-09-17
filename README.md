# ImageClassify
基于slim的图像分类

### slim的安装 [安装地址](https://github.com/tensorflow/models/tree/master/research/slim)
```
python setup.py
```
---


### 训练自己的数据集,制作数据集
```
python dataset/create_tfrecords.py
```
----


### 可视化tfrecord文件

```
python dataset/tfrecord_show.py
```
---


### 读取tfrecord文件，用于训练
需要自定义读取文件,模仿slim中的datasets文件
````
python dataset/sdfgoods.py
````
---

### 训练数据集
```
python train.py
```
* 需要注意的是dataset_name,就是自定义的读取文件名称
* model name 对应的是网络名称
* checkpoint_exclude_scopes 对应的是加载预训练模型，训练的层名称，其他层的参数使用checkpoint

---
### 预测分类结果
```
python predict.py
```
---
### 计算模型准确率
```
python cal_acc.py
```
---
### 数据集下载地址
只提供tfrecords分类数据集

---
### 模型下载地址

---