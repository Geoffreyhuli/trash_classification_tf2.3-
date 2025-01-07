# 基于tensorflow2.3的垃圾分类系统
课程设计要做一个垃圾分类系统，需要识别可回收垃圾、厨余垃圾、有害垃圾和其他垃圾等四个大类，在网上找到了很多开源的数据集，但是质量参差不齐，而且有坏图的存在，所以我就将这些数据集还有自己爬取的数据一起清洗了一遍，全部保存为了jpg的格式，一共有245个小类和4个大类。
模型训练使用的是tensorflow2.3，并使用pyqt5构建了图形化界面。

## 数据集
压缩包

## 代码结构

```
models 目录下放置训练好的模型
results 目录下放置的是训练的训练过程的一些可视化的图，txt文件是测试结果
utils 主要是中间的一些文件，对这个项目无实际的用途
window.py 是界面文件，主要是利用pyqt5完成的界面，通过电脑摄像头可以对图片种类进行预测
test_model.py 是测试文件
train_mobilenet.py 是训练代码
预训练模型是文件F:\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
```

## 效果
在result文件夹下

## 如何运行
requirements
本项目在pycharm上运行，建议新建一个conda环境