# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:04:36 2019

@author: user
"""

import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
 
cwd='D:/学习资料/论文/神经网络——信号调制方式识别/仿真/data/train/'  #图片存放路径
classes={'4PSK','4QAM','8PSK','8QAM','16PSK','16QAM','32QAM','64QAM','128QAM','256QAM'} #人为 设定 10类
writer= tf.python_io.TFRecordWriter("20190506_train.tfrecords") #要生成的文件路径
 
for index, name in enumerate(classes):       # enumerate()可以同时获得索引和元素
    class_path = cwd + name + '\\'
    for img_name in os.listdir(class_path): 
        img_path = class_path + img_name #每一个图片的地址
 
        img = Image.open(img_path)
        #img = img.convert("RGB")  # 将图片转成3通道的RGB图片
        img = img.resize((32, 32))
        img_raw = img.tobytes() #将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

print('OK')
