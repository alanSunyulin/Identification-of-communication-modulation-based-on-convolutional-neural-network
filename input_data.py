# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:09:43 2019

@author: user
"""

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
#============================================================================
#-----------------生成图片路径和标签的List------------------------------------
 

#============================================================================
#-----------------生成图片路径和标签的List------------------------------------
 
train_dir = 'D:\学习资料\论文\神经网络——信号调制方式识别\F_仿真\train'

PSK4 = []
label_PSK4 = []
QAM4 = []
label_QAM4 = []
PSK8 = []
label_PSK8 = []
QAM8 = []
label_QAM8 = []
PSK16 = []
label_PSK16 = []
QAM16 = []
label_QAM16 = []
QAM32 = []
label_QAM32 = []
QAM64 = []
label_QAM64 = []
QAM128 = []
label_QAM128 = []
QAM256 = []
label_QAM256 = []

#step1：获取'E:/Re_train/image_data/training_image'下所有的图片路径名，存放到
#对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir+'/PSK4'):
        PSK4.append(file_dir +'/PSK4'+'/'+ file) 
        label_PSK4.append(0)
    for file in os.listdir(file_dir+'/QAM4'):
        QAM4.append(file_dir +'/QAM4'+'/'+file)
        label_QAM4.append(1)
    for file in os.listdir(file_dir+'/PSK8'):
        PSK8.append(file_dir +'/PSK8'+'/'+ file) 
        label_PSK8.append(2)
    for file in os.listdir(file_dir+'/QAM8'):
        QAM8.append(file_dir +'/QAM8'+'/'+file)
        label_QAM8.append(3)
    for file in os.listdir(file_dir+'/PSK16'):
        PSK16.append(file_dir +'/PSK16'+'/'+file)
        label_PSK16.append(4)
    for file in os.listdir(file_dir+'/QAM16'):
        QAM16.append(file_dir +'/QAM16'+'/'+file)
        label_QAM16.append(5)
    for file in os.listdir(file_dir+'/QAM32'):
        QAM32.append(file_dir +'/QAM32'+'/'+file)
        label_QAM32.append(6)
    for file in os.listdir(file_dir+'/QAM64'):
        QAM64.append(file_dir +'/QAM64'+'/'+file)
        label_QAM64.append(7)
    for file in os.listdir(file_dir+'/QAM128'):
        QAM128.append(file_dir +'/QAM128'+'/'+file)
        label_QAM128.append(8)
    for file in os.listdir(file_dir+'/QAM256'):
        QAM256.append(file_dir +'/QAM256'+'/'+file)
        label_QAM256.append(9)
    
 
#step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((PSK4,QAM4,PSK8,QAM8,PSK16,QAM16,QAM32,QAM64,QAM128,QAM256))
    label_list = np.hstack((label_PSK4,label_QAM4,label_PSK4,label_QAM8,label_PSK16,
                            label_QAM16,label_QAM32,label_QAM64,label_QAM128,label_QAM256))
    #label_list = np.hstack((label_husky, label_jiwawa, label_poodle, label_qiutian))
 
    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    #从打乱的temp中再取出list（img和lab）
    #image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])
    #label_list = [int(i) for i in label_list]
    #return image_list, label_list
    
    #将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
 
    #将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    #ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))   #测试样本数
    n_train = n_sample - n_val   #训练样本数
 
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
 
    return tra_images, tra_labels, val_images, val_labels
    
    
#---------------------------------------------------------------------------
#--------------------生成Batch----------------------------------------------
 
#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    #转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
 
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
 
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0]) #read img from a queue  
    
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3) 
    
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
 
#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
#label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)
    #重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch   