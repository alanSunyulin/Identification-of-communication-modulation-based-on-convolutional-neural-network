# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:24:05 2019

@author: user
"""


#=============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files
 
#=======================================================================
#获取一张图片
def get_one_image(train):
    #输入参数：train,训练图片的路径
    #返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   #随机选择测试的图片
 
    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])  #由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image
 
#--------------------------------------------------------------------
#测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 10
 
       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 64, 64, 3])
 
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
 
       logit = tf.nn.softmax(logit)
 
       x = tf.placeholder(tf.float32, shape=[64, 64, 3])
 
       # you need to change the directories to yours.
       # logs_train_dir = 'E:/Re_train/image_data/inputdata/'
       logs_train_dir = 'D:/学习资料/论文/神经网络——信号调制方式识别/F_仿真/train'
 
       saver = tf.train.Saver()
 
       with tf.Session() as sess:
 
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')
 
           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a 4psk with possibility %.6f' %prediction[:, 0])
           elif max_index==1:
               print('This is a 4qam with possibility %.6f' %prediction[:, 1])
           elif max_index==2:
               print('This is a 8psk with possibility %.6f' %prediction[:, 2])
           elif max_index==3:
               print('This is a 8qam with possibility %.6f' %prediction[:, 3])
           elif max_index==4:
               print('This is a 16psk with possibility %.6f' %prediction[:, 4])
           elif max_index==5:
               print('This is a 16qam with possibility %.6f' %prediction[:, 5])
           elif max_index==6:
               print('This is a 32qam with possibility %.6f' %prediction[:, 6])               
           elif max_index==7:
               print('This is a 64qam with possibility %.6f' %prediction[:, 7])
           elif max_index==8:
               print('This is a 128qam with possibility %.6f' %prediction[:, 8]) 
           elif max_index==9:
               print('This is a 256qam with possibility %.6f' %prediction[:, 9])    
#------------------------------------------------------------------------
               
if __name__ == '__main__':
    
    train_dir = 'D:/学习资料/论文/神经网络——信号调制方式识别/F_仿真/train/'
    train, train_label, val, val_label = get_files(train_dir, 0.3)
    img = get_one_image(val)  #通过改变参数train or val，进而验证训练集或测试集
    evaluate_one_image(img)
#===========================================================================
