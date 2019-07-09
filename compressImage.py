# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:29:09 2019

@author: user
"""

import PIL.Image as Image
import os
 
#图片压缩批处理
def compressImage(srcPath,dstPath):
    for filename in os.listdir(srcPath):
        #如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)
 
        #拼接完整的文件或文件夹路径
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
 
        # 如果是文件就处理
        if os.path.isfile(srcFile):
            try:
                #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
                sImg=Image.open(srcFile)
                sImg=sImg.convert('RGB')#转换未灰度图
                w,h=sImg.size
                dImg=sImg.resize((int(64),int(64)),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
                dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
                print (dstFile+" 成功！")
            except Exception:
                print(dstFile+"失败！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
 
        # 如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile, dstFile)
 
if __name__=='__main__':
#    compressImage("D:/学习资料/论文/神经网络——信号调制方式识别/仿真/仿真练习/data/test/4PSK","D:/学习资料/论文/神经网络——信号调制方式识别/仿真/仿真练习/64X64data/test/4PSK")
    world1 = input("请输入要压缩的文件夹：")
    world2 = input("请输入压缩后要保存的文件夹：")
    compressImage(world1,world2)  
       
       
      
              
