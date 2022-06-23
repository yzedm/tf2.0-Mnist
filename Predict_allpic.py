# -*- coding = utf-8 -*-
# @Time:2022-02-21 11:53
# @Author:ym
# @File:Predict_allpic.py
# @Software :PyCharm
from  Mnist_train import conv_network
import numpy as np
import cv2
import os
#加载网络结构
network = conv_network()
#加载网络权重
network.load_weights('G:\MY_DL\Mnist-classifailer\Mnist_logs\conv_train_ep20_loss0.03_val_loss0.99.h5')
# 需要预测的图片路径，原始的图片(pic路径下的文件)需要使用pic_precross.py文件进行处理
PIC_path = 'G:\MY_DL\Mnist-classifailer/test-pic/'
pic_list = os.listdir(PIC_path)
for pic in pic_list:
    pridict_img = cv2.imread(PIC_path+pic,-1)
    cv2.imshow('test',pridict_img)
    #cv2.waitKey()
    pridict_img = pridict_img.reshape((1,28,28,1)).astype('float')/255
    res=network.predict(pridict_img)
    res = np.array(res)
    max = np.argmax(res)
    print('真实读数',pic[0:2],'预测读数为',max)