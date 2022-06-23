# -*- coding = utf-8 -*-
# @Time:2022-02-20 18:26
# @Author:ym
# @File:Predict.py
# @Software :PyCharm
from  Mnist_train import conv_network
import numpy as np
import cv2
network = conv_network()
# 加载权重
network.load_weights('G:\MY_DL\Mnist-classifailer\Mnist_logs\conv_train_ep20_loss0.04_val_loss0.99.h5')
# 输入的图片路径
pridict_img = cv2.imread('G:\MY_DL\Mnist-classifailer\pic/5.jpg')
gray = cv2.cvtColor(pridict_img, cv2.COLOR_BGR2GRAY)
retval,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
gray_resize = cv2.resize(thresh,(28,28))
gray_resize = gray_resize.reshape((1,28,28,1)).astype('float')/255
res=network.predict(gray_resize)
res = np.array(res)
max = np.argmax(res)
print(max)