# -*- coding = utf-8 -*-
# @Time:2022-02-20 19:20
# @Author:ym
# @File:pic_precross.py
# @Software :PyCharm
import cv2
import os
# 将自己的手写图片转换成网络输入的格式
pic_list = os.listdir('G:\MY_DL\Mnist-classifailer\pic')
for pic in pic_list:
    pridict_img = cv2.imread('G:\MY_DL\Mnist-classifailer\pic/'+pic)
    gray = cv2.cvtColor(pridict_img, cv2.COLOR_BGR2GRAY)
    retval,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    gray_resize = cv2.resize(thresh,(28,28))
    cv2.imwrite('G:\MY_DL\Mnist-classifailer/test-pic/' + pic[0:1]+'.png', gray_resize)