# -*- coding = utf-8 -*-
# @Time:2022-02-20 17:14
# @Author:ym
# @File:Mnist_train.py
# @Software :PyCharm
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras import  callbacks,models,layers,regularizers
from tensorflow.keras.optimizers import  RMSprop
from tensorflow.keras.datasets import mnist

# 加载数据集
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

# 转换格式 将图片装换为一维向量
train_images = train_images.reshape((60000,28,28,1)).astype('float')/255
test_images = test_images.reshape((10000,28,28,1)).astype('float')/255
# 转换格式 将标签转换为独热表示
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 搭建网络
def conv_network():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
    network.add(layers.AveragePooling2D(2,2))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D(2, 2))
    network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(units=32, activation='relu'))
    network.add(layers.Dense(units=10,activation='softmax'))
    return network
# 编译 训练网络 保存训练参数在Mnist_logs文件下
def train(callbacks=callbacks):
    checkpoint = callbacks.ModelCheckpoint(
            'Mnist_logs' + '/conv_train'+'_ep{epoch:02d}_loss{loss:.2f}_val_loss{accuracy:.2f}.h5',
            monitor='accuracy', save_weights_only=True, save_best_only=False, period=1)
    callbacks=[checkpoint]
    network = conv_network()
    network.compile(optimizer=RMSprop(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    network.fit(train_images,train_labels,epochs=20,batch_size=128,verbose = 2,callbacks=callbacks)

    print(network.summary())
net = conv_network()
net.summary()