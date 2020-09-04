import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score


def iamge_transform(x_train,x_test):

    # 将28*28图像转成32*32的格式
    x_train = np.pad(x_train, ((0,0), (2,2), (2,2)), 'constant', constant_values=0)
    x_test = np.pad(x_test, ((0,0), (2,2), (2,2)), 'constant', constant_values=0)
    # print(x_train.shape)

    # 数据类型转换 -> 换成tf需要的
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 数据正则化 -> 转换成(0,1)之间
    x_train /= 255
    x_test /= 255

    # 数据维度转换 四维 [n,h,w,c] n -> number, h -> 高度, w -> 宽度, c -> 通道
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    # print(x_test.shape)
    return x_train, x_test


def train_model(x_train, y_train,batch_size,num_epochs):
    '''
    构建模型，训练模型
    返回训练好的模型

    参数
    ——————————————
    x_trian:训练集
    y_trian:训练集标签
    batch_size:批大小
    num_epochs:训练次数
    filters:卷积核个数
    kernel_size:卷积核大小
    padding:填充方式
    activation：激活函数
    input_shape:输入数据格式
    pool_size：池化大小
    strides:步长
    units：输出的维数

    '''
    model = tf.keras.models.Sequential([
            # 第一卷积层
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation=tf.nn.relu, input_shape=(32,32,1)),
            # 第二层池化层 平均池化
            tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            # 第三层卷积层
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation=tf.nn.relu),
            # 第四层池化层 平均池化
            tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            # 扁平化层 将多维数据转化一维数据
            tf.keras.layers.Flatten(),
            # 第五层 全连接层  激活函数是relu
            tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
            # 第六层 全连接层  激活函数是relu
            tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
            # 第七层 全连接层  激活函数是softmax
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    # 优化器
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

    # 编译模型
    model.compile(optimizer=adam_optimizer,
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=['accuracy'])
    # 模型开始训练
    start_time = datetime.datetime.now()

    # 训练模型
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

    # 模型结束训练
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print('时间花费：', time_cost)

    return model


def save_model(model,filepath):
    '''
    保存模型
    model：模型
    filepath：保存路径
    '''
    model.save(filepath)


def load_models(filepath):
    '''
    加载模型
    filepath：模型所在的路径

    返回 加载的模型
    '''
    model = tf.keras.models.load_model(filepath)
    # print(model.summary()) # 打印模型结构
    return model


if __name__ == "__main__":

    # 保存模型的路径
    filepath = 'C:\\Users\\HKZ\\Desktop\\我的\\手写数字识别\\lenet_model.h5'

    # 超参数设置
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # 加载数据集 第一次加载稍微慢一点
    mnist = tf.keras.datasets.mnist
    # x_train是训练集数据，y_train是训练集标签,x_test是测试集图像，y_test是测试集标签
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    # 查看数据格式
    # print(x_train.shape, y_train.shape) 
    # print(x_test.shape, y_test.shape)

    # 随机显示一个图片并查看
    # image_index = 123
    # print(y_train[image_index]) 
    # plt.imshow(x_train[image_index]) # 显示图像
    # plt.show()

    # 将训练集和测试集转换成需要的格式
    x_train, x_test = iamge_transform(x_train, x_test) 
    
    # 构建并训练模型
    # model = train_model(x_train, y_train,batch_size,num_epochs)

    # 保存模型
    # save_model(model,filepath)

    # 加载模型  注意在加载模型之前 把上述两个语句注释掉
    model = load_models(filepath)

    # 模型展示
    print(model.summary())

    # 预测一张
    image_index = 100 # 自己随机输入

    pred = model.predict(x_test[image_index].reshape(1,32,32,1))
    print('预测结果',pred.argmax())

    plt.imshow(x_test[image_index].reshape(32,32), cmap='Greys')
    plt.show()

    #模型评估
    print(model.evaluate(x_test,  y_test, verbose=2))
