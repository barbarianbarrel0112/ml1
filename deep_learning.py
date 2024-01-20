# import the necessary packages
from keras.models import Sequential
from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import img_to_array
from keras.utils import to_categorical  # 用于one-hot编码
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd


class LeNet:
    def build(width, height, depth, classes):
        '''参数分别为：长 宽 高 分类'''
        # initialize the model
        model = Sequential()  # 建立线性堆叠模型
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        # 卷积1 过滤器大小为 5 * 5,会产生20个图像，卷积不会改变图像大小，起到了滤镜效果,设置ReLU激活函数
        model.add(Conv2D(filters=20, kernel_size=(5, 5), padding="same", input_shape=inputShape, activation='relu'))
        # 添加激活层
        # model.add(Activation("relu"))
        # 加入Dropout避免过拟合。
        model.add(Dropout(0.25))
        # 最大池化1 过滤器大小为 2 * 2，长和宽的步长均为2,不会改变图像的数量（仍旧是20），会改变大小（32*32变成16*16）
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        # 卷积2 过滤器大小为 5 * 5，会产生50个图像，卷积不会改变图像大小，起到了滤镜效果,设置ReLU激活函数
        model.add(Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation='relu'))
        # 激活函数
        # model.add(Activation("relu"))
        # 加入Dropout避免过拟合。
        # model.add(Dropout(0.25))
        # 最大池化2 过滤器大小为2 * 2，长和宽的步长均为2,不会改变图像的数量（仍旧是50），会改变大小（16*16变成8*8）
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        # Flatten层用来将输入“压平”
        model.add(Flatten())
        # Dense表示全连接层（500个神经元）
        model.add(Dense(500))
        model.add(Activation("relu"))
        # 加入Dropout避免过拟合。
        # model.add(Dropout(0.25))

        # softmax classifier
        # 建立输出层（分类数个神经元）,softmax可以将输出预测为每一个图像的概率
        model.add(Dense(classes, activation='softmax'))
        # 多分类
        model.add(Activation("softmax"))
        # 查看模型的摘要
        print(model.summary())
        # return the constructed network architecture
        return model

class VGG16:
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)

            # layer_1
            model.add(
                Conv2D(64, (3, 3), strides=(1, 1), input_shape=inputShape, padding='same', data_format='channels_last',
                       activation='relu', kernel_initializer='uniform'))
            model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                             kernel_initializer='uniform', activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            # layer_2
            model.add(
                Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu',
                       kernel_initializer='uniform'))
            model.add(
                Conv2D(128, (2, 2), strides=(1, 1), padding='same', data_format='channels_last', activation='relu',
                       kernel_initializer='uniform'))
            model.add(MaxPooling2D((2, 2)))

            # layer_3
            model.add(
                Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            # layer_4
            model.add(
                Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(512, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            # layer_5
            model.add(
                Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(
                Conv2D(512, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
            model.add(MaxPooling2D((2, 2)))


        model.add(Flatten())  # 拉平
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        # 建立输出层（分类数个神经元）,softmax可以将输出预测为每一个图像的概率
        model.add(Dense(classes, activation='softmax'))

        # 多分类
        model.add(Activation("softmax"))

        # model.summary()
        # print(model.summary())
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model




# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签
    # 直接读取数据，会节约时间
    if (os.path.exists('train_image_list1.csv.npy')
            & os.path.exists('test_image_list1.csv.npy')
            & os.path.exists('test_label_list.csv.npy')
            & os.path.exists('train_label_list.csv.npy')
            & os.path.exists('hunxiao.csv.npy')):
        train_image_list_1 = np.load('train_image_list1.csv.npy')
        train_label_list_1 = np.load('train_label_list.csv.npy')
        test_image_list_1 = np.load('test_image_list1.csv.npy')
        test_label_list_1 = np.load('test_label_list.csv.npy')
        test_label_list = np.load('hunxiao.csv.npy')
        print("训练集一共有%d张图\n" % len(train_label_list_1))
        print("测试集一共有%d张图\n" % len(test_label_list_1))
        return train_image_list_1, train_label_list_1, test_image_list_1, test_label_list_1, test_label_list
    image_list = []
    label_list = []
    name_dic = {'Black-grass': 0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 'Common wheat': 4,
                'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7, 'Scentless Mayweed': 8, 'Shepherds Purse': 9,
                'Small-flowered Cranesbill': 10, 'Sugar beet': 11}
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = str(file)
        name_count = 0
        for key in os.listdir(file_dir + file):
            name_count += 1
            image_list.append(file_dir + '\\' + file + '\\' + key)
            label_list.append(name_dic[file])
        print(name + "种类有" + str(name_count) + "张图片")
    print("一共有%d张图\n" % len(image_list))
    image_list = np.hstack(image_list)
    label_list = np.hstack(label_list)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    train_img, test_img = train_test_split(temp, train_size=0.7)
    train_image_list = list(train_img[:, 0])
    test_image_list = list(test_img[:, 0])
    train_label_list = list(train_img[:, 1])
    train_label_list = [int(i) for i in train_label_list]
    test_label_list = list(test_img[:, 1])
    test_label_list = [int(i) for i in test_label_list]

    train_image_list1 = []
    test_image_list1 = []
    for m in range(len(train_image_list)):
        image = cv2.imread(train_image_list[m])
        # print(image.shape) # 查看部分图片的shape
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        train_image_list1.append(image)
    for m in range(len(test_image_list)):
        image1 = cv2.imread(test_image_list[m])
        image1 = cv2.resize(image1, (norm_size, norm_size))
        image1 = img_to_array(image1)
        test_image_list1.append(image1)
    # 标准化：提高模型预测精准度，加快收敛
    train_image_list1 = np.array(train_image_list1, dtype="float") / 255.0
    test_image_list1 = np.array(test_image_list1, dtype="float") / 255.0
    # convert the labels from integers to vectors one-hot编码
    train_label_list1 = to_categorical(train_label_list, num_classes=CLASS_NUM)
    test_label_list1 = to_categorical(test_label_list, num_classes=CLASS_NUM)
    # 第一运行 把处理好的数据保存下来
    np.save('train_image_list1.csv', train_image_list1)
    np.save('test_image_list1.csv', test_image_list1)
    np.save('test_label_list.csv', test_label_list1)
    np.save('train_label_list.csv', train_label_list1)
    np.save('hunxiao.csv', test_label_list)
    return train_image_list1, train_label_list1, test_image_list1, test_label_list1, np.array(test_label_list)


def show_train_history(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS  # 训练周期数
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N) , H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')
    plt.show()


def train(aug, trainX, trainY, testX, testY, test_label_list):
    # initialize the model
    print("开始构建模型···")
    # model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    model = VGG16.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    # 加载已经存在的模型
    try:
        model.load_weights('saveModel/plant_sign.model')
        print("加载模型成功！继续训练模型")
    except:
        print("加载模型失败！开始训练一个新的模型")
    print("定义训练方式···")
    # 定义训练方式，三个参数，分别是loss：设置损失函数；optimizer：使用adam优化器收敛更快，metrics：设置评估模型的方式是准确率
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network，开始训练
    print("开始训练网络···")
    H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
                  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                  epochs=EPOCHS, verbose=1)
    # 输入训练数据集，划分方式是0.8+0.2 训练20个训练周期，每一个批次128项数据，verbose=2为显示训练过程
    predY = model.predict(testX)
    predY = np.argmax(predY, axis=1)
    # print(predY.shape)
    # print(test_label_list.shape)
    # 打印混淆矩阵
    matrix = pd.crosstab(test_label_list, predY, rownames=['label'], colnames=['predict'])
    print(matrix)
    # save the model to disk
    print("[INFO] serializing network...")
    model.save('saveModel/traffic_sign_result.model') # 保存模型
    # 画出准确率执行结果
    show_train_history(H)
    # prediction_probability = model.predict(True_Train_X) # 预测可能性
    # prediction = model.predict_classes(True_Train_X) # 直接预测分类结果

    ####测试模型

    name_dic = {'0': 'Black-grass', '1': 'Charlock', '2': 'Cleavers',
                '3': 'Common Chickweed', '4': 'Common wheat',
                '5': 'Fat Hen', '6': 'Loose Silky-bent', '7': 'Maize',
                '8': 'Scentless Mayweed', '9': 'Shepherds Purse',
                '10': 'Small-flowered Cranesbill', '11': 'Sugar beet'}
    path = "./test"
    test_list, test_name_list = get_file(path)
    # model = LeNet.build(width=32, height=32, depth=3, classes=12)
    # model = VGG16.build(width=32, height=32, depth=3, classes=12)
    # try:
    #     model.load_weights('saveModel/traffic_sign_result.model')
    #     print("加载模型成功！继续训练模型")
    # except:
    #     print("加载模型失败！开始训练一个新的模型")
    # 可视化模型
    # plot_model(model, to_file='model.png')
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    aug.flow(test_list)
    result = model.predict(test_list)
    result = np.argmax(result, axis=1)
    name_list = []
    for i in result:
        name_list.append(name_dic[str(i)])
    finally_result = pd.DataFrame({'file': test_name_list, 'species': name_list})
    print(finally_result)
    finally_result.to_csv("result.csv", index=False)
    # print(finally_result1)

def get_file(path):
    test_list = []
    test_name_list = []
    for file in os.listdir(path):
        image = cv2.imread(path + '/' + file)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        test_list.append(image)
        file_list_split = file.split(".")
        test_name_list.append(file)
    test_list = np.array(test_list, dtype="float") / 255.0
    return test_list, test_name_list

norm_size = 32


EPOCHS = 32  # 迭代次数
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 12
norm_size = 32
if __name__ == '__main__':
    train_file_path = "./train/"

    trainX, trainY, testX, testY, test_label_list = get_files(train_file_path)  # 导入数据集
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    train(aug, trainX, trainY, testX, testY, test_label_list)


