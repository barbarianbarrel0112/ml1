import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import  Image
import  xgboost
from sklearn.ensemble import AdaBoostClassifier
# from main import segment_plant,load_data
from sklearn.tree import  DecisionTreeClassifier


def get_category_dict(folder_path):
    category_dict = {}
    # 遍历文件夹中的所有文件
    for category, _, files in os.walk(folder_path):

        category = category.split(os.sep)[-1]
        if category == "":
            continue
        if category == "test":
            images = [os.path.join(folder_path, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        # 提取类别并创建图像列表
        else:
            images = [os.path.join(folder_path,category, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        # 将类别和图像列表添加到字典中
        category_dict[category] = images
    return category_dict


def load_data():
    train_data=get_category_dict('C:\\Users\\YLH04\\Documents\\Tencent Files\\2256501131\\FileRecv\\plant\\Nonsegmented_pack - k\\train\\')
    test_data=get_category_dict('C:\\Users\\YLH04\\Documents\\Tencent Files\\2256501131\\FileRecv\\plant\\Nonsegmented_pack - k\\test')
    image_type=len(train_data)
    return train_data,test_data,image_type

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def get_HOG_feature():
    def get_hog_feature(ima):
        img=cv2.imread(ima)
        img = segment_plant(img)
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature=hog.compute(img, winStride=(32,32), padding=(0,0)).flatten()
        return feature

    # 将数据dump保存  下次不需要重新读取
    if not os.path.exists('train_data.pkl'):
        train_data,test_data,image_type=load_data()
        pickle.dump(train_data,open("train_data.pkl",'wb'))
        pickle.dump(test_data,open("test_data.pkl",'wb'))
        pickle.dump(image_type,open("image_type.pkl",'wb'))
    else:
        train_data=pickle.load(open("train_data.pkl",'rb'))
        test_data=pickle.load(open("test_data.pkl",'rb'))
        image_type=pickle.load(open("image_type.pkl",'rb'))

    print(train_data.keys())
    print(len(test_data))
    print(image_type)
    #定义对象hog，同时输入定义的参数，剩下的默认即可
    winSize = (64,64)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor( winSize,blockSize,blockStride,cellSize, nbins )

    train_feature=[]
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_hog_feature(image).tolist()
            train_feature.append(feature)
    print("**************************")
    print(len(train_feature))


    # PCA降维
    pca = PCA(n_components=16)  # 自动选择特征个数  'mle'
    print("**************************")
    pca.fit(train_feature)
    print("**************************")
    train_feature = pca.transform(train_feature)



    test_feature=[]
    for image in test_data['test']:
            feature = get_hog_feature(image).tolist()
            test_feature.append(feature)
    print(len(test_feature))
    test_feature = np.array(test_feature)

    test_feature = pca.transform(test_feature)

    return train_feature,test_feature

def get_color_feature(img):
    r, g, b = cv2.split(img)
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    r_offset = (np.mean(np.abs((r - r_mean) ** 3))) ** (1. / 3)
    g_offset = (np.mean(np.abs((g - g_mean) ** 3))) ** (1. / 3)
    b_offset = (np.mean(np.abs((b - b_mean) ** 3))) ** (1. / 3)

    # # img= cv2.medianBlur(img, 3)  # 中值滤波
    # one = np.cumsum(cv2.calcHist([img], [1], None, [256], [0, 255],accumulate=True))
    # # one = np.std(one).tolist()
    # one = (one/(img.shape[0]*img.shape[1])).tolist()
    one = np.log1p([r_mean, g_mean, b_mean, r_std, g_std, b_std, r_offset, g_offset, b_offset]).tolist()
    return one
def get_image_paths(data_path, categories):
    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []
    print("获得训练集数据路径")
    for category in categories:
        # image_paths = glob(os.path.join(data_path, 'train', category, '*.png'))
        image_paths = glob(os.path.join(data_path, 'Nonsegmented_pack - k\\train', category, '*.png'))
        # path = os.path.join(data_path, 'train', category)
        path = os.path.join(data_path, 'Nonsegmented_pack - k\\train', category)
        files = os.listdir(path)

        for i in range(len(files)):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)
        print(path, '读取完成！', '->', len(files))
    

    
    print("获得测试集数据路径")
    image_paths = glob(os.path.join(data_path, 'Nonsegmented_pack - k\\test', '*.png'))
    path = os.path.join(data_path, 'Nonsegmented_pack - k\\test')
    files = os.listdir(path)
    for i in range(len(files)):
        test_image_paths.append(image_paths[i])
    print(path, '读取完成！', '->', len(files))
    return train_image_paths, test_image_paths, train_labels


def build_vocabulary(image_paths, k, length):
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
    sift = cv2.SIFT_create()
    for i in range(length):
        for j in [0, 248, 610, 878,1449 , 1652, 2083, 2696, 2902, 3338, 3608, 4069]:
            img = cv2.imread(image_paths[j + i])
            gray = segment_plant(img)
            kp = sift.detect(gray, None)
            if len(kp) != 0:
                bow_kmeans_trainer.add(sift.compute(gray, sift.detect(gray, None))[1])
    vocab = bow_kmeans_trainer.cluster()
    return vocab


def get_train_feat(image_paths, vocab, k):
    flann_params = dict(algorithm=1, tree=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    sift = cv2.SIFT_create()
    bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow_img_descriptor_extractor.setVocabulary(vocab)
    train=[]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = segment_plant(img)
        one = bow_img_descriptor_extractor.compute(gray, sift.detect(gray, None))
        if one is None:
            one = np.array([[0 for i in range(k)]])
            
        one=one.tolist()[0]
        two = get_color_feature(gray)
        train.append(one+two)

    return train


if __name__ == '__main__':
    data_path = "."
    categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                  'Loose Silky-bent',
                  'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    CATE2ID = {v: k for k, v in enumerate(categories)}
    train_image_paths, test_image_paths, train_labels = get_image_paths(data_path, categories)
    trian_image_labels_id = [CATE2ID[x] for x in train_labels]
    print("开始建立词汇表===========================>")
    if not os.path.exists('vocab.pkl'):
        vacob = build_vocabulary(train_image_paths, 100, 200)
        pickle.dump(vacob, open('vocab.pkl', 'wb'))
    else:
        vacob = pickle.load(open('vocab.pkl', 'rb'))
    print(vacob.shape)
    print("词汇表建立完成===========================>")




    # 打乱
    # train_image_paths, trian_image_labels_id= shuffle(train_image_paths, trian_image_labels_id, random_state=0)
    # HOG特征
    HOG_train,HOG_test=get_HOG_feature()


    print("开始提取训练集特征===========================>")
    if not os.path.exists('train_sift+color.pkl'):
        train_X = get_train_feat(train_image_paths, vacob,100)
        pickle.dump(train_X, open('train_sift+color.pkl', 'wb'))
    else:
        train_X = pickle.load(open('train_sift+color.pkl', 'rb'))
    # train_X = get_train_feat(train_image_paths, vacob, 50)
    # print(train_X)
    # exit(0)
    train = []
    # print()
    for i in range(len(train_X)):
        # print(type(train_X[i]),len(train_X[i]))
        # print(type(HOG_train[i]),len(HOG_train[i]))
        train.append(train_X[i]+HOG_train[i].tolist())
    # train = np.array(train)
    # print(train)
    print("提取完成===========================>")
    train_y = trian_image_labels_id
    # train_y = np.array(train_y)
    print("开始提取测试集特征===========================>")
    if not os.path.exists('test_sift+color.pkl'):
        test_X = get_train_feat(test_image_paths, vacob, 100)
        pickle.dump(test_X, open('test_sift+color.pkl', 'wb'))
    else:
        test_X = pickle.load(open('test_sift+color.pkl', 'rb'))
    #test_X = get_train_feat(test_image_paths, vacob, 50)
    test = []
    for i in range(len(test_X)):
        test.append(test_X[i]+HOG_test[i].tolist())
    # test = np.array(test)
    print("提取完成===========================>")
    print("开始训练===========================>")


    # 打乱顺序
    train, train_y = shuffle(train, train_y)
    # 划分
    train, x_test, train_y, y_test = train_test_split(train,train_y,test_size=0.2)

    # linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    # model = svm.SVC(kernel='linear', probability=True, gamma='auto', C=49).fit(train, train_y)
    model = svm.SVC(kernel='rbf', probability=True, gamma='auto', C=700).fit(train, train_y)
    # model1 = DecisionTreeClassifier(max_depth=8,random_state=0)
    # model=AdaBoostClassifier(baseestimator=model1,random_state=0,n_estimators=50).fit(train,train_y)
    print("训练结束")
    print("训练集准确率：", model.score(train, train_y))
    print("验证集准确率：", model.score(x_test, y_test))
    preds = model.predict(test)
    # print(preds)
    test = []
    for i in range(preds.shape[0]):
        test.append(categories[preds[i]])
    print(len(test))
    # sample = pd.read_csv("sample_submission.csv")
    # submission = pd.DataFrame({'file': sample['file'], 'species': test})
    #
    test_filenames = os.listdir("C:\\Users\\YLH04\\Documents\\Tencent Files\\2256501131\\FileRecv\\plant\\Nonsegmented_pack - k\\test")
    submission = pd.DataFrame({'ID':test_filenames,'Category':test})
    submission.to_csv('sift+color+HOG+SVM.csv', index=False)