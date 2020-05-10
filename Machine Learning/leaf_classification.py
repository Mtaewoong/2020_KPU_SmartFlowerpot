# from google.colab import drive
# drive.mount('/content/drive')
import tensorflow as tf
tf.__version__

import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger


ScaleTo = 70
seed = 7

path = '/content/drive/My Drive/ML   DL/PlantClassification/use/*/*.jpg'
#path = 'C:/Users/moont/Downloads/leafsnap-dataset/dataset/images/use/*/*.jpg'

files = glob(path)

trainImg = []
trainLabel = []
j = 1
num = len(files)


for img in files:
    print(str(j) + "/" + str(num),end="\r")
    trainImg.append(cv2.resize(cv2.imread(img), (ScaleTo, ScaleTo)))
    trainLabel.append(img.split("/")[-2])
    j += 1

trainImg = np.asarray(trainImg)
trainLabel = pd.DataFrame(trainLabel)

trainLabel.shape

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(trainImg[i])

plt.show()

clearTrainImg = []
examples = []
getEx = True
for img in trainImg:

    blurImg = cv2.GaussianBlur(img, (5, 5), 0)

    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    # https://ko.wikipedia.org/wiki/HSV_%EC%83%89_%EA%B3%B5%EA%B0%84

    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # https://hoony-gunputer.tistory.com/entry/opencv-pythonErosion%EA%B3%BC-Dilation
    # https://swprog.tistory.com/entry/Mathematical-morphology-%EB%AA%A8%ED%8F%B4%EB%A1%9C%EC%A7%80-%EC%97%B0%EC%82%B0

    bMask = mask > 0 # 배열을 불린언 형으로

    clear = np.zeros_like(img, np.uint8)
    clear[bMask] = img[bMask]

    clearTrainImg.append(clear)

    '''
    if getEx:
        plt.subplot(2, 3, 1);
        plt.imshow(img)
        plt.subplot(2, 3, 2);
        plt.imshow(blurImg)
        plt.subplot(2, 3, 3);
        plt.imshow(hsvImg)
        plt.subplot(2, 3, 4);
        plt.imshow(mask)
        plt.subplot(2, 3, 5);
        plt.imshow(bMask)
        plt.subplot(2, 3, 6);
        plt.imshow(clear)
        getEx = False
    '''

clearTrainImg = np.asarray(clearTrainImg) #실수형으로 변환됨

plt.show()

clearTrainImg = clearTrainImg / 255

le = preprocessing.LabelEncoder()
# https://medium.com/@john_analyst/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%A0%88%EC%9D%B4%EB%B8%94-%EC%9D%B8%EC%BD%94%EB%94%A9%EA%B3%BC-%EC%9B%90%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9-f0220df21df1
le.fit(trainLabel[0])
print("Classes: " + str(le.classes_))
encodeTrainLabels = le.transform(trainLabel[0])

clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)
# https://nabzacko.tistory.com/7  ex)1 => [1,0,0] 2=> [0,1,0] 3=>[0,0,1]
num_clases = clearTrainLabel.shape[1] # [1~19] [1~19] ~ [1~19] => 705개
print("Number of classes: " + str(num_clases))

trainX, testX, trainY, testY = train_test_split(clearTrainImg, clearTrainLabel,
                                                test_size=0.1, random_state=seed,
                                                stratify = clearTrainLabel)
# https://teddylee777.github.io/scikit-learn/train-test-split

trainX.shape

datagen = ImageDataGenerator(
        rotation_range=180,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )
datagen.fit(trainX)

np.random.seed(seed)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(ScaleTo, ScaleTo, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_clases, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#활성함수: relu  - 오버피팅 방지: dropout

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.4,
                                            min_lr=0.00001)

filepath="/content/drive/My Drive/ML   DL/PlantClassification/weights.best_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
filepath="/content/drive/My Drive/ML   DL/PlantClassification/weights.last_auto4.hdf5"
checkpoint_all = ModelCheckpoint(filepath, monitor='val_acc',
                                 verbose=1, save_best_only=False, mode='max')

callbacks_list = [checkpoint, learning_rate_reduction, checkpoint_all]

hist = model.fit_generator(datagen.flow(trainX, trainY, batch_size=75),
                            epochs=35, validation_data=(testX, testY),
                            steps_per_epoch=trainX.shape[0], callbacks=callbacks_list)

model.load_weights("/content/drive/My Drive/ML   DL/PlantClassification/weights.last_auto4.hdf5")

image=cv2.imread("/content/drive/My Drive/ML   DL/PlantClassification/test/test1.jpg")

plt.imshow(image)
plt.show()

image=cv2.resize(image, (ScaleTo, ScaleTo))

image.shape

image=image.reshape(-1,70,70,3)

result=model.predict(image)

print(le.classes_[np.argmax(result)])
