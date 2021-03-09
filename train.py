from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import pickle
import os
import cv2
from model.livenessNet import LivenessNet

image_paths = list(paths.list_images("data/face-antispoof-data"))
images = []
labels = []
for path in image_paths:
    label = path.split(os.path.sep)[-2]
    image = cv2.imread(path)
    image = cv2.resize(image, (112, 112))
    images.append(image)
    labels.append(label)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
images = np.array(images, dtype="float") / 255.0

(trainX,testX,trainY,testY) = train_test_split(images, labels, test_size=0.25, random_state=42)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

opt = Adam(lr=1e-4)

model = LivenessNet(112, 112, 3, 2).build()
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

H = model.fit_generator(aug.flow(trainX,trainY,batch_size=64),validation_data=(testX,testY), steps_per_epoch=len(trainX)//64,
                        validation_steps=len(testX)//64,epochs=100)

model.save_weights("model/model.h5")
f = open("data/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

