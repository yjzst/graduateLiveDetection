from keras.models import  Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
import keras.backend as k

class LivenessNet:
    def __init__(self, width, height, depth, classes):

        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes


    def build(self):
        chanDim = -1
        input_shape = (self.height, self.width, self.depth)
        if k.image_data_format() == "channels_first":
            input_shape = (self.depth, self.height, self.width)
            chanDim = 1

        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same",input_shape=input_shape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3,3),padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3,3),padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3,3),padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        return model
if __name__ == '__main__':
    model = LivenessNet(112,112,3,2).build()
    model.summary()