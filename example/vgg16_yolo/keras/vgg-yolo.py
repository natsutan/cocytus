from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, InputLayer, Dropout

# 入力サイズ等はここを変更
width = 416
height = 416
r_w = 13
r_h = 13
r_n = 5
classes = 20


def vgg_yolo_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 3)))

    model.add(Conv2D(64, data_format="channels_last", activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(64, data_format="channels_last",
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(128, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(128, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(256, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(256, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Conv2D(125, use_bias=True, data_format="channels_last", activation='relu',
                     padding='same', kernel_size=(1, 1), strides=(1, 1)))

    return model

vgg_yolo_model = vgg_yolo_model()
vgg_yolo_model.summary()