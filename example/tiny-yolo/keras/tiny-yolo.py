from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K
import numpy as np
from PIL import Image
import math

img_file = '../img/person.jpg'

width = 416
height = 416

r_w = 13
r_h = 13
r_n = 5

region_biases = (1.080000, 1.190000, 3.420000, 4.410000, 6.630000, 11.380000, 9.420000, 5.110000, 16.620001, 10.520000)


class box():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0

def tiny_yolo_model():
    model = Sequential()
    model.add(Conv2D(16, input_shape=(width, height, 3), use_bias=False, data_format="channels_last",
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(32, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(64, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(128, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(256, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(125, use_bias=True, data_format="channels_last", padding='same', kernel_size=(1, 1), strides=(1, 1)))

    return model


def get_region_box(x, biases, n, index, i, j, w, h):
    b = box()
    b.x = (i + logistic_activate(x[index + 0])) / w
    b.y = (j + logistic_activate(x[index + 1])) / h
    b.w = math.exp(x[index + 2]) * biases[2*n]   / w
    b.h = math.exp(x[index + 3]) * biases[2*n+1] / h
    return b


def logistic_activate(x):
    return 1./(1. + math.exp(-x))

def get_region_boxes(predications):
    classes = 20

    for raw in range(r_h):  # 13
        for col in range(r_w):  # 13
            i = raw * r_w + col
            for n in range(r_n):  # 5
                index = i * r_n + n
                # p_index = index * (classes + 5) + 4
                p_index = classes * n + 4 # 4, 29, 54, 79, 104
                scale = predications[raw][col][p_index]
                box_index = index * (classes * 5)

                class_index = index * (classes * 5) + 5




# create tiny-yolo model
tiny_yolo_model = tiny_yolo_model()
tiny_yolo_model.load_weights('weight/tiny-yolo.h5')

with open('tiny-yolo.json', 'w') as fp:
    json_string = tiny_yolo_model.to_json()
    fp.write(json_string)

tiny_yolo_model.summary()


# run yolo
image = Image.open(img_file)
resized_image = image.resize((width, height), Image.BICUBIC)
image_data = np.array(resized_image, dtype='float32') / 255.0
x = np.expand_dims(image_data, axis=0)

preds = tiny_yolo_model.predict(x)
boxes = [box()] * r_h * r_w * r_n
probs = [[[[0.0] * 13] * 13] * 125]


get_region_boxes(preds)

# nms_sort()

print(preds.shape)


