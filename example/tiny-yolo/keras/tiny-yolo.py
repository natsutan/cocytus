from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K
from functools import cmp_to_key
import numpy as np
from PIL import Image
import math

img_file = '../img/person.jpg'

width = 416
height = 416

r_w = 13
r_h = 13
r_n = 5
classes = 20

region_biases = (1.080000, 1.190000, 3.420000, 4.410000, 6.630000, 11.380000, 9.420000, 5.110000, 16.620001, 10.520000)


class box():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0


class sortable_bbox():
    def __init__(self):
        self.index = 0
        self.cls = 0
        self.probs = None

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


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    area = w*h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u


def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);



def get_region_box(x, biases, n, index, i, j, w, h):
    b = box()
    row = index // r_w
    col = index % r_w

    b.x = (i + logistic_activate(x[index + 0])) / w
    b.y = (j + logistic_activate(x[index + 1])) / h
    b.w = math.exp(x[index + 2]) * biases[2*n] / w
    b.h = math.exp(x[index + 3]) * biases[2*n+1] / h
    return b


def logistic_activate(x):
    return 1./(1. + math.exp(-x))


def get_region_boxes(predictions):
    for i in range(r_w*r_h):
        row = i // r_w
        col = i % r_w
        for n in range(r_n):
            index = i*r_n + n
            p_index = index * (classes + 5) + 4
            scale = predictions[0][row][col][p_index]
            box_index = index * (classes + 5)

            boxes[index] = get_region_box(predictions, region_biases, n, box_index, col, row, r_w, r_h)
            boxes[index].x *= r_w
            boxes[index].y *= r_h
            boxes[index].w *= r_w
            boxes[index].h *= r_h

            class_index = index * (classes * 5) + 5

            for j in range(classes):
                prob = scale * predictions[0][row][col][class_index + j]
                if prob > thresh:
                    probs[index][j] = prob
                else:
                    probs[index][j] = 0


def nms_comparator(a, b):
    ar, ac, an = div_rcn(a.index)
    br, bc, bn = div_rcn(b.index)

    diff = a.probs[ar][ac][an][b.cls] - b.probs[br][bc][bn][b.cls]
    if diff < 0:
        return 1
    elif diff > 0 :
        return -1
    return 0


def do_nms_sort(total, classes, thresh):
    s = [sortable_bbox()] * r_h * r_w * r_n
    for i, b in enumerate(s):
        b.index = i
        b.cls = 0
        b.probs = probs

    for k in range(classes):
        for i in range(total):
            s[i].cls = k
        s.sort(key=cmp_to_key(nms_comparator))

        for i in range(total):
            r, c, n  =div_rcn(s[i].index)
            if probs[r][c][n][k] == 0:
                continue
            a = boxes[s[i].index]
            for j in range(i+1, total):
                b = boxes[s[j].index]
                if box_iou(a, b) > thresh:
                    r2, c2, n2 = div_rcn(s[j].index)
                    probs[r2][c2][n2][k] = 0







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
probs = np.zeros((r_h * r_w * r_n * classes), dtype=np.float)
thresh = 0.6

print('probs ', probs.shape)

nms = 0.4
get_region_boxes(preds)
do_nms_sort(r_w*r_h*r_n, classes, nms)


# nms_sort()

print(preds.shape)


