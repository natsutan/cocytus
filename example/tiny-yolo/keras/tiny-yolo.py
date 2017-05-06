import sys
import math
from functools import cmp_to_key
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageDraw

from yad2k.yad2k_yolo import yolo_eval, voc_label

img_file = '../img/person.jpg'
#img_file = '../img/dog.jpg'

# 入力サイズ等はここを変更
width = 416
height = 416
r_w = 13
r_h = 13
r_n = 5
classes = 20


def tiny_yolo_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 3)))
    model.add(Conv2D(16, use_bias=False, data_format="channels_last",
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

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(125, use_bias=True, data_format="channels_last", padding='same', kernel_size=(1, 1), strides=(1, 1)))

    return model


def layer_dump(model, x, l, postfix=''):
    """
    :param model: Keras model
    :param x: input data
    :param l: layer number
    :param postfix: postfix for numpy file name
    :return: None
    """
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[l].output])

    layer_output = get_layer_output([x, 0])[0]
    last_dim = layer_output.shape[-1] - 1

    if l == 0:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, 2]
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
    else:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, last_dim]
        np.save('output/l%02d%s_all.npy' % (l, postfix), layer_output, allow_pickle=False)
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_%d.npy' % (l, postfix, last_dim), d2, allow_pickle=False)



file_post_fix = ''

# モデルの構築
tiny_yolo_model = tiny_yolo_model()
tiny_yolo_model.load_weights('weight/tyolo.h5')

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
probs = np.zeros((r_h * r_w * r_n, classes+1), dtype=np.float)
thresh = 0.3

layer_dump(tiny_yolo_model, x, 26)

np.save('output/preds%s.npy' % file_post_fix, preds)

out_boxes, out_scores, out_classes = yolo_eval(preds, image.size, score_threshold = thresh, iou_threshold = 0.5, classes = classes)

dr = ImageDraw.Draw(image)

for i in range(len(out_classes)):
    cls = out_classes[i]
    score = out_scores[i]
    box = out_boxes[i]

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(voc_label[cls], score, (left, top), (right, bottom))
    lt = (left, top)
    rt = (right, top)
    lb = (left, bottom)
    rb = (right, bottom)
    red = (255, 0, 0)
    dr.line((lt, rt), red, 2)
    dr.line((lt, lb), red, 2)
    dr.line((rt, rb), red, 2)
    dr.line((lb, rb), red, 2)

image.save(img_file+'out.png')


print("finish")

