import sys
import math
from functools import cmp_to_key
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from PIL import Image

img_file = '../img/person.jpg'

width = 416
height = 416

r_w = 13
r_h = 13
r_n = 5
classes = 20

region_biases = (1.080000, 1.190000, 3.420000, 4.410000, 6.630000, 11.380000, 9.420000, 5.110000, 16.620001, 10.520000)
voc_anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

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
#    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
 #   model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

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
        d0 = layer_output[:, :, 0]
        d1 = layer_output[:, :, 1]
        d2 = layer_output[:, :, 2]
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
    else:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, last_dim]
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_%d.npy' % (l, postfix, last_dim), d2, allow_pickle=False)



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
    row = index // (r_h * 125)
    col = (index // 125) % r_w

#    b.x = (i + logistic_activate(x[0][row][col][index + 0])) / w
#    b.y = (j + logistic_activate(x[0][row][col][index + 1])) / h
#    b.w = math.exp(x[0][row][col][index + 2]) * biases[2*n] / w
#    b.h = math.exp(x[0][row][col][index + 3]) * biases[2*n+1] / h
    b.x = (i + logistic_activate(x[0][row][col][0])) / w
    b.y = (j + logistic_activate(x[0][row][col][1])) / h
    b.w = math.exp(x[0][row][col][2]) * biases[2*n] / w
    b.h = math.exp(x[0][row][col][3]) * biases[2*n+1] / h

    return b


def logistic_activate(x):
    return 1./(1. + math.exp(-x))


def get_region_boxes(predictions):
    for i in range(r_w*r_h):
        row = i // r_w
        col = i % r_w
        for n in range(r_n):
            index = i*r_n + n
            #p_index = index * (classes + 5) + 4
            p_index = (classes + 5) + 4
            scale = predictions[0][row][col][p_index]
            box_index = index * (classes + 5)

            boxes[index] = get_region_box(predictions, region_biases, n, box_index, col, row, r_w, r_h)
            boxes[index].x *= r_w
            boxes[index].y *= r_h
            boxes[index].w *= r_w
            boxes[index].h *= r_h

            #class_index = index * (classes * 5) + 5
            class_index = (classes * 5) + 5

            for j in range(classes):
                prob = scale * predictions[0][row][col][class_index + j]
                if prob > thresh:
                    probs[index][j] = prob
                else:
                    probs[index][j] = 0


def nms_comparator(a, b):

    diff = a.probs[a.index][b.cls] - b.probs[b.index][b.cls]
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
            if probs[s[i].index][classes] == 0:
                continue
            a = boxes[s[i].index]
            for j in range(i+1, total):
                b = boxes[s[j].index]
                if box_iou(a, b) > thresh:
                    probs[s[j].index][k] = 0


def max_index(a, n):
    if n <= 0:
        return -1
    max_i = 0
    max = a[0]
    for i in range(1, n):
        if a[i] > max:
            max = a[i]
            max_i = i
    return max_i


def draw_detections(im, num, thresh, boxes, probs, classes):
    for i in range(num):
        cls = max_index(probs[i], classes)
        prob = probs[i][cls]
        if prob > thresh:
            width_i = int(width * 0.12)
            b = boxes[i]

            left = (b.x - b.w / 2.) * width
            right = (b.x + b.w / 2.) * width
            top = (b.y - b.h / 2.) * height
            bot = (b.y + b.h / 2.) * height

            if left < 0:
                left = 0
            if right >= width:
                right = width - 1
            if top < 0:
                top = 0
            if bot >= height:
                bot = height - 1

            print("%d %f %d, %d -> %d %d" %
                  (cls, prob*100, int(left), int(top), int(right), int(bot)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # shape (1, 13 , 13, 5, 20)
    dim = x.shape
    arr = np.copy(x)
    for raw in range(dim[1]):
        for col in range(dim[2]):
            for cls in range(dim[3]):
                a = x[0][raw][col][cls]
                e_x = np.exp(a - np.max(a))
                arr[0][raw][col][cls] = e_x / e_x.sum()
    return arr



def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = feats[0].shape  # assuming channels last

    # In YOLO the height index is the inner most iteration.
    conv_height_index = np.arange(0, stop=conv_dims[0])
    conv_width_index = np.arange(0, stop=conv_dims[1])
    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = np.tile(
        np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = np.transpose(conv_width_index).flatten()
    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    conv_index = np.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = conv_index.astype(np.float)

    feats = np.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])

    # natu atode
    #conv_dims = np.reshape(conv_dims, [1, 1, 1, 1, 2]).astype(np.float)
    conv_dims = np.reshape([13.0, 13.0], [1, 1, 1, 1, 2]).astype(np.float)

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = sigmoid(feats[..., :2])
    box_wh = np.exp(feats[..., 2:4])
    box_confidence = sigmoid(feats[..., 4:5])

    pb_deb = feats[..., 5:]

    box_class_probs = softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims


    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    ret =  np.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis = 4)
    return ret


def boolean_mask(xs, masks):
    ret = []
    for i in range(xs.shape[0]):
        xf = xs[i].flatten()
        mf = masks.flatten()
        for x, mask in zip(xf, mf):
            if mask:
                ret.append(x)
    return ret

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""

    box_scores = box_confidence * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    np.save('keras.npy', prediction_mask)

    # TODO: Expose tf.boolean_mask to Keras backend?
    #boxes = boolean_mask(boxes, prediction_mask)
    #scores = boolean_mask(box_class_scores, prediction_mask)
    #classes = boolean_mask(box_classes, prediction_mask)
    dim = boxes.shape

    boxes_f = []
    scores_f = []
    classes_f = []

    for r in range(dim[1]):
        for c in range(dim[2]):
            for n in range(dim[3]):
                if prediction_mask[0][r][c][n]:
                    # natu atode
                    #pos = [boxes[0][r][c][n][0], boxes[1][r][c][n][0],boxes[2][r][c][n][0], boxes[3][r][c][n][0]]
                    pos = boxes[0][r][c][n]
                    boxes_f.append(pos)
                    scores_f.append(box_class_scores[0][r][c][n])
                    classes_f.append(box_classes[0][r][c][n])

    return boxes_f, scores_f, classes_f


def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5,
              classes=20):
    """Evaluate YOLO model on given input batch and return filtered boxes."""


    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(yolo_outputs, voc_anchors, classes)
    boxes_t = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes_t, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = np.stack([height, width, height, width])
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    """
    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    """

    return boxes, scores, classes




use_ya2k = False
if use_ya2k:
    file_post_fix = '_ya2k'
else:
    file_post_fix =''

if use_ya2k:
    json_string = open('/home/natsutani/proj/YAD2K/model_data/tiny-yolo.json', 'r').read()
    tiny_yolo_model = model_from_json(json_string)
else:
    tiny_yolo_model = tiny_yolo_model()


#tiny_yolo_model.load_weights('weight/tiny-yolo.h5')
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
boxes = [box()] * r_h * r_w * r_n
probs = np.zeros((r_h * r_w * r_n, classes+1), dtype=np.float)
thresh = 0.3

np.save('output/preds%s.npy' % file_post_fix, preds)

out_boxes, out_scores, out_classes = yolo_eval(preds, (width, height), score_threshold = thresh, iou_threshold = 0.5, classes = classes)


for i in range(len(out_classes)):
    cls = out_classes[i]
    score = out_scores[i]
    box = out_boxes[i]
    print('%d %f %s' % (cls, score, str(box)))


sys.exit(1)
debug_layer = 28
#layer_dump(tiny_yolo_model, x, debug_layer, file_post_fix)


nms = 0.4
get_region_boxes(preds)
do_nms_sort(r_w*r_h*r_n, classes, nms)

draw_detections(image_data, r_w*r_h*r_n, thresh, boxes, probs, classes)

# nms_sort()

print(preds.shape)


