# This program was ported from the YAD2K project.
# https://github.com/allanzelener/YAD2K
#
# copyright
# https://github.com/allanzelener/YAD2K/blob/master/LICENSE
#
import numpy as np

region_biases = (1.080000, 1.190000, 3.420000, 4.410000, 6.630000, 11.380000, 9.420000, 5.110000, 16.620001, 10.520000)
voc_anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])


voc_label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']



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
    ], axis=4)
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

    dim = boxes.shape

    boxes_f = []
    scores_f = []
    classes_f = []

    for r in range(dim[1]):
        for c in range(dim[2]):
            for n in range(dim[3]):
                if prediction_mask[0][r][c][n]:
                    pos = boxes[0][r][c][n]
                    boxes_f.append(pos)
                    scores_f.append(box_class_scores[0][r][c][n])
                    classes_f.append(box_classes[0][r][c][n])

    return boxes_f, scores_f, classes_f

def non_max_surpression(boxes, scores, tresh):
    """    img = Image.open(img_file)

    see http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes:
    :param scores:
    :param tresh:
    :return:
    """
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > tresh)[0])))

    return pick


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
    height = image_shape[1]
    width = image_shape[0]
    image_dims = np.stack([height, width, height, width])
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    nms_index = non_max_surpression(boxes, scores, iou_threshold)

    boxes_last = []
    scores_last = []
    classes_last = []
    for i in nms_index:
        boxes_last.append(boxes[i])
        scores_last.append(scores[i])
        classes_last.append(classes[i])

    return boxes_last, scores_last, classes_last


# デバッグ用エントリー
if __name__ == '__main__':
    classes = 20
    preds = np.load('output/preds.npy')

    thresh = 0.3
    size = (620, 424)

    out_boxes, out_scores, out_classes = yolo_eval(preds, size, score_threshold = thresh, iou_threshold = 0.5, classes = classes)

    for i in range(len(out_classes)):
        cls = out_classes[i]
        score = out_scores[i]
        box = out_boxes[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(size[0], np.floor(right + 0.5).astype('int32'))
        print(voc_label[cls], score, (left, top), (right, bottom))

