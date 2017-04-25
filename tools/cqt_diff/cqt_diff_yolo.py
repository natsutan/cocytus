import numpy as np

cqt_file = '../../example/tiny-yolo/keras/output/preds.npy'
yolo_file = '/home/natsutani/hproj/tiny-yolo/darknet/output/predications.npy'


cqt = np.load(cqt_file)
yolo = np.load(yolo_file)

print('finish')