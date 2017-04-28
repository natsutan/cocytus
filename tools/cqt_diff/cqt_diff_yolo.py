import os
import numpy as np
import matplotlib.pyplot as plt
import sys

cqt_file = '../../example/tiny-yolo/keras/output/preds.npy'
keras_file = '/home/natsutani/proj/cocytus/example/tiny-yolo/keras/keras.npy'
yad2k_file = '/home/natsutani/proj/YAD2K/yad2k.npy'

yad2k = np.load(yad2k_file)
keras = np.load(keras_file)
dim = yad2k.shape

for r in range(dim[1]):
    for c in range(dim[2]):
        for n in range(dim[3]):
            y = yad2k[0][r][c][n]
            k = keras[0][r][c][n]
            if y != k:
                print(y, k, '0', r, c, n)



yad2k_t = np.where(yad2k==True)
keras_t = np.where(keras==True)

print(yad2k_t)
print(keras_t)





sys.exit(1)

k_f = cqt.flatten()
y_f = yolo.flatten()

x = np.arange(len(k_f))

plt.plot(x, k_f, color='r')
plt.plot(x, y_f, color='b')

plt.show()

print('finish')