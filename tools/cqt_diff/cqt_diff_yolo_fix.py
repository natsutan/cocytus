import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

keras_dir = '../../example/tiny-yolo/keras/output/'
cqt_dir = '../../example/tiny-yolo/c_fix/output/'

fix16mode = True
q = 9
layer_num = 32


def calc_statistics(l, d=0):
    """
    :param l:
    :return: (平均の差、分散の差、max, min, データ辺りのずれ)
    """
    keras = np.load(keras_dir + 'l%02d_%d.npy' % (l , d))
    cqt = np.load(cqt_dir+'l%02d_%d.npy' % (l, d))


    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    ave_diff = k_f.mean() - c_f.mean()
    max_diff = k_f.max() - c_f.max()
    min_diff = k_f.min() - c_f.min()
    var_diff = k_f.var() - c_f.var()

    return ave_diff, max_diff, min_diff, var_diff

aves = []
maxs = []
mins = []
vars = []

for l in range(layer_num):
    print(l)
    ave, max, min, var = calc_statistics(l, 1)
    aves.append(ave)
    maxs.append(max)
    mins.append(min)
    vars.append(vars)

x = np.arange(len(aves))


plt.plot(x, aves, color='b', label='average')
plt.plot(x, maxs, color='g', label='max')
plt.plot(x, mins, color='y', label='min')
plt.plot(x, mins, color='pink', label='variance')
plt.legend()

plt.show()

print('finish')


