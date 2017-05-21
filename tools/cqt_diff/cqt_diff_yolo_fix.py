import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

keras_dir = '../../example/tiny-yolo/keras/output/'
cqt_dir = '../../example/tiny-yolo/c_fix/output/'
qp_file = '../../example/tiny-yolo/c_fix/weight/'

fix16mode = True
layer_num = 32
oqs = []

def read_qpfile(odir):
    """qpファイルを読み込み、入力、出力、重みのＱ位置をリストにして返す"""
    iqs = []
    wqs = []
    oqs = []
    fname = os.path.join(odir, 'qp.txt')

    for i, l in enumerate(open(fname).readlines()):
        if i < 1:
            continue
        words = l.split(',')
        iqs.append(int(words[0]))
        oqs.append(int(words[1]))
        wqs.append(int(words[2]))

    return iqs, oqs, wqs


def calc_statistics(l, d=0):
    """
    :param l:
    :return: (平均の差、分散の差、max, min, データ辺りのずれ)
    """
    global oqs
    keras = np.load(keras_dir + 'l%02d_%d.npy' % (l , d))
    cqt = np.load(cqt_dir+'l%02d_%d.npy' % (l, d))

    q = oqs[l]

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
iqs, oqs, wqs = read_qpfile(qp_file)


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


