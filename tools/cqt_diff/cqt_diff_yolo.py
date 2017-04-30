import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

keras_dir = '../../example/tiny-yolo/keras/output/'
cqt_dir = '../../example/tiny-yolo/c/output/'


def layer0_comp():
    keras = np.load(keras_dir+'l00_2.npy')
    cqt = np.load(cqt_dir+'l00_2.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer1_comp():
    keras = np.load(keras_dir+'l01_15.npy')
    cqt = np.load(cqt_dir+'l01_15.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer2_comp():
    keras = np.load(keras_dir+'l02_15.npy')
    cqt = np.load(cqt_dir+'l02_15.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


layer2_comp()

print('finish')