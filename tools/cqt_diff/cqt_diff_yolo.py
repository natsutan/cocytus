import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

keras_dir = '../../example/tiny-yolo/keras/output/'
cqt_dir = '../../example/tiny-yolo/c_fix/output/'

fix16mode = True
q = 9

def layer0_comp():
    keras = np.load(keras_dir+'l00_2.npy')
    cqt = np.load(cqt_dir+'l00_2.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer1_comp():
    keras = np.load(keras_dir+'l01_4.npy')
    cqt = np.load(cqt_dir+'l01_4.npy')

    c_f = cqt.flatten()[:1024]
    k_f = keras.flatten()[:1024]

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer2_comp():
    keras = np.load(keras_dir+'l02_4.npy')
    cqt = np.load(cqt_dir+'l02_4.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer3_comp():
    keras = np.load(keras_dir+'l03_1.npy')
    cqt = np.load(cqt_dir+'l03_1.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    x = np.arange(len(k_f))

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer4_comp():
    keras = np.load(keras_dir+'l04_1.npy')
    cqt = np.load(cqt_dir+'l04_1.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    x = np.arange(len(k_f))

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer5_comp():
    keras = np.load(keras_dir+'l05_0.npy')
    cqt = np.load(cqt_dir+'l05_0.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    x = np.arange(len(k_f))

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer10_comp():
    keras = np.load(keras_dir+'l10_1.npy')
    cqt = np.load(cqt_dir+'l10_1.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    x = np.arange(len(k_f))

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer15_comp():
    keras = np.load(keras_dir+'l15_0.npy')
    cqt = np.load(cqt_dir+'l15_0.npy')

    c_f = cqt.flatten()[:512]
    k_f = keras.flatten()[:512]

    x = np.arange(len(k_f))

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer23_comp():
    keras = np.load(keras_dir+'l23_0.npy')
    cqt = np.load(cqt_dir+'l23_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer24_comp():
    keras = np.load(keras_dir+'l24_0.npy')
    cqt = np.load(cqt_dir+'l24_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)


    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()

def layer25_comp():
    keras = np.load(keras_dir+'l25_0.npy')
    cqt = np.load(cqt_dir+'l25_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)


    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer26_comp():
    keras = np.load(keras_dir+'l26_0.npy')
    cqt = np.load(cqt_dir+'l26_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)


    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()




def layer30_comp():
    keras = np.load(keras_dir+'l30_0.npy')
    cqt = np.load(cqt_dir+'l30_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


def layer31_comp():
    keras = np.load(keras_dir+'l31_0.npy')
    cqt = np.load(cqt_dir+'l31_0.npy')

    c_f = cqt.flatten()
    k_f = keras.flatten()

    if fix16mode:
        c_f = c_f.astype(np.float32) / (2 ** q)

    x = np.arange(len(k_f))

    plt.plot(x, c_f, color='r')
    plt.plot(x, k_f, color='b')

    plt.show()


layer4_comp()

print('finish')