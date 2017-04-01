import os
import numpy as np
import matplotlib.pyplot as plt

keras_dir = '../../example/vgg16/keras/output/'
cqt_dir = '../../example/vgg16/c/output/'

def layer1_comp():
    for i in [0,1,63]:
        keras_path = os.path.join(keras_dir, "dog_l01_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l01_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()

def layer2_comp():
    for i in [0,1,63]:
        keras_path = os.path.join(keras_dir, "dog_l02_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l02_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()


def layer3_comp():
    for i in [0,1,63]:
        keras_path = os.path.join(keras_dir, "dog_l03_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l03_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()



layer3_comp()
