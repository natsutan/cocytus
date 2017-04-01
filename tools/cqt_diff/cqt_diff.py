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

def layer14_comp():
    for i in [0,1,511]:
        keras_path = os.path.join(keras_dir, "dog_l14_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l14_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()


def layer15_comp():
    for i in [0,1,511]:
        keras_path = os.path.join(keras_dir, "dog_l15_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l15_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()

def layer17_comp():
    for i in [0,1,511]:
        keras_path = os.path.join(keras_dir, "dog_l17_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l17_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()

def layer19_comp():
    for i in [0,]:
        keras_path = os.path.join(keras_dir, "dog_l19_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l19_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        plt.show()

def layer21_comp():
    for i in [0,]:
        keras_path = os.path.join(keras_dir, "dog_l21_%d.npy" % i)
        cqt_path = os.path.join(cqt_dir, "dog_l21_%d.npy" % i)

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        weiths = np.load('/home/natu/proj/myproj/cocytus/example/vgg16/c/weight/predictions_W_1_z.npy')


        plt.show()

    diff = c_f - k_f
    plt.title('diff output')
    plt.plot(x, diff, color='b')
    plt.show()




def pred_comp():
    for i in [0,]:
        keras_path = os.path.join(keras_dir, "pred.npy")
        cqt_path = os.path.join(cqt_dir, "pred.npy")

        keras_data = np.load(keras_path)
        cqt_data = np.load(cqt_path)

        k_f = keras_data.flatten()
        c_f = cqt_data.flatten()

        x = np.arange(len(k_f))

        plt.plot(x, k_f, color='b')
        plt.plot(x, c_f, color='r')

        print("keras0 %f" % k_f[0])
        print("cqt0 %f" % c_f[0])
        print("keras1 %f" % k_f[1])
        print("cqt1 %f" % c_f[1])

        plt.show()

pred_comp()
#layer21_comp()