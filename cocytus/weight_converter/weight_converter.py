import h5py
import numpy as np
import sys
import itertools
import os
import math


class WeightConverter:
    def __init__(self, output_dir, h5file, dtype=""):
        self.output_dir = output_dir
        self.h5file = os.path.expanduser(h5file)
        self.dtype = dtype

    def convert(self):
        """
        outputdirに、変換したnumpyファイルを生成する。
        :return:
        """
        f = h5py.File(self.h5file, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        self.save_weights_from_hdf5_group(f)

        if hasattr(f, 'close'):
            f.close()

    def save_weights_from_hdf5_group(self, f):
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        print(layer_names)

        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                filtered_layer_names.append(name)

        layer_names = filtered_layer_names

        print(layer_names)

        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            print('')
            print(name)
            print(weight_names)
            for weight_name in weight_names:
                data = g[weight_name].value
                print(data.shape)
                data2 = self.tf_reshape(data)
                print(data2.shape)
                print('')

                if self.dtype == '':
                    # Kerasの重みをそのまま使う
                    filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                    filepath = os.path.join(self.output_dir, filename)
                    np.save(filepath, data2, allow_pickle=False)
                    print("save %s to %s" % (weight_name, filepath))
                elif self.dtype == 'fix8':
                    if weight_name.find('conv2d') == 0:
                        zerop = 5
                        print("convert conv2d zeropos = %d", zerop)
                        # 重みの型を変換する。
                        filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                        filepath = os.path.join(self.output_dir, filename)

                        #int8の範囲にクリップする。
                        int_bit = 8 - zerop
                        int_min = -(2 ** (int_bit - 1))
                        int_max = (2 ** (int_bit - 1)) - 1

                        cliped = data2.clip(int_min, int_max) * (2 ** zerop) - (1.0 / (2**zerop))
                        fix8_data = cliped.astype(np.int8)
                        np.save(filepath, fix8_data, allow_pickle=False)
                        print("save %s to %s(fix8)" % (weight_name, filepath))
                    elif weight_name.find('batch_') == 0:
                        zerop = 4

                        filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                        filepath = os.path.join(self.output_dir, filename)

                        if weight_name.find('moving_variance')!=-1:
                            # BatchNormalization のvarianceの重みは、重みデータではなく、 1.0 / sqrt(var + epsilon)の値を保存する。
                            zerop, data2 = self.calc_bach_invvar(data2)

                        int_bit = 8 - zerop
                        int_min = -(2 ** (int_bit - 1))
                        int_max = (2 ** (int_bit - 1)) - 1

                        cliped = data2.clip(int_min, int_max) * (2 ** zerop) - (1.0 / (2 ** zerop))
                        fix8_data = cliped.astype(np.int8)
                        np.save(filepath, fix8_data, allow_pickle=False)
                        print("save %s to %s(fix8) min = %f, max = %f" % (weight_name, filepath, data2.min(), data2.max()))
                else:
                    print("ERROR unkown weight dtype = %s" % self.dtype)

    def calc_bach_invvar(self, data):
        epsilon = 0.001
        outdata = 1.0 / np.sqrt(data + epsilon)
        print("calc_var min = %f, max = %f" % ( outdata.min(), outdata.max()))
        return 4, outdata


    def tf_reshape(self, data):
        """
        matrixの並びを変える.
        [3][3][1][32]->[32][1][3][3]
        """
        ori_shape = data.shape

        if len(ori_shape) == 1:
            return data

        new_shape = list(ori_shape)
        new_shape.reverse()
        new_data = np.zeros(new_shape, dtype=data.dtype)

        i_r = range(new_shape[0])
        j_r = range(new_shape[1])

        if len(ori_shape) == 2:
            for i,j in itertools.product(i_r, j_r):
                new_data[i][j] = data[j][i]
        elif len(ori_shape) == 3:
            for i, j, k in itertools.product(i_r, j_r, range(new_shape[2])):
                new_data[i][j][k] = data[k][j][i]
        elif len(ori_shape) == 4:
            # [32][1][3][3]:
            for i, j, k, l in itertools.product(i_r, j_r, range(new_shape[2]), range(new_shape[3])):
                new_data[i][j][k][l] = data[l][k][j][i]
        else:
            print("error")
            sys.exit(1)

        return new_data
