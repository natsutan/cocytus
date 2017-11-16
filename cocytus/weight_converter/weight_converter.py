import h5py
import numpy as np
import sys
import itertools
import os
import re


class WeightConverter:
    def __init__(self, output_dir, h5file, dtype, compiler):
        self.output_dir = output_dir
        self.h5file = os.path.expanduser(h5file)
        self.dtype = dtype
        self.compiler = compiler
        # BN計算用
        self.epsilon = 0.001

    def convert(self):
        """
        outputdirに、変換したnumpyファイルを生成する。
        :return:
        """
        f = h5py.File(self.h5file, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        self.save_weights_from_hdf5_group(f)
        self.save_qpoint_file()

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

                    # Zynq使用時の重み圧縮
                    # file名 batch_normalization_3_moving_variance_z.npy'
                    str_last = '_moving_variance_z.npy'
                    if filename.rfind(str_last) == (len(str_last) - 1) and self.compiler.is_batch_normalization_optimize_enable():
                        self.compress_bn(filename)

                elif self.dtype == 'fix16':
                    # convolutionの重みかどうかの判定。
                    fixconv_en, fix_name = self.is_fixconv(weight_name)

                    if fixconv_en:
                        name = fix_name

                        cl = self.compiler.get_cqt_layer_obj(name)
                        fix_q = cl.weight_q

                        print("convert weight Q = %d" % fix_q)
                        # 重みの型を変換する。
                        filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                        filepath = os.path.join(self.output_dir, filename)

                        d2_max = data2.max()
                        d2_min = data2.min()
                        d2_abs_max = max(d2_max, -d2_min)
                        qpos = calc_qpos(d2_abs_max)
                        print("WQ = %d, (max = %f, min = %f" % (qpos, d2_max, d2_min))

                        #intの範囲にクリップする。
                        int_bit = 16 - fix_q
                        int_min = -(2 ** (int_bit - 1))
                        int_max = (2 ** (int_bit - 1)) - 1

                        # 小数点以下の補正
                        adj = (1.0 / (2 ** fix_q)) / 2.0
                        data2 = data2 + adj

                        cliped = data2.clip(int_min, int_max) * (2 ** fix_q) - (1.0 / (2**fix_q))
                        fix16_data = cliped.astype(np.int16)
                        np.save(filepath, fix16_data, allow_pickle=False)
                        print("save %s to %s(fix16)" % (weight_name, filepath))
                    elif weight_name.find('batch_') == 0:
                        name = weight_name.split('/')[0]

                        cl = self.compiler.get_cqt_layer_obj(name)
                        fix_q = cl.weight_q

                        filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                        filepath = os.path.join(self.output_dir, filename)

                        if weight_name.find('moving_variance')!=-1:
                            # BatchNormalization のvarianceの重みは、重みデータではなく、 1.0 / sqrt(var + epsilon)の値を保存する。
                            data2 = self.calc_bach_invvar(data2)

                        int_bit = 16 - fix_q
                        int_min = -(2 ** (int_bit - 1))
                        int_max = (2 ** (int_bit - 1)) - 1

                        # 小数点以下の補正
                        adj = (1.0 / (2 ** fix_q)) / 2.0
                        data2 = data2 + adj

                        d2_max = data2.max()
                        d2_min = data2.min()
                        d2_abs_max = max(d2_max, -d2_min)
                        qpos = calc_qpos(d2_abs_max)
                        print("WQ = %d, (max = %f, min = %f" % (qpos, d2_max, d2_min))

                        #cliped = data2.clip(int_min, int_max) * (2 ** zerop) - (1.0 / (2 ** zerop))
                        cliped = data2.clip(int_min, int_max) * (2 ** fix_q) - (1.0 / (2 ** fix_q))
                        fix16_data = cliped.astype(np.int16)
                        np.save(filepath, fix16_data, allow_pickle=False)
                        print("save %s to %s(fix16) min = %f, max = %f" % (weight_name, filepath, data2.min(), data2.max()))
                elif self.dtype == 'fp16':
                    # Kerasの重みをそのまま使う
                    filename = weight_name.replace(':0', '_z').replace('/', '_') + '.npy'
                    filepath = os.path.join(self.output_dir, filename)

                    data_fp16 = data2.astype(np.float16)

                    np.save(filepath, data_fp16, allow_pickle=False)
                    print("save %s to %s(fp16)" % (weight_name, filepath))

                else:
                    print("ERROR unkown weight dtype = %s" % self.dtype)

    def compress_bn(self, filename):
        """
        Batch Normalizationの重み（gamma, beta, moving mean, moving variance)を1つの重みデータに
        圧縮する。 (0除算を防ぐためのepsilonは0.001に決め打ち）
        として、
        A = mean - (beta * sqrt(variance + epsilon))
        B = gamma / sqrt(variance + epsilon)
        とする。BNの計算は
        (X - A) * B
        で計算ができる。逆数、ルートの計算を実行時にやらなくてすみ、データ量を減らせる。
        データの並びは、先頭から
        A[0]B[0]A[1]B[1]・・・A[n]B[n]
        の順番
        :param filename: moving_varianceのファイル名。ここから残りのファイル名を生成する。
        :return:
        """
        print("Batch Normalization Compress %s" % filename)
        m = re.findall("\d+", filename)
        num = m[-1]
        fname_moving_variance = filename
        fname_moving_mean = 'batch_normalization_%s_moving_mean_z.npy' % num
        fname_gammma = 'batch_normalization_%s_gamma_z.npy' % num
        fname_beta = 'batch_normalization_%s_beta_z.npy' % num
        fname_bn_weight = 'batch_normalization_%s.npy' % num

        moving_variance = np.load(os.path.join(self.output_dir, fname_moving_variance))
        moving_mean = np.load(os.path.join(self.output_dir, fname_moving_variance))
        gammma = np.load(os.path.join(self.output_dir, fname_gammma))
        beta = np.load(os.path.join(self.output_dir, fname_beta))

        A = moving_mean - (beta * np.sqrt(moving_variance + self.epsilon))
        B = gammma / np.sqrt(moving_variance + self.epsilon)
        np_w = np.zeros(len(A) * 2)

        for i in range(len(A)):
            np_w[i*2] = A[i]
            np_w[i*2+1] = B[i]

        np.save(os.path.join(self.output_dir, fname_bn_weight), np_w, allow_pickle=False)

        os.remove(os.path.join(self.output_dir, fname_moving_variance))
        os.remove(os.path.join(self.output_dir, fname_moving_mean))
        os.remove(os.path.join(self.output_dir, fname_gammma))
        os.remove(os.path.join(self.output_dir, fname_beta))

    def calc_bach_invvar(self, data):
        epsilon = 0.001
        outdata = 1.0 / np.sqrt(data + epsilon)
        print("calc_var min = %f, max = %f" % (outdata.min(), outdata.max()))
        return outdata

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

    def save_qpoint_file(self):
        """デバッグ用にＱポイントの情報を書き出す"""
        fname = os.path.join(self.output_dir, 'qp.txt')
        f = open(fname, 'w')
        f.write("input, output, weight, layer\n")
        first = True

        last_oq = 0

        for cl in self.compiler.cqt_layers:
            name = cl.l.name
            if first:
                wq = 0
                iq = 8
                oq = 8
                first = False
            else:
                wq = cl.weight_q
                iq = cl.input_q
                if cl.keras_layer_type in ['LeakyReLU', 'MaxPooling2D']:
                    oq = last_oq
                else:
                    oq = cl.output_q

                last_oq = oq

            f.write("%3s, %3s, %3s, %s\n" % (iq, oq, wq, name))
        f.close()

    def is_fixconv(self, weight_name):
        """
        重みの名前がconv2dかどかの判定
        重みのフォーマットが古いバージョンかどうかで判定ルーチンを変える。
        戻り地は、真偽とレイヤー名のペア
        """
        config = self.compiler.config
        if 'weight_filename_mode' in config['Cocyuts']:
            weight_filename_mode = int(config['Cocyuts']['weight_filename_mode'])
        else:
            # default
            weight_filename_mode = 0

        # バージョンによって場合分け
        if weight_filename_mode == 0:
            if weight_name.find('conv') != -1:
                names = weight_name.split('_')
                name = names[0] + '_' + names[1]
                return True, name
            elif weight_name.find('fc') == 0  or weight_name.find('predictions') == 0:
                names = weight_name.split('_')
                name = names[0]
                return True, name
            else:
                return False, ''
        else:
            if weight_name.find('conv2d') == 0:
                name = weight_name.split('/')[0]
                return True, name
            else:
                return False, ''

def calc_qpos(x, bit = 16):
    """
    引数の数値を表現できる最大のＱ位置を返す。
    :param x: float
    :return: int
    """
    for q in range(bit):
        maxv = (2 ** (q - 1)) - 1
        if x > maxv:
            continue
        return bit - q

    return bit

