import argparse
import sys
import itertools

import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Image converter')
    parser.add_argument('img', help='input image')

    args = parser.parse_args()

    img_file = args.img
    img = Image.open(img_file)
    width = 224
    height = 224

    if img.size != (width, height):
        img = img.resize((width, height))

    # 次元の反転
    arr = np.asarray(img, dtype=np.float32)

    ori_shape = arr.shape

    if len(ori_shape) == 1:
        np.save(img_file + '.npy', arr, allow_pickle=False)
        sys.exit(0)

    new_shape = list(ori_shape)
    new_shape.reverse()
    new_data = np.zeros(new_shape, dtype=np.float16)

    i_r = range(new_shape[0])
    j_r = range(new_shape[1])

    if len(ori_shape) == 3:
        for i, j, k in itertools.product(i_r, j_r, range(new_shape[2])):
            # RGB -> BGR
            if i == 0:
                id = 2
            elif i == 2:
                id = 0
            else:
                id = 1

            new_data[id][k][j] = arr[k][j][i]
    else:
        print("error")
        sys.exit(1)

    # ＶＧＧ平均値の減算
    vgg_mean = [103.939, 116.779, 123.68]
    new_data[0,:,:] = new_data[0,:,:] - vgg_mean[0]
    new_data[1,:,:] = new_data[1,:,:] - vgg_mean[1]
    new_data[2,:,:] = new_data[2,:,:] - vgg_mean[2]

    np.save(img_file+'_fp16.npy', new_data, allow_pickle=False)

if __name__ == '__main__':
    main()

