import argparse
import sys
import itertools
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Image converter')
    parser.add_argument('-x', nargs='?', help='output image width')
    parser.add_argument('-y', nargs='?', help='output image height')
    parser.add_argument('img', help='input image')

    args = parser.parse_args()

    img_file = args.img
    img = Image.open(img_file)

    # サイズ変更
    if 'x' in args:
        width = int(args.x)
    else:
        width = img.size[0]

    if 'y' in args:
        height = int(args.y)
    else:
        height = img.size[1]

    if img.size != (width, height):
        img = img.resize((width, height))

    # 次元の反転
    arr = np.asarray(img, dtype=np.float32)

    ori_shape = arr.shape

    np.save(img_file + '.npy', arr, allow_pickle=False)

    if len(ori_shape) == 1:
        np.save(img_file + '.npy', arr, allow_pickle=False)
        sys.exit(0)

    new_shape = list(ori_shape)
    new_shape.reverse()
    new_data = np.zeros(new_shape, dtype=np.float32)

    i_r = range(new_shape[0])
    j_r = range(new_shape[1])

    if len(ori_shape) == 3:
        for i, j, k in itertools.product(i_r, j_r, range(new_shape[2])):
            new_data[i][k][j] = arr[k][j][i]
    else:
        print("error")
        sys.exit(1)

    # ＶＧＧ平均値の減算
    vgg_mean = [103.939, 116.779, 123.68]
    new_data[0,:,:] = new_data[0,:,:] - vgg_mean[2]
    new_data[1,:,:] = new_data[1,:,:] - vgg_mean[1]
    new_data[2,:,:] = new_data[2,:,:] - vgg_mean[0]


    np.save(img_file+'.npy', new_data, allow_pickle=False)


if __name__ == '__main__':
    main()

