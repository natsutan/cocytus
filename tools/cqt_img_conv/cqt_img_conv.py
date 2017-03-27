import argparse
import sys
import itertools
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Image converter')
    parser.add_argument('-x', nargs='?', help='output image width')
    parser.add_argument('-y', nargs='?', help='output image height')

#    parser.add_argument('format', help='output format. rgb or mono')
#    parser.add_argument('dtype', help='output data type. float32 or uint8')
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

    # format変更

    # 次元の反転
    arr = np.asarray(img)

    ori_shape = arr.shape

    if len(ori_shape) == 1:
        np.save(img_file + '.npy', arr, allow_pickle=False)
        sys.exit(0)

    new_shape = list(ori_shape)
    new_shape.reverse()
    new_data = np.zeros(new_shape, dtype=arr.dtype)

    i_r = range(new_shape[0])
    j_r = range(new_shape[1])

    if len(ori_shape) == 2:
        for i, j in itertools.product(i_r, j_r):
            new_data[i][j] = arr[j][i]
    elif len(ori_shape) == 3:
        for i, j, k in itertools.product(i_r, j_r, range(new_shape[2])):
            new_data[i][j][k] = arr[k][j][i]
    elif len(ori_shape) == 4:
        # [32][1][3][3]:
        for i, j, k, l in itertools.product(i_r, j_r, range(new_shape[2]), range(new_shape[3])):
            new_data[i][j][k][l] = arr[l][k][j][i]
    else:
        print("error")
        sys.exit(1)

    np.save(img_file+'.npy', new_data, allow_pickle=False)





if __name__ == '__main__':
    main()

