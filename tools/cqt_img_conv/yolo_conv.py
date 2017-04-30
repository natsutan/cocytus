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
    width = 416
    height = 416

    image = Image.open(img_file)
    resized_image = image.resize((width, height), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32') / 255.0

    arr = np.asarray(image_data, dtype=np.float32)

    ori_shape = arr.shape

    new_shape = list(ori_shape)
    new_shape.reverse()
    new_data = np.zeros(new_shape, dtype=np.float32)

    i_r = range(new_shape[0])
    j_r = range(new_shape[1])

    if len(ori_shape) == 3:
        for i, j, k in itertools.product(i_r, j_r, range(new_shape[2])):
            if i == 0:
                id = 0
            elif i == 1:
                id = 1
            else:
                id = 2

            new_data[id][k][j] = arr[k][j][i]


    np.save(img_file+'.npy', new_data, allow_pickle=False)

if __name__ == '__main__':
    main()

