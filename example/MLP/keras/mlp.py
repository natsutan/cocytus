# -*- coding: utf-8 -*-

import keras
import sys
import numpy as np
from PIL import Image
from PIL import ImageOps


# 各自の環境に置き換えてください
train_csv = '/media/natu/data/data/src/output/train.csv'
test_csv = '/media/natu/data/data/src/output/test.csv'


def input_data_from_csv(file):
    """
    引数で指定されたcsvファイルから、画像データとラベルを読み込み、numpyの配列として返す。
    """
    images = []
    labels = []
    try:
        fp = open(file, 'r')
    except FileNotFoundError:
        print("Error %s がオープンできません" % file)
        sys.exit()

    for l in fp.readlines():
        l = l.rstrip()
        img_name, label = l.split(',')

        # 画像ファイルを開き、モノクロ化、リサイズ
        img_ori = Image.open(img_name)
        img_gray = ImageOps.grayscale(img_ori)
        img_resize = img_gray.resize((image_size, image_size))

        # imageををnumpyに変更して、imagesに追加
        img_ary = np.asarray(img_resize)
        # neg pos 反転
        img_ary_inv = 255 - img_ary
        images.append(img_ary_inv.flatten().astype(np.float32) / 255.0)

        # ラベルの追加
        label_array = np.zeros(kana_num)
        label_array[int(label)] = 1
        labels.append(label_array)

    return images, labels


def main():
    None


if __name__ == '__main__':
    main()
