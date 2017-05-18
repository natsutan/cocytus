import sys
import math
import numpy as np
from PIL import Image, ImageDraw


img_file = './sample/person.jpg'
#img_file = '../img/dog.jpg'
0
# 入力サイズ等はここを変更
width = 416
height = 416

image = Image.open(img_file)

dr = ImageDraw.Draw(image)

regions = [((181, 150), (229, 270)), ((458, 162), (519, 262)), ((169, 128), (239, 345))]

for r in regions:
    left = r[0][0]
    top  = r[0][1]
    right = r[1][0]
    bottom = r[1][1]

    lt = (left, top)
    rt = (right, top)
    lb = (left, bottom)
    rb = (right, bottom)
    red = (255, 0, 0)
    dr.line((lt, rt), red, 2)
    dr.line((lt, lb), red, 2)
    dr.line((rt, rb), red, 2)
    dr.line((lb, rb), red, 2)

image.save(img_file+'fix_out.png')