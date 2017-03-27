from PIL import Image
import numpy as np

size = (224, 224)

img = Image.new('RGB', size, 0x000000)

arr = np.array(img)
arr[2][2][0] = 255
arr[2][2][1] = 255
arr[2][2][2] = 255

arr[2][10][0] = 255
arr[2][15][1] = 255
arr[2][20][2] = 255

pilImg = Image.fromarray(np.uint8(arr))

pilImg.save("test_00.png")