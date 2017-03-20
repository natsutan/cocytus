from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

img_file = '../img/3.jpg'
model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


img = image.load_img(img_file, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)

# for cocytus
# 重みファイルは、 ~/.Keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5
# に保存される。
model.summary()

with open('vgg16.json', 'w') as fp:
    json_string = model.to_json()
    fp.write(json_string)

np.save(img_file+'.npy', x, allow_pickle=False)
print('finish.')
