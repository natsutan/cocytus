from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import numpy as np

#img_file = '../img/3.jpg'
#img_file = '../img/test_00.png'
img_file = '../img/dog.png'

model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


img = image.load_img(img_file, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# for cocytus
# 各レイヤの出力を出す


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


l = 3
get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[l].output])
layer_output = get_layer_output([x, 0])


print('L ', l, ' ', layer_output[0].shape)
if l == 2:
    arr = layer_output[0][0]
    d0 = layer_output[0][0][:,:,0]
    d1 = layer_output[0][0][:,:,1]
    d2 = layer_output[0][0][:,:,63]

    np.save('output/dog_l%02d_0.npy' % l, d0, allow_pickle=False)
    np.save('output/dog_l%02d_1.npy' % l, d1, allow_pickle=False)
    np.save('output/dog_l%02d_63.npy' % l, d2, allow_pickle=False)


if l == 3:
    arr = layer_output[0][0]
    d0 = layer_output[0][0][:,:,0]
    d1 = layer_output[0][0][:,:,1]
    d2 = layer_output[0][0][:,:,63]

    np.save('output/dog_l%02d_0.npy' % l, d0, allow_pickle=False)
    np.save('output/dog_l%02d_1.npy' % l, d1, allow_pickle=False)
    np.save('output/dog_l%02d_63.npy' % l, d2, allow_pickle=False)



print('finish.')
