import os
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import numpy as np

#img_file = '../img/3.jpg'
#img_file = '../img/test_00.png'
img_file = '../img/dog.png'


width = 224
height = 224

def layer_dump(model, x, l, postfix=''):
    """
    :param model: Keras model
    :param x: input data
    :param l: layer number
    :param postfix: postfix for numpy file name
    :return: None
    """
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[l].output])

    layer_output = get_layer_output([x, 0])[0]
    last_dim = layer_output.shape[-1] - 1

    if l == 0:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, 2]
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
    else:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, 2]
        d3 = layer_output[0,:, :, 3]
        d4 = layer_output[0,:, :, 4]
        dl = layer_output[0,:, :, last_dim]
        np.save('output/l%02d%s_all.npy' % (l, postfix), layer_output, allow_pickle=False)
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
        np.save('output/l%02d%s_3.npy' % (l, postfix), d3, allow_pickle=False)
        np.save('output/l%02d%s_4.npy' % (l, postfix), d4, allow_pickle=False)

        np.save('output/l%02d%s_%d.npy' % (l, postfix, last_dim), dl, allow_pickle=False)



file_post_fix = ''



def main():
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pre_x = preprocess_input(x)

    model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

    preds = model.predict(pre_x)
    results = decode_predictions(preds, top=5)[0]
    for result in results:
        print(result)

    # モデルの保存
    model.summary()
    with open('vgg16.json', 'w') as fp:
        json_string = model.to_json()
        fp.write(json_string)

    # ディレクトリの作成
    if not os.path.exists('output'):
        os.mkdir("output")

    # 出力するレイヤーを選択

    for l in range(19):
        layer_dump(model, pre_x, l)

    print('finish.')


if __name__ == '__main__':
    main()
