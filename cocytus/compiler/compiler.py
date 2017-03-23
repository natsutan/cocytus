from enum import Enum
import tensorflow as tf
from keras.models import model_from_json

class CQT_Dtype(Enum):
    """
    型の定義
    """
    NONE = 0
    INT8 = 1
    UINT8 = 2
    INT32 = 3
    FLOAT32 = 4

class CocytusLayerInfo:
    """
    コキュートス用の追加レイヤー情報。
    Kerasから取得できるレイヤー情報と合わせて使用する。
    """
    def __init__(self):
        self.input_dtypes = []
        self.output_dtypes = []
        self.weight_dtypes = []


class CocytusCompiler:
    def __init__(self, config, nn_prefix='cqt_'):
        self.config = config
        json_file = config['Cocyuts']['keres_json']
        json_string = open(json_file, 'r').read()
        print("JSON:open %s" % json_file)
        self.model = model_from_json(json_string)
#        self.model = model_from_json(json_string, custom_objects={"Normalize": Normalize, "PriorBox": PriorBox})

        self.layers = self.model.layers
        self.nn_prefix = nn_prefix
        self.cqt_layers = []

    @property
    def compile(self):
        """
        self.modelから後工程に必要な情報を抜き出す。
        :return:bool
        """
        self.model.summary()
        for l in self.layers:
            keras_layer_type = l.__class__.__name__
            print("%s:%s" % (keras_layer_type, l.name))

            cl = CocytusLayerInfo()

            # 型のチェック
            # 将来的には各層を任意の型に買えられるようにする。
            input_type = l.input.dtype
            cl.input_dtypes.append(conv_type_np_to_cqt(input_type))
            output_type = l.output.dtype
            cl.output_dtypes.append(conv_type_np_to_cqt(output_type))

            if not l.weights:
                cl.weight_dtypes.append(CQT_Dtype.NONE)
            else:
                for w in l.weights:
                    wtype = w.dtype
                    cl.weight_dtypes.append(conv_type_np_to_cqt(wtype))

            self.cqt_layers.append(cl)

        return True


def conv_type_np_to_cqt(tf_type):
    """
    numpyの型を、cqtの型情報に変換する。
    :param type:dtype
    :return:CQT_Dtype
    """
    dtype = str(tf_type)

    conv_dic = {"<dtype: 'float32'>": CQT_Dtype.FLOAT32, "<dtype: 'float32_ref'>": CQT_Dtype.FLOAT32}

    return conv_dic[dtype]
