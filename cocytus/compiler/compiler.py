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
    def __init__(self, l):
        """
        :param l:  keras layer
        """
        self.input_size = []
        self.input_dtypes = []
        self.output_dtypes = []
        self.weight_dtypes = []
        self.l = l

    def get_Wshape(self):
        return self.l.weights[0]._keras_shape

    def get_conv2d_weight_variable_name(self):
        """
        重みの変数名
        重みのnumpyヘッダー変数名
        biasの変数名
        biasのnumpyヘッダー変数名
        :return: str
        """
        name = self.l.name
        w_name = "w_%s_W" % name
        w_nph_name = "nph_%s_W" % name
        b_name = "w_%s_b" % name
        b_nph_name = "nph_%s_b" % name
        return w_name, w_nph_name, b_name, b_nph_name

    def get_output_variable_name(self):
        """
        出力用変数の文字列を返す。
        """
        name = self.l.name
        return name + '_output'

    def get_weight_type_str(self):
        type = self.weight_dtypes[0]
        if type == CQT_Dtype.INT8:
            return 'signed char'
        elif type == CQT_Dtype.UINT8:
            return 'unsigned char'
        elif type == CQT_Dtype.INT32:
            return 'int'
        elif type == CQT_Dtype.FLOAT32:
            return 'float'

        raise ValueError("Error layer %s type is not supported" % type)

    def get_output_shape(self):
        return self.l.output_shape

    def get_input_shape(self):
        return self.l.input_shape

    def get_output_type_str(self):
        type = self.output_dtypes[0]
        if type == CQT_Dtype.INT8:
            return 'signed char'
        elif type == CQT_Dtype.UINT8:
            return 'unsigned char'
        elif type == CQT_Dtype.INT32:
            return 'int'
        elif type == CQT_Dtype.FLOAT32:
            return 'float'

        raise ValueError("Error layer %s tpye is not supported" % type)

    def get_prev_layer_output_dimension(self, i):
        """
        前の層の出力数を返す。
        :param i: int
        :return: int
        """
        if i == 0 :
            # 入力層は1
            return 1

        return self.l.input_shape[-1]


class CocytusCompiler:
    def __init__(self, config, nn_prefix='cqt_'):
        self.config = config
        json_file = config['Cocyuts']['keres_json']
        json_string = open(json_file, 'r').read()
        print("JSON:open %s" % json_file)
        self.model = model_from_json(json_string)
#        self.model = model_from_json(json_string, custom_objects={"Normalize": Normalize, "PriorBox": PriorBox})
        #conf = self.model.get_config()

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

            cl = CocytusLayerInfo(l)

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

    def get_layer_obj(self, name):
        """
        引数のレイヤー名から、KerasのLayerオブジェクトを返す。
        get_configだけでは取れない情報を取得するために使う。
        :param name:
        :return:NN_DTYPE
        """
        for l in self.model.layers:
            if l.name == name:
                return l

        # not found
        raise ValueError("Error layer  %s layer is not found" % name)

    def get_cqt_layer_obj(self, name):
        """
        引数のレイヤー名から、CocytuのLayerオブジェクトを返す。
        get_configだけでは取れない情報を取得するために使う。
        :param name:
        :return:
        """
        for (i, l) in enumerate(self.model.layers):
            if l.name == name:
                return self.cqt_layers[i]

        # not found
        raise ValueError("Error layer  %s layer is not found" % name)

    def get_model_name(self):
        """
        ニューラルネットの名前を返す。
        一つのプログラムで複数のネットを使いたいときなどは、ここで調整する。
        """
        name = "g_cqt_" + self.model.name
        return name


def conv_type_np_to_cqt(tf_type):
    """
    numpyの型を、cqtの型情報に変換する。
    :param type:dtype
    :return:CQT_Dtype
    """
    dtype = str(tf_type)

    conv_dic = {"<dtype: 'float32'>": CQT_Dtype.FLOAT32, "<dtype: 'float32_ref'>": CQT_Dtype.FLOAT32}

    return conv_dic[dtype]
