from enum import Enum
from keras.models import model_from_json

class Dtype(Enum):
    """
    型の定義
    """
    NONE = 0
    INT8 = 1
    INT32 = 2
    FLOAT32 = 3

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
    def __init__(self, config, name = 'cqt_'):
        self.config = config
        json_file = config['Cocyuts']['keres_json']
        json_string = open(json_file, 'r').read()
        print("JSON:open %s" % json_file)
#        self.model = model_from_json(json_string)
        self.model = model_from_json(json_string, custom_objects={"Normalize": Normalize, "PriorBox": PriorBox})

        self.layers = self.model.layers
        self.nn_name = name
        self.cqt_layers = []

    def compile(self):
        """
        self.modelから後工程に必要な情報を抜き出す。
        :return:bool
        """
        self.model.summary()
        for l in self.layers:
            print("%s:%s" % (l.__class__.__name__, l.name))

            cl = CocytusLayerInfo()

            # 型のチェック
            # 将来的には各層を任意の型に買えられるようにする。
            if l.dtype == 'float32':
                cl.input_dtypes.append(Dtype.FLOAT32)
                cl.output_dtypes.append(Dtype.FLOAT32)
                cl.weight_dtypes.append(Dtype.FLOAT32)
            else:
                print("ERROR:dtype %s is not supported")
                return False

            self.cqt_layers.append(cl)

        return True