import datetime
import os
import string

ctype_dic = {'CQT_FLOAT32': 'float', 'CAT_UINT8': 'unsigned char'}


class FunctionGenerator:
    def __init__(self, compiler, config, target_dir, template_dir):
        self.compiler = compiler
        self.config = config
        self.target_dir = target_dir
        self.template_dir = template_dir

    def get_config(self):
        """
        Keras Modelのコンフィグ情報を返す。
        :return:
        """
        return self.compiler.model.get_config()

    def generate(self):
        print('function generate')
        model_config = self.get_config()

        func_list = []

        for i, l in enumerate(model_config['layers']):
            class_name = l['class_name']
            config = l['config']
            name = config['name']
            layer_detal = self.compiler.get_cqt_layer_obj(name)

            func_name = layer_detal.make_func_name()
            if not func_name in func_list:
                print('generating int %s(CQT_LAYER *lp, void *inp, void *outp);' % func_name)
                if class_name == 'InputLayer':
                    self.generate_input_layer(func_name)

                func_list.append(func_name)

    def generate_input_layer(self, func_name):
        output_file = os.path.join(self.target_dir, 'cqt_lib', 'InputLayer.c')
        template_file = os.path.join(self.template_dir, 'InputLayer', 'InputLayer.c')

        with open(output_file, 'w') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name)
            fpout.write(func_str)



