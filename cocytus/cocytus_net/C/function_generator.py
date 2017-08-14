import datetime
import os
import string

from compiler.compiler import CQT_Dtype

ctype_dic = {CQT_Dtype.FLOAT32: 'float', CQT_Dtype.UINT8: 'unsigned char',
             CQT_Dtype.FIX16:'FIXP16', CQT_Dtype.FIX8: 'FIXP8'}


class FunctionGenerator:
    def __init__(self, compiler, config, target_dir, template_dir):
        self.compiler = compiler
        self.config = config
        self.target_dir = target_dir
        self.template_dir = template_dir

        self.conv2d_same_3x3_first = True
        self.conv2d_same_1x1_first = True
        self.maxpoolong2d_first = True
        self.flatten_first = True
        self.dense_first = True
        self.batchnormalization_first = True
        self.leakyrelu_first = True

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
        layers = self.compiler.get_layers()

        for i, l in enumerate(layers):
            name = l.name
            config = l.get_config()
            layer_detal = self.compiler.get_cqt_layer_obj(name)
            class_name = layer_detal.keras_layer_type

            func_name = layer_detal.make_func_name()
            if not func_name in func_list:
                print('generating int %s(CQT_LAYER *lp, void *inp, void *outp);' % func_name)
                if class_name == 'InputLayer':
                    self.generate_input_layer(layer_detal)
                elif class_name == 'Conv2D':
                    self.generate_conv2d(layer_detal)
                elif class_name == 'MaxPooling2D':
                    self.generate_maxpooling2d(layer_detal)
                elif class_name == 'Flatten':
                    self.generate_flatten(layer_detal)
                elif class_name == 'Dense':
                    self.generate_dense(layer_detal)
                elif class_name == 'BatchNormalization':
                    self.generate_batchnormalization(layer_detal)
                elif class_name == 'LeakyReLU':
                    self.generate_leakyrelu(layer_detal)

                func_list.append(func_name)

    def generate_input_layer(self, layer_detail):
        func_name = layer_detail.make_func_name()
        output_file = os.path.join(self.target_dir, 'cqt_lib', 'InputLayer.c')
        template_file = os.path.join(self.template_dir, 'InputLayer', 'InputLayer.c')

        with open(output_file, 'w') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name)
            fpout.write(func_str)

    def generate_conv2d(self, layer_detail):

        kernel_size = layer_detail.l.kernel_size
        padding = layer_detail.l.padding
        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        weight_type = layer_detail.weight_dtypes[0]
        output_type = layer_detail.output_dtypes[0]
        shift_val = 0

        if (kernel_size == (3, 3) or kernel_size == (1, 1)) and padding == 'same':
            if kernel_size == (3, 3):
                output_file = os.path.join(self.target_dir, 'cqt_lib', 'Conv2d_same_3x3.c')
            else:
                output_file = os.path.join(self.target_dir, 'cqt_lib', 'Conv2d_same_1x1.c')


            # templete fileの選択
            conv2d_template_fname = ''
            opt_level = self.compiler.get_conv2d_optlevel()
            if opt_level == 'dash' and kernel_size == (3, 3):
                conv2d_template_fname = 'Conv2d_same_3x3_dash.c'
            else:
                if opt_level != '':
                    print("WARNING unkown Conv2d optimze level %s, use dafalut(no optimize)." % opt_level)

                if self.compiler.is_fix16_mode():
                    shift_val = layer_detail.weight_q
                    if kernel_size == (3, 3):
                        conv2d_template_fname = 'Conv2d_same_3x3_fixed.c'
                    elif kernel_size == (1, 1):
                        conv2d_template_fname = 'Conv2d_same_1x1_fixed.c'
                else:
                    if kernel_size == (3, 3):
                        if self.compiler.is_output_channel_last():
                            conv2d_template_fname = 'Conv2d_same_3x3_cl.c'
                        else:
                            conv2d_template_fname = 'Conv2d_same_3x3.c'
                    elif kernel_size == (1, 1):
                        conv2d_template_fname = 'Conv2d_same_1x1.c'

            template_file = os.path.join(self.template_dir, 'Conv2d', conv2d_template_fname)

            if kernel_size == (3, 3) and self.conv2d_same_3x3_first:
                with open(output_file, 'w') as fp:
                    fp.write('#include <string.h>\n')
                    fp.write('#include <limits.h>\n')
                    fp.write('#include <assert.h>\n')
                    fp.write('#include "cqt.h"\n')
                    fp.write('#include "cqt_net.h"\n')
                    fp.write('\n')
                self.conv2d_same_3x3_first = False

            if kernel_size == (1, 1) and self.conv2d_same_1x1_first:
                with open(output_file, 'w') as fp:
                    fp.write('#include <string.h>\n')
                    fp.write('#include <assert.h>\n')
                    fp.write('#include "cqt.h"\n')
                    fp.write('#include "cqt_net.h"\n')
                    fp.write('\n')
                self.conv2d_same_1x1_first = False
        else:
            name = layer_detail.l.name
            print('ERROR unsupported Conv2d %s kernel = %s padding = %s' % (name, str(kernel_size), padding))
            return

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    weight_type=ctype_dic[weight_type],
                                    output_type=ctype_dic[output_type],
                                    shift_val=shift_val)
            fpout.write(func_str)

    def generate_maxpooling2d(self, layer_detail):

        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        output_type = layer_detail.output_dtypes[0]

        output_file = os.path.join(self.target_dir, 'cqt_lib', 'MaxPooling2D.c')
        template_file = os.path.join(self.template_dir, 'MaxPooling2D', 'MaxPooling2D.c')

        if self.maxpoolong2d_first:
            with open(output_file, 'w') as fp:
                fp.write('#include <string.h>\n')
                fp.write('#include <assert.h>\n')
                fp.write('#include "cqt.h"\n')
                fp.write('#include "cqt_net.h"\n')
                fp.write('\n')
            self.maxpoolong2d_first = False

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    output_type=ctype_dic[output_type])
            fpout.write(func_str)

    def generate_flatten(self, layer_detail):

        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        output_type = layer_detail.output_dtypes[0]

        output_file = os.path.join(self.target_dir, 'cqt_lib', 'Flatten.c')
        template_file = os.path.join(self.template_dir, 'Flatten', 'Flatten.c')

        if self.flatten_first:
            with open(output_file, 'w') as fp:
                fp.write('#include <string.h>\n')
                fp.write('#include <assert.h>\n')
                fp.write('#include "cqt.h"\n')
                fp.write('#include "cqt_net.h"\n')
                fp.write('\n')
            self.flatten_first = False

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    output_type=ctype_dic[output_type])
            fpout.write(func_str)

    def generate_dense(self, layer_detail):

        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        output_type = layer_detail.output_dtypes[0]
        weight_type = layer_detail.weight_dtypes[0]

        output_file = os.path.join(self.target_dir, 'cqt_lib', 'Dense.c')

        if self.compiler.is_fix16_mode():
            tfile = 'Dense_fixed.c'
        else:
            tfile = 'Dense.c'

        template_file = os.path.join(self.template_dir, 'Dense', tfile)



        if self.dense_first:
            with open(output_file, 'w') as fp:
                fp.write('#include <string.h>\n')
                fp.write('#include <assert.h>\n')
                fp.write('#include <math.h>\n')
                fp.write('#include <limits.h>\n')
                fp.write('#include "cqt.h"\n')
                fp.write('#include "cqt_net.h"\n')
                fp.write('\n')
            self.dense_first = False

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    weight_type=ctype_dic[weight_type],
                                    output_type=ctype_dic[output_type])
            fpout.write(func_str)

    def generate_batchnormalization(self, layer_detail):

        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        output_type = layer_detail.output_dtypes[0]
        weight_type = layer_detail.weight_dtypes[0]
        int_max = 0
        int_min = 0
        shift_val = 0

        output_file = os.path.join(self.target_dir, 'cqt_lib', 'BatchNormalization.c')
        if self.compiler.is_fix16_mode():
            shift_val = layer_detail.weight_q
            int_bit = 16 - shift_val
            int_min = -(2 ** (int_bit - 1))
            int_max = (2 ** (int_bit - 1)) - 1

            template_file = os.path.join(self.template_dir, 'BatchNormalization', 'BatchNormalization_fixed.c')
        else:
            template_file = os.path.join(self.template_dir, 'BatchNormalization', 'BatchNormalization.c')

        if self.batchnormalization_first:
            with open(output_file, 'w') as fp:
                fp.write('#include <string.h>\n')
                fp.write('#include <assert.h>\n')
                fp.write('#include <limits.h>\n')
                fp.write('#include <math.h>\n')
                fp.write('#include "cqt.h"\n')
                fp.write('#include "cqt_net.h"\n')
                fp.write('\n')
            self.batchnormalization_first = False

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    weight_type=ctype_dic[weight_type],
                                    output_type=ctype_dic[output_type],
                                    shift_val=shift_val,
                                    max=int_max, min=int_min)
            fpout.write(func_str)


    def generate_leakyrelu(self, layer_detail):
        func_name = layer_detail.make_func_name()
        input_type = layer_detail.input_dtypes[0]
        output_type = layer_detail.output_dtypes[0]

        output_file = os.path.join(self.target_dir, 'cqt_lib', 'LeakyReLU.c')
        template_file = os.path.join(self.template_dir, 'LeakyReLU', 'LeakyReLU.c')

        if self.leakyrelu_first:
            with open(output_file, 'w') as fp:
                fp.write('#include <string.h>\n')
                fp.write('#include <assert.h>\n')
                fp.write('#include <math.h>\n')
                fp.write('#include "cqt.h"\n')
                fp.write('#include "cqt_net.h"\n')
                fp.write('\n')
            self.leakyrelu_first = False

        with open(output_file, 'a') as fpout:
            t = string.Template(open(template_file).read())
            func_str = t.substitute(func_name=func_name,
                                    input_type=ctype_dic[input_type],
                                    output_type=ctype_dic[output_type])
            fpout.write(func_str)


