import os

from compiler.compiler import CQT_Dtype

ctype_dic = {CQT_Dtype.FLOAT32: 'float', CQT_Dtype.UINT8: 'unsigned char',
             CQT_Dtype.FIX16:'FIXP16', CQT_Dtype.FIX8: 'FIXP8', CQT_Dtype.FLOAT16: 'FP16'}


class SDSOC_gen():
    def __init__(self):
        self.funcname_list = []
        self.header_name = ""
        self.fp_c = None
        self.fp_h = None

    def generate(self, output_file, layer_detail, l):
        if self.fp_c is None:
            self.open_file(output_file)
            self.write_include()

        func_name = 'CQT_' + l.name + "_3x3"
        hw_func_name = func_name + '_hw'

        print('[SDODC] %s' % func_name)

        prototype = self.make_function_prottype(hw_func_name, layer_detail)
        self.add_funciton_to_header(prototype)

        self.write_func(func_name, layer_detail)
        self.write_func_hw(hw_func_name, layer_detail)

    def write_func(self, func_name, layer_detail):
        self.wr('int %s(CQT_LAYER *lp, void *inp, void *outp){\n' % func_name)

        self.wr('\treturn CQT_RET_OK;\n')
        self.wr('}\n')
        self.wr('\n')


    def write_func_hw(self, hw_func_name, layer_detail):
        prototype = self.make_function_prottype(hw_func_name, layer_detail)

        self.wr(prototype + '{\n')



        self.wr('}\n')
        self.wr('\n')

    def make_function_prottype(self, func, layer_detail):
        input_type = ctype_dic[layer_detail.input_dtypes[0]]
        weight_type = ctype_dic[layer_detail.weight_dtypes[0]]
        output_type = ctype_dic[layer_detail.output_dtypes[0]]
        size1 = layer_detail.l.input_shape[1]
        size2 = layer_detail.l.input_shape[2]

        #assert(size1 == size2)
        prot = 'void %s(%s ip[%d], %s op[%d], %s weight[9], int bias, int act, int last)' % (func, input_type, size1 * size2, output_type, size1 * size2, weight_type )
        return prot


    def open_file(self, output_file):
        self.fp_c = open(output_file, 'w')
        header_name = output_file[0:-2] + ".h"
        self.fp_h = open(header_name, 'w')
        self.header_name = os.path.basename(header_name)

    def write_include(self):
        self.wr('#include <string.h>\n')
        self.wr('#include <limits.h>\n')
        self.wr('#include <assert.h>\n')
        self.wr('#include "cqt.h"\n')
        self.wr('#include "cqt_net.h"\n')
        self.wr('#include "%s"\n' % self.header_name)
        self.wr('\n')

    def wr(self, s):
        self.fp_c.write(s)


    def add_funciton_to_header(self, prototype):

        self.fp_h.write("%s;\n" % prototype)

