#!/bin/env python3
import sys
import configparser

from compiler.compiler import CocytusCompiler
from cocytus_net.C.c_generator import CGenerator
from weight_converter.weight_converter import WeightConverter


def open_inifile(ini_file):
    """
    inifileをオープンし、パーサーを返す。
    :param ini_file:str
    :return:ConfigParser
    """
    print("INI:open %s" % ini_file)
    config = configparser.ConfigParser()
    config.read(ini_file)
    return config


def check_config(config):
    """
    configの必須データを確認する。
    必須出たが存在しないとｋは、エラーを表示してプログラムを終了する。
    :param config:ConfigParser
    :return:bool
    """
    cqt_options = ['keres_json', 'output_dir', 'c_lib_dir']
    ret = True

    for opt in cqt_options:
        try:
            config.get('Cocyuts', opt)
        except (configparser.NoSectionError, configparser.NoOptionError):
            print("ERROR:no entry [%s] %s" % ('Cocyuts', opt))
            ret = False
    return ret


def main(argv):
    # TODO オプションの処理作成
    global weight_q
    if len(argv) != 2:
        print('usage:python3 cocytus.py ini')
        sys.exit()
    ini_file = argv[1]
    config = open_inifile(ini_file)

    # TODO ここにオプションで上書きの処理を入れる。
    if not check_config(config):
        print("ERROR:ini file %s" % ini_file)
        sys.exit(1)

    # jsonの読み込み
    compiler = CocytusCompiler(config)
    if not compiler.compile():
        print("ERROR:compaile failed")
        sys.exit(1)

    # Cソースの生成
    c_generator = CGenerator(compiler)
    c_generator.generate()

    # 重みの変換
    try:
        w_out_dir = config.get('Cocyuts', 'weight_output_dir')
        hdf_file = config.get('Cocyuts', 'keras_weight')

        if 'weight_dtype' in config['Cocyuts']:
            # iniファイルの設定を優先
            type = config.get('Cocyuts', 'weight_dtype')
            if 'weight_q' in config['Cocyuts']:
                weight_q = config.get('Cocyuts', 'weight_q')
            else:
                weight_q = 8

        else:
            type = ""

        w_converter = WeightConverter(w_out_dir, hdf_file, type, weight_q=weight_q)
        w_converter.convert()

    except (configparser.NoSectionError, configparser.NoOptionError):
        print("Weight Conversion skipped.")


    print('finish')



if __name__ == '__main__':
    main(sys.argv)





