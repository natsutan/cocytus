import os
import shutil

class CGenerator:
    def __init__(self, compiler):
        self.compiler = compiler
        self.config = self.compiler.config

    def generate(self):
        """
        Cソースファイルを作成する
        """
        # ディレクトリの作成
        target_dir = self.config['Cocyuts']['output_dir']
        create_c_dir(target_dir)

        # ヘッダーファイルの作成、コピー
        template_dir = self.config['Cocyuts']['c_lib_dir']

        self.generate_hedarfiles(target_dir, template_dir)

        # Cソースの作成
        self.generate_cqt_gen()

        # ライブラリの作成
        self.generate_cqt_lib()

    def generate_hedarfiles(self, target_dir, template_dir):
        """
        template_dirから、target_dirへヘッダファイルをコピーする。
        :param target_dir: str
        :param template_dir: str
        :return:
        """
        headers = ['cqt.h', 'cqt_type.h', 'cqt_keras.h', 'numpy.h']
        for h in headers:
            shutil.copy(os.path.join(template_dir, h),
                        os.path.join(target_dir, 'inc'))


    def generate_cqt_gen(self):
        """
        ## コキュートス実行用ソース
        ### cqt_gen.h
        ### cqt_gen.c
        """

    def generate_cqt_lib(self):
       """
        ## Ｃライブラリ(cqt_lib)
        ### cqt_lib.h
        ### cqt_lib.c
       """


def create_c_dir(tdir):
    """
    tdir以下に以下のディレクトリを作成する。
    inc, cqt_gen, cqt_lib
    :param tdir: str
    :return:
    """
    dirs = ['inc', 'cqt_gen', 'cqt_lib']
    for d in dirs:
        path = os.path.join(tdir, d)
        if not os.path.isdir(path):
            os.makedirs(path)

