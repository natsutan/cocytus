import os

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

        # ヘッダーファイルの作成

        # Cソースの作成

        # ライブラリの作成


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

