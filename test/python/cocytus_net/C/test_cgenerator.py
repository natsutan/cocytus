import unittest
import sys
import os

sys.path.append('../../cocytus')
from cocytus_net.C.c_generator import create_c_dir

ini_path = 'cocytus/ini/'


class FunctionTest(unittest.TestCase):
    def test_make_dirs(self):
        cdir = os.getcwd()
        tdir = os.path.join(cdir, 'cocytus_net', 'C', 'work')

        if not os.path.isdir(tdir):
            os.makedirs(tdir)

        dirs = ['inc', 'cqt_gen', 'cqt_lib']
        for d in dirs:
            path = os.path.join(tdir, d)
            if os.path.isdir(path):
                   os.rmdir(path)

        create_c_dir(tdir)

        for d in dirs:
            path = os.path.join(tdir, d)
            self.assertTrue(os.path.isdir(path))

        for d in dirs:
            path = os.path.join(tdir, d)
            if os.path.isdir(path):
                os.rmdir(path)


if __name__ == '__main__':
    unittest.main()
