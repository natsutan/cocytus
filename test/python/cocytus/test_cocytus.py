import unittest
import sys

sys.path.append('../../cocytus')
import cocytus as cqt

ini_path = 'cocytus/ini/'

class InifileTest(unittest.TestCase):
    def test_normal(self):
        config = cqt.open_inifile(ini_path + 'ok.ini')
        self.assertTrue(cqt.check_config(config))

    def test_err(self):
        config = cqt.open_inifile(ini_path + 'err0.ini')
        self.assertFalse(cqt.check_config(config))
        config = cqt.open_inifile(ini_path + 'err1.ini')
        self.assertFalse(cqt.check_config(config))

if __name__ == '__main__':
    unittest.main()
