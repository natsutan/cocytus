import os


predlist = '/home/natsutani/proj/cocytus/example/tiny-yolo/c_fix/pred.txt'
filelist = '/home/natsutani/proj/cocytus/tools/tyolo_conv/2007_test.txt'




for l in open(filelist).readlines():
    basename = os.path.basename(l.strip())
    print(basename)
