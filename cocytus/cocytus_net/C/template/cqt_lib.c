// compileを通すためのダミーファイル
#include "cqt.h"
#include "cqt_net.h"

int CQT_InputLayer_if_of(CQT_LAYER *lp, float *ip, float *op) { return 0;}
int CQT_Conv2D_3x3_if_of(CQT_LAYER *lp, float *ip, float *op) { return 0;}
int CQT_MaxPooling2D_if_of(CQT_LAYER *lp, float *ip, float *op) { return 0;}
int CQT_Flatten_if_of(CQT_LAYER *lp, float *ip, float *op) { return 0;}
int CQT_Dense_if_of(CQT_LAYER *lp, float *ip, float *op) { return 0;}
