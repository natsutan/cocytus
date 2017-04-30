// compileを通すためのダミーファイル
#include "cqt.h"
#include "cqt_net.h"

int CQT_BatchNormalization_if_wf_wf_wf_wf_of(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_BatchNormalization *bnp = lp->param_p;

    float *ip = inp;
    float *op = outp;
    float data;

    int input_size_x;
    int input_size_y;
    int input_size_num;

    int n, x, y;
    int idx_i, idx_o;

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    for(n=0;n<input_size_num;n++) {
        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                data = *(ip + idx_i);
                *(op + idx_o) = data;
            }
        }
    }

    return 1;
}

int CQT_LeakyReLU_if_of(CQT_LAYER *lp, void *inp, void *outp)
{
    return 1;

}

int CQT_Conv2D_same_1x1_if_wf_wf_of(CQT_LAYER *lp, void *inp, void *outp)
{
    return 1;
}
