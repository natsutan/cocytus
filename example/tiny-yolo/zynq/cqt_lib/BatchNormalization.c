#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include "cqt.h"
#include "cqt_net.h"


int CQT_BatchNormalization_if_wf_wf_wf_wf_of(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_BatchNormalization *bnp = lp->param_p;

    float *ip = inp;
    float *op = outp;
    float i_data;
    float o_data;

    //  A = mean - (beta * sqrt(variance + epsilon))
    //  B = gamma / sqrt(variance + epsilon)
    //  BNの計算は (X - A) * B で求められる。
    float A;
    float B;

    int input_size_x;
    int input_size_y;
    int input_size_num;

    int n, x, y;
    int idx_i, idx_o;

    assert(bnp->scale==true);
    assert(bnp->center==true);

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    for(n=0;n<input_size_num;n++) {
        A = *((float *)bnp->beta_p + (n * 2));
        B = *((float *)bnp->beta_p + (n * 2) + 1);

        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                i_data = *(ip + idx_i);

                o_data = (i_data - A) * B;
                *(op + idx_o) = o_data;
            }
        }
    }

    return CQT_RET_OK;
}
