#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include "cqt.h"
#include "cqt_net.h"


int CQT_BatchNormalization_ifp16_wfp16_wfp16_wfp16_wfp16_ofp16(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_BatchNormalization *bnp = lp->param_p;

    FP16 *ip = inp;
    FP16 *op = outp;
    FP16 i_data;
    FP16 normalized_data;
    FP16 o_data;

    FP16 mean;
    FP16 var;
    FP16 gamma;
    FP16 beta;
    FP16 inv_denomin;

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
        beta = *((FP16 *)bnp->beta_p + n);
        gamma = *((FP16 *)bnp->gamma_p + n);
        mean = *((FP16 *)bnp->moving_mean_p + n);
        var = *((FP16 *)bnp->moving_variance_p + n);

        inv_denomin = 1.0 / sqrt(var + bnp->epsilon);

        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                i_data = *(ip + idx_i);

                normalized_data = (i_data - mean) * inv_denomin;
                o_data = normalized_data * gamma + beta;
                *(op + idx_o) = o_data;
            }
        }
    }

    return CQT_RET_OK;
}
