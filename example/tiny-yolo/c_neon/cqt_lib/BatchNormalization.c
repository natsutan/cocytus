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
    float normalized_data;
    float o_data;

    float mean;
    float var;
    float gamma;
    float beta;
    float inv_denomin;

    int input_size_x;
    int input_size_y;
    int input_size_num;
    int data_size_x;
    int data_size_y;
    int padding;

    int n, x, y;
    int idx_i, idx_o;

    assert(bnp->scale==true);
    assert(bnp->center==true);

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    padding = lp->neon_padding_hi;

    data_size_x = NEON_HTR + input_size_x + padding; //確保している画像サイズ
    data_size_y = input_size_y + NEON_VTR * 3; //確保している画像サイズ

    for(n=0;n<input_size_num;n++) {
        beta = *((float *)bnp->beta_p + n);
        gamma = *((float *)bnp->gamma_p + n);
        mean = *((float *)bnp->moving_mean_p + n);
        var = *((float *)bnp->moving_variance_p + n);

        inv_denomin = 1.0 / sqrt(var + bnp->epsilon);

        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = n * (data_size_y * data_size_x) + ((y + NEON_VTR) * data_size_x) + (x + NEON_HTR);
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
