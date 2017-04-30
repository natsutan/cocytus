// compileを通すためのダミーファイル
#include "cqt.h"
#include "cqt_net.h"

#include <math.h>
#include <assert.h>

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

    int n, x, y;
    int idx_i, idx_o;

    assert(bnp->scale==true);
    assert(bnp->center==true);

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    //bn = (i_f - i_mean) / math.sqrt(i_var + epsilon)


    for(n=0;n<input_size_num;n++) {
        beta = *((float *)bnp->beta_p + n);
        gamma = *((float *)bnp->gamma_p + n);
        mean = *((float *)bnp->moving_mean_p + n);
        var = *((float *)bnp->moving_variance_p + n);

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
