#include <string.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"


int CQT_Conv2D_same_1x1_if_wf_wf_of (CQT_LAYER *lp, void *inp, void *outp)
{
    float filter;
    float data;
    float bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    float *ip = (float *)inp;
    float *op = outp;
    float *wp = cnvp->weight_p;
    float *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;
    int data_size_x;
    int data_size_y;
    int padding;

    int f, x, y, n;
    int idx_i,idx_o, idx_w;
    float o_data;

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    //parameter check o_data
    assert(cnvp->kernel_size[0]==1);
    assert(cnvp->kernel_size[1]==1);
    assert(cnvp->padding==PD_SAME);
    assert(cnvp->strides[0]==1);
    assert(cnvp->strides[1]==1);
    assert(fill_num==lp->cqt_output_shape[3]);

    padding = lp->neon_padding_hi;

    data_size_x = NEON_HTR + input_size_x + padding; //確保している画像サイズ
    data_size_y = input_size_y + NEON_VTR * 3; //確保している画像サイズ

    memset(op, 0.0, fill_num * data_size_y * data_size_x * sizeof(float));

    for(f=0;f<fill_num;f++) {
        for(n=0;n<input_size_num;n++){
            idx_w = (f * input_size_num) + n;
            filter = *(wp+idx_w);
            bias = *(bp+f);

            //apply filter
            for(y=0;y<input_size_y;y++) {
                for(x=0;x<input_size_x;x++) {
                    //get data
                    idx_i = n * (data_size_y * data_size_x) + ((y + NEON_VTR) * data_size_x) + (x + NEON_HTR);
                    idx_o = f * (data_size_y * data_size_x) + ((y + NEON_VTR) * data_size_x) + (x + NEON_HTR);

                    o_data = *(op + idx_o);
                    data = *(ip + idx_i);

                    o_data += filter * data;

                    if(n==(input_size_num-1)) {
                        //bais
                        if(cnvp->use_bias) {
                                o_data += bias;
                        }

                        //activattion
                        if(cnvp->activation == ACT_RELU) {
                            if(o_data < 0) {
                                o_data = 0.0;
                            }
                        }
                    }

                    *(op + idx_o) = o_data;
                }
            }
        }
    }
    return CQT_RET_OK;
}
