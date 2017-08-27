#include <string.h>
#include <limits.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"


int CQT_Conv2D_same_3x3_if_wf_of (CQT_LAYER *lp, void *inp, void *outp)
{
    float filter3x3[3][3];
    float data3x3[3][3];
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
    int idx_i,idx_o;
    float w_data;
    float o_data;

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    padding = lp->neon_padding_h;

    data_size_x = NEON_HTR + input_size_x + padding; //確保している画像サイズ
    data_size_y = input_size_y + NEON_VTR * 3; //確保している画像サイズ


    //parameter check o_data
    assert(cnvp->kernel_size[0]==3);
    assert(cnvp->kernel_size[1]==3);
    assert(cnvp->padding==PD_SAME);
    assert(cnvp->strides[0]==1);
    assert(cnvp->strides[1]==1);
    assert(fill_num==lp->cqt_output_shape[3]);

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(float));

    for(f=0;f<fill_num;f++) {
        for(n=0;n<input_size_num;n++){
            // get filter
            for(x=0;x<3;x++) {
                for(y=0;y<3;y++) {
                    idx_i = (f * input_size_num * 3 * 3) + (n * 3 * 3) + (y * 3) + x;
                    w_data = *(wp+idx_i);
                    filter3x3[x][y] = w_data;
                }
            }
            bias = *(bp+f);

            //apply filter
            for(y=0;y<input_size_y;y++) {
                for(x=0;x<input_size_x;x++) {
                    //get data
                    idx_i = n * (input_size_y * input_size_x) + ((y-1) * input_size_x) + x;
                    idx_o = f * (input_size_y * input_size_x) + (y * input_size_x) + x;
                    o_data = *(op + idx_o);

                    data3x3[0][0] = *(ip + idx_i - 1);
                    data3x3[0][1] = *(ip + idx_i);
                    data3x3[0][2] = *(ip + idx_i + 1);

                    idx_i = n * (input_size_y * input_size_x) + y * input_size_y + x;
                    data3x3[1][0] = *(ip + idx_i - 1);
                    data3x3[1][1] = *(ip + idx_i);
                    data3x3[1][2] = *(ip + idx_i + 1);

                    idx_i = n * (input_size_y * input_size_x) + (y + 1) * input_size_y + x;
                    data3x3[2][0] = *(ip + idx_i - 1);
                    data3x3[2][1] = *(ip + idx_i);
                    data3x3[2][2] = *(ip + idx_i + 1);

                    //border == 'same
                    //zero padding
                    if (x == 0) {
                        data3x3[0][0] = 0;
                        data3x3[1][0] = 0;
                        data3x3[2][0] = 0;
                    }
                    if (x == (input_size_x - 1)) {
                        data3x3[0][2] = 0;
                        data3x3[1][2] = 0;
                        data3x3[2][2] = 0;
                    }
                    if (y == 0) {
                        data3x3[0][0] = 0;
                        data3x3[0][1] = 0;
                        data3x3[0][2] = 0;
                    }
                    if (y == (input_size_y - 1)) {
                        data3x3[2][0] = 0;
                        data3x3[2][1] = 0;
                        data3x3[2][2] = 0;
                    }


                    o_data += filter3x3[0][0] * data3x3[0][0];
                    o_data += filter3x3[0][1] * data3x3[0][1];
                    o_data += filter3x3[0][2] * data3x3[0][2];
                    o_data += filter3x3[1][0] * data3x3[1][0];
                    o_data += filter3x3[1][1] * data3x3[1][1];
                    o_data += filter3x3[1][2] * data3x3[1][2];
                    o_data += filter3x3[2][0] * data3x3[2][0];
                    o_data += filter3x3[2][1] * data3x3[2][1];
                    o_data += filter3x3[2][2] * data3x3[2][2];

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
