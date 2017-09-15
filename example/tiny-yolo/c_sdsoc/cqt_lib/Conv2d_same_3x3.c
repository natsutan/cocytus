#include <string.h>
#include <limits.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"
#include "../cqt_gen/cqt_gen.h"

void CQT_Conv2D_same_3x3_if_wf_of_hw(float ip[MAX_OUT_SIZE], float op[MAX_OUT_SIZE], float weight[9], float bias, int input_size_x, int input_size_y, int act, int last);

int CQT_Conv2D_same_3x3_if_wf_of (CQT_LAYER *lp, void *inp, void *outp)
{
    float filter3x3[3*3];
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

    int f, x, y, n;
    int idx_i,idx_o;

    float w_data;
    int last;

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

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
                    filter3x3[(y * 3) + x] = w_data;
                }
            }

            if(cnvp->use_bias) {
                bias = *(bp+f);
            } else {
                bias = 0.0;
            }

            if (n==(input_size_num-1)) {
                last = 1;
            } else {
                last = 0;
            }

           idx_i = n * (input_size_y * input_size_x);
           idx_o = f * (input_size_y * input_size_x);

           CQT_Conv2D_same_3x3_if_wf_of_hw(ip + idx_i, op  + idx_o, filter3x3, bias, input_size_x, input_size_y, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_Conv2D_same_3x3_if_wf_of_hw(float ip[MAX_OUT_SIZE], float op[MAX_OUT_SIZE], float weight[9], float bias, int input_size_x, int input_size_y, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<input_size_y;y++) {
        for(x=0;x<input_size_x;x++) {
            //get data
            idx_o = (y * input_size_x) + x;
            o_data = *(op + idx_o);

            if(y != 0) {
                idx_i = ((y-1) * input_size_x) + x;
            } else {
                idx_i = (y * input_size_x) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (input_size_x - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }
            idx_i = y * input_size_y + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);
            if (x != (input_size_x - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            if(y != (input_size_y - 1)) {
                idx_i =  (y + 1) * input_size_y + x;
            } else {
                idx_i =  y * input_size_y + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (input_size_x - 1)) {
                data3x3[2][2] = *(ip + idx_i + 1);
            } else {
                data3x3[2][2] = 0.0;
            }

            //border == 'same
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

            o_data += weight[0] * data3x3[0][0];
            o_data += weight[3] * data3x3[0][1];
            o_data += weight[6] * data3x3[0][2];
            o_data += weight[1] * data3x3[1][0];
            o_data += weight[4] * data3x3[1][1];
            o_data += weight[7] * data3x3[1][2];
            o_data += weight[2] * data3x3[2][0];
            o_data += weight[5] * data3x3[2][1];
            o_data += weight[8] * data3x3[2][2];

            if(last) {
                 o_data += bias;

                //activattion
                if(act == ACT_RELU) {
                    if(o_data < 0) {
                        o_data = 0.0;
                    }
                }
            }

            *(op + idx_o) = o_data;
        }

    }

}
