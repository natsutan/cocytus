#include <string.h>
#include <limits.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"
#include "Conv2d_same_3x3.h"


int CQT_conv2d_1_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_1_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_1_3x3_hw(float ip[173056], float op[173056], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<416;y++) {
        for(x=0;x<416;x++) {
            //get data
            idx_o = (y * 416) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 416) + x;
            } else {
            //dummy
                idx_i = (y * 416) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (416 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 416 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (416 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (416 - 1)) {
                idx_i =  (y + 1) * 416 + x;
            } else {
                idx_i =  y * 416 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (416 - 1)) {
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
            if (y == (416 - 1)) {
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

int CQT_conv2d_2_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_2_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_2_3x3_hw(float ip[43264], float op[43264], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<208;y++) {
        for(x=0;x<208;x++) {
            //get data
            idx_o = (y * 208) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 208) + x;
            } else {
            //dummy
                idx_i = (y * 208) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (208 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 208 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (208 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (208 - 1)) {
                idx_i =  (y + 1) * 208 + x;
            } else {
                idx_i =  y * 208 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (208 - 1)) {
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
            if (y == (208 - 1)) {
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

int CQT_conv2d_3_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_3_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_3_3x3_hw(float ip[10816], float op[10816], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<104;y++) {
        for(x=0;x<104;x++) {
            //get data
            idx_o = (y * 104) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 104) + x;
            } else {
            //dummy
                idx_i = (y * 104) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (104 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 104 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (104 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (104 - 1)) {
                idx_i =  (y + 1) * 104 + x;
            } else {
                idx_i =  y * 104 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (104 - 1)) {
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
            if (y == (104 - 1)) {
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

int CQT_conv2d_4_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_4_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_4_3x3_hw(float ip[2704], float op[2704], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<52;y++) {
        for(x=0;x<52;x++) {
            //get data
            idx_o = (y * 52) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 52) + x;
            } else {
            //dummy
                idx_i = (y * 52) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (52 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 52 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (52 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (52 - 1)) {
                idx_i =  (y + 1) * 52 + x;
            } else {
                idx_i =  y * 52 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (52 - 1)) {
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
            if (y == (52 - 1)) {
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

int CQT_conv2d_5_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_5_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_5_3x3_hw(float ip[676], float op[676], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<26;y++) {
        for(x=0;x<26;x++) {
            //get data
            idx_o = (y * 26) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 26) + x;
            } else {
            //dummy
                idx_i = (y * 26) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (26 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 26 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (26 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (26 - 1)) {
                idx_i =  (y + 1) * 26 + x;
            } else {
                idx_i =  y * 26 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (26 - 1)) {
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
            if (y == (26 - 1)) {
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

int CQT_conv2d_6_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_6_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_6_3x3_hw(float ip[169], float op[169], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            idx_o = (y * 13) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 13) + x;
            } else {
            //dummy
                idx_i = (y * 13) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 13 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (13 - 1)) {
                idx_i =  (y + 1) * 13 + x;
            } else {
                idx_i =  y * 13 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
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
            if (y == (13 - 1)) {
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

int CQT_conv2d_7_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_7_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_7_3x3_hw(float ip[169], float op[169], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            idx_o = (y * 13) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 13) + x;
            } else {
            //dummy
                idx_i = (y * 13) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 13 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (13 - 1)) {
                idx_i =  (y + 1) * 13 + x;
            } else {
                idx_i =  y * 13 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
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
            if (y == (13 - 1)) {
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

int CQT_conv2d_8_3x3 (CQT_LAYER *lp, void *inp, void *outp)
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

           CQT_conv2d_8_3x3_hw(ip + idx_i, op  + idx_o, filter3x3, bias, 0, last);

        }
    }
    return CQT_RET_OK;
}

void CQT_conv2d_8_3x3_hw(float ip[169], float op[169], float weight[9], int bias, int act, int last)
{

    float data3x3[3][3];
    float o_data;
    int x, y;
    int idx_i, idx_o;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            idx_o = (y * 13) + x;
            o_data = *(op + idx_o);


            //capture data 1
            if(y != 0) {
                idx_i = ((y-1) * 13) + x;
            } else {
            //dummy
                idx_i = (y * 13) + x;
            }

            if(x != 0) {
                data3x3[0][0] = *(ip + idx_i - 1);
            } else {
                data3x3[0][0] = 0.0;
            }

            data3x3[0][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[0][2] = *(ip + idx_i + 1);
            } else {
                data3x3[0][2] = 0.0;
            }


            //capture data 2
            idx_i = y * 13 + x;
            if(x != 0) {
                data3x3[1][0] = *(ip + idx_i - 1);
            } else {
                data3x3[1][0] = 0.0;
            }

            data3x3[1][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
                data3x3[1][2] = *(ip + idx_i + 1);
            } else {
                data3x3[1][2] = 0.0;
            }

            //capture data 3
            if(y != (13 - 1)) {
                idx_i =  (y + 1) * 13 + x;
            } else {
                idx_i =  y * 13 + x;
            }

            if (x != 0) {
                data3x3[2][0] = *(ip + idx_i - 1);
            } else {
                data3x3[2][0] = 0.0;
            }
            data3x3[2][1] = *(ip + idx_i);

            if (x != (13 - 1)) {
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
            if (y == (13 - 1)) {
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
