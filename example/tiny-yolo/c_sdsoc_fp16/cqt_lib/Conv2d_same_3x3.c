#include <string.h>
#include <limits.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"
#include "Conv2d_same_3x3.h"


int CQT_conv2d_1_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_1_3x3_hw(FP16 ip[173056], FP16 op[173056], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[416][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<416;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<416;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<416;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<416;y++) {
        for(x=0;x<416;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(416 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(416 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(416 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (416 - 1)) {
            for(li=0;li<416;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<416;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_2_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_2_3x3_hw(FP16 ip[43264], FP16 op[43264], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[208][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<208;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<208;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<208;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<208;y++) {
        for(x=0;x<208;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(208 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(208 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(208 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (208 - 1)) {
            for(li=0;li<208;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<208;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_3_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_3_3x3_hw(FP16 ip[10816], FP16 op[10816], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[104][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<104;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<104;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<104;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<104;y++) {
        for(x=0;x<104;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(104 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(104 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(104 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (104 - 1)) {
            for(li=0;li<104;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<104;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_4_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_4_3x3_hw(FP16 ip[2704], FP16 op[2704], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[52][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<52;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<52;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<52;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<52;y++) {
        for(x=0;x<52;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(52 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(52 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(52 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (52 - 1)) {
            for(li=0;li<52;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<52;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_5_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_5_3x3_hw(FP16 ip[676], FP16 op[676], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[26][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<26;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<26;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<26;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<26;y++) {
        for(x=0;x<26;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(26 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(26 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(26 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (26 - 1)) {
            for(li=0;li<26;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<26;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_6_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_6_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[13][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<13;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (13 - 1)) {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_7_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_7_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[13][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<13;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (13 - 1)) {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}

int CQT_conv2d_8_3x3 (CQT_LAYER *lp, void *inp, void *outp)
{
    FP16 filter3x3[3*3];
    FP16 bias;

    LY_Conv2D *cnvp;
    cnvp = lp->param_p;

    FP16 *ip = (FP16 *)inp;
    FP16 *op = outp;
    FP16 *wp = cnvp->weight_p;
    FP16 *bp = cnvp->bias_p;

    int fill_num = cnvp->filters;
    int input_size_x;
    int input_size_y;
    int input_size_num;

    int f, x, y, n;
    int idx_i,idx_o;

    FP16 w_data;
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

    memset(op, 0.0, fill_num * input_size_y * input_size_x * sizeof(FP16));

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

void CQT_conv2d_8_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last)
{

    FP16 data3x3[3][3];
    FP16 o_data;
    int x, y;
    int idx_i, idx_o;
    int li; // for line buffer
    int write_buf_idx; //次に書き込むラインバッファ 2bit
    int read_buf_idx0; //読み出し位置のインデックス 2bit
    int read_buf_idx1; //読み出し位置のインデックス 2bit
    int read_buf_idx2; //読み出し位置のインデックス 2bit

    idx_o = 0;
    idx_i = 0;

    static FP16 line_buffer[13][4]; // line-buffers
    #pragma HLS ARRAY_PARTITION variable=line_buffer block factor=4

    for(li=0;li<13;li++) {
        line_buffer[li][3] = 0;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][1] = *(ip + idx_i);
        idx_i++;
    }

    for(li=0;li<13;li++) {
        line_buffer[li][2] = *(ip + idx_i);
        idx_i++;
    }

    write_buf_idx = 3;

    //apply filter
    for(y=0;y<13;y++) {
        for(x=0;x<13;x++) {
            //get data
            o_data = *(op + idx_o);

            //メモリのラインバッファ選択
            switch (y % 4) {
                case 0:
                    read_buf_idx0 = 0;
                    read_buf_idx1 = 1;
                    read_buf_idx2 = 2;
                    break;
                case 1:
                    read_buf_idx0 = 1;
                    read_buf_idx1 = 2;
                    read_buf_idx2 = 3;
                    break;
                case 2:
                    read_buf_idx0 = 2;
                    read_buf_idx1 = 3;
                    read_buf_idx2 = 0;
                    break;
                default:  //case 3
                    read_buf_idx0 = 3;
                    read_buf_idx1 = 0;
                    read_buf_idx2 = 1;
                    break;
            }

            // データ選択
            data3x3[0][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx0];
            data3x3[0][1] = line_buffer[x][read_buf_idx0];
            data3x3[0][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx0];

            data3x3[1][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx1];
            data3x3[1][1] = line_buffer[x][read_buf_idx1];
            data3x3[1][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx1];

            data3x3[2][0] = (x==0) ? 0 : line_buffer[x-1][read_buf_idx2];
            data3x3[2][1] = line_buffer[x][read_buf_idx2];
            data3x3[2][2] = (x==(13 - 1)) ? 0 : line_buffer[x+1][read_buf_idx2];

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
            idx_o++;

        }
        //次のデータの書き込み
        //パラレル化可能
        if (y != (13 - 1)) {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = *(ip + idx_i);
                idx_i++;
            }
        } else {
            for(li=0;li<13;li++) {
                line_buffer[li][write_buf_idx] = 0;
            }
        }

        if (write_buf_idx == 3) {
            write_buf_idx = 0;
        } else {
            write_buf_idx++;
        }
    }

}
