#include <assert.h>
#include <memory.h>

#include "cqt.h"
#include "cqt_net.h"

// InputLayer
int CQT_InputLayer_if_of (CQT_LAYER *np, void *inp, void *outp)
{

    int fill_num;
    int input_size_x;
    int input_size_y;
    int output_size_x;
    int output_size_y;

    int x, y, f;
    float data;
    int idx_i;
    int idx_o;

    float *ip;
    float *op;

    fill_num = np->cqt_input_shape[3];
    input_size_x = np->cqt_input_shape[1];
    input_size_y = np->cqt_input_shape[2];

    output_size_x = NEON_HTR + input_size_x + NEON_HPADDING_0;
    output_size_y = input_size_y + NEON_VTR * 3;

    ip = (float *)inp;
    op = (float *)outp;

    //inputレイヤーはパディングなしのみ対応
    assert(input_size_x % 4 == 0);

    memset(op, 0.0, fill_num * output_size_y * output_size_x * sizeof(float));


    for(f=0;f<fill_num;f++) {
        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = f * input_size_y * input_size_x + y * input_size_x + x;
                idx_o = f * output_size_y * output_size_x + (y + NEON_VTR) * output_size_x + (x + NEON_HTR);
                data = *(ip + idx_i);
                *(op + idx_o) = data;
            }
        }
    }

    return CQT_RET_OK;
}