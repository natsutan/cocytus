#include <string.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"

int CQT_MaxPooling2D_if_of (CQT_LAYER *lp, void *inp, void *outp)
{

    LY_MaxPooling2D *mpp;
    mpp = lp->param_p;
    float *ip = inp;
    float *op = outp;


    assert(mpp->pool_size[0]==2);
    assert(mpp->pool_size[1]==2);
    assert(mpp->strides[0]==2);
    assert(mpp->strides[1]==2);
    assert(mpp->padding==PD_VALID);

	int input_size_x = lp->cqt_input_shape[1];  //入力　x size
	int input_size_y = lp->cqt_input_shape[2];  //入力　y size
	int input_size_num = lp->cqt_input_shape[3];  //入力の数

	int x, y, n;
	int idx_i,idx_o;
	float data[4]; //2x2
	float max;

	for(n=0;n<input_size_num;n++) {
		for(y=0;y<input_size_y;y+=2) {
			for(x=0;x<input_size_x;x+=2){
				idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
				idx_o = n * (input_size_y * input_size_x / 4) + (y * input_size_x / 4) + (x/2);

				data[0] = *(ip + idx_i);
				data[1] = *(ip + idx_i + 1);
				data[2] = *(ip + idx_i + input_size_x);
				data[3] = *(ip + idx_i + input_size_x + 1);

				//max
				max = data[0];
				if(max < data[1]) {
					max = data[1];
				}
				if(max < data[2]) {
					max = data[2];
				}
				if(max < data[3]) {
					max = data[3];
				}

				*(op + idx_o) = max;
			}
		}
	}


    return CQT_RET_OK;
}