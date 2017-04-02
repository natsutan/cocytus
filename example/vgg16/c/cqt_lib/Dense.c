#include <string.h>
#include <assert.h>
#include <math.h>
#include "cqt.h"
#include "cqt_net.h"

int CQT_Dense_if_of (CQT_LAYER *lp, void *inp, void *outp)
{
	LY_Dense *dp = lp->param_p;

	int f, n;

	int filter_num = lp->cqt_output_shape[1];
	int input_num = lp->cqt_input_shape[1];

	float *ip = (float *)inp;
	float *op = outp;
	float *wp = dp->weight_p;
	float *bp = dp->bias_p;

	int idx_i, idx_o;
	float data;
	float weight;
	float bias;
	float accumulator;
	float sm_data;

    float sum = 0.0;

	assert((dp->activation == ACT_RELU) || (dp->activation == ACT_SOFTMAX) || (dp->activation == ACT_LINEAR));
	assert(dp->use_bias == true);

    memset(op, 0.0, filter_num);

	for(f=0;f<filter_num;f++) {
		accumulator = 0;
		for(n=0;n<input_num;n++) {
			idx_i = n;
			data = *(ip + idx_i );
			weight = *(wp + (f * input_num) + idx_i);
			accumulator += data * weight;
		}
        bias = *(bp + f);
        accumulator += bias;

         //activattion fillter
         //ACT_LINEARの時は何も計算をしない
         if(dp->activation == ACT_RELU) {
            if(accumulator < 0) {
                accumulator = 0.0;
            }
         } else if (dp->activation == ACT_SOFTMAX) {
            //一度exp(accumulator)を書き出す。
            accumulator = (float)exp(accumulator);
            sum += accumulator;
         }

		idx_o = f;

		*(op + idx_o) = accumulator;
	}

    //softmax出力時は、出力用の配列の値を再度書き換える。
    if(dp->activation == ACT_SOFTMAX) {
    	for(f=0;f<filter_num;f++) {
            idx_o = f;
            sm_data = *(op + idx_o);
            *(op + idx_o) = sm_data / sum;
        }
    }

	return CQT_RET_OK;
}


