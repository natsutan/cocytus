int $func_name (CQT_LAYER *lp, void *inp, void *outp)
{
    LY_Dense *dp = lp->param_p;

    int f, n;

    int filter_num = lp->cqt_output_shape[1];
    int input_num = lp->cqt_input_shape[1];

    $input_type *ip = ($input_type *)inp;
    $output_type *op = outp;
    $weight_type *wp = dp->weight_p;
    $weight_type *bp = dp->bias_p;

    int idx_i, idx_o;
    $input_type data;
    $weight_type weight;
    $weight_type bias;
    $output_type accumulator;

    int max, min;
    int acc;

    int mul_shift = lp->input_q + lp->weight_q - lp->output_q;
    int add_shift = lp->weight_q - lp->output_q;

    assert((dp->activation == ACT_RELU) || (dp->activation == ACT_SOFTMAX) || (dp->activation == ACT_LINEAR));
    assert(dp->use_bias == true);

    max = SHRT_MAX;
    min = SHRT_MIN;
    memset(op, 0.0, filter_num);

    for(f=0;f<filter_num;f++) {
        accumulator = 0;
        for(n=0;n<input_num;n++) {
            idx_i = n;
            data = *(ip + idx_i );
            weight = *(wp + (f * input_num) + idx_i);
            acc = (data * weight);

            if (acc>max) {
                acc = max;
            } else if (acc < min) {
                acc = min;
            }

            accumulator += (acc >> mul_shift);
        }
        bias = *(bp + f);
        accumulator += (bias >> add_shift);

         //activattion fillter
         //ACT_LINEARの時は何も計算をしない
         if(dp->activation == ACT_RELU) {
            if(accumulator < 0) {
                accumulator = 0.0;
            }
         }

        idx_o = f;

        *(op + idx_o) = accumulator;
    }

    //固定少数点時、ソフトマックスは呼び出し側で行う。


    return CQT_RET_OK;
}


