
int $func_name(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_BatchNormalization *bnp = lp->param_p;

    $input_type *ip = inp;
    $output_type *op = outp;
    $input_type i_data;
    $output_type o_data;

    $weight_type mean;
    $weight_type var;
    $weight_type gamma;
    $weight_type beta;
    $weight_type inv_denomin;

    //SIMD時には、32bitで宣言し演算時に飽和命令を使う。
    int normalized_data;
    int data_sub_mean;
    int normalized_data_pre;
    int mul_gamma;
    int mul_gamma_beta;

    int mean_adj;
    int beta_adj;

    int input_size_x;
    int input_size_y;
    int input_size_num;
    int beta_shift = (lp->weight_q - lp->output_q);
    int inv_beta_shift = -beta_shift;

    int n, x, y;
    int idx_i, idx_o;

    int max, min;

    int mul_shift = lp->input_q + lp->weight_q - lp->output_q;
    assert(bnp->scale==true);
    assert(bnp->center==true);
    assert(sizeof(int) < sizeof(long));

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    max = SHRT_MAX;
    min = SHRT_MIN;

    for(n=0;n<input_size_num;n++) {
        beta = *(($weight_type *)bnp->beta_p + n);
        gamma = *(($weight_type *)bnp->gamma_p + n);
        mean = *(($weight_type *)bnp->moving_mean_p + n);
        var = *(($weight_type *)bnp->moving_variance_p + n);

        //重み変換時に計算済
        //inv_denomin = 1.0 / sqrt(var + bnp->epsilon);
        inv_denomin = var;

        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                i_data = *(ip + idx_i);

                //もともとの計算式
                //normalized_data = (i_data - mean) * inv_denomin;
                //o_data = normalized_data * gamma + beta;

                //ここはinput_qに合わせる
                mean_adj = mean >> (lp->input_q - lp->weight_q);
                data_sub_mean = (i_data - mean_adj);

                //ここからoutpu_qに合わせる
                normalized_data_pre = data_sub_mean * inv_denomin;
                normalized_data  = normalized_data_pre >> mul_shift;

                mul_gamma = (normalized_data * gamma) >> lp->weight_q;
                if (beta_shift >= 0) {
                    beta_adj = beta >> beta_shift;
                } else {
                    beta_adj = beta << inv_beta_shift;
                }
                mul_gamma_beta = mul_gamma + beta_adj;


                if(mul_gamma_beta > max) {
                    lp->overflow_cnt++;
                    o_data = max;
                } else if (mul_gamma_beta < min) {
                    o_data = min;
                    lp->overflow_cnt++;
                } else {
                    o_data = mul_gamma_beta;
                }
                *(op + idx_o) = o_data;
            }
        }
    }

    return CQT_RET_OK;
}
