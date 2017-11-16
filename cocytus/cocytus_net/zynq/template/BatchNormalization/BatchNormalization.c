
int $func_name(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_BatchNormalization *bnp = lp->param_p;

    $input_type *ip = inp;
    $output_type *op = outp;
    $input_type i_data;
    $output_type o_data;

    //  A = mean - (beta * sqrt(variance + epsilon))
    //  B = gamma / sqrt(variance + epsilon)
    //  BNの計算は (X - A) * B で求められる。
    $weight_type A;
    $weight_type B;

    int input_size_x;
    int input_size_y;
    int input_size_num;

    int n, x, y;
    int idx_i, idx_o;

    assert(bnp->scale==true);
    assert(bnp->center==true);

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    for(n=0;n<input_size_num;n++) {
        A = *(($weight_type *)bnp->beta_p + (n * 2));
        B = *(($weight_type *)bnp->beta_p + (n * 2) + 1);

        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                i_data = *(ip + idx_i);

                o_data = (i_data - A) * B;
                *(op + idx_o) = o_data;
            }
        }
    }

    return CQT_RET_OK;
}
