
int $func_name(CQT_LAYER *lp, void *inp, void *outp)
{
    LY_LeakyReLU *lrp = lp->param_p;

    $input_type *ip = inp;
    $output_type *op = outp;
    $input_type i_data;
    $output_type o_data;

    float alpha;

    int input_size_x;
    int input_size_y;
    int input_size_num;

    int n, x, y;
    int idx_i, idx_o;

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

    alpha = lrp->alpha;

    for(n=0;n<input_size_num;n++) {
        for(y=0;y<input_size_y;y++) {
            for(x=0;x<input_size_x;x++) {
                idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                idx_o = idx_i;
                i_data = *(ip + idx_i);
                if (i_data > 0.0) {
                    o_data = i_data;
                } else {
                    o_data = i_data * alpha;
                }
                *(op + idx_o) = o_data;
            }
        }
    }

    return CQT_RET_OK;
}
