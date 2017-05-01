int $func_name (CQT_LAYER *lp, void *inp, void *outp)
{

    LY_MaxPooling2D *mpp;
    mpp = lp->param_p;
    $input_type *ip = inp;
    $output_type *op = outp;


    assert(mpp->pool_size[0]==2);
    assert(mpp->pool_size[1]==2);

    int input_size_x = lp->cqt_input_shape[1];  //入力　x size
    int input_size_y = lp->cqt_input_shape[2];  //入力　y size
    int input_size_num = lp->cqt_input_shape[3];  //入力の数

    int x, y, n;
    int idx_i,idx_o;
    $input_type data[4]; //2x2
    $output_type max;

    bool stride_1_padding_same_mode = false;

    //パラメータチェック
    //strideが1x1, padding=PD_SAMEの時は別ルーチンを通る。
    //場合分けが増えてきたら別ファイルへ分離
    if((mpp->strides[0]==1) && (mpp->strides[1]==1) && (mpp->padding==PD_SAME)) {
        assert(mpp->pool_size[0]==2);
        assert(mpp->pool_size[1]==2);
        stride_1_padding_same_mode = true;
    } else {
        assert(mpp->strides[0]==2);
        assert(mpp->strides[1]==2);

        if(mpp->padding!=PD_VALID) {
            //領域が2x2のプーピングで、サイズが偶数であれば、パディングが発生せず、
            //PD_SAMEでも同じ処理を実行する。そうでなければ、assertで落ちる。
            assert(mpp->padding==PD_SAME);
            assert(input_size_x%2==0);
            assert(input_size_y%2==0);
        }
    }


    if(stride_1_padding_same_mode == false) {
        //通常の処理
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
    } else {
        //stride_1_padding_same_mode
        assert(stride_1_padding_same_mode);
        for(n=0;n<input_size_num;n++) {
            for(y=0;y<input_size_y;y++) {
                for(x=0;x<input_size_x;x++){
                    idx_i = (n * input_size_y * input_size_x) + (y * input_size_x) + x;
                    idx_o = idx_i;

                    if(y==(input_size_y-1)) {
                        //一番下
                        data[0] = *(ip + idx_i);
                        if(x==(input_size_x-1)) {
                            data[1] = data[0];
                        } else {
                            data[1] = *(ip + idx_i + 1);
                        }
                        data[2] = data[0];
                        data[3] = data[0];
                    } else if (x==(input_size_x-1)) {
                        //右端
                        data[0] = *(ip + idx_i);
                        data[1] = data[0];
                        data[2] = *(ip + idx_i + input_size_x);
                        data[3] = data[0];
                    } else {
                        //通常の処理
                        data[0] = *(ip + idx_i);
                        data[1] = *(ip + idx_i + 1);
                        data[2] = *(ip + idx_i + input_size_x);
                        data[3] = *(ip + idx_i + input_size_x + 1);
                    }

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
    }
    return CQT_RET_OK;
}