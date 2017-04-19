//Conv2d Conv2d_OPTLEVEL = dash version

int $func_name (CQT_LAYER *lp, void *inp, void *outp)
{
	$weight_type  filter3x3[3][3];
	$input_type data3x3[4][4];
	$weight_type bias;

	LY_Conv2D *cnvp;
	cnvp = (LY_Conv2D*)lp->param_p;

	$input_type *ip = ($input_type *)inp;
	$output_type *op = ($output_type *)outp;
	$weight_type  *wp = ($weight_type *)cnvp->weight_p;
	$weight_type  *bp = ($weight_type *)cnvp->bias_p;

	int fill_num = cnvp->filters;
	int input_size_x;
	int input_size_y;
	int input_size_num;

	int f, x, y, n;
	int idx_i,idx_o1,idx_o2;
	$weight_type w_data;
	$output_type o_data[2][2];

    input_size_x = lp->cqt_input_shape[1];  //画像サイズ
    input_size_y = lp->cqt_input_shape[2];  //画像サイズ
    input_size_num = lp->cqt_input_shape[3]; //入力の数

	//parameter check o_data
	assert(cnvp->kernel_size[0]==3);
	assert(cnvp->kernel_size[1]==3);
	assert(cnvp->padding==PD_SAME);
	assert(cnvp->activation==ACT_RELU);
	assert(cnvp->strides[0]==1);
	assert(cnvp->strides[1]==1);
	assert(fill_num==lp->cqt_output_shape[3]);

    memset(op, 0.0, fill_num * input_size_y * input_size_x);

	for(f=0;f<fill_num;f++) {
	    for(n=0;n<input_size_num;n++){
			// get filter
			for(x=0;x<3;x++) {
				for(y=0;y<3;y++) {
					idx_i = (f * input_size_num * 3 * 3) + (n * 3 * 3) + (y * 3) + x;
					w_data = *(wp+idx_i);
					filter3x3[x][y] = w_data;
				}
			}
			bias = *(bp+f);

			//apply filter
			for(y=0;y<input_size_y;y+=2) {
				for(x=0;x<input_size_x;x+=2) {
					//get data(for axis=n)
					idx_o1 = f * (input_size_y * input_size_x) + ( y    * input_size_x) + x;
					idx_o2 = f * (input_size_y * input_size_x) + ((y+1) * input_size_x) + x;
					o_data[0][0] = *(op + idx_o1);			//x  , y
					o_data[0][1] = *(op + idx_o1 + 1);		//x+1, y
					o_data[1][0] = *(op + idx_o2);			//x  , y+1
					o_data[1][1] = *(op + idx_o2 + 1);		//x+1, y+1


					//get data
					idx_i = n * (input_size_y * input_size_x) + ((y-1) * input_size_x) + x;
                    data3x3[0][0] = *(ip + idx_i - 1);		//x-1, y-1
                    data3x3[0][1] = *(ip + idx_i);			//x  , y-1
                    data3x3[0][2] = *(ip + idx_i + 1);		//x+1, y-1
                    data3x3[0][3] = *(ip + idx_i + 2);		//x+2, y-1	//add

                    idx_i = n * (input_size_y * input_size_x) +   y    * input_size_x + x;
                    data3x3[1][0] = *(ip + idx_i - 1);		//x-1, y
                    data3x3[1][1] = *(ip + idx_i);    		//x  , y
                    data3x3[1][2] = *(ip + idx_i + 1);		//x+1, y
                    data3x3[1][3] = *(ip + idx_i + 2);		//x+2, y	//add

                    idx_i = n * (input_size_y * input_size_x) +  (y+1) * input_size_x + x;
                    data3x3[2][0] = *(ip + idx_i - 1);		//x-1, y+1
                    data3x3[2][1] = *(ip + idx_i);    		//x  , y+1
                    data3x3[2][2] = *(ip + idx_i + 1);		//x+1, y+1
                    data3x3[2][3] = *(ip + idx_i + 2);		//x+2, y+1	//add

                    idx_i = n * (input_size_y * input_size_x) +  (y+2) * input_size_x + x;
                    data3x3[3][0] = *(ip + idx_i - 1);		//x-1, y+2
                    data3x3[3][1] = *(ip + idx_i);    		//x  , y+2
                    data3x3[3][2] = *(ip + idx_i + 1);		//x+1, y+2
                    data3x3[3][3] = *(ip + idx_i + 2);		//x+2, y+2	//add

                    //border == 'same
                    //zero padding
                    if (x == 0) {
                        data3x3[0][0] = 0;
                        data3x3[1][0] = 0;
                        data3x3[2][0] = 0;
                        data3x3[3][0] = 0;	//add
                    }
                    if (x == (input_size_x - 2)) {
                        data3x3[0][3] = 0;
                        data3x3[1][3] = 0;
                        data3x3[2][3] = 0;
                        data3x3[3][3] = 0;
                    }
                    if (y == 0) {
                        data3x3[0][0] = 0;
                        data3x3[0][1] = 0;
                        data3x3[0][2] = 0;
                        data3x3[0][3] = 0;	//add
                    }
                    if (y == (input_size_y - 2)) {
                        data3x3[3][0] = 0;
                        data3x3[3][1] = 0;
                        data3x3[3][2] = 0;
                        data3x3[3][3] = 0;
                    }


					//for x, y
					o_data[0][0] += filter3x3[0][0] * data3x3[0][0];
					o_data[0][0] += filter3x3[0][1] * data3x3[0][1];
					o_data[0][0] += filter3x3[0][2] * data3x3[0][2];
					o_data[0][0] += filter3x3[1][0] * data3x3[1][0];
					o_data[0][0] += filter3x3[1][1] * data3x3[1][1];
					o_data[0][0] += filter3x3[1][2] * data3x3[1][2];
					o_data[0][0] += filter3x3[2][0] * data3x3[2][0];
					o_data[0][0] += filter3x3[2][1] * data3x3[2][1];
					o_data[0][0] += filter3x3[2][2] * data3x3[2][2];

					//for x+1, y
					o_data[0][1] += filter3x3[0][0] * data3x3[0][1];
					o_data[0][1] += filter3x3[0][1] * data3x3[0][2];
					o_data[0][1] += filter3x3[0][2] * data3x3[0][3];
					o_data[0][1] += filter3x3[1][0] * data3x3[1][1];
					o_data[0][1] += filter3x3[1][1] * data3x3[1][2];
					o_data[0][1] += filter3x3[1][2] * data3x3[1][3];
					o_data[0][1] += filter3x3[2][0] * data3x3[2][1];
					o_data[0][1] += filter3x3[2][1] * data3x3[2][2];
					o_data[0][1] += filter3x3[2][2] * data3x3[2][3];

					//for x, y+1
					o_data[1][0] += filter3x3[0][0] * data3x3[1][0];
					o_data[1][0] += filter3x3[0][1] * data3x3[1][1];
					o_data[1][0] += filter3x3[0][2] * data3x3[1][2];
					o_data[1][0] += filter3x3[1][0] * data3x3[2][0];
					o_data[1][0] += filter3x3[1][1] * data3x3[2][1];
					o_data[1][0] += filter3x3[1][2] * data3x3[2][2];
					o_data[1][0] += filter3x3[2][0] * data3x3[3][0];
					o_data[1][0] += filter3x3[2][1] * data3x3[3][1];
					o_data[1][0] += filter3x3[2][2] * data3x3[3][2];

					//for x+1, y+1
					o_data[1][1] += filter3x3[0][0] * data3x3[1][1];
					o_data[1][1] += filter3x3[0][1] * data3x3[1][2];
					o_data[1][1] += filter3x3[0][2] * data3x3[1][3];
					o_data[1][1] += filter3x3[1][0] * data3x3[2][1];
					o_data[1][1] += filter3x3[1][1] * data3x3[2][2];
					o_data[1][1] += filter3x3[1][2] * data3x3[2][3];
					o_data[1][1] += filter3x3[2][0] * data3x3[3][1];
					o_data[1][1] += filter3x3[2][1] * data3x3[3][2];
					o_data[1][1] += filter3x3[2][2] * data3x3[3][3];


					//bais
					if(n==(input_size_num-1)) {
    					o_data[0][0] += bias;
    					o_data[0][1] += bias;
    					o_data[1][0] += bias;
    					o_data[1][1] += bias;
                    }

                    //activattion
					if(n==(input_size_num-1)) {

					    if(o_data[0][0] < 0) { o_data[0][0] = 0.0; }
					    if(o_data[0][1] < 0) { o_data[0][1] = 0.0; }
					    if(o_data[1][0] < 0) { o_data[1][0] = 0.0; }
					    if(o_data[1][1] < 0) { o_data[1][1] = 0.0; }

                    }

					*(op + idx_o1)   = o_data[0][0];
					*(op + idx_o1+1) = o_data[0][1];
					*(op + idx_o2)   = o_data[1][0];
					*(op + idx_o2+1) = o_data[1][1];
				}
			}
		}
	}
    return CQT_RET_OK;
}

