int $func_name (CQT_LAYER *lp, void *inp, void *outp)
{

   //loop counter
	int k, l, m, n;

    //コメントは、入力の形が 40x12x12の時
	int k_max = lp->cqt_input_shape[0]; //1
	int l_max = lp->cqt_input_shape[1]; //12
	int m_max = lp->cqt_input_shape[2]; //12
	int n_max = lp->cqt_input_shape[3]; //40
	//assert(k_max!=0);
	assert(l_max!=0);
	assert(m_max!=0);
	assert(n_max!=0);

    if (k_max==0) {
        k_max = 1;
    }

	$input_type *ip = (float *)inp;
	$output_type *op = outp;
	int idx_i;
	int idx_o;
	$output_type data;

	//テンソルの並びを逆にする。3次元でないと計算合わないかも
	assert(k_max==1);
	assert(l_max==m_max);


	idx_o = 0;
	for(k=0;k<k_max;k++) {
		for(l=0;l<l_max;l++) {
			for(m=0;m<m_max;m++) {
				for(n=0;n<n_max;n++) {
					idx_i = (k * l_max * m_max * n_max) + (n * m_max * l_max)  + (l * m_max) + m;
					data = *(ip + idx_i);
					*(op + idx_o) = data;
					idx_o++;
				}
			}
		}
	}


	return CQT_RET_OK;
}
