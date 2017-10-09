#include <string.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"
#include <omp.h>

/// Super version
//#include <emmintrin.h> 	///SSE2
//#include <pmmintrin.h> 	//SSE3
//#include <tmmintrin.h> 	//SSSE3
//#include <smmintrin.h> 	//SSE4.1
//#include <nmmintrin.h> 	//SSE4.2
//#include <wmmintrin.h> 	//AES
//#include <immintrin.h> 	//AVX, AVX2, FMA
//#include <zmmintrin.h> 	//AVX-512
#include "SSE2NEON.h"


int CQT_Conv2D_same_3x3_if_wf_of (CQT_LAYER *lp, void *inp, void *outp)
///int CQT_Conv2D_same_3x3_if_of_step4x2_SSE_v2 (CQT_LAYER *lp, void *inp, void *outp)
{
	// *** raspi :  36.0810 sec use mla+openmp exp ok!!! *** //
	// *** raspi :  97.4727 sec use mla *** //
	// *** raspi : 110.7601 sec use mul+add *** //
	// *** speed : 5.054 *** //

	LY_Conv2D *cnvp;
	cnvp = (LY_Conv2D*)lp->param_p;

	float *ip = (float *)inp;
	float *op = (float *)outp;
	float *wp = (float *)cnvp->weight_p;
	float *bp = (float *)cnvp->bias_p;

	int fill_num = cnvp->filters;
	int input_size_x;
	int input_size_y;
	int input_size_num;

	int f;///, x, y, n;
///	int idx_i,idx_o;

	input_size_x = lp->cqt_input_shape[1];  //画像サイズ
	input_size_y = lp->cqt_input_shape[2];  //画像サイズ
	input_size_num = lp->cqt_input_shape[3]; //入力の数
	int input_h_padding = lp->neon_padding_hi;
	int output_h_padding = lp->neon_padding_ho;


	int input_next_line  = 4 + input_size_x +  input_h_padding;
	int output_next_line = 4 + input_size_x + output_h_padding;
	int input_next_frame  = (4 + input_size_x +  input_h_padding) * (2 + input_size_y) + 4;
	int output_next_frame = (4 + input_size_x + output_h_padding) * (2 + input_size_y) + 4;

	int output_memsize = fill_num * output_next_frame; 
	
	//parameter check o_data
	assert(cnvp->kernel_size[0]==3);
	assert(cnvp->kernel_size[1]==3);
	assert(cnvp->padding==PD_SAME);
	assert(cnvp->strides[0]==1);
	assert(cnvp->strides[1]==1);
	assert(fill_num==lp->cqt_output_shape[3]);

///    memset(op, 0.0, fill_num * input_size_y * input_size_x);
///    memset(op, 0.0, output_memsize );
//	int i =  output_memsize / sizeof(float);	//ok fast?
//	float *tmp = op;
//	while( i-- > 0 ){ *tmp++ = 0; }
	memset(op, 0, output_memsize * sizeof(float));

	
	#pragma omp parallel for	//segment fault
///	#pragma omp parallel for private(n,y,x)	//exp error and slow
///	#pragma omp parallel for private(n)	//segmento fault
	for(f=0;f<fill_num;f++) {

		int n,y,x;	//for memory leak at openmp
		int idx_i,idx_o1,idx_o2;
		int idx_L0,idx_L1,idx_L2,idx_L3;

		__m128	_data3x3_L0_b1, _data3x3_L0_nw, _data3x3_L0_f1;
		__m128	_data3x3_L1_b1, _data3x3_L1_nw, _data3x3_L1_f1;
		__m128	_data3x3_L2_b1, _data3x3_L2_nw, _data3x3_L2_f1;
		__m128	_data3x3_L3_b1, _data3x3_L3_nw, _data3x3_L3_f1;
		__m128	_o_data1;
		__m128	_o_data2;

		__m128	_flt_L0w0;
		__m128	_flt_L0w1;
		__m128	_flt_L0w2;
		__m128	_flt_L1w0;
		__m128	_flt_L1w1;
		__m128	_flt_L1w2;
		__m128	_flt_L2w0;
		__m128	_flt_L2w1;
		__m128	_flt_L2w2;

		__m128	_fo1;
		__m128	_fo2;
		__m128	_fo3;

		__m128	_bias;

		__m128	_fltw0361;
		__m128	_fltw4725;
		__m128	_fltw8xxx;

	    for(n=0;n<input_size_num;n++){
			// get filter
			idx_i = (f * input_size_num * 3 * 3) + (n * 3 * 3);
			// weight : w0 w1 w2, w3 w4 w5, w6 w7 w8
			//+0: w0w3w6w1 w4w7w2w5 w8

			_fltw0361 = _mm_load_ps( wp+idx_i+0);
			_fltw4725 = _mm_load_ps( wp+idx_i+4);
			_fltw8xxx = _mm_load_ps( wp+idx_i+8);

			_flt_L0w0  = /*w0*/ _mm_shuffle_ps( _fltw0361, _fltw0361, _MM_SHUFFLE(0,0,0,0) );
			_flt_L0w1  = /*w1*/ _mm_shuffle_ps( _fltw0361, _fltw0361, _MM_SHUFFLE(3,3,3,3) );
			_flt_L0w2  = /*w2*/ _mm_shuffle_ps( _fltw4725, _fltw4725, _MM_SHUFFLE(2,2,2,2) );
			_flt_L1w0  = /*w3*/ _mm_shuffle_ps( _fltw0361, _fltw0361, _MM_SHUFFLE(1,1,1,1) );
			_flt_L1w1  = /*w4*/ _mm_shuffle_ps( _fltw4725, _fltw4725, _MM_SHUFFLE(0,0,0,0) );
			_flt_L1w2  = /*w5*/ _mm_shuffle_ps( _fltw4725, _fltw4725, _MM_SHUFFLE(3,3,3,3) );
			_flt_L2w0  = /*w6*/ _mm_shuffle_ps( _fltw0361, _fltw0361, _MM_SHUFFLE(2,2,2,2) );
			_flt_L2w1  = /*w7*/ _mm_shuffle_ps( _fltw4725, _fltw4725, _MM_SHUFFLE(1,1,1,1) );
			_flt_L2w2  = /*w8*/ _mm_shuffle_ps( _fltw8xxx, _fltw8xxx, _MM_SHUFFLE(0,0,0,0) );

			_bias = _mm_load_ps( bp+f );	//ld bias [0]
			_bias = _mm_shuffle_ps( _bias, _bias, 0 );	//bias[0],[0],[0],[0]


			//apply filter
			for(y=0;y<input_size_y;y+=2) {
				for(x=0;x<input_size_x;x+=4) {

					//get data
///					idx_o = f * output_next_frame + ((y+1) * output_next_line ) + x + 4;
///					o_data = *(op + idx_o);
					idx_o1 = f * output_next_frame + ((y+1) * output_next_line ) + x + 4;
					idx_o2 = f * output_next_frame + ((y+2) * output_next_line ) + x + 4;
					_o_data1 = _mm_load_ps( op + idx_o1 );
					_o_data2 = _mm_load_ps( op + idx_o2 );

///					idx_i = n * (input_size_y * input_size_x) + ((y-1) * input_size_x) + x;
					idx_L0 = n * input_next_frame  + ((y+0) * input_next_line ) + x + 4;
					idx_L1 = n * input_next_frame  + ((y+1) * input_next_line ) + x + 4;
					idx_L2 = n * input_next_frame  + ((y+2) * input_next_line ) + x + 4;
					idx_L3 = n * input_next_frame  + ((y+3) * input_next_line ) + x + 4;

///	if( (output_memsize / sizeof(float)) <= idx_L3 ){
///		printf(" error @@ %d,%d,%d,%d, %d,%d \n", f, n, y, x, idx_L0, idx_L3);
///		printf("       @@ %d \n", (output_memsize / sizeof(float)) );
///		exit(1);
///	}
///
///	if( (output_memsize / sizeof(float)) <= idx_L0 ){
///		printf(" error @@ %d,%d,%d,%d, %d,%d \n", f, n, y, x, idx_L0, idx_L3);
///		printf("       @@ %d \n", (output_memsize / sizeof(float)) );
///		exit(1);
///	}

					_data3x3_L0_b1 = _mm_loadu_ps( ip + idx_L0 - 1 );
					_data3x3_L1_b1 = _mm_loadu_ps( ip + idx_L1 - 1 );
					_data3x3_L2_b1 = _mm_loadu_ps( ip + idx_L2 - 1 );
					_data3x3_L3_b1 = _mm_loadu_ps( ip + idx_L3 - 1 );
					_data3x3_L0_nw = _mm_load_ps( ip + idx_L0 );
					_data3x3_L1_nw = _mm_load_ps( ip + idx_L1 );
					_data3x3_L2_nw = _mm_load_ps( ip + idx_L2 );
					_data3x3_L3_nw = _mm_load_ps( ip + idx_L3 );
					_data3x3_L0_f1 = _mm_loadu_ps( ip + idx_L0 + 1 );
					_data3x3_L1_f1 = _mm_loadu_ps( ip + idx_L1 + 1 );
					_data3x3_L2_f1 = _mm_loadu_ps( ip + idx_L2 + 1 );
					_data3x3_L3_f1 = _mm_loadu_ps( ip + idx_L3 + 1 );

					_o_data1 = vmlaq_f32( _o_data1, _flt_L0w0, _data3x3_L0_b1 );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L0w1, _data3x3_L0_nw );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L0w2, _data3x3_L0_f1 );

					_o_data1 = vmlaq_f32( _o_data1, _flt_L1w0, _data3x3_L1_b1 );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L1w1, _data3x3_L1_nw );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L1w2, _data3x3_L1_f1 );

					_o_data1 = vmlaq_f32( _o_data1, _flt_L2w0, _data3x3_L2_b1 );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L2w1, _data3x3_L2_nw );
					_o_data1 = vmlaq_f32( _o_data1, _flt_L2w2, _data3x3_L2_f1 );

					// for LINE 2
					_o_data2 = vmlaq_f32( _o_data2, _flt_L0w0, _data3x3_L1_b1 );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L0w1, _data3x3_L1_nw );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L0w2, _data3x3_L1_f1 );

					_o_data2 = vmlaq_f32( _o_data2, _flt_L1w0, _data3x3_L2_b1 );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L1w1, _data3x3_L2_nw );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L1w2, _data3x3_L2_f1 );

					_o_data2 = vmlaq_f32( _o_data2, _flt_L2w0, _data3x3_L3_b1 );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L2w1, _data3x3_L3_nw );
					_o_data2 = vmlaq_f32( _o_data2, _flt_L2w2, _data3x3_L3_f1 );

					if(n==(input_size_num-1)) {
						_o_data1 =  _mm_add_ps( _o_data1, _bias );
						_o_data2 =  _mm_add_ps( _o_data2, _bias );

						//0 clip
						//_o_data1 = _mm_max_ps( _o_data1, _mm_setzero_ps() );
						//_o_data2 = _mm_max_ps( _o_data2, _mm_setzero_ps() );
					}


					_mm_store_ps( op + idx_o1, _o_data1 );
					_mm_store_ps( op + idx_o2, _o_data2 );

				}

			}

		}

		//zero fill for padding area
		for(x=0;x<output_h_padding;x++) {
			for(y=0;y<input_size_y;y+=2) {
				idx_o1 = f * output_next_frame + ((y+1) * output_next_line ) + x + input_size_x + 4;
				idx_o2 = f * output_next_frame + ((y+2) * output_next_line ) + x + input_size_x + 4;
				*( op + idx_o1 ) = 0;
				*( op + idx_o2 ) = 0;

			}
		}

	}
    return CQT_RET_OK;
}


