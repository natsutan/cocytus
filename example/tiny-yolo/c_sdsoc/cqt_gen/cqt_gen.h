//----------------------------------------------------------------------------------------------------
// This file is automatically generated.
// 2017/09/15 00:00:00
//----------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "cqt.h"
#include "cqt_net.h"


CQT_NET* cqt_init(void);
int cqt_load_weight_from_files(CQT_NET* np, const char *path);
int cqt_run(CQT_NET* np, void *dp);
void cqt_close(CQT_NET* np);

extern int cqt_process;

// cocytus network
extern CQT_NET g_cqt_sequential_1;

//Layers
extern LY_InputLayer input_1;
extern LY_Conv2D conv2d_1;
extern LY_BatchNormalization batch_normalization_1;
extern LY_LeakyReLU leaky_re_lu_1;
extern LY_MaxPooling2D max_pooling2d_1;
extern LY_Conv2D conv2d_2;
extern LY_BatchNormalization batch_normalization_2;
extern LY_LeakyReLU leaky_re_lu_2;
extern LY_MaxPooling2D max_pooling2d_2;
extern LY_Conv2D conv2d_3;
extern LY_BatchNormalization batch_normalization_3;
extern LY_LeakyReLU leaky_re_lu_3;
extern LY_MaxPooling2D max_pooling2d_3;
extern LY_Conv2D conv2d_4;
extern LY_BatchNormalization batch_normalization_4;
extern LY_LeakyReLU leaky_re_lu_4;
extern LY_MaxPooling2D max_pooling2d_4;
extern LY_Conv2D conv2d_5;
extern LY_BatchNormalization batch_normalization_5;
extern LY_LeakyReLU leaky_re_lu_5;
extern LY_MaxPooling2D max_pooling2d_5;
extern LY_Conv2D conv2d_6;
extern LY_BatchNormalization batch_normalization_6;
extern LY_LeakyReLU leaky_re_lu_6;
extern LY_MaxPooling2D max_pooling2d_6;
extern LY_Conv2D conv2d_7;
extern LY_BatchNormalization batch_normalization_7;
extern LY_LeakyReLU leaky_re_lu_7;
extern LY_Conv2D conv2d_8;
extern LY_BatchNormalization batch_normalization_8;
extern LY_LeakyReLU leaky_re_lu_8;
extern LY_Conv2D conv2d_9;

//weights
extern NUMPY_HEADER nph_conv2d_1_W;
extern NUMPY_HEADER nph_conv2d_1_b;
extern float w_conv2d_1_W[16][3][3][3];
extern float w_conv2d_1_b[16];
extern NUMPY_HEADER nph_beta_batch_normalization_1_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_1_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_1_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_1_W;
extern float beta_batch_normalization_1_W[16];
extern float gamma_batch_normalization_1_W[16];
extern float moving_mean_batch_normalization_1_W[16];
extern float moving_variance_batch_normalization_1_W[16];
extern NUMPY_HEADER nph_conv2d_2_W;
extern NUMPY_HEADER nph_conv2d_2_b;
extern float w_conv2d_2_W[32][16][3][3];
extern float w_conv2d_2_b[32];
extern NUMPY_HEADER nph_beta_batch_normalization_2_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_2_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_2_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_2_W;
extern float beta_batch_normalization_2_W[32];
extern float gamma_batch_normalization_2_W[32];
extern float moving_mean_batch_normalization_2_W[32];
extern float moving_variance_batch_normalization_2_W[32];
extern NUMPY_HEADER nph_conv2d_3_W;
extern NUMPY_HEADER nph_conv2d_3_b;
extern float w_conv2d_3_W[64][32][3][3];
extern float w_conv2d_3_b[64];
extern NUMPY_HEADER nph_beta_batch_normalization_3_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_3_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_3_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_3_W;
extern float beta_batch_normalization_3_W[64];
extern float gamma_batch_normalization_3_W[64];
extern float moving_mean_batch_normalization_3_W[64];
extern float moving_variance_batch_normalization_3_W[64];
extern NUMPY_HEADER nph_conv2d_4_W;
extern NUMPY_HEADER nph_conv2d_4_b;
extern float w_conv2d_4_W[128][64][3][3];
extern float w_conv2d_4_b[128];
extern NUMPY_HEADER nph_beta_batch_normalization_4_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_4_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_4_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_4_W;
extern float beta_batch_normalization_4_W[128];
extern float gamma_batch_normalization_4_W[128];
extern float moving_mean_batch_normalization_4_W[128];
extern float moving_variance_batch_normalization_4_W[128];
extern NUMPY_HEADER nph_conv2d_5_W;
extern NUMPY_HEADER nph_conv2d_5_b;
extern float w_conv2d_5_W[256][128][3][3];
extern float w_conv2d_5_b[256];
extern NUMPY_HEADER nph_beta_batch_normalization_5_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_5_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_5_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_5_W;
extern float beta_batch_normalization_5_W[256];
extern float gamma_batch_normalization_5_W[256];
extern float moving_mean_batch_normalization_5_W[256];
extern float moving_variance_batch_normalization_5_W[256];
extern NUMPY_HEADER nph_conv2d_6_W;
extern NUMPY_HEADER nph_conv2d_6_b;
extern float w_conv2d_6_W[512][256][3][3];
extern float w_conv2d_6_b[512];
extern NUMPY_HEADER nph_beta_batch_normalization_6_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_6_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_6_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_6_W;
extern float beta_batch_normalization_6_W[512];
extern float gamma_batch_normalization_6_W[512];
extern float moving_mean_batch_normalization_6_W[512];
extern float moving_variance_batch_normalization_6_W[512];
extern NUMPY_HEADER nph_conv2d_7_W;
extern NUMPY_HEADER nph_conv2d_7_b;
extern float w_conv2d_7_W[1024][512][3][3];
extern float w_conv2d_7_b[1024];
extern NUMPY_HEADER nph_beta_batch_normalization_7_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_7_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_7_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_7_W;
extern float beta_batch_normalization_7_W[1024];
extern float gamma_batch_normalization_7_W[1024];
extern float moving_mean_batch_normalization_7_W[1024];
extern float moving_variance_batch_normalization_7_W[1024];
extern NUMPY_HEADER nph_conv2d_8_W;
extern NUMPY_HEADER nph_conv2d_8_b;
extern float w_conv2d_8_W[1024][1024][3][3];
extern float w_conv2d_8_b[1024];
extern NUMPY_HEADER nph_beta_batch_normalization_8_W;
extern NUMPY_HEADER nph_gamma_batch_normalization_8_W;
extern NUMPY_HEADER nph_moving_mean_batch_normalization_8_W;
extern NUMPY_HEADER nph_moving_variance_batch_normalization_8_W;
extern float beta_batch_normalization_8_W[1024];
extern float gamma_batch_normalization_8_W[1024];
extern float moving_mean_batch_normalization_8_W[1024];
extern float moving_variance_batch_normalization_8_W[1024];
extern NUMPY_HEADER nph_conv2d_9_W;
extern NUMPY_HEADER nph_conv2d_9_b;
extern float w_conv2d_9_W[125][1024][1][1];
extern float w_conv2d_9_b[125];

//outputs
extern float *input_1_output; //input_1_output[3][416][416]
extern float *conv2d_1_output; //conv2d_1_output[16][416][416]
extern float *batch_normalization_1_output; //batch_normalization_1_output[16][416][416]
extern float *leaky_re_lu_1_output; //leaky_re_lu_1_output[16][416][416]
extern float *max_pooling2d_1_output; //max_pooling2d_1_output[16][208][208]
extern float *conv2d_2_output; //conv2d_2_output[32][208][208]
extern float *batch_normalization_2_output; //batch_normalization_2_output[32][208][208]
extern float *leaky_re_lu_2_output; //leaky_re_lu_2_output[32][208][208]
extern float *max_pooling2d_2_output; //max_pooling2d_2_output[32][104][104]
extern float *conv2d_3_output; //conv2d_3_output[64][104][104]
extern float *batch_normalization_3_output; //batch_normalization_3_output[64][104][104]
extern float *leaky_re_lu_3_output; //leaky_re_lu_3_output[64][104][104]
extern float *max_pooling2d_3_output; //max_pooling2d_3_output[64][52][52]
extern float *conv2d_4_output; //conv2d_4_output[128][52][52]
extern float *batch_normalization_4_output; //batch_normalization_4_output[128][52][52]
extern float *leaky_re_lu_4_output; //leaky_re_lu_4_output[128][52][52]
extern float *max_pooling2d_4_output; //max_pooling2d_4_output[128][26][26]
extern float *conv2d_5_output; //conv2d_5_output[256][26][26]
extern float *batch_normalization_5_output; //batch_normalization_5_output[256][26][26]
extern float *leaky_re_lu_5_output; //leaky_re_lu_5_output[256][26][26]
extern float *max_pooling2d_5_output; //max_pooling2d_5_output[256][13][13]
extern float *conv2d_6_output; //conv2d_6_output[512][13][13]
extern float *batch_normalization_6_output; //batch_normalization_6_output[512][13][13]
extern float *leaky_re_lu_6_output; //leaky_re_lu_6_output[512][13][13]
extern float *max_pooling2d_6_output; //max_pooling2d_6_output[512][13][13]
extern float *conv2d_7_output; //conv2d_7_output[1024][13][13]
extern float *batch_normalization_7_output; //batch_normalization_7_output[1024][13][13]
extern float *leaky_re_lu_7_output; //leaky_re_lu_7_output[1024][13][13]
extern float *conv2d_8_output; //conv2d_8_output[1024][13][13]
extern float *batch_normalization_8_output; //batch_normalization_8_output[1024][13][13]
extern float *leaky_re_lu_8_output; //leaky_re_lu_8_output[1024][13][13]
extern float *conv2d_9_output; //conv2d_9_output[125][13][13]

#define MAX_OUT_SIZE (2768896)

