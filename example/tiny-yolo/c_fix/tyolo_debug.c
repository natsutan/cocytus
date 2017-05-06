//
// Created by natu on 17/04/01.
//
#include <stdbool.h>
#include "numpy.h"
#include "cqt_gen/cqt_gen.h"

extern NUMPY_HEADER np;
#define IMG_SIZE 416

void layer0_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = IMG_SIZE * IMG_SIZE;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(input_1_output[0], "output/l00_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer0_output %d\n", ret);

    }
    ret = save_to_numpy(input_1_output[1], "output/l00_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer0_output %d\n", ret);

    }
    ret = save_to_numpy(input_1_output[2], "output/l00_2.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer0_output %d\n", ret);

    }


}

void layer1_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = IMG_SIZE * IMG_SIZE;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(conv2d_1_output[0], "output/l01_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(conv2d_1_output[1], "output/l01_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(conv2d_1_output[15], "output/l01_15.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer2_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = IMG_SIZE * IMG_SIZE;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(batch_normalization_1_output[0], "output/l02_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_foutput %d\n", ret);

    }
    ret = save_to_numpy(batch_normalization_1_output[1], "output/l02_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(batch_normalization_1_output[15], "output/l02_15.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer3_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = IMG_SIZE * IMG_SIZE;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(leaky_re_lu_1_output[0], "output/l03_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_1_output[1], "output/l03_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_1_output[15], "output/l03_15.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer4_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = (IMG_SIZE / 2) * (IMG_SIZE / 2);
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(max_pooling2d_1_output[0], "output/l04_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(max_pooling2d_1_output[1], "output/l04_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(max_pooling2d_1_output[15], "output/l04_15.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer5_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = (IMG_SIZE / 2) * (IMG_SIZE / 2);
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(conv2d_2_output[0], "output/l05_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(conv2d_2_output[1], "output/l05_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(conv2d_2_output[31], "output/l05_31.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer10_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 104 * 104;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(batch_normalization_3_output[0], "output/l10_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(batch_normalization_3_output[1], "output/l10_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(batch_normalization_3_output[63], "output/l10_63.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}


void layer15_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 52 * 52;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(leaky_re_lu_4_output[0], "output/l15_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_4_output[1], "output/l15_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_4_output[15], "output/l15_127.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer23_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 13 * 13;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(leaky_re_lu_6_output[0], "output/l23_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_6_output[1], "output/l23_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);

    }
    ret = save_to_numpy(leaky_re_lu_6_output[511], "output/l23_511.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}



void layer24_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 13 * 13;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(max_pooling2d_6_output[0], "output/l24_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(max_pooling2d_6_output[1], "output/l24_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(max_pooling2d_6_output[511], "output/l24_511.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer26_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 13 * 13;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(batch_normalization_7_output[0], "output/l26_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(batch_normalization_7_output[1], "output/l26_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(batch_normalization_7_output[1023], "output/l26_1023.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}



void layer30_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 13 * 13;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(leaky_re_lu_8_output[0], "output/l30_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(leaky_re_lu_8_output[1], "output/l30_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(leaky_re_lu_8_output[1023], "output/l30_1023.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}

void layer31_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 13 * 13;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(conv2d_9_output[0], "output/l31_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(conv2d_9_output[1], "output/l31_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
    ret = save_to_numpy(conv2d_9_output[124], "output/l31_124.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}
