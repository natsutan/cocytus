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
        printf("ERROR in layer_output %d\n", ret);

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
    ret = save_to_numpy(conv2d_9_output[1023], "output/l31_1023.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer_output %d\n", ret);
    }
}
