//
// Created by natu on 17/03/22.
//

#include <stdio.h>
#include <stdlib.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"

NUMPY_HEADER np;

void layer0_output(void);
void layer1_output(void);
void layer2_output(void);
void layer3_output(void);
void layer15_output(void);
void layer17_output(void);

int main(void)
{
    CQT_NET *vgg16_p;
    int ret;

    vgg16_p = cqt_init();
    printf("hello cqt\n");

    //input layer の出力に画像データを格納する。

    ret = load_from_numpy(input_1_output, "../img/dog.png.npy", 3*224*224, &np);
    if(ret != CQT_RET_OK) {
        printf("error in load_from_numpy %d\n", ret);
        exit(1);
    }

    //ret = cqt_load_weight_from_files(vgg16_p, "weight/");
    //if (ret != CQT_RET_OK) {
    //    printf("ERROR in cqt_load_weight_from_files %d\n", ret);
    //}

    ret = cqt_run(vgg16_p, NULL);
    if(ret != CQT_RET_OK){
        printf("ERROR in cqt_run %d\n", ret);
    }

    layer15_output();

    return 0;
}

void layer0_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 224 * 224;
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

    np_0.shape[0] = 224 * 224;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block1_conv1_output[0], "output/dog_l01_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_conv1_output[1], "output/dog_l01_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_conv1_output[63], "output/dog_l01_63.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}

void layer2_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 224 * 224;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block1_conv2_output[0], "output/dog_l02_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_conv2_output[1], "output/dog_l02_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_conv2_output[63], "output/dog_l02_63.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}

void layer3_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 112 * 112;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block1_pool_output[0], "output/dog_l03_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_pool_output[1], "output/dog_l03_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block1_pool_output[63], "output/dog_l03_63.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}

void layer14_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 14 * 14;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block4_pool_output[0], "output/dog_l14_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block4_pool_output[1], "output/dog_l14_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block4_pool_output[511], "output/dog_l14_511.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}

void layer15_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 14 * 14;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block5_conv1_output[0], "output/dog_l15_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block5_conv1_output[1], "output/dog_l15_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block5_conv1_output[511], "output/dog_l15_511.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}

void layer17_output(void)
{
    NUMPY_HEADER np_0 = np;
    int ret;

    np_0.shape[0] = 14 * 14;
    np_0.shape[1] = 0;
    np_0.shape[2] = 0;
    np_0.shape[3] = 0;

    ret = save_to_numpy(block5_conv3_output[0], "output/dog_l17_0.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block5_conv3_output[1], "output/dog_l17_1.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
    ret = save_to_numpy(block5_conv3_output[512], "output/dog_l17_512.npy", &np_0);
    if(ret != CQT_RET_OK) {
        printf("ERROR in layer1_output %d\n", ret);
    }
}
