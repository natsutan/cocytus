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

int main(void)
{
    CQT_NET *vgg16_p;
    int ret;

    vgg16_p = cqt_init();
    printf("hello cqt\n");

    //input layer の出力に画像データを格納する。

    ret = load_from_numpy(input_1_output, "../img/dog.jpg.npy", 3*224*224, &np);
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

    layer0_output();

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
