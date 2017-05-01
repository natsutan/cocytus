//
// Created by natu on 17/04/28.
//

#include <stdio.h>
#include <stdlib.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"


NUMPY_HEADER np;

#define IMG_SIZE 416

extern void layer0_output(void);
extern void layer1_output(void);
extern void layer2_output(void);
extern void layer3_output(void);
extern void layer23_output(void);
extern void layer24_output(void);
extern void layer30_output(void);
extern void layer31_output(void);


int main(void)
{
    CQT_NET *tyolo_p;
    int ret;

    tyolo_p = cqt_init();
    printf("hello cqt\n");

    //input layer の出力に画像データを格納する。

    ret = load_from_numpy(input_1_output, "../img/person.jpg.npy", 3*IMG_SIZE*IMG_SIZE, &np);
    if(ret != CQT_RET_OK) {
        printf("error in load_from_numpy %d\n", ret);
        exit(1);
    }

    ret = cqt_load_weight_from_files(tyolo_p, "weight/");
    if (ret != CQT_RET_OK) {
        printf("ERROR in cqt_load_weight_from_files %d\n", ret);
    }

    printf("start run\n");
    ret = cqt_run(tyolo_p, NULL);
    if(ret != CQT_RET_OK){
        printf("ERROR in cqt_run %d\n", ret);
    }
//    layer30_output();
    layer31_output();

    return 0;
}



