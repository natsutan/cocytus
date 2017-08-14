//
// Created by natu on 17/04/28.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"
#include "cqt_gen/cqt_debug.h"
#include "ya2k_yolo.h"

NUMPY_HEADER np;
#define IMG_SIZE 416


int main(void)
{
    CQT_NET *tyolo_p;
    int ret;
    YOLO_PARAM  yolo_parameter;

    tyolo_p = cqt_init();
    printf("hello cqt\n");

    //input layer の出力に画像データを格納する。

    ret = load_from_numpy(input_1_output, "../img/person.jpg_cl.npy", 3*IMG_SIZE*IMG_SIZE, &np);
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

    cqt_layer29_dump();
    cqt_layer30_dump();
    cqt_layer31_dump();

    // ここから領域の計算
    yolo_parameter.width = 620;
    yolo_parameter.height = 424;
    yolo_parameter.score_threshold = 0.3;
    yolo_parameter.iou_threshold = 0.5;
    yolo_parameter.classes = 20;

    ret = yolo_eval(conv2d_9_output, &yolo_parameter);
    printf("yolo eval %d\n", ret);

    if(ret < 0) {
        printf("ERROR %d\n", ret);
        exit(1);
    }

    for(int i=0;i<ret;i++) {
        int class = yolo_result[i].class;
        float score = yolo_result[i].score;
        BOX b = yolo_result[i].box;

        int top, left, bottom, right;

        top = (int)floor(b.top + 0.5);
        if(top < 0) {
            top = 0;
        }
        left = (int)floor(b.left + 0.5);
        if(left < 0) {
            left = 0;
        }
        bottom = (int)floor(b.bottom + 0.5);
        if(bottom >= yolo_parameter.height) {
            bottom = yolo_parameter.height - 1;
        }
        right = (int)floor(b.right + 0.5);
        if(right >= yolo_parameter.width) {
            right = yolo_parameter.width - 1;
        }
        printf("%s %f (%d, %d), (%d, %d)\n",
               voc_class[class], score, left, top, right, bottom);
    }



    return 0;
}



