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
//tiny-yolo
#define IMG_SIZE 416

float div_q = 512;


int main(int argc, char *argv[])
{
    CQT_NET *tyolo_p;
    int ret;
    YOLO_PARAM  yolo_parameter;
    CQT_LAYER *lp;
    FILE *fp_in;
    char fpath[256];
    char *cp;
    char *nl_p;

    if(argc != 2) {
        printf("usage:cqt_tyolo filelist\n");
        exit(1);
    }
    cp = *(argv + 1);

    fp_in = fopen(cp, "r");
    printf("%s\n", cp);
    if(fp_in == NULL) {
        printf("can't open %s\n", argv[1]);
        exit(1);
    }

    assert(sizeof(FIXP8)==1);
    assert(sizeof(FIXP16)==2);

    tyolo_p = cqt_init();
    printf("cqt yolo fixed point\n");


    ret = cqt_load_weight_from_files(tyolo_p, "weight/");
    if (ret != CQT_RET_OK) {
        printf("ERROR in cqt_load_weight_from_files %d\n", ret);
    }

    printf("start run\n");


    while(fgets(fpath, sizeof(fpath), fp_in)!=NULL) {
        nl_p =  strchr(fpath, '\n');
        if(nl_p!=NULL) {
            *nl_p = '\0';
        }

        //input layer の出力に画像データを格納する。
        ret = load_from_numpy(input_1_output, fpath, 3 * IMG_SIZE * IMG_SIZE, &np);
        if (ret != CQT_RET_OK) {
            printf("error in load_from_numpy %d\n", ret);
            exit(1);
        }

        //overflow clear
        for (int i = 0; i < tyolo_p->layernum; i++) {
            lp = &(tyolo_p->layer[i]);
            lp->overflow_cnt = 0;
        }


        ret = cqt_run(tyolo_p, NULL);
        if (ret != CQT_RET_OK) {
            printf("ERROR in cqt_run %d\n", ret);
        }

        //cqt_layer1_dump();
        for (int i = 0; i < 32; i++) {
            cqt_layerdump(i);
        }

        // ここから領域の計算
        yolo_parameter.width = IMG_SIZE;
        yolo_parameter.height = IMG_SIZE;
        yolo_parameter.score_threshold = 0.3;
        yolo_parameter.iou_threshold = 0.5;
        yolo_parameter.classes = 20;

        ret = yolo_eval(conv2d_9_output, &yolo_parameter);
        printf("yolo eval %d\n", ret);

        //overflow check
        for (int i = 0; i < tyolo_p->layernum; i++) {
            lp = &(tyolo_p->layer[i]);
            if (lp->overflow_cnt != 0) {
                printf("Overflow:l=%d, %s, cnt = %d, inq = %d, outq = %d, wq = %d\n",
                       i, lp->name, lp->overflow_cnt, lp->input_q, lp->output_q, lp->weight_q);
            }
        }


        if (ret < 0) {
            printf("ERROR %d\n", ret);
            continue;
        }

        for (int i = 0; i < ret; i++) {
            int class = yolo_result[i].class;
            float score = yolo_result[i].score;
            BOX b = yolo_result[i].box;

            int top, left, bottom, right;

            top = (int) floor(b.top + 0.5);
            if (top < 0) {
                top = 0;
            }
            left = (int) floor(b.left + 0.5);
            if (left < 0) {
                left = 0;
            }
            bottom = (int) floor(b.bottom + 0.5);
            if (bottom >= yolo_parameter.height) {
                bottom = yolo_parameter.height - 1;
            }
            right = (int) floor(b.right + 0.5);
            if (right >= yolo_parameter.width) {
                right = yolo_parameter.width - 1;
            }
            printf("%s %f (%d, %d), (%d, %d)\n",
                   voc_class[class], score, left, top, right, bottom);
        }
    }

    return 0;
}



