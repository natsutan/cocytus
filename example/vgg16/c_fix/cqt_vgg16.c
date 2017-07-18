//
// Created by natu on 17/03/22.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"
#include "cqt_gen/cqt_debug.h"

float result[1000];


typedef struct ret_pair
{
    int no;
    float rate;
} ret_pair;


typedef struct{
    int no;
    char tag[100];
    char class[100];
} VGG16_CLASS;

#include "vgg_table.h"

NUMPY_HEADER np;

//debug関数

void print_result(void);


int main(void)
{
    CQT_NET *vgg16_p;
    int ret;
    CQT_LAYER *lp;

    vgg16_p = cqt_init();
    printf("hello cqt\n");

    //input layer の出力に画像データを格納する。
    ret = load_from_numpy(input_1_output, "../img/dog.png_fix_q7.npy", 3*224*224, &np);
//    ret = load_from_numpy(input_1_output, "../img/cat.png_fix_q7.npy", 3*224*224, &np);
    if(ret != CQT_RET_OK) {
        printf("error in load_from_numpy %d\n", ret);
        exit(1);
    }

    ret = cqt_load_weight_from_files(vgg16_p, "weight/");
    if (ret != CQT_RET_OK) {
        printf("ERROR in cqt_load_weight_from_files %d\n", ret);
    }


    //overflow clear
    for (int i = 0; i < vgg16_p->layernum; i++) {
        lp = &(vgg16_p->layer[i]);
        lp->overflow_cnt = 0;
    }

    printf("start run\n");
    ret = cqt_run(vgg16_p, NULL);
    if(ret != CQT_RET_OK){
        printf("ERROR in cqt_run %d\n", ret);
    }

    float sum = 0.0;
    for(int i=0;i<1000;i++) {
        float d = predictions_output[i] / 32.0; //Q=5
        result[i] = exp(d);
        sum += result[i];
    }
    for(int i=0;i<1000;i++) {
        result[i] /= sum;
    }

    //overflow check
    for (int i = 0; i < vgg16_p->layernum; i++) {
        lp = &(vgg16_p->layer[i]);
        if (lp->overflow_cnt != 0) {
            printf("Overflow:l=%d, %s, cnt = %d, inq = %d, outq = %d, wq = %d\n",
                   i, lp->name, lp->overflow_cnt, lp->input_q, lp->output_q, lp->weight_q);
        }
    }

    print_result();

    return 0;
}


int compare_int(const void *a, const void *b)
{
    ret_pair *ap = (ret_pair *)a;
    ret_pair *bp = (ret_pair *)b;

    if ((ap->rate) > (bp->rate)) {
        return -1;
    } else {
        return 1;
    }
}

void print_result(void)
{
    int i;
    ret_pair results[1000];

    for(i=0;i<1000;i++) {
        results[i].no = i;
        results[i].rate = result[i];
    }

    qsort(results, 1000, sizeof(ret_pair), compare_int);

    for(i=0;i<5;i++) {

        printf("%s, rate = %f\n", vgg16_table[results[i].no].class, results[i].rate);

    }

}
