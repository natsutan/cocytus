#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cqt_gen/cqt_gen.h"
#include "ya2k_yolo.h"


BOX out_boxes[YOLO_MAX_RESULT];
float out_scores[YOLO_MAX_RESULT];
int out_classes[YOLO_MAX_RESULT];

//yolo_eval_output
//float box_xy[YOLO_REGION_SIZE][YOLO_REGION_SIZE][YOLO_CLUSTERS][2];
//float box_wh[YOLO_REGION_SIZE][YOLO_REGION_SIZE][YOLO_CLUSTERS][2];
//float box_confidence[YOLO_REGION_SIZE][YOLO_REGION_SIZE][YOLO_CLUSTERS];
//float box_class_probs[YOLO_REGION_SIZE][YOLO_REGION_SIZE][YOLO_CLUSTERS][YOLO_CLASSES];

//決め打ちにしないとデバッガが落ちる
float box_xy[13][13][5][2];
float box_wh[13][13][5][2];
float box_confidence[13][13][5];
float box_class_probs[13][13][5][20];

char voc_class[YOLO_CLASSES][YOLO_BUFSIZE] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

void yolo_head(void *predp);


//処理結果は、box_xy, box_wh, box_confidence, box_class_probs
void yolo_head(void *predp)
{
    int row, col;
    int k, i, idx_k;
    float data0, data1;

    //配列の並びをKerasに合わせる。
    for(row=0;row<YOLO_REGION_SIZE;row++) {
        for(col=0;col<YOLO_REGION_SIZE;col++) {
            for(k=0;k<YOLO_CLUSTERS;k++) {
                //yoloの出力結果にアクセスするときは、idx_kを使う。
                idx_k = k * (YOLO_CLASSES + 5);
                printf("idx_k = %d\n", idx_k);
                fflush(stdout);

                //box_xy = sigmoid(feats[..., :2])
                data0 = conv2d_9_output[idx_k+0][row][col];
                data1 = conv2d_9_output[idx_k+1][row][col];
                box_xy[row][col][k][0] = data0;
                box_xy[row][col][k][1] = data1;

                //box_wh = np.exp(feats[..., 2:4])
                data0 = conv2d_9_output[idx_k+2][row][col];
                data1 = conv2d_9_output[idx_k+3][row][col];
                box_wh[row][col][k][0] = data0;
                box_wh[row][col][k][1] = data1;

                //box_confidence = sigmoid(feats[..., 4:5])
                data0 = conv2d_9_output[idx_k+4][row][col];
                box_confidence[row][col][k] = data0;

                //box_class_probs = softmax(feats[..., 5:])
                for(i=0;i<YOLO_CLASSES;i++) {
                    data0 = conv2d_9_output[idx_k+5+i][row][col];
                    box_class_probs[row][col][k][i] = data0;
                }
            }
        }
    }
    printf("end of yolo_head\n");
}


int yolo_eval(void *predp, YOLO_PARAM *pp)
{
    assert(predp!=NULL);
    assert(pp!=NULL);
    assert(pp->classes==YOLO_CLASSES);
    yolo_head(predp);


    return -1;
}
