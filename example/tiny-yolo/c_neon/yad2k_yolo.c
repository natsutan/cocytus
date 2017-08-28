#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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


//yolo_boxes_to_cornersの出力
BOX boxes_t [13][13][5];

//yolo_filter_boxesの出力
YOLO_RESULT filtered_boxes[YOLO_MAX_RESULT];

//non_max_surpressionの出力


//最終出力
YOLO_RESULT yolo_result[YOLO_MAX_RESULT];


const char voc_class[YOLO_CLASSES][YOLO_BUFSIZE] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

const float voc_anchors[YOLO_CLUSTERS][2] = {{1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38}, {9.42, 5.11}, {16.62, 10.52}};

//関数プロトタイプ
void yolo_head(void *predp);
void yolo_head_cl(void);
float sigmoid(float x);
void yolo_boxes_to_corners(void);
int yolo_filter_boxes(float thresh);
int non_max_surpression(int num, float iou_thresh);


int non_max_surpression(int num, float iou_thresh)
{
    float area[YOLO_MAX_RESULT];
    bool remove_flg[YOLO_MAX_RESULT];
    int idxs_len = num;

    int i, j;
    int tmp;
    float a, b;
    float x1, y1, x2, y2;
    int last;
    int idxs[YOLO_MAX_RESULT];
    int idxs_work[YOLO_MAX_RESULT];
    float xx1, yy1, xx2, yy2;
    float w, h, overlap;

    int idwork_idx;
    int ret_idx = 0;

    //エリアの計算
    for(i=0;i<num;i++) {
        x1 = filtered_boxes[i].box.left;
        y1 = filtered_boxes[i].box.top;
        x2 = filtered_boxes[i].box.right;
        y2 = filtered_boxes[i].box.bottom;
        area[i] = (x2 - x1 + 1) * (y2 - y1 + 1);
    }

    //とりえあず、バブルソート
    for(i=0;i<idxs_len;i++) {
        idxs[i] = i;
    }

    for(i=0;i<idxs_len;i++) {
        for(j=0;j<i;j++) {
            a = filtered_boxes[idxs[i]].score;
            b = filtered_boxes[idxs[j]].score;
            if(a < b) {
                //swap index
                tmp = idxs[i];
                idxs[i] = idxs[j];
                idxs[j] = tmp;
            }
        }
    }

    while(0 < idxs_len) {
        last = idxs_len - 1;
        i = idxs[last];
        //結果の追加
        yolo_result[ret_idx] = filtered_boxes[idxs[last]];
        ret_idx++;

        for(j=0;j<last;j++) {

            //xx1, yy1 xx2, yy2が入れ替わっている。(Keras版のバグ？)
            if(filtered_boxes[idxs[last]].box.left > filtered_boxes[idxs[j]].box.left) {
                yy1 = filtered_boxes[idxs[last]].box.left;
            } else {
                yy1 = filtered_boxes[idxs[j]].box.left;
            }
            if(filtered_boxes[idxs[last]].box.top > filtered_boxes[idxs[j]].box.top) {
                xx1 = filtered_boxes[idxs[last]].box.top;
            } else {
                xx1 = filtered_boxes[idxs[j]].box.top;
            }
            if(filtered_boxes[idxs[last]].box.right < filtered_boxes[idxs[j]].box.right) {
                yy2 = filtered_boxes[idxs[last]].box.right;
            } else {
                yy2 = filtered_boxes[idxs[j]].box.right;
            }
            if(filtered_boxes[idxs[last]].box.bottom < filtered_boxes[idxs[j]].box.bottom) {
                xx2 = filtered_boxes[idxs[last]].box.bottom;
            } else {
                xx2 = filtered_boxes[idxs[j]].box.bottom;
            }

            w = xx2 - xx1 + 1;
            if(w < 0) {
                w = 0;
            }
            h = yy2 - yy1 + 1;
            if(h < 0) {
                h = 0;
            }
            overlap = (w * h) / area[idxs[j]];

            //スレッショルド以下は削除
            if(overlap < iou_thresh) {
                remove_flg[j] = false;
            } else {
                remove_flg[j] = true;
            }
        }

        //要素の削除
        idwork_idx = 0;
        for(j=0;j<last;j++) {
            if(remove_flg[j]) {
                continue;
            }
            idxs_work[idwork_idx] = idxs[j];
            idwork_idx++;
        }
        for(j=0;j<idwork_idx;j++) {
            idxs[j] = idxs_work[j];
        }
        idxs_len = idwork_idx;
    }

    return ret_idx;
}

int yolo_filter_boxes(float thresh)
{
    int row, col, k, i;
    int idx_r = 0;
    float max_prob;
    int max_class;
    float prob;
    float box_class_scores;

    for(row=0;row<YOLO_REGION_SIZE;row++) {
        for(col=0;col<YOLO_REGION_SIZE;col++) {
            for(k=0;k<YOLO_CLUSTERS;k++) {
                //最大のクラスを選択する。
                max_prob = 0.0;
                max_class = 0;
                for (i = 0; i < YOLO_CLASSES; i++) {
                    prob = box_class_probs[row][col][k][i];
                    if (prob > max_prob) {
                        max_prob = prob;
                        max_class = i;
                    }
                }
                box_class_scores = max_prob * box_confidence[row][col][k];
                if (box_class_scores > thresh) {
                    //結果の追加
                    filtered_boxes[idx_r].class = max_class;
                    filtered_boxes[idx_r].score = box_class_scores;
                    filtered_boxes[idx_r].box.top = boxes_t[row][col][k].top;
                    filtered_boxes[idx_r].box.left = boxes_t[row][col][k].left;
                    filtered_boxes[idx_r].box.bottom = boxes_t[row][col][k].bottom;
                    filtered_boxes[idx_r].box.right = boxes_t[row][col][k].right;

                    //数のチェック
                    idx_r++;
                    if(idx_r>=YOLO_MAX_RESULT) {
                        return RET_YOLO_MAX_RESULT_OVER;
                    }
                }
            }
        }
    }
    return idx_r;
}


void yolo_boxes_to_corners(void)
{
    int row, col, k;
    float box_mins0, box_mins1, box_maxes0, box_maxes1;
    for(row=0;row<YOLO_REGION_SIZE;row++) {
        for(col=0;col<YOLO_REGION_SIZE;col++) {
            for(k=0;k<YOLO_CLUSTERS;k++) {
                box_mins0 = box_xy[row][col][k][0] - (box_wh[row][col][k][0] / 2);
                box_mins1 = box_xy[row][col][k][1] - (box_wh[row][col][k][1] / 2);
                box_maxes0 = box_xy[row][col][k][0] + (box_wh[row][col][k][0] / 2);
                box_maxes1 = box_xy[row][col][k][1] + (box_wh[row][col][k][1] / 2);
                boxes_t[row][col][k].left   = box_mins0;
                boxes_t[row][col][k].top    = box_mins1;
                boxes_t[row][col][k].right  = box_maxes0;
                boxes_t[row][col][k].bottom = box_maxes1;
            }
        }
    }
}



float sigmoid(float x)
{
    return (float) (1.0 / (1 + exp(-x)));

}

//処理結果は、box_xy, box_wh, box_confidence, box_class_probs
void yolo_head(void *predp)
{
    int row, col;
    int k, i, idx_k;
    float data0, data1;
    float softmax_work[YOLO_CLASSES];
    float softmax_sum;
    float softmax_max;

    //配列の並びをKerasに合わせる。
    for(row=0;row<YOLO_REGION_SIZE;row++) {
        for(col=0;col<YOLO_REGION_SIZE;col++) {
            for(k=0;k<YOLO_CLUSTERS;k++) {
                //yoloの出力結果にアクセスするときは、idx_kを使う。
                idx_k = k * (YOLO_CLASSES + 5);

                //box_xy = sigmoid(feats[..., :2])
                data0 = conv2d_9_output[idx_k+0][row+NEON_VTR][NEON_HTR+col];
                data1 = conv2d_9_output[idx_k+1][row+NEON_VTR][NEON_HTR+col];

                box_xy[row][col][k][0] = (sigmoid(data0) + col) / YOLO_REGION_SIZE;
                box_xy[row][col][k][1] = (sigmoid(data1) + row) / YOLO_REGION_SIZE;

                //box_wh = np.exp(feats[..., 2:4])
                data0 = conv2d_9_output[idx_k+2][row+NEON_VTR][NEON_HTR+col];
                data1 = conv2d_9_output[idx_k+3][row+NEON_VTR][NEON_HTR+col];

                box_wh[row][col][k][0] = (float) ((exp(data0) * voc_anchors[k][0]) / YOLO_REGION_SIZE);
                box_wh[row][col][k][1] = (float) ((exp(data1) * voc_anchors[k][1]) / YOLO_REGION_SIZE);

                //box_confidence = sigmoid(feats[..., 4:5])
                data0 = conv2d_9_output[idx_k+4][row+NEON_VTR][NEON_HTR+col];
                box_confidence[row][col][k] = sigmoid(data0);

                //box_class_probs = softmax(feats[..., 5:])
                softmax_max = 0.0;
                //一回目のループでmaxを求める。
                for(i=0;i<YOLO_CLASSES;i++) {
                    data0 = conv2d_9_output[idx_k + 5 + i][row+NEON_VTR][NEON_HTR+col];
                    if (softmax_max < data0) {
                        softmax_max = data0;
                    }
                }
                //２回目のループでexpとsumを求める。
                softmax_sum = 0.0;
                for(i=0;i<YOLO_CLASSES;i++) {
                    data0 = conv2d_9_output[idx_k + 5 + i][row+NEON_VTR][NEON_HTR+col];
                    softmax_work[i] = (float) exp(data0 - softmax_max);
                    softmax_sum += softmax_work[i];
                }

                for(i=0;i<YOLO_CLASSES;i++) {
                    box_class_probs[row][col][k][i] = softmax_work[i] / softmax_sum;
                }
            }
        }
    }
}

//処理結果は、box_xy, box_wh, box_confidence, box_class_probs
//yolo_headのコラムラストバージョン
void yolo_head_cl(void)
{
    int row, col;
    int k, i, idx_k;
    float data0, data1;
    float softmax_work[YOLO_CLASSES];
    float softmax_sum;
    float softmax_max;

    assert(0);

    //配列の並びをKerasに合わせる。
    for(row=0;row<YOLO_REGION_SIZE;row++) {
        for(col=0;col<YOLO_REGION_SIZE;col++) {
            for(k=0;k<YOLO_CLUSTERS;k++) {
                //yoloの出力結果にアクセスするときは、idx_kを使う。
                idx_k = k * (YOLO_CLASSES + 5);

                //box_xy = sigmoid(feats[..., :2])
                data0 = conv2d_9_output[row][col][idx_k+0];
                data1 = conv2d_9_output[row][col][idx_k+1];

                box_xy[row][col][k][0] = (sigmoid(data0) + col) / YOLO_REGION_SIZE;
                box_xy[row][col][k][1] = (sigmoid(data1) + row) / YOLO_REGION_SIZE;

                //box_wh = np.exp(feats[..., 2:4])
                data0 = conv2d_9_output[row][col][idx_k+2];
                data1 = conv2d_9_output[row][col][idx_k+3];

                box_wh[row][col][k][0] = (float) ((exp(data0) * voc_anchors[k][0]) / YOLO_REGION_SIZE);
                box_wh[row][col][k][1] = (float) ((exp(data1) * voc_anchors[k][1]) / YOLO_REGION_SIZE);

                //box_confidence = sigmoid(feats[..., 4:5])
                data0 = conv2d_9_output[row][col][idx_k+4];
                box_confidence[row][col][k] = sigmoid(data0);

                //box_class_probs = softmax(feats[..., 5:])
                softmax_max = 0.0;
                //一回目のループでmaxを求める。
                for(i=0;i<YOLO_CLASSES;i++) {
                    data0 = conv2d_9_output[row][col][idx_k + 5 + i];
                    if (softmax_max < data0) {
                        softmax_max = data0;
                    }
                }
                //２回目のループでexpとsumを求める。
                softmax_sum = 0.0;
                for(i=0;i<YOLO_CLASSES;i++) {
                    data0 = conv2d_9_output[row][col][idx_k + 5 + i];
                    softmax_work[i] = (float) exp(data0 - softmax_max);
                    softmax_sum += softmax_work[i];
                }

                for(i=0;i<YOLO_CLASSES;i++) {
                    box_class_probs[row][col][k][i] = softmax_work[i] / softmax_sum;
                }
            }
        }
    }
}

int yolo_eval(void *predp, YOLO_PARAM *pp)
{
    int yolo_filter_boxes_ret; //領域のの数
    int nms_ret;
    int i;

    assert(predp!=NULL);
    assert(pp!=NULL);
    assert(pp->classes==YOLO_CLASSES);

    //コラムラスト時は、yolo_head_clを呼び出す。
#ifdef CQT_CHANNEL_LAST
    yolo_head_cl();
#else
    yolo_head(predp);
#endif //CQT_CHANNEL_LAST

    yolo_boxes_to_corners();
    yolo_filter_boxes_ret = yolo_filter_boxes(pp->score_threshold);
    if(yolo_filter_boxes_ret <= 0) {
        return yolo_filter_boxes_ret;
    }

    for(i=0;i<yolo_filter_boxes_ret;i++) {
        filtered_boxes[i].box.left   *= pp->width;
        filtered_boxes[i].box.right  *= pp->width;
        filtered_boxes[i].box.top    *= pp->height;
        filtered_boxes[i].box.bottom *= pp->height;
    }

    nms_ret = non_max_surpression(yolo_filter_boxes_ret, pp->iou_threshold);
    if(nms_ret) {
        return nms_ret;
    }

    return -1;
}
