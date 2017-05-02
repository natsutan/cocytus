//
// Created by natu on 17/05/01.
//

#ifndef CQT_TYOLO_YA2K_YOLO_H
#define CQT_TYOLO_YA2K_YOLO_H

#define YOLO_MAX_RESULT (128)
#define YOLO_CLASSES (20)
#define YOLO_BUFSIZE (128)

//デフォルトのtiny-yoloは13x13
#define YOLO_REGION_SIZE (13)
//クラスタ数。論文中のk
#define YOLO_CLUSTERS (5)

typedef struct yolo_param_t {
    int width;  // 入力画像の幅（縮小前）
    int height; // 入力画像の高さ（縮小前）
    float score_threshold; //信頼度のスレッショルド　0.3等
    float iou_threshold; //IOUのスレッショルド　0.5程度
    int classes; //クラス数　VOCであれば20
} YOLO_PARAM;

typedef struct box_t {
    float top;
    float left;
    float bottom;
    float right;
} BOX;

extern BOX out_boxes[YOLO_MAX_RESULT];
extern float out_scores[YOLO_MAX_RESULT];
extern int out_classes[YOLO_MAX_RESULT];
extern const char voc_class[YOLO_CLASSES][YOLO_BUFSIZE];

// yolo_eval
// CNNの出力から、領域提案を行う。
// 引数
//     void *predp: CNNの出力を指すポインター。通常のtiny-yoloであればconv2d_9_output
//     YOLO_PARAM *pp: YOLOのパラメータ
// 戻り値
//     正数：見つけた提案領域数
//     0:提案領域無し
//     負数：エラー
//
// 関数を呼び出すと、グローバル変数 out_boxes,  out_scores, out_classesに領域情報を書き込む。
// 有効な領域の数は、yolo_evalの戻り値で判別する。
// yolo_evalが3を返した時、先頭からout_boxes[2], out_scores,[2] out_classe[2]までが
// 有効なデータとなる。戻り値が0の時は、提案領域が見つからなかった事を示す。戻り値が負の場合は
// 処理中にエラー発生した事を示す。

int yolo_eval(void *predp, YOLO_PARAM *pp);



#endif //CQT_TYOLO_YA2K_YOLO_H
