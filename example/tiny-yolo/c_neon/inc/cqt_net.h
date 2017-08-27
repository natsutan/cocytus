#pragma once
//コキュートスでネットワークを記述するための型宣言
#include "cqt_net.h"
#include "cqt_keras.h"

typedef struct cqt_net_layer_tag {
    KR_LAYER_TYPE type;
    char name[CQT_MAX_LAYER_NAME];
    int cqt_input_shape[4];
    int cqt_output_shape[4];

    CQT_DTYPE input_dtypes[4];
    CQT_DTYPE weight_dtypes[4];
    CQT_DTYPE output_dtypes[4];

    int input_q;
    int weight_q;
    int output_q;
    int overflow_cnt;
    //neon only
    int neon_padding_hi;
    int neon_padding_ho;

    void *param_p;  //pointer to parameters
    void *data_p;   //pointer to data, eg:weights
} CQT_LAYER;


typedef struct cqt_net_tag {
    int layernum;
    CQT_LAYER layer[CQT_MAX_LAYER_NUM];
} CQT_NET;