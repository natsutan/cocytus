#pragma once
//コキュートスでネットワークを記述するための型宣言
#include "cqt_net.h"
#include "cqt_keras.h"

typedef struct cqt_net_layer_tag {
    KR_LAYER_TYPE type;
    char name[CQT_MAX_LAYER_NAME];
    int cqt_input_shape[4];
    int cqt_output_shape[4];

    NN_DTYPE input_dtype;
    NN_DTYPE weight_dtype;
    NN_DTYPE output_dtype;

    void *p_param;  //pointer to parameters
    void *p_data;   //pointer to data, eg:weights
} CQT_LAYER;


typedef struct cqt_net_tag {
    int layernum;
    CQT_LAYER layer[CQT_MAX_LAYER_NUM];
} CQT_NET;