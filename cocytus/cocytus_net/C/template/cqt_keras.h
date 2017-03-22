#pragma once
#include <stdbool.h>

#include "cqt_type.h"
#include "numpy.h"

typedef enum {
    //Core Layers
    LT_Dense,
    LT_Activation,
    LT_Dropout,
    LT_Flatten,
    LT_Reshape,
    LT_Permute,
    LT_RepeatVector,
    LT_Lambda,
    LT_ActivityRegularization,
    LT_Masking,

    //Convolutional
    LT_Conv1D,
    LT_Conv2D,
    LT_SeparableConv2D,
    LT_Conv2DTranspose,
    LT_Conv3D,
    LT_Cropping1D,
    LT_Cropping2D,
    LT_Cropping3D,
    LT_UpSampling1D,
    LT_UpSampling2D,
    LT_UpSampling3D,
    LT_ZeroPadding1D,
    LT_ZeroPadding2D,
    LT_ZeroPadding3D,

    //Pooling Layers
    LT_MaxPooling1D,
    LT_MaxPooling2D,
    LT_MaxPooling3D,
    LT_AveragePooling1D,
    LT_AveragePooling2D,
    LT_AveragePooling3D,
    LT_GlobalMaxPooling1D,
    LT_GlobalAveragePooling1D,
    LT_GlobalMaxPooling2D,

    //Locally-connected Layers
    LT_LocallyConnected1D,
    LT_LocallyConnected2D,

    //Recurrent Layers
    LT_Recurrent,
    LT_SimpleRNN,
    LT_GRU,
    LT_LSTM,

    //Embedding Layers
    LT_Embedding,

    //Merge Layers
    LT_Add,
    LT_Multiply,
    LT_Average,
    LT_Maximum,
    LT_Concatenate,
    LT_Dot,
    LT_add,
    LT_multiply,
    LT_average,
    LT_maximum,
    LT_concatenate,
    LT_dot,

    //Advanced Activations Layers
    LT_LeakyReLU,
    LT_PReLU,
    LT_ELU,
    LT_ThresholdedReLU,

    //Normalization Layers
    LT_BatchNormalization,

    //Noise layers
    LT_GaussianNoise,
    LT_GaussianDropout,

    //Layer wrappers
    LT_TimeDistributed,
    LT_Bidirectional
} KR_LAYER_TYPE;
/*


//活性化関数
typedef enum {
    SOFTMAX,
    SOFTPLUS,
    SOFTSIGN,
    RELU,
    TANH,
    SIGMOID,
    HARD_SIGMOID,
    LINEAR,
    //-------
            NO_ACTIVATION
} KR_ACTIVATION;

typedef enum {
    BD_VALID,
    BD_SAME,
    //-------
            BD_NONE
} KR_BOADER_MODE;


typedef enum {
    RG_L1,
    RG_L2,
    RG_L1L2,
    RG_ACTIVITY_L1,
    RG_ACTIVITY_L2,
    RG_ACTIVITY_L1L2,
    //--------
            RG_NONE
} KR_REGULARIZER;


// 未使用　init, , W_constraint,  dim_ordering, b_constraint, trainable,
typedef struct LY_Convolution2D_tag {
    int nb_filter;
    int nb_row;
    int nb_col;
    KR_ACTIVATION activation;
    int batch_input_shape[4];
    KR_BOADER_MODE border_mode;
    KR_REGULARIZER activity_regularizer;
    KR_REGULARIZER W_regularizer;
    KR_REGULARIZER b_regularizer;
    bool bias;
    NN_DTYPE input_dtype;
    int subsample[2];
    //
    int nnn_input_shape[4];
    void *nnn_whp;
    void *nnn_wp;
    void *nnn_bhp;
    void *nnn_bp;
} LY_Convolution2D;


typedef struct LY_Activation_tag {
    KR_ACTIVATION activation;
    //
    int nnn_input_shape[4];
} LY_Activation;


typedef struct LY_Dropout_tag {
    float p;
    //
    int nnn_input_shape[4];
} LY_Dropout;


typedef struct LY_Flatten_tag {
    /* nothing */
    int nnn_input_shape[4];
} LY_Flatten;


typedef struct LY_Dense_tag {
    int input_dim;
    int output_dim;
    KR_ACTIVATION activation;
    KR_REGULARIZER W_regularizer;
    KR_REGULARIZER b_regularizer;
    KR_REGULARIZER activity_regularizer;
    bool bias;
    //
    void *nnn_whp;
    void *nnn_wp;
    void *nnn_bhp;
    void *nnn_bp;

} LY_Dense;

typedef struct LY_MaxPooling2D_tag {
    int strides[2];
    int pool_size[2];
    KR_BOADER_MODE border_mode;
    //
    int nnn_input_shape[4];
} LY_MaxPooling2D;
*/