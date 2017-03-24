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


typedef struct {


} LY_InputLayer;

typedef struct {


} LY_Convolution2D;

typedef struct {

} LY_MaxPooling2D;

typedef struct {


} LY_Flatten;

typedef struct {

} LY_Dense;
