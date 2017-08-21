// Cocytus Header
#pragma once
//-----------------------------
//compile時のdefine上書きで変更可能
#define CQT_MAX_LAYER_NUM   32  //最大レイヤー数
#define CQT_MAX_LAYER_NAME 256  //レイヤー名の最大文字数

//-----------------------------
//戻り値定義
#define CQT_RET_OK (0)      //正常終了
#define CQT_RET_ERR (1)     //エラー

#define CQT_ERR_NO_FILE (2)   //ファイルがオープンできない
#define CQT_NP_HEADER_ERR (3) //numpy読み込み時にヘッダー情報が異常
#define CQT_FREAD_ERR (4) //fread時にエラー発生

//----------------------------
//ファイルのR/W用定義
#define CQT_MAX_PATH 256
#define CQT_NP_BUF_SIZE 256  //numpyで使うバッファサイズ

//---------------------------
//型の宣言
typedef short FIXP16;
typedef signed char FIXP8;
typedef short FP16;

//#include "nnnet_def.h"
//#include "numpy.h"
//#include "nnn_func.h"

//-------------------------
//NEON用外枠、パディング定義
#define NEON_VTR (2)
#define NEON_HTR (4)
#define NEON_HPADDING_0 (0)
#define NEON_HPADDING_1 (1)
#define NEON_HPADDING_2 (2)
#define NEON_HPADDING_3 (3)