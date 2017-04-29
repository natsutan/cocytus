# はじめに
コキュートスとは、組込向けDeep Learningフレームワークである。Kerasから、ネットワーク構成と学習結果を読みだし、移植性の高いＣソースコードを生成する。コキュートスは以下の特徴を持つ。
- 組込み向けＣソースコードの生成機能
- 生成されたＣソースとKerasの出力を比較するデバッグ機能

本マニュアルは、コキュートスの使用方法について記述したものである。

# コキュートスとは
## コキュートスの特徴



## コキュートスの処理手順

コキュートスの処理手順を以下に示す。

![処理手順](./img/cocytus_arch.png)

以下の順序で処理を行う。

- 最初にKerasでネットワークの定義、および学習を行う。
- Kerasからネットワークの情報をjson形式で、重みデータをhdf5形式で出力する。
- コキュートスは2つのファイルを読み込み、コキュートス初期設定ファイル(Cqt_gen)、コキュートスライブラリ(Cqt_lib)、コキュートス重みファイル(numpy形式)を生成する。
- コキュートス初期設定ファイル(Cqt_gen)、コキュートスライブラリ(Cqt_lib)と、ユーザープログラムをリンクし、実行ファイルを作成する。
- 実行時にはコキュートス重みファイルを読みこみ、ＮＮを動作させる。
- 必要に応じて、Ｋｅｒａｓの出力と、コキュートスの出力を比較する事ができる。

# 使い方

## 起動方法
コマンドラインからcocytus.pyを実行する。
```
python cocuytus.py iniファイル
```


## 設定一覧
設定の一覧を以下に示す。

Table: オプション一覧

| オプション名 |  iniファイルセクション|iniファイルエントリー |設定例|内容|
|:-----------|:------------|:------------|:------------|:--------|
| ネットワーク指定|  [Cocyuts] | keres_json | ~/foo/baa.json | 変換するネットワーク(jsonファイル)を指定する。（必須)|
| 重み指定|  [Cocyuts] | keras_weight | ~/foo/baa.h5 | 変換する重み(hdf5ファイル)を指定する。（必須)|
| 出力ディレクトリ |  [Cocyuts] | output_dir | ~/proj/ | ファイルの出力先（必須)|
| Ｃライブラリディレクトリ |  [Cocyuts] | c_lib_dir | ./../../cocytus/cocytus_net/C/template | Cライブラリ指定（必須)|
| 重み出力ディレクトリ|  [Cocyuts] | weight_output_dir | ~/proj/weight/ | コキュートス重みファイルの出力先。省略されるとコキュートス重みファイルを生成しません。|
| 重みファイル名モード |[Cocyuts] | weight_filename_mode | 1 |Kerasから生成される重みファイル名のmodeを指定する。0:VGG16モード（デフォルト）。Conv2dの重みファイル末尾が_W_1_z.npyの時 1:YOLOモード　Conv2dの重みファイル末尾が_kernel_z.npyの時 |
| Conv2d最適化レベル |  [CGEN] | Conv2d_OPTLEVEL | dash | CONV2dレイヤの最適化レベルを指定します。dashで最適化をONにします。デフォルト値は最適化無しです。|



## iniファイル例
iniファイルの例を以下に示す。

```
[Cocyuts]
; 変換するネットワーク(jsonファイル)を指定する。
keres_json=keras/vgg16.json

; ファイルの出力先
output_dir=c

; コキュートス重みファイルの出力先。省略されるとコキュートス重みファイルを生成しません。
;weight_output_dir=c/weight

; 変換する重み(hdf5ファイル)を指定する。
keras_weight=~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5

; C言語のテンプレートディレクトリ
c_lib_dir=../../cocytus/cocytus_net/C/template/
```

# コキュートスAPI
## cqt_init

## cqt_load_weight_from_files

## cqt_run

## load_from_numpy

## save_to_numpy

# コキュートスが生成するファイル
## 共通Cヘッダーファイル
### cqt.h
### cqt_type.h
### cqt_keras.h
### numpy.h

## コキュートス実行用ソース
### cqt_gen.h
### cqt_gen.c

## Ｃライブラリ(cqt_lib)
### cqt_lib.h
### cqt_lib.c

# ディレクトリ構成

- cocytus
    - *<b>cocytus.py</b>*
    - compiler
        - *<b>compiler.py</b>*
    - cocytus_net
        - acivation
        - convolution
        - etc..
    - weight_converter
    - util
- test
    - *<b>test.py</b>*
    - python_test
    - c_test
- doc
    - manual マニュアル
    - spec　　仕様書
    - users_guide ユーザーズガイド

- tools
- example
    - vgg16

参考：
http://www.hexacosa.net/pph_ja/
