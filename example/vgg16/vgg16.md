# VGG 16 サンプル

## 使い方
keras ディレクトリへ移動し、vgg16.pyを実行する。
```
cd keras
python vgg16.py
```
初回の実行時には、学習済重みファイルが、 ~/.Keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5 にダウンロードされる。

## Cソースの作成
vgg16ディレクトリに戻り、コキュートスを起動する。
```
cd ..
python ../../cocytus.py vgg16.ini
```

