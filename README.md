# tensor-flow-mnist-model-for-android-maker

MNISTのモデルをKerasを使って構築し、TensorFlowで読める形式のファイルに変換するためのスクリプトです。

python3で動きます。

# コンポーネントの紹介

- make_model_keras.py <BR> Kerasによるmodel構築スクリプト。`python3 make_model_keras.py` でモデル作成を行い、DFH5ファイル(h5ファイル)として保存します。
  
  
- keras_to_tensorflow.py<BR>DFH5ファイルをpbファイルに変換します。`python3 keras_to_tensorflow.py [入力h5ファイル] [出力先]` で変換処理を行います。
 
# 必要なpythonライブラリ

- TensorFlow
- Keras
- h5py
