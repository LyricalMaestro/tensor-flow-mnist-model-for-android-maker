#coding: utf-8

import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from keras import optimizers
from keras.utils.np_utils import to_categorical
import os

SAVE_MODEL_DIR = "keras_model"

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 入力データ(画像データ)を1次元化
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    # 0,1,…9という整数値でなく、「nである確率」を表す出力ユニットを作るための処理。
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation("sigmoid"))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)
    score = model.evaluate(X_test, y_test, verbose=1)

    print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

    # 学習済みのモデルの保存
    print("model saving…")
    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)

    model.save(os.path.join(SAVE_MODEL_DIR, "modeldata.h5"), overwrite=True)
    print("model saved")

if __name__ == '__main__':
    main()
