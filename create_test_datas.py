#coding: utf-8

#
#   データ作成用スクリプト
#

from PIL import Image
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range(len(X_test[1:100])):
    test = X_test[i]
    img = Image.new('RGB', (28, 28))
    for j in range(28):
        for k in range(28):
            pixel = test[j, k]
            img.putpixel((k,j), (pixel, pixel, pixel))

    img.save('test' + str(i) + ".jpg")
