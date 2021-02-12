import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
import tensorflow as tf
import extra_keras_datasets.emnist as emnist
# from extra_keras_datasets import emnist

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

batch_size = 128  # 訓練データを128(仮)ずつのデータに分けて学習させる
num_classes = 26 # 分類させる数。アルファベットなので26。
epochs = 20 # 訓練データを繰り返し学習させる数

# 訓練データ,ラベル(t_images, t_labels)とテストデータ,ラベル（v_images, v_labels)を取得する
# type="letters"の場合、28×28の3次元の画像データでtrainが124800、testが20800である
(t_images, t_labels), (v_images, v_labels) = emnist.load_data(type="letters")

# 元のデータは3次元の配列なので、4次元配列に整形する
def reshape_image(imgs):
    imgs = imgs.astype("float32")
    imgs = imgs / 255.0
    imgs = imgs.reshape(-1, 28, 28, 1)
    return imgs

# t_images = t_images.reshape(124800, 28, 28, 1)
# v_images = v_images.reshape(20800, 28, 28, 1)
t_images = reshape_image(t_images)
v_images = reshape_image(v_images)
t_images = t_images.astype('float32')
v_images = v_images.astype('float32')
print(t_images.shape[0], 'train samples')
print(v_images.shape[0], 'test samples')


# モデル構築
# 畳み込み、プーリング、ドロップアウトの組み合わせで、最後は完全結合でsoftmaxでクラス分け
# 隠れ層の活性化関数には定番のReLUを使用
# TensorFlowを使う場合、最初の隠れ層だけ入力データの形式をinput_shapeで指定する必要がある
# 28x28ピクセルで1チャンネルの画像データが入力データになるため、input_shape=(28, 28, 1)と指定。2層目以降は、前の層の出力から自動で判断
# 畳み込み層を12層、プーリング層を3層用いた
model = keras.Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28,28, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2), padding="same"),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2), padding="same"),
        Dropout(0.4),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2), padding="same"),
        Dropout(0.4),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(27, activation="softmax"),
    ]
)

# モデルのコンパイル
#モデルはoptimizersを使用
opt = keras.optimizers.Adam()
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

# モデルの概要
model.summary()

# 学習は、scrkit-learnと同様fitで記述できる
history = model.fit(t_images, t_labels,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(v_images, v_labels))

# 評価はevaluateで行う
score = model.evaluate(v_images, v_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
