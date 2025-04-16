# Face Emotion Classification モデル

## 概要
このプロジェクトは顔の表情から感情を分類するための畳み込みニューラルネットワーク(CNN)モデルを実装しています。事前学習済みモデルを読み込み、必要に応じて微調整（ファインチューニング）を行うことができます。

## モデル構造
- 入力サイズ: 64x64ピクセルのグレースケール画像
- 畳み込み層、バッチ正規化、活性化関数、セパラブル畳み込み層の組み合わせ
- 7クラス分類（感情カテゴリ）
- パラメータ総数: 58,423

## 必要条件
- Python 3.6
- Keras (TensorFlowバックエンド)
- numpy
- matplotlib
- OpenCV（画像処理用）

## 使用方法

### モデルの読み込み
```python
from keras.models import load_model
model = load_model("./trained_models/emotion_models/fer2013_mini_XCEPTION.hdf5")
```

### ファインチューニング
```python
# 最初の10層を凍結
for layer in model.layers[:10]:
    layer.trainable = False

# オプティマイザーの設定
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# データ拡張の設定
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1.0/255
)

# 訓練
train_generator = datagen.flow_from_directory(
    "データセットのパス",
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

model.fit(train_generator, epochs=10, verbose=1)

# モデルの保存
model.save("fine_tuned_model.hdf5")
```

## データセット
FER2013データセット（または同様の表情データセット）を使用して訓練できます。データセットは7つの感情カテゴリに分類された顔画像で構成されています。

## 推論方法
```python
import cv2
import numpy as np

# 画像の読み込みと前処理
image = cv2.imread("test_image.jpg", 0)  # グレースケールで読み込み
image = cv2.resize(image, (64, 64))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=-1)

# 予測
prediction = model.predict(image)
emotion_labels = ['怒り', '嫌悪', '恐怖', '喜び', '悲しみ', '驚き', '無表情']
predicted_emotion = emotion_labels[np.argmax(prediction)]
```

## 参考文献
このモデルはXceptionアーキテクチャを基にした軽量版モデルを使用しています。

- [オリジナルモデル（GitHub）](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)
- [FER2013データセット（Kaggle）](https://www.kaggle.com/datasets/msambare/fer2013)
- [画像例](https://github.com/user-attachments/assets/4a52bdd1-ffbe-4ad5-bb41-7fb1f4b1195f)
- [表情認識の実装（Qiita）](https://qiita.com/k-keita/items/e27e4eefc8c009ecdeab) 
