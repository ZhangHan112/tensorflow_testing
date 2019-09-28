from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

total_epoch = 5
backup_epoch = 1

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#設計模型
def create_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
  model.add(tf.keras.layers.Dense(units=128,activation='sigmoid'))
  model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model




#備份資訊初始化
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=backup_epoch)



#從頭train model，並將結果放入callback中
# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit(x_train, 
#               y_train,
#               batch_size=100,
#               epochs=total_epoch, 
#               callbacks=[cp_callback],
#               validation_data=(x_test,y_test),
#               verbose=0)


#從被中斷的地方，找出最新train出來的model版本，繼續train下去
# model = create_model()
# model.load_weights(latest)

# model.fit(x_train, 
#           y_train,  
#           epochs=10,
#           validation_data=(x_test,y_test),
#           callbacks=[cp_callback]) 
# loss, acc = model.evaluate(x_test, y_test)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))



#檢查還沒train好的model目前的accuracy

# model = create_model()
# model.load_weights(latest)
# loss, acc = model.evaluate(x_test,y_test)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))