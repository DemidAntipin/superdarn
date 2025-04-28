import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
@tf.keras.utils.register_keras_serializable()
def abs_qty(y_true, y_pred):
  qty=y_true[:, :,1]
  mae=tf.abs(y_true[:,:, 0]-tf.squeeze(y_pred))
  loss=mae*qty
  return tf.reduce_mean(loss)

@keras.saving.register_keras_serializable()
@tf.keras.utils.register_keras_serializable()
def phase_qty(y_true, y_pred):
  qty = y_true[:,:,1]
  mae = tf.abs(y_true[:,:,0] - tf.squeeze(y_pred))
  loss = mae * qty
  return tf.reduce_mean(loss)

@keras.saving.register_keras_serializable()
@tf.keras.utils.register_keras_serializable()
def extract_first_param(x):
    return x[:, 0, 0]
