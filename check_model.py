import sys
import numpy as np
import matplotlib.pyplot as pp
import tensorflow as tf

#@tf.keras.utils.register_keras_serializable()
#def mae_qty(y_true, y_pred):
#  qty=y_true[:,:,2]
#  mae=tf.reduce_sum(tf.abs((y_true[:,:, :2]-y_pred)), axis=-1)
#  loss=mae*qty
#  return tf.reduce_sum(loss)

@tf.keras.utils.register_keras_serializable()
def abs_qty(y_true, y_pred):
  qty=y_true[:, :,1]
  mae=tf.abs(y_true[:,:, 0]-y_pred[:, 0])
  loss=mae*qty
  return tf.reduce_sum(loss)

@tf.keras.utils.register_keras_serializable()
def phase_qty(y_true, y_pred):
  qty=y_true[:,:,1]
  mae=tf.abs(y_true[:,:, 0]-y_pred[:,1])
  loss=mae*qty
  return tf.reduce_sum(loss)

model = tf.keras.models.load_model('prototype2.keras', custom_objects={'abs_qty': abs_qty, 'phase_qty': phase_qty})
model.summary()
with open('20210503.0000.14.ekb.fitacf', 'r') as f:
    for s in f:
        d = np.array(s.split(',')).astype(float)
        p = d[3:]
        x = np.array([p[0] * p[1], p[0] * p[2], p[3]]).reshape((1, 1, 3))

        y_pred = model.predict(x)
        y_pred = np.array(y_pred)

        fig, axs = pp.subplots(1, 2, constrained_layout=True, figsize=(8, 4))

        axs[0].set_title('Abs Prediction')
        axs[0].plot(np.log(p[:-1:3] + 1), label='Actual Abs', color='blue')
        axs[0].plot(y_pred[0][0][:, 0], color='green', label='Predicted Abs')
        axs[0].legend()

        axs[1].set_title('Phase Prediction')
        axs[1].plot(p[1:-1:3] + np.pi, label='Actual Phase', color='blue')
        axs[1].plot(y_pred[1][0][:, 0], color='green', label='Predicted Phase')
        axs[1].legend()

        pp.show()
