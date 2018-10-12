import tensorflow as tf
import numpy as np
from keras import backend as K

# Custom metric: Mean IOU found for thresholds: 0.5 0.55 ... 0.95
def mean_iou(y_true, y_pred):
	prec = []
	for t in np.arange(0.50, 1.0, 0.05):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(labels=y_true,predictions = y_pred_, num_classes = 2, weights = y_true) # Confusion matrix of [num_classes, num_classes]
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
		return K.mean(K.stack(prec), axis=0)
    