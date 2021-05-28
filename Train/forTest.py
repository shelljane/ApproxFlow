import tensorflow as tf
from tensorflow.python.ops.array_ops import fake_quant_with_min_max_vars

a = tf.Variable([0.0, 0.1, 0.3, 0.49, 0.5, 0.8, 1.1, 1.23, 1.49, 1.5, 1.51, 2.0])
qa = fake_quant_with_min_max_vars(a, tf.reduce_min(a), tf.reduce_max(a), num_bits=3, narrow_range=False)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(a))
print(sess.run(qa))
