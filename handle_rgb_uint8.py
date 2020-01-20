import tensorflow as tf
import numpy as np
import imageio

in_img = imageio.imread('input.jpg')
in_data = in_img[np.newaxis, :]

x = tf.placeholder(tf.uint8, shape=[1, None, None, 3], name='dnn_in')
y = tf.math.add(x, 10, name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'handle_rgb_uint8.pb', as_text=False)

output = sess.run(y, feed_dict={x: in_data})
imageio.imsave("out.jpg", np.squeeze(output))
