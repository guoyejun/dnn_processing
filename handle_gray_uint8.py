import tensorflow as tf
import numpy as np
from skimage import color
from skimage import io

in_img = io.imread('input.jpg')
in_img = color.rgb2gray(in_img)
io.imsave('ori_gray.jpg', np.squeeze(in_img))

in_img = in_img * 255;
in_img = in_img.astype(np.uint8)
in_data = np.expand_dims(in_img, axis=0)
in_data = np.expand_dims(in_data, axis=3)

x = tf.placeholder(tf.uint8, shape=[1, None, None, 1], name='dnn_in')
y = x + 2
y = tf.identity(y, name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'handle_gray_uint8.pb', as_text=False)

output = sess.run(y, feed_dict={x: in_data})
output = output.astype(np.uint8)
io.imsave("out.jpg", np.squeeze(output))
