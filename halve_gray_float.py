import tensorflow as tf
import numpy as np
from skimage import color
from skimage import io

in_img = io.imread('input.jpg')
in_img = color.rgb2gray(in_img)
io.imsave('ori_gray.jpg', np.squeeze(in_img))

in_data = np.expand_dims(in_img, axis=0)
in_data = np.expand_dims(in_data, axis=3)

filter_data = np.array([0.5]).reshape(1,1,1,1).astype(np.float32)
filter = tf.Variable(filter_data)
x = tf.placeholder(tf.float32, shape=[1, None, None, 1], name='dnn_in')
y = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='VALID', name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'halve_gray_float.pb', as_text=False)

print("halve_gray_float.pb generated, please use \
path_to_ffmpeg/tools/python/convert.py to generate halve_gray_float.model\n")

output = sess.run(y, feed_dict={x: in_data})
output = output * 255.0
output = output.astype(np.uint8)
io.imsave("out.jpg", np.squeeze(output))
