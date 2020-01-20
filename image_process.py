import tensorflow as tf
import numpy as np
import imageio

in_img = imageio.imread('input.jpg')
in_img = in_img.astype(np.float32)/255.0
in_data = in_img[np.newaxis, :]

filter_data = np.array([0.5, 0, 0, 0, 1., 0, 0, 0, 1.]).reshape(1,1,3,3).astype(np.float32)
filter = tf.Variable(filter_data)
x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
y = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='VALID', name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'image_process.pb', as_text=False)

print("image_process.pb generated, please use \
path_to_ffmpeg/tools/python/convert.py to generate image_process.model\n")

output = sess.run(y, feed_dict={x: in_data})
output = output * 255.0
output = output.astype(np.uint8)
imageio.imsave("out.jpg", np.squeeze(output))
