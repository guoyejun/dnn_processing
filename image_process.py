import tensorflow as tf
import numpy as np
import imageio

in_img = imageio.imread('input.jpg')
in_img = in_img.astype(np.float32)/255.0
in_data = in_img[np.newaxis, :]

x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
x1 = tf.minimum(0.7, x)
x2 = tf.maximum(x1, 0.4)
y = tf.identity(x2, name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'image_process.pb', as_text=False)

print("image_process.pb generated, please use \
path_to_ffmpeg/tools/python/convert.py to generate image_process.model\n")

output = sess.run(y, feed_dict={x: in_data})
imageio.imsave("out.jpg", np.squeeze(output))
