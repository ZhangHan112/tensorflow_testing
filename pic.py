# import matplotlib.image as mp_image
# import matplotlib.pyplot as plt
# import tensorflow as tf
# filename = "download.jpeg"
# input_image = mp_image.imread(filename)



import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf

filename = "download.jpeg"
input_image = mp_image.imread(filename)
my_image = tf.placeholder("int64",[None,None,3])

slice = tf.slice(my_image,[10,0,0],[16,-1,-1])

with tf.Session() as sess:
    result = sess.run(slice,feed_dict={my_image: input_image})
    print(result.shape)

plt.imshow(result)
plt.show()
# print ('input dim = {}'.format(input_image.ndim))
# print ('input shape = {}'.format(input_image.shape))