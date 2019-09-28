import tensorflow as tf
import numpy as np

matrix1 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype='int32')
matrix2 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype='int32')

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

matrix_product = tf.matmul(matrix1,matrix2)
matrix_sum = tf.add(matrix1,matrix2)


# with tf.Session() as sess:
#     result = sess.run()

# print(matrix_product)
# print(matrix_sum)


with tf.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)


print ("matrix1*matrix2")
print (result1)

print ("matrix1+matrix2")
print (result2)

