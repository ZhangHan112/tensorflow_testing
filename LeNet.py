import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
  
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
  
sess = tf.InteractiveSession()  
  
# 訓練資料
x = tf.placeholder("float", shape=[None, 784])  
# 訓練標籤資料
y_ = tf.placeholder("float", shape=[None, 10])  
# 把x更改為4維張量，第1維代表樣本數量，第2維和第3維代表影象長寬， 第4維代表影象通道數, 1表示黑白
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一層：卷積層
# 過濾器大小為5*5, 當前層深度為1， 過濾器的深度為32
conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))
# 移動步長為1, 使用全0填充
conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
# 啟用函式Relu去線性化
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
  
#第二層：最大池化層  
#池化層過濾器的大小為2*2, 移動步長為2，使用全0填充  
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
#第三層：卷積層  
conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1)) #過濾器大小為5*5, 當前層深度為32， 過濾器的深度為64  
conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))  
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #移動步長為1, 使用全0填充  
relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  
  
#第四層：最大池化層  
#池化層過濾器的大小為2*2, 移動步長為2，使用全0填充  
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
#第五層：全連線層  
fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1)) #7*7*64=3136把前一層的輸出變成特徵向量  
fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))  
pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)  
  
#為了減少過擬合，加入Dropout層  
keep_prob = tf.placeholder(tf.float32)  
fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
  
#第六層：全連線層  
fc2_weights = tf.get_variable("fc2_weights", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1)) #神經元節點數1024, 分類節點10  
fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))  
fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
  
#第七層：輸出層  
# softmax  
y_conv = tf.nn.softmax(fc2)  
  
#定義交叉熵損失函式  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  
  
#選擇優化器，並讓優化器最小化損失函式/收斂, 反向傳播  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
# tf.argmax()返回的是某一維度上其資料最大所在的索引值，在這裡即代表預測值和真實值  
# 判斷預測值y和真實值y_中最大數的索引是否一致，y的值為1-10概率  
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  
  
# 用平均值來統計測試準確率  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  
#開始訓練  
sess.run(tf.global_variables_initializer())  
for i in range(10000):  
    batch = mnist.train.next_batch(100)  
    if i%100 == 0:  
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}) #評估階段不使用Dropout  
        print("step %d, training accuracy %g" % (i, train_accuracy))  
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #訓練階段使用50%的Dropout  
  
  
#在測試資料上測試準確率  
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))