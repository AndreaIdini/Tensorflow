import tensorflow as tf
# tutorial from:
# https://www.tensorflow.org/get_started/mnist/pros
# how to deep learn writing recognition

from tensorflow.examples.tutorials.mnist import input_data #import the example

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #read it


"""Here we instead use the convenient InteractiveSession class, which makes TensorFlow
more flexible about how you structure your code. It allows you to interleave operations
which build a computation graph with ones that run the graph. This is particularly
convenient when working in interactive contexts like IPython.
If you are not using an InteractiveSession, then you should build the entire
computation graph before starting a session and launching the graph."""

sess = tf.InteractiveSession()

# These placeholder are tensors of rank 2 (matrix)
x = tf.placeholder(tf.float32, shape=[None, 784]) # first dimension None can be of any size,
                                                  # second dimension is 784 = 28x28, pixel size of image
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 0 to 10 for the possible outcomes

# Trainable Variables, fill it with zeros
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Initialize variables
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b #Again linear model

# cost function is the softmax of the cross_entroy
# softmax_i(y)=exp(y_i)/sum_j(exp(y_j))
# cross entropy is \sum_i q_i * log(p_i), over two distributions q and p
# cf. http://colah.github.io/posts/2015-09-Visual-Information/

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# define training step as gradient descent over cross entropy respect to
# the traning set and the linear regression.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  batch = mnist.train.next_batch(100) #we get 100 random data from the training set
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #run the traning defined above

# check if the prediction of y matches the training set y_, creates an array of [True,False...]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# if you cast a float value the array becomes [0,1,...], its mean is the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Accuracy should be around 92%, we can do better with next training #

# ------ Convolutional Network ------ #

# We have to initialize weight in the neural networks slightly positive, possibly randomized,
# instead of pure zeros. Since in tensorflow variable have to be inizialized,
# it is convenient to define functions.

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# computes a 2D convolution of data x and filter W.
# https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
#
# uses a stride of one (scans through every index) and is zero padded (no borders of zeros to preserve size)
# so the output is the same size as the input

# input:
# x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
# W is the filter tensor - should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
# strides is a 4-D tensor that defines how the filter slides over the input tensor in each of the 4 dimensions
# ex. if strides=[1, 1, 1, 1] and padding='SAME' the filter is centered at every pixel from the image
# padding - if it is set to "VALID" it means that there is no padding.

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# performs max pooling over 2x2 blocks, in other words it downsamples by striding 2 by 2.
# this will lighten the calculation and reduce overfitting

# input shapes:
# x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
# ksize has the same dimensionality as the input tensor. It defines the patch size. It extracts the max
# value out of each such patch. Here the patch we define is a 2x2 block
# strides is a 4-D tensor that defines how the patch slides over the input tensor
# if the padding is "SAME" there is padding, if it is "VALID" there is no padding
# For the SAME padding, the output height and width are computed as:
#     out_height = ceil(float(in_height) / float(strides1))
#     out_width = ceil(float(in_width) / float(strides[2]))
# For the VALID padding, the output height and width are computed as:
#     out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
#     out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
# example: if x is an image of shape [2,3] and has 1 channel (so the input shape is [1, 2, 3, 1])
# , we max pool with 2x2 kernel and the stride is 2
# if the pad is VALID (valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID'))
#  the output is of shape [1, 1, 1, 1]
# if the pad is SAME we pad the image to the shape [2, 4];
# (same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')) the output is of shape [1, 1, 2, 1]
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# ------ Actual Network ------ #

# - First Layer

W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5 patch, 1 in channel, 32 outputs.
b_conv1 = bias_variable([32])            # bias for each input
# reshape the variable to a 4-tensor, with same size, 28x28 size for image, and 1 color channel (B&W picture)
x_image = tf.reshape(x, [-1,28,28,1])

#use ReLU f(x)=(max(0,x)) as activator of the node for every node
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# - Second Layer
W_conv2 = weight_variable([5, 5, 32, 64]) # Num of in channels as the output of the first layer
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# - Densily Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 64 Input layer, of 7x7 images, 1024 output layers
b_fc1 = bias_variable([1024])               # Bias for 1024 output layers

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy % 0.4f"%(i, train_accuracy))
#    print sess.run(h_conv1, feed_dict={
#                    x:batch[0], y_: batch[1], keep_prob: 1.0})

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy % 0.4f"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
