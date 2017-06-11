import tensorflow as tf

#Tutorial from:
#https://www.tensorflow.org/get_started/get_started

#Defines constants, that are set once and for all
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

#output: only the type of tensors
print(node1,node2) #(<tf.Tensor 'Const:0' shape=() dtype=float32>, <tf.Tensor 'Const_1:0' shape=() dtype=float32>)

#Tensors have to be evaluated to print their content
sess = tf.Session()
print(sess.run([node1, node2])) #[3.0,4.0]

#Adding operation
node3 = tf.add(node1, node2)
#output: again print only the tensor type, that is an "add" type
print("node3: ", node3) #'node3: ', <tf.Tensor 'Add:0' shape=() dtype=float32>)
#output: in this way it evaulates the tensor and prints the output
print("sess.run(node3): ",sess.run(node3)) #7.0

""" Placeholder are constants that are defined in a second moment.
They can be set at a later moment, eventually changed,
but do not evolve during training and tensorflow networks
"""
#set two placeholder float32
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#set an adder node
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

#output: are evaluated, thus outputs are content of the adder node, with placeholder set to value
#        takes tensors of whatever ranks
print(sess.run(adder_node, {a: 3, b:4.5}))         #  7.5
print(sess.run(adder_node, {a: [1,3], b: [2, 4]})) # [3., 7.]

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))     # 22.5

""" Variables allow to add trainable parameters (unlike placeholders) to a graph.
They are constructed with a type and initial value (unlike placeholders).
However the value assigned has to be initialized before training."""

#Setting up two variables for fitting the parameters of the linear model
#and a placeholder for the "x" and "y" "actual variables" of the model.

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b #setup linear model
y = tf.placeholder(tf.float32) #define y

squared_deltas = tf.square(linear_model - y) #(rmsq deviation)
loss = tf.reduce_sum(squared_deltas)         #reduced sum of rmsq is the Chi (cost function)

#Initializes variables
init = tf.global_variables_initializer()
sess.run(init)    #initializes variables

#defines training sets
x_train = [1,2,3,4]; y_train= [0,-1,-2,-3]

print("starting values") #At the beginning I can calculate y for every x
print(sess.run(linear_model, {x:x_train}))
print(sess.run(loss, {x:x_train, y:y_train})) # and the loss function of the random W and b I put

optimizer = tf.train.GradientDescentOptimizer(0.01) # I can use the gradient descent to train the linear_model
train = optimizer.minimize(loss)                    # minimize the cost function

sess.run(init)  # reset values to defaults.

for i in range(1000):
  sess.run(train, {x:x_train, y:y_train}) # run cost function minimization 1000 times

W_fin, b_fin, loss_fin = sess.run([W, b,loss],{x:x_train, y:y_train}) #assignes values to parking variables

print 'W   = ', W_fin[0]
print 'b   = ', b_fin[0]
print 'err = ', loss_fin
