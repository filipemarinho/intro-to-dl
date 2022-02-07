import numpy as np
import tensorflow as tf

tf.reset_default_graph()
x = tf.get_variable("x", shape=(), dtype=tf.float32) #trainable = True by default
f = x**2

#minimizing the value of f with respect to x:
optimizer = tf.train.GradientDescentOptimizer(0.1) #learning rate of 0.1

#one step operation:
step = optimizer(f, var_list=[x]) #var_list can be omitted because all variables are trainable by default

#get all trainable variables
tf.trainable_variables()

#Creating a session and initializing variables:
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(10):
    _, curr_x, curr_f = s.run([step, x, f])
    print(curr_x, curr_f)


