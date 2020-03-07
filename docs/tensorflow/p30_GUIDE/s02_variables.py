# # Variables (https://www.tensorflow.org/alpha/guide/variables)

import tensorflow as tf

# ## Creating a Variable
my_variable = tf.Variable(tf.zeros([1, 2, 3]))

# -
with tf.device("/device:GPU:0"):
    v = tf.Variable(tf.zeros([10, 10]))

# ## Using variables
v = tf.Variable(0.0)
w = v + 1
# !w is a tf.Tensor which is computed based on the value of v.
# !Any time a variable is used in an expression it gets automatically converted to a
# !tf.Tensor representing its value.

# -
v = tf.Variable(0.0)
v.assign_add(1)
v.read_value()


# ## Keeping track of variables
class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]


class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)


m = MyOtherModule()
len(m.variables)  # 12; 11 from m.m and another from m.v
