""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Set random seed
op_seed = 3
graph_seed = 42
tf.set_random_seed(graph_seed)

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 50

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input], name="input_image")
X_2dimage = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image("input", X_2dimage, 3)
Y = tf.placeholder("float", [None, num_classes], name="input_label")

def create_weight(num_in, num_out):
    return tf.Variable(tf.random_normal([num_in, num_out], seed=op_seed, name="weights"))

def create_biase(num):
    return tf.Variable(tf.random_normal([num], seed=op_seed, name="biases"))

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    with tf.name_scope('hidden_layer_1'):
        w1 = create_weight(num_input, n_hidden_1)
        b1 = create_biase(n_hidden_1)
        tf.summary.histogram("weights", w1)
        tf.summary.histogram("biases", b1)
        layer_1 = tf.add(tf.matmul(x, w1), b1)

    # Hidden fully connected layer with 256 neurons
    with tf.name_scope('hidden_layer_2'):
        w2 = create_weight(n_hidden_1, n_hidden_2)
        b2 = create_biase(n_hidden_2)
        tf.summary.histogram("weights", w2)
        tf.summary.histogram("biases", b2)
        layer_2 = tf.add(tf.matmul(layer_1, w2), b2)

    # Output fully connected layer with a neuron for each class
    with tf.name_scope('out_layer'):
        w_out = create_weight(n_hidden_2, num_classes)
        b_out = create_biase(num_classes)
        out_layer = tf.matmul(layer_2, w_out) + b_out
    return out_layer


# Construct model
logits = neural_net(X)

prediction = tf.nn.softmax(logits, name="prediction")

# Define loss and optimizer
with tf.name_scope('softmax_cross_entropy_loss'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    tf.summary.scalar('loss', loss_op)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Evaluate model
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Add session graph to tensorboard
    train_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)
    # Run the initializer
    sess.run(init)

    merged_summary = tf.summary.merge_all()

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=False)
        # Run optimization op (backprop)
        summary, _ = sess.run([merged_summary,train_op], feed_dict={X: batch_x, Y: batch_y})
        train_writer.add_summary(summary, step)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))
