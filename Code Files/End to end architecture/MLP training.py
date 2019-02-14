# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from PIL import Image
import numpy as np
import random

# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
path_source2='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
num_notmap=222
num_map=66

# batch_size=10
# mnist=input_data.read_data_sets('/tmp/data/')
# (x_batch, y_batch)=mnist.train.next_batch(batch_size)
# print(type(x_batch))
# print((x_batch[0]))
# print((y_batch))

X_batches=[]
y_batches=[]
data_pair=[]
# Get the image data and store data into X_batches and y_batches
for i in range(num_notmap):
    name_source='NotMap'+str(i+1)+'.png'
    img = Image.open(path_source1+name_source)
    pixel_values=list(img.getdata())
    # X_batches.append(pixel_values)
    # y_batches.append(0)
    data_pair.append(pixel_values+[0])

for i in range(num_map):
    name_source='Map'+str(i+1)+'.png'
    img = Image.open(path_source2+name_source)
    pixel_values=list(img.getdata())
    # X_batches.append(pixel_values)
    # y_batches.append(1)
    data_pair.append(pixel_values+[1])

# Shuffle data_pair as input of Neural Network
random.seed(42)
random.shuffle(data_pair)
data_pair=np.array(data_pair)
len_x=len(data_pair[0])-1
X_batches_255=[data_pair[i][0:len_x] for i in range(data_pair.shape[0])]  
y_batches=[data_pair[i][len_x] for i in range(data_pair.shape[0])]
# data get from last step is with the total value of pixel 255 

for i in range(data_pair.shape[0]):
    X_1img=[X_batches_255[i][j][0]/256.0 for j in range(len_x)]
    X_batches.append(X_1img)
X_batches=np.array(X_batches)
y_batches=np.array(y_batches)
print(X_batches.shape)
print(y_batches.shape)
# print(data_pair[0])

n_inputs=300*250
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs",
                            activation_fn=None)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 400
batch_size = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(num_map+num_notmap // batch_size):
            index_s=iteration*batch_size
            X_batch, y_batch = X_batches[index_s:index_s+batch_size],y_batches[index_s:index_s+batch_size]
            for i in range(len(y_batch)):
                if y_batch[i]==1:
                    y_batch[i]=y_batch[i]-0.0001
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        # acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
        #                             y: mnist.test.labels})
        # print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print(epoch, "Train accuracy:", acc_train)
    save_path = saver.save(sess, "./my_model_final.ckpt")

