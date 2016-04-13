import numpy as np
import tensorflow as tf
from skflow.io.data_feeder import get_batch

class StackedAutoEncoder:
    """A deep autoencoder
    Parameters:
        x: A 2D feature matrix
        dims: array containing dimensions of each hidden layer. The size of the array will provide the no. of layers to use.
        epoch: default 1000. this is the epoch to be used for each layer-wise training
        batch_size: Mini batch size.
        noise: "gaussian" or "mask-0.4". mask-0.4 means that 40% random input for each example will be set to 0. if noise is not given,
	it acts like a autoencoder without denoising capability.
     """

    def __init__(self, x, dims, epoch=1000, noise=None):
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.x = x
        self.depth = len(dims)

    def add_noise(self):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(self.x), len(self.x[0])))
            return self.x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(self.x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def encode(self):
        """Returns the final encoded learnt feature representations."""
        ae=None
        for i in range(self.depth):
            if self.noise is None:
                ae = BasicAutoEncoder(data_x=self.x, data_x_=self.x, hidden_dim=self.dims[i], epoch=self.epoch)
            else:
                ae = BasicAutoEncoder(data_x=self.add_noise(), data_x_=self.x, hidden_dim=self.dims[i], epoch=self.epoch)
            ae.run()
            self.x = ae.get_hidden_feature()
        return self.x


class BasicAutoEncoder:
    """A basic autoencoder with a single hidden layer"""

    def __init__(self, data_x, data_x_, hidden_dim, epoch=1000, batch_size=50):
        self.data_x_ = data_x_
        self.data_x = data_x
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.input_dim = len(data_x[0])
        self.hidden_feature = []
        # print(data_x.shape, data_x_.shape, hidden_dim)

    def forward(self, x):
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim], dtype=tf.float32),
                                  name='weights')

            biases = tf.Variable(tf.zeros([self.hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases, name='encoded')

        with tf.name_scope('decode'):
            biases = tf.Variable(tf.zeros([self.input_dim]), name='biases')
            decoded = tf.matmul(encoded, tf.transpose(weights)) + biases
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x, name='cross_entropy'))
        return encoded, decoded

    def train(self, x_, decoded):
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        return loss, train_op

    def run(self):
        with tf.Graph().as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x')
            x_ = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x_')
            encoded, decoded = self.forward(x)
            loss, train_op = self.train(x_, decoded)
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                for i in range(self.epoch):
                    for j in range(50):
                        b_x, b_x_ = get_batch(self.data_x, self.data_x_, self.batch_size)
                        sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
                    if i % 100 == 0:
                        l = sess.run(loss, feed_dict={x: self.data_x, x_: self.data_x_})
                        print('epoch {0}: global loss = {1}'.format(i, l))
                self.hidden_feature = sess.run(encoded, feed_dict={x: self.data_x_})
                # print(sess.run(decoded, feed_dict={x: self.data_x})[0])

    def get_hidden_feature(self):
        return self.hidden_feature