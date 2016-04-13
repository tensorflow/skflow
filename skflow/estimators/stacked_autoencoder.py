import numpy as np
import tensorflow as tf
from skflow.io.data_feeder import get_batch

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


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

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert self.epoch > 0, "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert self.noise in allowed_noises, "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse', lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.ae = None
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        """Fits the autoencoder on a data"""
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                self.ae = BasicAutoEncoder(data_x=x, activation=self.activations[i], data_x_=x,
                                           hidden_dim=self.dims[i], epoch=self.epoch, loss=self.loss,
                                           batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)
            else:
                self.ae = BasicAutoEncoder(data_x=self.add_noise(x), activation=self.activations[i], data_x_=x,
                                           hidden_dim=self.dims[i],
                                           epoch=self.epoch, loss=self.loss, batch_size=self.batch_size, lr=self.lr,
                                           print_step=self.print_step)
            x, w, b = self.ae.run()
            self.weights.append(w)
            self.biases.append(b)

    def transform(self, data):
        """Transforms the autoencoder on a data"""
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.ae.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        """Fits and transforms the autoencoder on same data"""
        self.fit(x)
        return self.transform(x)


class BasicAutoEncoder:
    """A basic autoencoder with a single hidden layer. This is not to be externally used but internally by
    StackedAutoEncoder"""

    def __init__(self, data_x, data_x_, hidden_dim, activation, loss, lr, print_step, epoch=1000, batch_size=50):
        self.print_step = print_step
        self.lr = lr
        self.loss = loss
        self.activation = activation
        self.data_x_ = data_x_
        self.data_x = data_x
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.input_dim = len(data_x[0])
        self.hidden_feature = []
        self.encoded = None
        self.decoded = None
        self.x = None
        self.x_ = None

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def train(self, x_, decoded):
        if self.loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        else:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, train_op

    def run(self):
        sess = tf.Session()
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x')
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal([self.input_dim, self.hidden_dim], dtype=tf.float32)),
                  'biases': tf.Variable(tf.truncated_normal([self.hidden_dim], dtype=tf.float32))}
        encoded_vals = tf.matmul(self.x, encode['weights']) + encode['biases']
        self.encoded = self.activate(encoded_vals, self.activation)
        decode = {'biases': tf.Variable(tf.truncated_normal([self.input_dim], dtype=tf.float32))}
        self.decoded = tf.matmul(self.encoded, tf.transpose(encode['weights'])) + decode['biases']
        loss, train_op = self.train(self.x_, self.decoded)
        sess.run(tf.initialize_all_variables())
        for i in range(self.epoch):
            b_x, b_x_ = get_batch(self.data_x, self.data_x_, self.batch_size)
            sess.run(train_op, feed_dict={self.x: b_x, self.x_: b_x_})
            if (i + 1) % self.print_step == 0:
                l = sess.run(loss, feed_dict={self.x: self.data_x, self.x_: self.data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
                # print(sess.run(encode['weights'])[0])
        # debug
        # print('Decoded', sess.run(self.decoded, feed_dict={self.x: self.data_x_})[0])
        return sess.run(self.encoded, feed_dict={self.x: self.data_x_}), sess.run(
            encode['weights']), sess.run(encode['biases'])
