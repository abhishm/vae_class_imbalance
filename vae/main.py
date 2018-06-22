import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

digits_low_samples = 6
num_low_samples_digits_generate = 4900
low_samples = 100

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)

def get_test_images(digits_high_samples, batch_size):
    images = mnist.test.images[mnist.test.labels == digits_low_samples][:low_samples]
    def gen_images(batch_size):
        j = 0
        n = len(images)
        while True:
            if j + batch_size < n:
                j += batch_size
                yield images[j:j + batch_size]
            else:
                first_batch = images[j:n]
                second_batch = images[:batch_size - n + j]
                j = batch_size - n + j
                yield np.vstack((first_batch, second_batch))
    return gen_images(batch_size)

class LatentAttention():
    def __init__(self):
        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])
        self.num_samples = tf.placeholder(tf.int32, None)

        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        self.z_mean, self.z_stddev = z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)

        gaussian_samples = tf.random_normal([self.num_samples,self.n_z],0,1,dtype=tf.float32)
        with tf.variable_scope("", reuse=True):
            self.sample_images = self.generation(gaussian_samples)

        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            n = tf.shape(input_images)[0]
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[n, 7*7*32])
            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")
        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            n = tf.shape(z)[0]
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [n, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [n, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [n, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        next_batch = get_test_images(digits_low_samples, self.batchsize)
        visualization = next(next_batch)
        print(visualization.shape)
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        self.sess = tf.Session()
        sess = self.sess
        sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            batch = next(next_batch)
            _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
            if epoch % 100 == 0:
                print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                generated_test = generated_test.reshape(self.batchsize,28,28)
                ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))
        sample_images = sess.run(self.sample_images, feed_dict={self.num_samples: num_low_samples_digits_generate})
        np.save("../vae_samples.npy", np.reshape(sample_images, (-1, 28 * 28)))
        sample_images = sample_images.reshape(num_low_samples_digits_generate, 28, 28)
        ims("results/"+"vae_samples.jpg",merge(sample_images[:64],[8,8]))


model = LatentAttention()
model.train()
