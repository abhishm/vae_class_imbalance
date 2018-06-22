import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
batch_size = 128
high_samples = 5000
low_samples = 100
num_itr = 1000
learning_rate = 0.001
digits_high_samples=0
digits_low_samples=6

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_train_images(digits_high_samples=0, digits_low_samples=1, training_type="without_vae"):
    train_images_high_samples = mnist.train.images[mnist.train.labels == digits_high_samples][:high_samples]
    train_images_low_samples = mnist.train.images[mnist.train.labels == digits_low_samples][:low_samples]
    if training_type == "with_vae":
        vae_samples = np.load("vae_samples.npy")
        train_images_low_samples = np.vstack((train_images_low_samples, vae_samples))
    return train_images_high_samples, train_images_low_samples


def get_test_images(digits_high_samples=0, digits_low_samples=1):
    test_images_high_samples = mnist.test.images[mnist.test.labels == digits_high_samples]
    test_images_low_samples = mnist.test.images[mnist.test.labels == digits_low_samples]
    len_high_samples = len(test_images_high_samples)
    len_low_samples = len(test_images_low_samples)
    images = np.vstack((test_images_high_samples,
                        test_images_low_samples))
    labels = np.vstack((np.zeros((len_high_samples, 1)),
                        np.ones((len_low_samples, 1))))
    return images, labels


def get_balanced_batch(digits_high_samples, digits_low_samples, batch_size, training_type):
    train_images_high_samples, train_images_low_samples = get_train_images(digits_high_samples, digits_low_samples, training_type)
    m, n = len(train_images_high_samples), len(train_images_low_samples)
    def next_batch(batch_size):
        np.random.shuffle(train_images_high_samples)
        np.random.shuffle(train_images_low_samples)
        i = 0
        j = 0
        assert batch_size % 2 == 0, "batch size should be even"
        half_batch_size = batch_size // 2
        while True:
            if (i + half_batch_size < m) and (j + half_batch_size < n):
                images = np.vstack((train_images_high_samples[i:i + half_batch_size],
                                    train_images_low_samples[j:j + half_batch_size]))
                labels = np.vstack((np.zeros((half_batch_size, 1)),
                                    np.ones((half_batch_size, 1))))
                i += half_batch_size
                j += half_batch_size
                yield images, labels
            elif (i + half_batch_size >= m) and (j + half_batch_size < n):
                np.random.shuffle(train_images_high_samples)
                i = 0
            elif (i + half_batch_size < m) and (j + half_batch_size >= n):
                np.random.shuffle(train_images_low_samples)
                j = 0
            else:
                np.random.shuffle(train_images_low_samples)
                np.random.shuffle(train_images_high_samples)
                i = 0
                j = 0
    return next_batch(batch_size)


def build_and_train(training_type):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
    h1 = tf.layers.dense(x, 800, tf.nn.relu, name="h1")
    h2 = tf.layers.dense(h1, 800, tf.nn.relu, name="h2")
    logits = tf.layers.dense(h2, 1, name="logit")
    pred_ = tf.cast(logits > 0, dtype=tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_, y), tf.float32))
    missclassification_error = tf.reduce_sum(tf.cast(tf.not_equal(pred_, y), tf.float32))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    sess = get_session()
    sess.run(tf.global_variables_initializer())

    x_test, y_test = get_test_images(digits_high_samples, digits_low_samples)

    accs = []
    miss_classes = []
    next_batch = get_balanced_batch(digits_high_samples, digits_low_samples, batch_size, training_type)
    for itr in range(num_itr):
        x_train, y_train = next(next_batch)
        _, miss_class, acc = sess.run([train_op, missclassification_error, accuracy],
                                      feed_dict={x: x_train,
                                                 y: y_train})
        accs.append(acc)
        miss_classes.append(miss_class)
        if itr % 100 == 0:
            test_acc, test_miss_class = sess.run([accuracy, missclassification_error],
                                                 feed_dict={x: x_test,
                                                            y: y_test})
            print("itr: {0}, train_acc: {1:0.4f}, test_acc: {2:0.4f}, train_miss_class: {3:0.2f}, test_miss_class: {4:0.2f}"
                  .format(itr, np.mean(accs), test_acc, np.mean(miss_classes), test_miss_class))
            accs = []
            miss_classes = []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training type for the model")
    parser.add_argument("--training_type", default="without_vae", type=str,
                        dest="training_type", choices=["with_vae", "without_vae"])
    args = parser.parse_args()
    build_and_train(args.training_type)




































    #
