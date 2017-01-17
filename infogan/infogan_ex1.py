import tensorflow as tf
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d_transpose, conv_2d
from tflearn.layers.normalization import batch_normalization
from infogan import leaky_relu, InfoGAN


def generator(net):
    net = fully_connected(net, 28 * 28 * 512, scope='fc', weights_init='xavier', bias=False)
    net = tf.reshape(net, [tf.shape(net)[0], 28, 28, 512])

    # for i in range(2):
    #     i2 = 2**i
    #     net = conv_2d_transpose(
    #         net, 256 // i2, 5, [14 * i2, 14 * i2], strides=2, scope="l%d" % (i + 1), bias=False, weights_init='xavier')
    #     net = batch_normalization(net, scope="bn_%d" % (i + 1))
    #     net = tf.nn.relu(net)

    net = conv_2d(net, 1, 1, scope="l4")
    return net


def dq_common(net):
    net = conv_2d(net, 64, 5, strides=2, scope='l1')
    net = leaky_relu(net)

    for i in range(4):
        net = conv_2d(net, 64 * 2**i, 5, strides=2, scope='l%d' % (i + 2),
                      bias=False)  # BN handles bias
        net = batch_normalization(net, scope='bn_l%d' % (i + 2))
        net = leaky_relu(net)

    return net


if __name__ == "__main__":
    model = InfoGAN(
        "model_1",
        generator_fn=generator,
        dq_common_fn=dq_common,
        n_gauss=0,
        n_bernulli=0,
        clear_logdir=True,
    )
    try:
        model.init_session()
        model.restore()
        for i in range(model.get_global_step() + 1, 10**5 + 1):
            model.train_loop(summary_every=10)
    finally:
        model.close_session()
