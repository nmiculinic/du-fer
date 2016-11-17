from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, activation
from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import base64

import os

import uuid

from tflearn.datasets import cifar10
from hyperopt import fmin, tpe, hp


(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

space = {
    'lambda': hp.loguniform('lambda', -3, 5),
    'dropout_cnn': hp.uniform('dropout CNN', 0.5, 1),
    'dropout_fc': hp.uniform('dropout FC', 0.5, 1),
    'bn_cnn': hp.choice("BatchNorm CNN", [True, False]),
    'bn_fc': hp.choice("BatchNorm FC", [True, False]),
    "batch_size": hp.choice("batch_size", [32, 64, 96, 128]),
    "lr": hp.loguniform("learning rate", -8, -4),
    "cnn_layers": hp.quniform("CNN layers", 1, 4, 1)
}


def trial(params):
    run_id = str(base64.b64encode(uuid.uuid4().bytes, b"__")[:10])
    tdir = os.path.join(os.path.expanduser('~'), 'logs')
    mdir = os.path.join(tdir, run_id)
    os.makedirs(mdir)

    with open(os.path.join(mdir, 'README.md'), 'w') as f:
        print(params, file=f)

    print(run_id)
    print(params)

    x = input_data(
        shape=[None, 32, 32, 3],
        data_preprocessing=img_prep,
        data_augmentation=img_aug
    )

    network = x
    for i in range(int(params['cnn_layers'])):
        network = conv_2d(x, 2**(i + 5), 3, regularizer='L2', weight_decay=params['lambda'])
        if params['bn_cnn']:
            network = batch_normalization(network)
        network = dropout(network, params['dropout_cnn'])
        network = activation(network, 'relu')
        network = max_pool_2d(network, 2)

    network = fully_connected(network, 512, regularizer='L2', weight_decay=params['lambda'])
    if params['bn_fc']:
        network = batch_normalization(network)
    network = activation(network, 'relu')
    network = dropout(network, params['dropout_fc'])
    logits = fully_connected(network, 10, activation='softmax', regularizer='L2', weight_decay=params['lambda'])

    network = regression(logits, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=params['lr'])

    model = tflearn.DNN(
        network,
        tensorboard_verbose=0,
        tensorboard_dir=tdir
    )
    model.fit(X, Y, n_epoch=25, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=params['batch_size'], run_id=run_id)

    model.save(os.path.join(mdir, 'model.save'))
    return model.evaluate(X_test, Y_test)

best = fmin(fn=trial, space=space, algo=tpe.suggest, max_evals=100)
print(best)
