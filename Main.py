import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import NeuralNet as nn
import pickle


def vectorize(num):
    output = np.zeros(10)
    output[num] = 1
    return output

train, test = tf.keras.datasets.mnist.load_data()

train_x = [image.reshape(-1) / 255 for image in train[0]]
train_y = [vectorize(y) for y in train[1]]
train_data = [x for x in zip(train_x, train_y)]
test_x = [image.reshape(-1) / 255 for image in test[0]]
test_y = [vectorize(y) for y in test[1]]
test_data = [x for x in zip(test_x, test_y)]

net = nn.NeuralNetwork([784, 16, 16, 10])
net.train(15, 10, 0.1, train_data, test_data=test_data)

pickle.dump(net, open('trainedNet.pkl', 'wb'))

# data = [([2, 1], [1])]
# net = nnf.NeuralNetwork([2, 2 ,1])
# net.train(1, 1, 1, data)
