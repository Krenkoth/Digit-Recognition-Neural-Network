import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import random


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))
def sig_prime(x):
    return sig(x) * (1 - sig(x))

class NeuralNetwork:
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.layers = []
        for i  in range(len(sizes)):
            self.layers.append(Layer(sizes[i], self, i))
    
    # forward propogate, return array with values of final layer
    def forProp(self, input):
        for layer in self.layers[1:]:
            input = layer.eval(input)[1]
        return input
    
    def error(self, test_data):
        total_err = 0
        total_correct = 0
        for data in test_data:
            output = self.forProp(data[0])
            max = 0
            for i in range(len(output)):
                if output[i] > output[max]:
                    max = i
            if data[1][max] == 1:
                total_correct += 1
            err = [(y - e) ** 2 for y, e in zip(output, data[1])]
            total_err += sum(err) / 10
        return total_correct, total_err / len(test_data)
    
    # trains network
    # data should be a list of tuples (input, expected)
    # input and expected should be array-like
    def train(self, epochs: int, batch_size: int, lr: float, train_data, test_data=None):
        if test_data:
            n = len(test_data)
        print("\n-----Training-----\n")
        for epoch in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, len(train_data), batch_size)]
            
            for i in range(len(batches)):
                # for each layer
                self.update_batch(batches[i], lr)
                print("Batch {0}/{1}".format(i, len(batches)), end="\r")
            print("Epoch {0} complete".format(epoch))
            if test_data:
                accuracy, error = self.error(test_data)
                print("   Accuracy: {0}/{1} \n   Error: {2}".format(accuracy, n, error))
                
                
    def update_batch(self, batch, lr):
        for data in batch:
            self.backprop(data)
        for layer in self.layers[1:]:
            layer.update(lr)

    def backprop(self, data):
        activations = [data[0]]
        zs = []
        for layer in self.layers[1:]:
            z, a = layer.eval(activations[-1])
            zs.append(z)
            activations.append(a)
        for i in range(1, self.numLayers):
            layer = self.layers[-i]
            # for each node in layer
            for j in range(layer.size):
                dCdz = self.z_derivative(self.numLayers-i, j, data[1], activations, zs)
                # n = 1/0
                layer.b_changes[j] += dCdz
                layer.w_changes[j] = dCdz * activations[-i-1] + layer.w_changes[j]
        # for layer in self.layers[1:]:
            # print("b_changes: {0}, w_changes: {1}".format(layer.b_changes, layer.w_changes))
    

    def z_derivative(self, layerNum, node_index, expected, activations, zs):
        # print("LayerNum: {0}, node_index: {1}".format(layerNum, node_index))
        if layerNum == self.numLayers - 1:
            # dC/dA * dA/dz
            activation = activations[layerNum][node_index]
            z = zs[layerNum - 1][node_index]
            # print("activation: {0}, z: {1}, expected: {2}".format(activation, z, expected[node_index]))
            # print("expected[node_index]: {0}".format(expected[node_index]))
            return (activation - expected[node_index]) * sig_prime(z)
        else:
            # dC/dz (l+1) * dz/dA * dA/dz
            total = 0
            for i in range(self.layers[layerNum + 1].size):
                layer = self.layers[layerNum + 1]
                total += layer.weights[i][node_index] * self.z_derivative(layerNum + 1, i, expected, activations, zs)
            return total * sig_prime(zs[layerNum-1][node_index])

                    

class Layer:
    def __init__(self, size, network, layerNum):
        self.size = size
        self.network = network
        if layerNum != 0:
            self.biases = np.random.randn(size)
            self.weights = np.random.randn(size, network.layers[layerNum - 1].size)
            self.b_changes = np.zeros(size)
            self.w_changes = np.zeros(self.weights.shape)
    def eval(self, prevActivation):
        z = [np.dot(prevActivation, w)+b for w, b in zip(self.weights, self.biases)]
        return (z, np.vectorize(sig)(z))
    def update(self, lr):
        for i in range(self.size):
            self.biases[i] -= lr * self.b_changes[i]
            # print("w_changes: {0}".format(self.w_changes.shape))
            self.weights[i] = self.weights[i] - self.w_changes[i] * lr
        self.b_changes = np.zeros(self.size)
        self.w_changes = np.zeros(self.weights.shape)