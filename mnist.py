"""This module impements a simple neural network for handwritten digits recognition using the MNIST dataset. """

import mnist_data
import numpy as np
import pickle

class Mnist():
    """ class for working with the MNIST dataset"""
    
    def __init__(self):
        """Initializes the Mnist class with data and empty parameters."""
        self.data = mnist_data.MnistData()
        self.params = {}

    def sigmoid(self, x):
        """Applies sigmoid activation functio"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        """Applies softmax activation function"""
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)

    def load(self):
        """Loads the MNIST training and test datasets"""
        (x_train, y_train), (x_test, y_test) = self.data.load()
        return (x_train, y_train), (x_test, y_test)
    
    def init_network(self):
        """Initializes network weights and biases from a pre-trained file."""
        with open('sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)
    
    def predict(self, x):
        """function to predict class probabilities."""
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)

        return y    
