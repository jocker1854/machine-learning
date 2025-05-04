import numpy as np
from helper import generate_linear_data

class Perceptron:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        nIn = self.inputs.shape[1]
        nOut = self.targets.shape[1]
        self.weights = np.random.rand(nIn+1, nOut)*0.1-0.05
        nData = inputs.shape[0]
        self.inputs = np.concatenate((self.inputs, -np.ones((nData, 1))), axis=1)
    
    @staticmethod
    def accuracy(y_pred, y_true):
        return (y_pred==y_true).mean()

    def train(self, max_iters=100, lr=0.25):
        for i in range(max_iters):
            activations = np.dot(self.inputs, self.weights)
            outputs = np.where(activations>0, 1, 0)
            self.weights -= lr * np.dot(self.inputs.T, (outputs-self.targets))
            acc = self.accuracy(outputs, self.targets)
            #if (i+1) % 100 == 0:
            print(f'iter {i+1}:')
            print('accuracy:', acc)

            if acc == 1.0:
                print(outputs)
                print(self.weights)
                break


X,Y = generate_linear_data(n_samples=100, separation_line=(1, -1, 0), noise=0.1)

p = Perceptron(X, Y)
p.train(max_iters=100, lr=0.01)