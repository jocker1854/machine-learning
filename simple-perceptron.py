import numpy as np

class Perceptron:
    def __init__(self, nIn, nOut):
        self.nIn = nIn
        self.nOut = nOut
        # Initialize weights randomly in range [-0.05, 0.05]
        self.weights = np.random.rand(nIn + 1, nOut) * 0.1 - 0.05

    def train(self, inputs, targets, learning_rate=0.1, epochs=10):
        self.nData = inputs.shape[0]
        # Add bias input of -1
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

        for _ in range(epochs):
            # Forward pass
            activations = np.dot(inputs, self.weights)
            outputs = np.where(activations > 0, 1, 0)

            # Weight update rule
            self.weights += learning_rate * np.dot(inputs.T, (targets - outputs))

    def predict(self, inputs):
        inputs = np.concatenate((inputs, -np.ones((inputs.shape[0], 1))), axis=1)
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)

# Example usage: AND logic gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# For single output
Y = np.array([[0], [0], [0], [1]])

p = Perceptron(nIn=2, nOut=1)
p.train(X, Y, learning_rate=0.1, epochs=20)

# Test predictions
predictions = p.predict(X)
for x, pred in zip(X, predictions):
    print(f"Input: {x}, Prediction: {pred[0]}")

