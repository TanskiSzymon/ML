print ("hw")
import numpy as np
import matplotlib.pyplot as plt

# Definicja funkcji aktywacji i jej pochodnej
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inicjalizacja klasy perceptronu
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.randn(input_size + 1) * 0.01
        self.lr = lr
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return sigmoid(summation)
    
    def train(self, training_inputs, labels):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                adjustments = self.lr * error * sigmoid_derivative(prediction)
                self.weights[1:] += adjustments * inputs
                self.weights[0] += adjustments

# Definicja zbioru danych XOR
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0,1,1,0])

# Utworzenie i trening perceptronu
perceptron = Perceptron(input_size=2)
epochs = 100000
perceptron.train(inputs, labels)

# Testowanie perceptronu
predictions = np.array([perceptron.predict(x) for x in inputs])
print(predictions)
