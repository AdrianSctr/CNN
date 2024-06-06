import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.25):
        self.dropout_rate = dropout_rate
        self.mask = None
        
    def forward(self, input):
        self.mask = (np.random.rand(*input.shape) < self.dropout_rate) / self.dropout_rate
        return input * self.mask
    
    def backward(self, grad_output, learning_rate):
        return grad_output * self.mask