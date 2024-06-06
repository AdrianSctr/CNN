import numpy as np

class Flatten:
    def forward(self, input):
        self.input = input
        #print(input.shape)
        return np.reshape(input, (input.shape[0], -1))

    def backward(self, grad_output, learning_rate):
        return np.reshape(grad_output, self.input.shape)