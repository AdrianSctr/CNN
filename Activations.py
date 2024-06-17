import numpy as np

class Softmax:
    def __init__(self):
        self.input = None
        self.result = None
        
    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.result = exps / np.sum(exps, axis=1, keepdims=True)
        return self.result

    def backward(self, grad_output, learning_rate):
        
        num_samples = grad_output.shape[0]
        grad_input = np.empty_like(grad_output)

        for i in range(num_samples):
            y = self.result[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian_matrix, grad_output[i])
        
        return grad_input
    
class Sigmoid:
    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))
        
    def backward(self, grad_output, learning_rate):
        sigmoid_output = 1 / (1 + np.exp(-self.input))
        return gradient_sortie * (1 - sigmoid_output) * sigmoid_output
