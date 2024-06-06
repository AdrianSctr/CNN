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
        #probabilities = self.input
        #size = self.input.shape[0]
        #return grad_output #* probabilities * (1 - probabilities)
        # Calcul de la dérivée de la fonction softmax par rapport à ses entrées
        """"
        softmax_output = self.result
        batch_size = softmax_output.shape[0]
        grad_input = np.empty_like(softmax_output)
        for i in range(batch_size):
            jacobian_matrix = -np.outer(softmax_output[i], softmax_output[i]) + np.diag(softmax_output[i])
            grad_input[i] = np.dot(grad_output[i], jacobian_matrix)
        """
        """
        batch_size = self.result.shape[0]
        grad_input = (self.result - grad_output) / batch_size
        """
        softmax_output = self.result
        batch_size = softmax_output.shape[0]
        grad_input = np.empty_like(softmax_output)
        
        for i in range(batch_size):
            # Calcul de la matrice jacobienne pour le i-ème exemple
            s = softmax_output[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            # Calcul du gradient de l'entrée
            grad_input[i] = np.dot(jacobian_matrix, grad_output[i])
        
        return grad_input
        return grad_input
    
class Sigmoid:
    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))
        
    def backward(self, grad_output, learning_rate):
        sigmoid_output = 1 / (1 + np.exp(-self.input))
        return gradient_sortie * (1 - sigmoid_output) * sigmoid_output