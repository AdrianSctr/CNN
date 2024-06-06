import numpy as np
import random

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, input_channels=1, padding=0, activation='relu', optimizer=None, reg_type = 'L1', lambda_reg=0.001, alpha_leaky_relu=0.01):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels).astype(np.float64) * np.sqrt(2. / (filter_size * filter_size))
        self.filters /= np.sqrt(np.sum(np.square(self.filters)) + 1e-5)
        
        self.biases = np.zeros((1,1,1,num_filters))
        self.stride = stride
        self.padding = 0
        self.input = None
        self.activation = activation
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.lambda_reg = lambda_reg
        self.alpha_leaky_relu = alpha_leaky_relu
        self.activations = None
        self.convolution = []
        
    def forward(self, input):
        self.input = input
        #print(input.shape)
        batch_size, input_height, input_width, input_channels = input.shape
        filter_height, filter_width = self.filter_size, self.filter_size
        padded_input = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        output_height = (input_height + 2 * self.padding - filter_height) // self.stride + 1
        output_width = (input_width + 2 * self.padding - filter_width) // self.stride + 1

        output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for w in range(output_width):
            for h in range(output_height):
                h_start = h * self.stride
                h_end = h_start + filter_height
                w_start = w * self.stride
                w_end = w_start + filter_width
                for f in range(self.num_filters):
                    output[:, h, w, f] = np.sum(padded_input[:, h_start:h_end, w_start:w_end, :] * self.filters[f], axis=(1,2,3)) + self.biases[:, 0, 0, f]

        if self.activation == 'relu':
            self.activations = np.maximum(0,output)
        elif self.activation == 'leaky_relu':
            self.activations = np.where(output > 0, output, output * self.alpha_leaky_relu)
        #print(output.shape)
        self.convolution.append(self.activations)
        return self.activations
    
   
    def backward(self, grad_output, learning_rate):
        
        batch_size, output_height, output_width, _ = grad_output.shape
        _, input_height, input_width, input_channels = self.input.shape
        #padded_input = np.pad(grad_output, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        grad_filters = np.zeros_like(self.filters)
        grad_biases = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.input)
        
        grad_biases = np.sum(grad_output, axis=(0, 1, 2), keepdims=True)
        
        if self.activation == 'relu':
            grad_output = grad_output * (self.activations > 0)
        elif self.activation == 'leaky_relu':
            grad_output = grad_output * np.where(self.activations > 0, 1, self.alpha_leaky_relu)
            
        #print(grad_output.shape, padded_input.shape, grad_filters.shape)
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.filter_size
                        w_end = w_start + self.filter_size
                        
                        # Mise à jour des gradients des filtres
                        window = self.input[b, h_start:h_end, w_start:w_end, :]
                        grad_filters[f] += np.sum(grad_output[b, i, j, f] * window, axis=(0,1,2), keepdims=True)
                        # Mise à jour du gradient de l'entrée (ce qui sera renvoyé à la couche précédente)
                        grad_input[b, h_start:h_end, w_start:w_end, :] += np.sum(grad_output[b, i, j, f] * self.filters[f], axis=(0,1,2))
        
        if self.reg_type =='L1':
                self.filters -= learning_rate * (grad_filters + self.lambda_reg * np.sign(self.filters))
        elif self.reg_type == 'L2':
                self.filters -= learning_rate * (grad_filters + self.lambda_reg * 2 * self.filters)
                 
        if self.optimizer:
            params = [self.filters, self.biases]
            grads = [grad_filters, grad_biases]
            self.filters, self.biases = self.optimizer.update(params, grads)
            
        else:
            self.filters -= learning_rate * grad_filters
            self.biases -= learning_rate * grad_biases #/ batch_size
       
            
        return grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]