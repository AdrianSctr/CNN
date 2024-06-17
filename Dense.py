import numpy as np

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

class Dense:
    def __init__(self, units, activation, optimizer=None, reg_type='L1', lambda_reg=0.001, alpha_leaky_relu=0.01):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input = None
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.lambda_reg = lambda_reg
        self.alpha_leaky_relu = alpha_leaky_relu
        
        if activation == "softmax":
            self.activation_function = Softmax()
        elif activation == "sigmoid":
            self.activation_function = Sigmoid()
        else:
            self.activation_function = None
        
    def initialize(self, input_shape):
        input_units = input_shape[1]
        if self.activation == 'relu':
            self.weights = np.random.randn(input_units, self.units) * np.sqrt(2. / input_units)
        else:
            self.weights = np.random.randn(input_units, self.units) * 0.01  
        self.bias = np.zeros((1, self.units))

    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.initialize(input.shape)
        #print(input.shape)
        output = np.dot(input, self.weights) + self.bias

        if self.activation == 'relu':
            self.activations = np.maximum(0, output)
        elif self.activation == 'leaky_relu':
            self.activations = np.where(output > 0, output, output * self.alpha_leaky_relu)
        elif self.activation in ['softmax', 'sigmoid']:
            self.activations = self.activation_function.forward(output)
        else:
            self.activations = output
        return self.activations

    def backward(self, grad_output, learning_rate):
       
        if self.activation == 'relu':
            grad_output = grad_output * (self.activations > 0)
        elif self.activation == 'leaky_relu':
            grad_output = grad_output * np.where(self.activations > 0, 1, self.alpha_leaky_relu)
        elif self.activation == 'softmax':
            grad_output = grad_output
        elif self.activation == 'sigmoid':
            grad_output = self.activation_function.backward(grad_output,learning_rate)

        grad_weights = np.dot(self.input.T, grad_output)
        grad_input = np.dot(grad_output, self.weights.T)
        #print("b:",self.input.shape,grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        #print(grad_weights.shape,learning_rate)
        
        if self.reg_type == 'L1':
            grad_weights += self.lambda_reg * np.sign(self.weights)
        elif self.reg_type == 'L2':
            grad_weights += self.lambda_reg * self.weights * 2

        if self.optimizer:
            params = [self.weights, self.bias]
            grads = [grad_weights, grad_bias]
            self.weights, self.bias = self.optimizer.update(params, grads)
        else:
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias

        return grad_input
