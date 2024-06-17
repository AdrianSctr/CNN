import numpy as np

class AveragePooling2D:
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        pool_height, pool_width = self.pool_size

        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        self.output = np.zeros((batch_size, output_height, output_width, input_channels))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        h_start = i * self.stride
                        h_end = h_start + pool_height
                        w_start = j * self.stride
                        w_end = w_start + pool_width

                        patch = input[b, h_start:h_end, w_start:w_end, c]
                        self.output[b, i, j, c] = np.mean(patch)

        return self.output

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, input_channels = grad_output.shape

        grad_input = np.zeros_like(self.input)

        pool_height, pool_width = self.pool_size

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        h_start = i * self.stride
                        h_end = h_start + pool_height
                        w_start = j * self.stride
                        w_end = w_start + pool_width

                        grad_input[b, h_start:h_end, w_start:w_end, c] += grad_output[b, i, j, c] / (pool_height * pool_width)

        return grad_input
    
class MaxPooling2D:
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.mask_h = None  
        self.mask_w = None

    def forward(self, input):
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        pool_height, pool_width = self.pool_size

        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        self.output = np.zeros((batch_size, output_height, output_width, input_channels))
        self.mask_h = np.zeros_like(self.output, dtype=int)
        self.mask_w = np.zeros_like(self.output, dtype=int)
        
        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        h_start = i * self.stride
                        h_end = h_start + pool_height
                        w_start = j * self.stride
                        w_end = w_start + pool_width

                        # Trouver l'indice du maximum et le mettre dans la sortie
                        patch = input[b, h_start:h_end, w_start:w_end, c]
                        max_value = np.max(patch)
                        self.output[b, i, j, c] = max_value

                        # Garder une trace de l'indice du maximum pour la rétropropagation
                        max_index = np.unravel_index(np.argmax(patch), patch.shape)
                        self.mask_h[b, i, j, c] = h_start + max_index[0]
                        self.mask_w[b, i, j, c] = w_start + max_index[1]

        return self.output

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, input_channels = grad_output.shape

        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        h = self.mask_h[b, i, j, c]  # Obtenir l'indice en hauteur
                        w = self.mask_w[b, i, j, c]  # Obtenir l'indice en hauteur
                        #w_start = j * self.stride
                        #w_end = w_start + self.pool_size[1]
                        grad_input[b, h, w, c] += grad_output[b, i, j, c]
        return grad_input
