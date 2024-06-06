import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_gradients=None, clip_params=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.clip_gradients = clip_gradients
        self.clip_params = clip_params
        
    def update(self, params, grads):
        if self.m is None:  # Initialisation au premier appel
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        self.t += 1
        updated_params = []
        
        for param, grad, m, v in zip(params, grads, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Clip des gradients si spécifié
            if self.clip_gradients:
                m_hat = np.clip(m_hat, -self.clip_gradients, self.clip_gradients)
                v_hat = np.clip(v_hat, -self.clip_gradients, self.clip_gradients)
            
            # Mise à jour des paramètres
            param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Clip des mises à jour des paramètres si spécifié
            if self.clip_params:
                param_update = np.clip(param_update, -self.clip_params, self.clip_params)
            
            param -= param_update
            updated_params.append(param)
        
        # Mise à jour des listes de moments
        self.m = [m * self.beta1 + (1 - self.beta1) * grad for m, grad in zip(self.m, grads)]
        self.v = [v * self.beta2 + (1 - self.beta2) * (grad ** 2) for v, grad in zip(self.v, grads)]

        return updated_params
    
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        updated_params = []
        for param, grad in zip(params, grads):
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)
        return updated_params