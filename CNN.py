import pickle
import math
import numpy as np

class CNN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
            #print(output.shape)
        return output

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            #print(type(layer).__name__)
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output

    def compile(self, optimizer='Adam', loss='categorical_cross_entropy'):
        self.optimizer = optimizer
        self.loss = loss
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.001, patience=5):
        num_batches = math.ceil(len(X_train)/batch_size)#len(X_train) // batch_size if len(X_train) // batch_size !=0 else 1
        num_batches_val = math.ceil(len(X_val) / batch_size)
        #supp  = 1 if (len(X_train)%num_batches!=0 and len(X_train)>=batch_size) else 0
        #num_batches += supp
        
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(total=len(X_train), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = min((batch+1) * batch_size, len(X_train))  
                    #print(start,end)
                    
                    # Forward pass
                    output = self.forward(X_train[start:end])

                    # Calcul de la perte
                    loss = self.calculate_loss(output, y_train[start:end])
                    total_loss += loss
          
                    # Backward pass
                    grad_output = self.calculate_gradient(output, y_train[start:end])
                    self.backward(grad_output, learning_rate)
                    pbar.update(batch_size)
                    pbar.set_postfix({'Loss': loss})
                    
            # Affichage de la perte moyenne par epoch
            average_loss = total_loss / num_batches
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")
            
            val_accuracy = self.evaluate(X_val, y_val, num_batches_val, batch_size)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
           
    def evaluate(self, X_val, y_val, num_batches, batch_size):
        total_correct = 0
        total_samples = 0

        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch+1) * batch_size, len(X_val))

            output = self.forward(X_val[start:end])
            predicted_labels = np.argmax(output, axis=1)
            true_labels = np.argmax(y_val[start:end], axis=1)

            total_correct += np.sum(predicted_labels == true_labels)
            total_samples += end - start

        accuracy = total_correct / total_samples
        return accuracy   
    
    def calculate_loss(self, predicted_output, true_output):
        # Utilisation de la fonction de perte de l'entropie croisée pour les prédictions softmax
        batch_size = predicted_output.shape[0]
        if self.loss == 'categorical_cross_entropy':
            # Ajouter une petite valeur epsilon pour éviter les valeurs nulles
            epsilon = 1e-9
            predicted_output = np.clip(predicted_output, epsilon, 1 - epsilon)
            # Calcul de la perte
            loss = -np.sum(true_output * np.log(predicted_output + epsilon)) / batch_size
            
        elif self.loss == 'binary_cross_entropy': 
            loss = -np.sum(true_output * np.log(predicted_output) + (1 - true_output) * np.log(1 - predicted_output)) / batch_size

        return loss

    def calculate_gradient(self, predicted_output, true_output):
        # Calcul du gradient en fonction de la sortie prédite et des étiquettes réelles
        grad_loss = (predicted_output - true_output) / len(true_output)
        return grad_loss
    
    def set_training_mode(self, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.set_training(training)
                
    def predict(self, X):
        self.set_training_mode(training=False)
        # Effectue une passe avant pour obtenir les prédictions
        predictions = self.forward(X)
        #print(predictions)
        # Trouve l'indice du label avec la probabilité la plus élevée pour chaque prédiction
        predicted_labels = np.argmax(predictions)
        return predicted_labels
    
    def save_model(self, filepath):
        model_params = {
            'layers': self.layers,
            'optimizer': self.optimizer,
            'loss': self.loss
        }

        with open(filepath, 'wb') as file:
            pickle.dump(model_params, file)
            
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as file:
            model_params = pickle.load(file)

        model = cls()
        model.layers = model_params['layers']
        model.optimizer = model_params['optimizer']
        model.loss = model_params['loss']
        return model
