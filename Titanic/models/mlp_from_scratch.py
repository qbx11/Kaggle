import numpy as np
import pickle
"""
Multi-Layer Perceptron (MLP) implementation from scratch
- Feedforward neural network
- Fully connected layers
- Sigmoid activation
- Adam optimizer
- Binary classification
"""

class Neuron:
    def __init__(self, num_of_inputs, name="Neuron"):
        self.weights = np.random.rand(num_of_inputs) * 0.1  # małe wagi
        self.bias = np.random.randn() * 0.1
        self.name = name
        self.last_output = None

        # Adam optimizer
        self.m_weights = np.zeros(num_of_inputs)
        self.v_weights = np.zeros(num_of_inputs)
        self.m_bias = 0
        self.v_bias = 0
        self.t = 0

    def update_weights(self, learning_rate, gradient, bias_gradient, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.t += 1

        # Wagi
        self.m_weights = beta_1 * self.m_weights + (1 - beta_1) * gradient
        self.v_weights = beta_2 * self.v_weights + (1 - beta_2) * (gradient ** 2)

        m_corrected = self.m_weights / (1 - beta_1 ** self.t)
        v_corrected = self.v_weights / (1 - beta_2 ** self.t)

        self.weights += learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)

        # Bias
        self.m_bias = beta_1 * self.m_bias + (1 - beta_1) * bias_gradient
        self.v_bias = beta_2 * self.v_bias + (1 - beta_2) * (bias_gradient ** 2)

        m_bias_corrected = self.m_bias / (1 - beta_1 ** self.t)
        v_bias_corrected = self.v_bias / (1 - beta_2 ** self.t)

        self.bias += learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + epsilon)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-np.clip(input, -500, 500)))  # clip żeby uniknąć overflow

    def sigmoid_deriv(self, input):
        return input * (1 - input)

    def forward(self, input):
        total = np.dot(self.weights, input) + self.bias
        output = self.sigmoid(total)
        self.last_output = output
        return output


class Network:
    def __init__(self, architecture):
        self.architecture = architecture
        self.layers = []

        for layer_num in range(1, len(architecture)):
            layer = []
            num_of_inputs = architecture[layer_num - 1]
            num_of_neurons = architecture[layer_num]

            for neuron_num in range(num_of_neurons):
                neuron = Neuron(num_of_inputs, name=f"L{layer_num}N{neuron_num}")
                layer.append(neuron)
            self.layers.append(layer)

    def forward(self, input):
        self.layer_outputs = [input]

        for layer in self.layers:
            outputs = []
            for neuron in layer:
                output = neuron.forward(self.layer_outputs[-1])
                outputs.append(output)
            self.layer_outputs.append(outputs)

        return self.layer_outputs[-1]

    def compute_output_deltas(self, target, layer_output):
        deltas = []
        errors = []
        for i, output in enumerate(layer_output):
            error = target[i] - output
            errors.append(error)
            delta = error * output * (1 - output)
            deltas.append(delta)
        return deltas, errors

    def compute_hidden_deltas(self, layer_index, layer, layer_output):
        deltas = []
        next_layer = self.layers[layer_index + 1]
        next_layer_deltas = self.layer_deltas[layer_index + 1]

        for i, neuron in enumerate(layer):
            delta_sum = 0
            for j, neuron_next in enumerate(next_layer):
                delta_from_next = next_layer_deltas[j]
                weight = neuron_next.weights[i]
                delta_sum += delta_from_next * weight

            output = layer_output[i]
            delta = delta_sum * output * (1 - output)
            deltas.append(delta)

        return deltas

    def compute_gradients(self):
        gradients = {}

        for i, layer in enumerate(self.layers):
            layer_deltas = self.layer_deltas[i]
            layer_inputs = self.layer_outputs[i]

            layer_gradients = []
            for j, neuron in enumerate(layer):
                delta = layer_deltas[j]
                weight_grads = np.array([delta * inp for inp in layer_inputs])
                bias_grad = delta
                layer_gradients.append((weight_grads, bias_grad))

            gradients[f'layer_{i}'] = layer_gradients

        return gradients

    def backward(self, target, learning_rate):
        # 1. Oblicz delty dla wszystkich warstw
        self.layer_deltas = [None] * len(self.layers)

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer_output = self.layer_outputs[i + 1]

            if i == len(self.layers) - 1:  # Warstwa output
                deltas, errors = self.compute_output_deltas(target, layer_output)
                mse = np.mean([e ** 2 for e in errors])
            else:  # Warstwy ukryte
                deltas = self.compute_hidden_deltas(i, layer, layer_output)

            self.layer_deltas[i] = deltas

        #Oblicz gradienty
        gradients = self.compute_gradients()

        #AKTUALIZUJ WAGI
        for i, layer in enumerate(self.layers):
            layer_gradients = gradients[f'layer_{i}']
            for j, neuron in enumerate(layer):
                weight_grads, bias_grad = layer_gradients[j]
                neuron.update_weights(learning_rate, weight_grads, bias_grad)

        return mse



    def train(self, X_train, y_train, epochs, learning_rate, verbose=True,final_error=0.002):

        history = {'loss': []}

        for epoch in range(epochs):
            total_loss = 0

            for x_sample, y_sample in zip(X_train, y_train):
                # Forward pass
                output = self.forward(x_sample)

                # Backward pass + aktualizacja wag
                loss = self.backward([y_sample], learning_rate)
                total_loss += loss

            avg_loss = total_loss / len(X_train)
            history['loss'].append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

            # Zatrzymywanie uczenia
                # Wczesne zatrzymanie uczenia
            if avg_loss < final_error:
                print(f"Wczesne zatrzymanie uczenia po epoce {epoch}, Loss: {avg_loss:.6f}")
                break


        return history

    def predict(self, X):

        predictions = []
        for x_sample in X:
            output = self.forward(x_sample)
            predictions.append(output[0])  # output to lista, bierzemy pierwszy element
        return np.array(predictions)

    def predict_classes(self, X, threshold=0.5):

        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test, y_test):

        predictions = self.predict_classes(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy

    def save_weights(self, filepath):

        weights_data = {
            'architecture': self.architecture,
            'layers': []
        }

        for layer in self.layers:
            layer_data = []
            for neuron in layer:
                neuron_data = {
                    'weights': neuron.weights,
                    'bias': neuron.bias,

                    'm_weights': neuron.m_weights,
                    'v_weights': neuron.v_weights,
                    'm_bias': neuron.m_bias,
                    'v_bias': neuron.v_bias,
                    't': neuron.t
                }
                layer_data.append(neuron_data)
            weights_data['layers'].append(layer_data)

        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)


    def load_weights(self, filepath):

        with open(filepath, 'rb') as f:
            weights_data = pickle.load(f)


        if weights_data['architecture'] != self.architecture:
            raise ValueError(f"Architecture mismatch! File has {weights_data['architecture']}, "
                             f"but model has {self.architecture}")


        for i, layer in enumerate(self.layers):
            layer_data = weights_data['layers'][i]
            for j, neuron in enumerate(layer):
                neuron_data = layer_data[j]
                neuron.weights = neuron_data['weights']
                neuron.bias = neuron_data['bias']

                neuron.m_weights = neuron_data['m_weights']
                neuron.v_weights = neuron_data['v_weights']
                neuron.m_bias = neuron_data['m_bias']
                neuron.v_bias = neuron_data['v_bias']
                neuron.t = neuron_data['t']





if __name__ == "__main__":
    #XOR test
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    #network 2 inputs, 4 hidden, 1 output
    network = Network([2, 4, 1])


    history = network.train(X, y, epochs=2000, learning_rate=0.1,final_error = 0.002)

    print("\nPredykcje po treningu:")
    predictions = network.predict(X)
    for i, (x_sample, pred, true) in enumerate(zip(X, predictions, y)):
        print(f"{x_sample} -> {pred:.4f} ({true})")

    print(f"\nAccuracy: {network.evaluate(X, y):.2%}")