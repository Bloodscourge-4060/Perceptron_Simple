import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=50):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def accuracy(self, inputs, labels):
        correct_predictions = 0
        for inp, label in zip(inputs, labels):
            prediction = self.predict(inp)
            if prediction == label:
                correct_predictions += 1
        return correct_predictions / len(labels) * 100.0
    


if __name__ == "__main__":
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    #Compuerta OR y AND respectivamente
    #labels = np.array([0, 1, 1, 1])
    labels = np.array([0, 0, 0, 1])


    perceptron = Perceptron(input_size=2)

    perceptron.train(training_inputs, labels)

    test_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    for test_input in test_inputs:
        prediction = perceptron.predict(test_input)
        print(f"Entrada: {test_input}, Prediccion: {prediction}")

    acc = perceptron.accuracy(test_inputs, labels)
    print(f"Exactitud del modelo: {acc}%")
