import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self, input_nodes, output_nodes, learning_rate=0.1, epochs=1000):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.normal(0.0, pow(self.input_nodes, -0.5),
                                        (self.output_nodes, self.input_nodes))

    def train(self, X_train, y_train):
        for _ in range(self.epochs):
            for i in range(len(X_train)):
                x = X_train[i].reshape(-1, 1)
                target = y_train[i].reshape(-1, 1)

                output = expit(np.dot(self.weights, x))

                error = target - output

                self.weights += self.learning_rate * error * output * (1 - output) * x.T

    def predict(self, X):
        outputs = expit(np.dot(self.weights, X.T))
        return outputs.T


data = np.loadtxt('data06.csv', delimiter=';')
X = data[:, :2]
y = data[:, 2].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLP(input_nodes=2, output_nodes=1, learning_rate=0.1, epochs=50)
mlp.train(X_train, y_train)

predictions = mlp.predict(X_test)
preds_binary = (predictions > 0.5).astype(int)
correct = np.sum(preds_binary == y_test)
print(f"{correct / len(y_test) * 100:.2f}%")
