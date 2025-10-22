import numpy as np
from scipy.special import expit


class MLP:
    def __init__(self, in_l, hid_l, out_l, learning_rate, epoch=1):
        self.in_l = in_l
        self.hid_l = hid_l
        self.out_l = out_l
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.in_h_weights = np.random.normal(0.0, pow(self.in_l, -0.5), (self.hid_l, self.in_l))
        self.h_out_weights = np.random.normal(0.0, pow(self.hid_l, -0.5), (self.out_l, self.hid_l))

    def train(self, data_file):
        for _ in range(self.epoch):
            with open(data_file, 'r') as file:
                for i in file.readlines():
                    inp = np.array(i.replace('\n', '').split(','), ndmin=2, dtype=int)
                    out = np.zeros((10, 1))
                    out[inp[0, 0], 0] = 0.99
                    inp = inp[0, 1:] / 255 * 0.99 + 0.01
                    inp.resize((784, 1))
                    hidden_out = expit(np.dot(self.in_h_weights, inp))
                    calc_out = expit(np.dot(self.h_out_weights, hidden_out))

                    error = out - calc_out
                    hid_out_dir = -2 * np.dot(error * calc_out * (1 - calc_out), hidden_out.T)
                    inp_gid_dir = -2 * np.dot(np.dot(self.h_out_weights.T, error) * hidden_out * (1 - hidden_out),
                                              inp.T)

                    self.in_h_weights -= self.learning_rate * inp_gid_dir
                    self.h_out_weights -= self.learning_rate * hid_out_dir

    def predict(self, test_file):
        with open(test_file, 'r') as file:
            lines = file.readlines()

        correct = 0
        total = len(lines)

        for line in lines:
            data = np.array(line.strip().split(','), dtype=int)
            label = data[0]
            inputs = data[1:] / 255 * 0.99 + 0.01
            inputs = inputs.reshape((self.in_l, 1))

            hidden_out = expit(np.dot(self.in_h_weights, inputs))
            outputs = expit(np.dot(self.h_out_weights, hidden_out))

            prediction = np.argmax(outputs)
            if prediction == label:
                correct += 1

        accuracy = correct / total
        print(f"{accuracy * 100:.2f}%")
        return accuracy


mlp = MLP(784, 500, 10, 0.01)

# Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
mlp.train('mnist_train.csv')
mlp.predict('mnist_test.csv')
