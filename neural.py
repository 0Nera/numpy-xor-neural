import numpy as np
from funcs import sigmoid, sigmoid_prime, tanh, tanh_prime


class neural_network:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Установка весов
        self.weights = []

        # Слои(2 входных, 2 скрытых, 1 выходной) = [2,2,1]
        # Диапазон значений веса (-1,1)
        # Входной и выходной слой - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        
        # Выходной слой - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)


    def fit(self, X, y, learning_rate=0.2, epochs=50000):
        # Добавим столбец из единиц для X
        # Это делается для добавления единицы смещения к входному слою
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        # Обучение
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
        
            # Выходной слой
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # Начнем с предпоследнего слоя(слой до выходного слоя)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # Переворачиваем конструкцию
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # Обратное распространение
            # 1. Умножьте его выходную дельту и входную активацию, 
            #    чтобы получить градиент веса.
            # 2. Вычтите соотношение (процент) градиента из веса.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: 
                print('Эпоха:', k)


    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)   

        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a