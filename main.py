import numpy as np
import time
from neural import neural_network




if __name__ == '__main__':

    nn = neural_network([2,2,1])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([0, 1, 1, 0])

    print("Обучение нейросети...")
    _start = time.time()

    nn.fit(X, y, epochs=50000)
    _end = time.time()

    print(f"Обучение закончено: {round(_end - _start,4)}")

    print("Веса:")

    for i in nn.weights:
        print(f"->{i}")
    
    print("Результат:")
    for e in X:
        print(f"|{e}->{round(nn.predict(e)[0], 4)}")