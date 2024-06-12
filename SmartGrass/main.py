import numpy as np

from activation_layer import ActivationLayer
from activations import relu, relu_derivative
from activations import tanh, tanh_derivative
from fc_layer import FCLayer
from losses import mse, mse_derivative
from network import Network

# szabály: Ha az első és utolsó bit 1 -> 1
x_train = np.array([
    [[1,0,1,1,1]], 
    [[0,1,0,0,0]], 
    [[1,0,1,0,0]], 
    [[1,1,1,1,0]], 
    [[0,0,1,1,1]], 
    [[1,1,1,0,1]], 
    [[0,0,1,0,1]], 
    [[1,0,0,1,0]],
    [[1,0,1,1,1]]
])
y_train = np.array([
    [[1]],
    [[0]],
    [[0]],
    [[0]],
    [[0]],
    [[1]],
    [[0]],
    [[0]],
    [[1]]
    ])

model = Network()
model.add(FCLayer(5, 5)) # 4 es input, 5 neuron
model.add(ActivationLayer(tanh, tanh_derivative))
model.add(FCLayer(5, 1))
model.add(ActivationLayer(tanh, tanh_derivative))
model.use_loss(mse, mse_derivative)
model.fit(x_train, y_train, epochs = 1000, learning_rate = 0.01)

"""
lista = []
for i in range(1, 6):
    lista.append(int(input(f"Add meg a(z) {i}. bitet: ")))
input = np.array(lista).reshape(1,5)
#print(input)
result = model.predict(input)
print(round(result[0][0][0]))
"""

