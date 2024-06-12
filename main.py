import numpy as np
import SmartGrass as sg
from SmartGrass import network
from SmartGrass import activations
from SmartGrass import activation_layer
from SmartGrass import fc_layer
from SmartGrass import losses

"""
from SmartGrass.activation_layer import ActivationLayer
from SmartGrass.activations import relu, relu_derivative
from SmartGrass.fc_layer import FCLayer
from SmartGrass.losses import mse, mse_derivative
from SmartGrass.network import Network
"""
x_train = np.array([
    [[1,0,1,1]], 
    [[0,1,0,0]], 
    [[1,0,1,0]], 
    [[1,1,1,1]], 
    [[0,0,1,1]], 
    [[1,1,1,0]], 
    [[1,0,0,1]]
])
y_train = np.array([
    [[1]],
    [[0]],
    [[1]],
    [[1]],
    [[0]],
    [[1]],
    [[1]]
    ])

model = network.Network()
model.add(sg.FCLayer(4, 5)) # 4 es input, 5 neuron
model.add(sg.ActivationLayer(sg.relu, sg.relu_derivative))
model.add(sg.FCLayer(5, 1))
model.add(sg.ActivationLayer(sg.relu, sg.relu_derivative))

model.use(sg.mse, sg.mse_derivative)
model.fit(x_train, y_train, epochs = 1000, learning_rate = 0.1)

result = model.predict(x_train)
print(result)