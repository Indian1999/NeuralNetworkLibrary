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
model.add(FCLayer(5, 10)) # 4 es input, 5 neuron
model.add(ActivationLayer(tanh, tanh_derivative))
model.add(FCLayer(10, 1))
model.add(ActivationLayer(tanh, tanh_derivative))
model.use_loss(mse, mse_derivative)
model.fit(x_train, y_train, epochs = 1000, learning_rate = 0.1)


#Teszteljünk az összes lehetséges inputra:
test_x = []
for i in range(2**5):
    bit_num = bin(i)
    bit_num = str(bit_num[2:])
    while len(bit_num) < 5:
        bit_num = "0" + bit_num
    print(bit_num)
    lista = []
    for j in range(5):
        lista.append(int(bit_num[j]))
    test_x.append(lista)

test_x = np.array(test_x).reshape(32,1,5)
result = model.predict(test_x)
output = []
for item in result:

    output.append(max(0,round(item[0][0])))
print(output)
test_y = [0 for i in range(16)]
test_y.extend([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
print(test_y)


"""
lista = []
for i in range(1, 6):
    lista.append(int(input(f"Add meg a(z) {i}. bitet: ")))
input = np.array(lista).reshape(1,5)
#print(input)
result = model.predict(input)
print(round(result[0][0][0]))
"""

