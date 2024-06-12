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

print(test_x)