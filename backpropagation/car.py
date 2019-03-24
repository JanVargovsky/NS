import pickle
import socket
import xml.etree.ElementTree as ET
from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm import tqdm

from NeuralNetwork import NeuralNetworkBuilder, Sigmoid

plt.style.use('ggplot')


def load_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    input_names = []
    for node in root.find('inputDescriptions').findall('inputDescription'):
        input_names.append(node.find('name').text)

    output_names = []
    for node in root.find('outputDescriptions').findall('outputDescription'):
        output_names.append(node.text)

    train_x = []
    train_y = []
    for element in root.find('trainSet').findall('trainSetElement'):
        inputs = []
        for node in element.find('inputs').findall('value'):
            inputs.append(float(node.text))
        outputs = []
        for node in element.find('outputs').findall('value'):
            outputs.append(float(node.text))
        train_x.append(inputs)
        train_y.append(outputs)

    return train_x, train_y, input_names, output_names


def save(nn, name):
    with open(name, 'wb') as f:
        pickle.dump(nn, f)


def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def create_and_train(x, y):
    nn = NeuralNetworkBuilder.build(len(x[0]), [8, 8, len(y[0])], Sigmoid())
    loss = []
    with tqdm(range(10000)) as epochs:
        for _ in epochs:
            loss.append(nn.train(x, y, 3))
            epochs.set_postfix_str(f'loss:{loss[-1]:1.5f}')
            if loss[-1] <= 0.003:
                epochs.close()
                break

    plt.plot(loss)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    print(f"error={loss[-1]}")
    return nn


filename = '2'
base_path = 'demo'
train_x, train_y, input_names, output_names = load_file(f"{base_path}/{filename}.txt")

assert len(train_x[0]) == 28
assert len(train_y[0]) == 2

nn = load(f"{base_path}/{filename}.pickle")
# nn = create_and_train(train_x, train_y)
# save(nn, f"{base_path}/{filename}.pickle")

host = 'localhost'
port = 9461
race_name = 'Race'
driver_name = 'var0065'
color = 'FFD700'
car_type = None

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print('Connecting')
    s.connect((host, port))
    print('Connected')

    s_in = s.makefile('r')
    s_out = s.makefile('w')


    def write(data):
        s_out.write(data)


    def writeln(data):
        s_out.write(data + '\n')


    def write_end():
        write('\n')
        s_out.flush()


    def read():
        return s_in.readline().strip('\n')


    print("Sending driver info")

    writeln('driver')
    writeln(f'race:{race_name}')
    writeln(f'driver:{driver_name}')
    writeln(f'color:{color}')
    if car_type:
        writeln(f'car:{car_type}')
    write_end()

    print("Driver info sent")

    data = read()
    if data != 'ok':
        raise ValueError(f'Driver sync failed, server error: "{data}"')
    assert read() == ''

    print("Entering main loop")

    # main loop
    while True:
        data = read()
        if data == 'round':
            # parse inputs
            inputs = OrderedDict([(name, 0) for name in input_names])
            while True:
                line = read()
                if line == '':
                    break
                name, value = line.split(':')
                inputs[name] = float(value)

            # call NN
            result = nn.predict_single(inputs.values())

            # write output
            writeln("ok")
            for name, value in zip(output_names, result):
                writeln(f"{name}:{value}")
            write_end()
        elif data == 'finish':
            print("Finish")
            break
        elif data != '':
            raise ValueError(f'Server error: "{data}"')

    print("Done")
