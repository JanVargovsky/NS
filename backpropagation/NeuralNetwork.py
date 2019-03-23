import itertools
import numpy as np
import math

class Node:
    def __init__(self, name, activation):
        self.name = name
        self.activation = activation
        self.b = 0
        self.inputs = []
        self.outputs = []
        self.output = None
        self.outputDerivative = None
        
    def forward(self):
        self.totalInput = sum([edge.w * edge.nodeFrom.output for edge in self.inputs]) + self.b
        self.output = self.activation.output(self.totalInput)
        
    def calculateOutputDerivative(self):
        self.outputDerivative = self.activation.derivative(self.output)
        
    def __repr__(self):
        return self.name

class Edge:
    def __init__(self, name, nodeFrom, nodeTo):
        self.name = name
        self.nodeFrom = nodeFrom
        self.nodeFo = nodeTo
        self.w = np.random.normal(loc=0.5, scale=0.1)
        self.error = 0
        self.delta = 0
        
    def __repr__(self):
        return self.name
        
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, data_x, data_y, learning_rate):
        loss = 0
        for x, y_expected in zip(data_x, data_y):
            loss += self._backpropagation(x, y_expected, learning_rate)
        return loss
                    
    def _backpropagation(self, x, y_expected, learning_rate):
        y_real = self.predict_single(x)
        errors = 0.5 * (np.subtract(y_real, y_expected) ** 2)
        error_total = sum(errors)
        
        #print(f"errors = {errors}")
        #print(f"total error = {error_total}")
        
        # output layer
        for i, node in enumerate(self.layers[-1]):
            node.calculateOutputDerivative()
            for edge in node.inputs:
                edge.error = (y_real[i] - y_expected[i]) * node.outputDerivative
                edge.delta = edge.error * edge.nodeFrom.output
        
        # hidden layers
        for layer in reversed(self.layers[:-1]):
            for node in layer:
                node.calculateOutputDerivative()
                error = sum([edge.error * edge.w for edge in node.outputs])
                #print(f"{node} error = {error}")
                for edge in node.inputs:
                    edge.error = error * node.outputDerivative
                    edge.delta = edge.error * edge.nodeFrom.output
        
        # update weights
        for layer in self.layers:
            for node in layer:
                for edge in node.outputs:
                    edge.w -= learning_rate * edge.delta
                    
        return error_total
    
    def predict(self, x):        
        return [self.predict_single(item) for item in x]
    
    def predict_single(self, x):
        assert len(self.layers[0]) == len(x)
        
        for node, xi in zip(self.layers[0], x):
            node.output = xi
        
        for layer in self.layers[1:]:
            for node in layer:
                node.forward()
                
        return [node.output for node in self.layers[-1]]
        
class NeuralNetworkBuilder:
    @staticmethod
    def build(input_dim, layerDims, activation):
        layers = []
        
        layers.append([Node(f"i{i+1}", activation) for i in range(input_dim)])
        for li, layer in enumerate(layerDims[:-1]):
            layers.append([Node(f"h{li+1}{i+1}", activation) for i in range(layer)])
        layers.append([Node(f"o{i+1}", activation) for i in range(layerDims[-1])])
            
        w = 1
        for i in range(1, len(layers)):
            for previous, current in itertools.product(layers[i - 1], layers[i]):
                edge = Edge(f"w{w}", previous, current)
                #print(f"created {edge}")
                previous.outputs.append(edge)
                current.inputs.append(edge)
                w+=1
        
        return NeuralNetwork(layers)

class Sigmoid:
    def output(self, x):
        return 1 / (1 + math.exp(-x))
    def derivative(self, x):
        return x * (1 - x)
    
class ReLU:
    def output(self, x):
        return max(0, x)
    def derivative(self, x):
        #return 0 if x <= 0 else 1
        return 1 if x > 0 else 0
    
class LeakyReLU:
    def output(self, x):
        return max(0.1 * x, x)
    def derivative(self, x):
        return 1/10 if x <= 0 else 1
