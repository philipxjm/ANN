#! /usr/bin/env python

import math
import random
import string
import numpy as np
import csv
import binascii

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class Network:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights matrices
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = float(inputs[i]);

        # hidden activations
        for j in range(self.nh-1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:];


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = float(targets[k])-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(float(targets[k])-self.ao[k])**2
        return error


    def test(self, data):
        for p in data:
            binList = self.update(p[1:]);
            binStr = "";
            for x in binList:
                if(x <= 0):
                    binStr = binStr + str(0);
                else:
                    binStr = binStr + str(1);
            print(binStr);
            #print(str(p[0]) + '->' + self.fromBinaryToCharacter(binStr));

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('\nOutput weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, data, iterations=100, N=0.05, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            #print(data[1:]);
            inputs = data[1:];
            targets = self.fromCharacterToBinary(data[0]);
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
            #if i % 100 == 0:
            #    print('error %-.5f' % error)

    def fromCharacterToBinary(self, char):
        binStr = str(bin(int(binascii.hexlify(char), 16)))[2:];
        while(len(binStr) != 7):
            binStr = str(0) + binStr;
        #print(binStr);
        return binStr;

    def fromBinaryToCharacter(self, binStr):
        n = int("0b" + binStr, 2);
        #print(binStr);
        character = binascii.unhexlify(str(n));
        return character;

def demo():
    trainingData = importCSV("param1.csv");
    #print(trainingData);

    # create a network with two input, two hidden, and one output nodes
    n = Network(13, 26, 7)
    # train it with some data
    for x in range(len(trainingData)):
        n.train(trainingData[x], iterations=1, N=0.05);
    # test it
    n.test(trainingData);

def importCSV(csvURL):
    with open(csvURL, 'rb') as f:
        reader = csv.reader(f);
        trainingData = list(reader);
        return trainingData;

if __name__ == '__main__':
    demo()
