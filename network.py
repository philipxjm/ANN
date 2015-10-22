#! /usr/bin/env python

import math
import random
import string
import numpy as np
import csv
import binascii
import time

random.seed(0);

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a;

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = [];
    for i in range(I):
        m.append([fill]*J);
    return m;

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x);

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2;

class Network:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1; # +1 for bias node
        self.nh = nh + 1; # +1 for bias node
        self.no = no;

        # activations for nodes
        self.ai = [1.0]*self.ni;
        self.ah = [1.0]*self.nh;
        self.ao = [1.0]*self.no;

        # create weights matrices
        self.wi = makeMatrix(self.ni, self.nh);
        self.wo = makeMatrix(self.nh, self.no);
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.5, 0.5);
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-0.1, 0.1);

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh);
        self.co = makeMatrix(self.nh, self.no);

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs');

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(float(inputs[i]))
            self.ai[i] = float(inputs[i]);
        self.ai.append(float(1.0));

        # hidden activations
        for j in range(self.nh):
            sum = 0.0;
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j];
            self.ah[j] = sigmoid(sum);

        # output activations
        for k in range(self.no):
            sum = 0.0;
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k];
            #print("sigmoid: " + str(sigmoid(sum)));
            self.ao[k] = sigmoid(sum);

        return self.ao[:];


    def backPropagate(self, tar, N, M):
        targets = [];
        for t in tar:
            targets.append((float(t)*2) - 1);
        if len(targets) != self.no:
            raise ValueError('wrong number of target values');

        # calculate error terms for output
        output_deltas = [0.0] * self.no;
        for k in range(self.no):
            error = float(targets[k])-self.ao[k];
            output_deltas[k] = dsigmoid(self.ao[k]) * error;

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh;
        for j in range(self.nh):
            error = 0.0;
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k];
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error;

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j];
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k];
                self.co[j][k] = change;
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i];
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j];
                self.ci[i][j] = change;

        # calculate error
        error = 0.0;
        for k in range(len(targets)):
            error = error + 0.5*(float(targets[k])-self.ao[k])**2;
        return error;


    def recognize(self, data):
        numCorrect = 0.0;
        errorList = [];
        for p in data:
            binList = self.update(p[1:]);
            binStr = "";
            for x in binList:
                if(x <= 0):
                    binStr = binStr + str(0);
                else:
                    binStr = binStr + str(1);
            #print(binStr);
            print(str(p[0]) + '->' + self.fromBinaryToCharacter(binStr));
            if(str(p[0]) == self.fromBinaryToCharacter(binStr)):
                numCorrect += 1;
            else:
                errorList.append(str(p[0]) + '->' + self.fromBinaryToCharacter(binStr));
        for l in errorList:
            print("Bad Recognition: " + str(l));
        print("Accuracy: " + str((numCorrect/len(data))*100.0));

    def weights(self):
        print('Input weights:');
        for i in range(self.ni):
            print(self.wi[i]);
        print('\nOutput weights:');
        for j in range(self.nh):
            print(self.wo[j]);

    def saveWeights(self):
        if((self.nwi is not None) and (self.nwo is not None)):
            np.savetxt(self.nwi, self.wi, delimiter=",");
            np.savetxt(self.nwo, self.wo, delimiter=",");
        else:
            np.savetxt("wi.csv", self.wi, delimiter=",");
            np.savetxt("wo.csv", self.wo, delimiter=",");
        #print(len(self.wi));
        #print(len(self.wo));

    def importWeights(self, wiURL, woURL):
        self.wi = np.loadtxt(open(wiURL,"rb"),delimiter=",");
        self.wo = np.loadtxt(open(woURL,"rb"),delimiter=",");
        #print(len(self.wi));
        #print(len(self.wo));

    def train(self, data, iterations=100, N=0.00005, M=0.01):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0;
            #print(data[1:]);
            inputs = data[1:];
            targets = self.fromCharacterToBinary(data[0]);
            self.update(inputs);
            error = error + self.backPropagate(targets, N, M);
            if i % 100 == 0:
                print('error %-.5f' % error);

    def fromCharacterToBinary(self, char):
        binStr = str(bin(int(binascii.hexlify(char), 16)))[2:];
        while(len(binStr) != 7):
            binStr = str(0) + binStr;
        #print(binStr);
        return binStr;

    def fromBinaryToCharacter(self, binStr):
        st = "0" + binStr;
        #print st;
        n = int("0b" + st, 2);
        #print(n);
        #print(binStr);
        character = binascii.unhexlify('%x' % n);
        return character;

def importCSV(csvURL):
    with open(csvURL, 'rb') as f:
        reader = csv.reader(f);
        trainingData = list(reader);
        return trainingData;

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Neural Network for Optical Character Recognition");
    parser.add_argument('-t', type=str, help="train this network, set name of training csv file");
    parser.add_argument('-nwi', type=str, default="wi.csv", help="set prefered name of wi output file");
    parser.add_argument('-nwo', type=str, default="wo.csv", help="set prefered name of wo output file");
    parser.add_argument('-r', type=str, help="use this network for recognition, set name of target csv file");
    parser.add_argument('-wi', type=str, default="wi.csv", help="set the input weights of this network, set name of wi csv input file");
    parser.add_argument('-wo', type=str, default="wo.csv", help="set the out weights of this network, set name of wo csv input file");
    parser.add_argument('-ic', type=float, default=9, help="iteration number, default value 9");
    parser.add_argument('-nc', type=float, default=0.00001, help="learning constant, default value 0.00001");
    parser.add_argument('-mc', type=float, default=0.001, help="momentum constant, default value 0.001");
    parser.add_argument('-w', type=bool, default=False, help="whether print weight or not");
    opts = parser.parse_args();

    n = Network(24, 150, 7);
    trainingData = None;
    testingData = None;

    if(opts.t is not None):
        print(opts.t);
        trainingData = importCSV(opts.t);
    if((opts.nwi is not None) and (opts.nwo is not None)):
        n.nwi = opts.nwi;
        n.nwo = opts.nwo;
    if((opts.r is not None)):
        testingData = importCSV(opts.r);
    if((opts.wi is not None) and (opts.wo is not None)):
        n.importWeights(opts.wi, opts.wo);

    if((trainingData is not None)):
        start = time.time();
        for x in range(len(trainingData)):
            n.train(trainingData[x], opts.ic, opts.nc, opts.mc);
        n.saveWeights();
        end = time.time();
        print("\nTime took to train: " + str(end - start) + " seconds");
    if((testingData is not None)):
        start = time.time();
        n.recognize(testingData);
        end = time.time();
        print("\nTime took to recognize: " + str(end - start) + " seconds");
    #print(trainingData);
    if(opts.w):
        n.weights();













