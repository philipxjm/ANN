#! /usr/bin/env python

# neural network with flexible node configurations
# optimized for optical character recognition
# author: Philip Xu

import os
import math
import random
import numpy as np
import csv
import binascii
import time
import sys
from progressbar import Bar, Percentage, ProgressBar, SimpleProgress

random.seed(0);

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a;

# Make a matrix
def makeMatrix(I, J, fill=0.0):
    m = [];
    for i in range(I):
        m.append([fill]*J);
    return m;

# sigmoid function, uses tanh
def sigmoid(x):
    return math.tanh(x);

# derivative of sigmoid function, in terms of the output (i.e. y)
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

    # feed forward data
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

    # backpropagate errors
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

    # recognition sequence but with error display
    def recognizeDebug(self, data):
        # set vars for error calculations
        numCorrect = 0.0;
        totalNum = 0.0;
        errorList = [];

        # recognition process
        print("Recognition (Debug) Progress:");
        pbar = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=len(data)).start();
        for p in data:
            binList = self.update(p[1:]);
            binStr = "";
            for x in binList:
                if(x <= 0):
                    binStr = binStr + str(0);
                else:
                    binStr = binStr + str(1);

            # print(str(p[0]) + ' -> ' + self.fromBinaryToCharacter(binStr));

            if(str(p[0]) == self.fromBinaryToCharacter(binStr)):
                numCorrect += 1;
            else:
                errorList.append(str(p[0]) + ' -> ' + self.fromBinaryToCharacter(binStr));
            totalNum += 1;
            pbar.update(totalNum);
        pbar.finish();

        # print all errors made
        for l in errorList:
            print("Bad Recognition: " + str(l));

        # print results
        print("\n");
        print("Accuracy: " + str(round((numCorrect/len(data))*100.0, 2)) + "%");

    # recognition sequence
    def recognize(self, data, filename):
        # set vars for error calculations
        totalNum = 0;
        outlist = [];

        # recognition process
        print("Recognition Progress:");
        pbar = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=len(data)).start();
        for p in data:
            binList = self.update(p[1:]);
            binStr = "";
            for x in binList:
                if(x <= 0):
                    binStr = binStr + str(0);
                else:
                    binStr = binStr + str(1);
            # print(binStr);
            # print("number: " + str(idx) + ' -> ' + self.fromBinaryToCharacter(binStr));
            outlist.append(self.fromBinaryToCharacter(binStr));
            totalNum += 1;
            pbar.update(totalNum);

        # saving recognized string as textfile
        out = "".join(outlist);
        if not os.path.exists("out"):
            os.makedirs("out");
        text_file = open("out/" + filename, "w");
        text_file.write(out);
        text_file.close();
        pbar.finish();
        print("Wrote recognized text to: " + "out/" + filename);

    # print weights matrices
    def weights(self):
        print('Input weights:');
        print(self.wi);
        print('\nOutput weights:');
        print(self.wo);

    # save weight matrices as csv
    def saveWeights(self):
        # save matrices to specified location
        if((self.ww is not None)):
            newpath = "weights/" + self.ww;
            if not os.path.exists(newpath):
                os.makedirs(newpath);
            np.savetxt("weights/" + self.ww + "/wi.csv", self.wi, delimiter=",");
            np.savetxt("weights/" + self.ww + "/wo.csv", self.wo, delimiter=",");
            print("\nSaved weight matrices in: " + "weights/" + self.ww + "/");

        # save matrices to default folder
        else:
            newpath = "weights/defaultweights";
            if not os.path.exists(newpath):
                os.makedirs(newpath);
            np.savetxt("weights/defaultweights/wi.csv", self.wi, delimiter=",");
            np.savetxt("weights/defaultweights/wo.csv", self.wo, delimiter=",");
            print("\nSaved weight matrices in: " + "weights/defaultweights/");

    # read weight csv into matrices
    def importWeights(self, wiURL, woURL):
        self.wi = np.loadtxt(open(wiURL,"rb"),delimiter=",");
        self.wo = np.loadtxt(open(woURL,"rb"),delimiter=",");

    # training sequence
    def train(self, data, N=0.00001, M=0.001):
        # N: learning rate
        # M: momentum factor
        error = 0.0;
        inputs = data[1:];

        # target output node activations
        targets = self.fromCharacterToBinary(data[0]);

        # feedforward inputs
        self.update(inputs);

        # backpropagate errors
        error = error + self.backPropagate(targets, N, M);

    # take character string and convert to binary string
    def fromCharacterToBinary(self, char):
        binStr = str(bin(int(binascii.hexlify(char), 16)))[2:];

        # making sure binStr is 7 digits long
        while(len(binStr) != 7):
            binStr = str(0) + binStr;
        return binStr;

    # takes binary string and convert to ascii characters
    def fromBinaryToCharacter(self, binStr):

        # making sure binStr is 8 digits long
        st = "0" + binStr;

        # converting binStr to ascii char
        n = int("0b" + st.strip(), 2);
        if len(str(hex(n).split('x')[1])) == 0:
            character = binascii.unhexlify("00");
        elif len(str(hex(n).split('x')[1])) == 1:
            character = binascii.unhexlify("0" + (hex(n).split('x')[1]));
        else:
            character = binascii.unhexlify((hex(n).split('x')[1]).strip());
        return character;

# imports csv as list
def importCSV(csvURL):
    with open(csvURL, 'rb') as f:
        reader = csv.reader(f);
        trainingData = list(reader);
        return trainingData;

# move console cursor up
def up():
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

# move console cursor down
def down():
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    # adding arguments
    import argparse
    parser = argparse.ArgumentParser(description="Neural Network for Optical Character Recognition");
    parser.add_argument('-t', type=str, help="train this network, set name of training csv file");
    parser.add_argument('-ww', type=str, help="set prefered names of directory to save weights");
    parser.add_argument('-rw', type=str, help="set the input weights of this network, set name of directory to read weights");
    parser.add_argument('-r', type=str, help="use this network for recognition, set name of target csv file");
    parser.add_argument('-ic', type=float, default=1, help="iteration number, default value 1");
    parser.add_argument('-nc', type=float, default=0.00001, help="learning constant, default value 0.00001");
    parser.add_argument('-mc', type=float, default=0.001, help="momentum constant, default value 0.001");
    parser.add_argument('-d', action="store_true", default=False, help="whether to recognize using debug algorithm or not");
    parser.add_argument('-o', type=str, default="default.txt", help="set name of output file");
    opts = parser.parse_args();

    # creating a network with input, hidden, and output nodes
    n = Network(26, 150, 7);

    # setting data vars
    trainingData = None;
    testingData = None;

    # setting vars according to args
    if(opts.t is not None):
        trainingData = importCSV("encodedcsv/" + opts.t);
    if((opts.ww is not None)):
        n.ww = opts.ww;
    else:
        n.ww = None;
    if((opts.r is not None)):
        testingData = importCSV("encodedcsv/" + opts.r);
    if((opts.rw is not None)):
        n.importWeights("weights/" + opts.rw + "/wi.csv", "weights/" + opts.rw + "/wo.csv");

    # train
    if((trainingData is not None)):
        start = time.time();
        print("Training Progress: ");
        down();
        pbar2 = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=int(opts.ic)).start();
        for i in range(int(opts.ic)):
            up();
            pbar1 = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=len(trainingData)).start();
            for x in range(len(trainingData)):
                n.train(trainingData[x], opts.nc, opts.mc);
                pbar1.update(float(x));
            pbar1.finish();
            pbar2.update(float(i));
            # print('error %-.5f' % error);
        pbar2.finish();
        n.saveWeights();
        end = time.time();
        print("\nTime took to train: " + str(end - start) + " seconds");

    # test
    if((testingData is not None)):
        start = time.time();
        if opts.d:
            n.recognizeDebug(testingData);
        else:
            n.recognize(testingData, opts.o);
        end = time.time();
        print("\nTime took to recognize: " + str(end - start) + " seconds");
