import math
import random
import string
import numpy as np
import scipy as sc
import os
import csv
import sys
import time
from PIL import Image

class Receptor:
    def __init__(self, inputURL, letter):
        self.inputImg = Image.open(inputURL);
        self.inputArr = np.array(self.inputImg);
        self.output = [];
        self.letter = letter;
        #output is a list of all receptor values
        #to be fed into the neural net
        #x1, x2, x3, x4, x5 = horizontal values
        #x6, x7, x8, x9, x10 = vertical values
        #x11 = horizontal symmetry value
        #x12 = vertical symmetry value
        sys.setrecursionlimit(50000);
        np.set_printoptions(linewidth = 1000);
        np.set_printoptions(threshold=np.nan)
        self.inputArr.astype(int);
        #print("Original Image: \n" + str(self.inputArr));
        self._createHorizontalValueReceptors();
        self._createVerticalValueReceptors();
        self._createHorizontalSymmetryReceptors();
        self._createVerticalSymmetryReceptors();
        self._createHadamardTransformReceptors();
        self._createCavityReceptors();
        #print("Letter: " + letter + ", Output Array: " + str(self.output));

    def _createHorizontalValueReceptors(self):
        self.hParams = [];
        self.yIndices = [];
        for x in range(5):
            self.hParams.append(0);
            self.yIndices.append(math.ceil(self.inputArr.shape[0]*(-0.1 + (x+1)*0.2)) - 1);
        for i in range(self.inputArr.shape[1]):
            for y in range(5):
                if(self.inputArr[self.yIndices[y], i] == 0):
                    self.hParams[y] += 1;

        self.output.extend(self.hParams);
        #print("yIndices: " + str(self.yIndices));
        #print("hParams: " + str(self.hParams));

    def _createVerticalValueReceptors(self):
        self.vParams = [];
        self.xIndices = [];
        for x in range(5):
            self.vParams.append(0);
            self.xIndices.append(math.ceil(self.inputArr.shape[1]*(-0.1 + (x+1)*0.2)) - 1);
        for i in range(self.inputArr.shape[0]):
            for y in range(5):
                if(self.inputArr[i, self.xIndices[y]] == 0):
                    self.vParams[y] += 1;

        self.output.extend(self.vParams);
        #print("xIndices: " + str(self.xIndices));
        #print("vParams: " + str(self.vParams));

    def _createHorizontalSymmetryReceptors(self):
        totalElements = self.inputArr.size;
        totalReflectedElements = 0.0;
        self.hSym = 0;
        horizontalSplitOriginal = self._blockshaped(self.inputArr, self.inputArr.shape[0], math.floor(self.inputArr.shape[1]/2.0));
        leftOriginal = horizontalSplitOriginal[0];
        rightOriginal = horizontalSplitOriginal[1];
        leftFlipped = np.fliplr(leftOriginal);
        rightFlipped = np.fliplr(rightOriginal);

        for i in range(int(math.floor(self.inputArr.shape[1]/2.0))):
            for j in range(self.inputArr.shape[0]):
                if(leftFlipped[j,i] == rightOriginal[j,i]):
                    totalReflectedElements += 1;
                if(rightFlipped[j,i] == leftOriginal[j,i]):
                    totalReflectedElements += 1;

        self.hSym = totalReflectedElements/totalElements;

        self.output.append(self.hSym);
        #print("leftOriginal: \n" + str(leftOriginal));
        #print("rightOriginal: \n" + str(rightOriginal));
        #print("hSym: " + str(self.hSym));


    def _createVerticalSymmetryReceptors(self):
        totalElements = self.inputArr.size;
        totalReflectedElements = 0.0;
        self.vSym = 0;
        verticalSplitOriginal = self._blockshaped(self.inputArr, math.floor(self.inputArr.shape[0]/2.0), self.inputArr.shape[1]);
        topOriginal = verticalSplitOriginal[0];
        bottomOriginal = verticalSplitOriginal[1];
        topFlipped = np.flipud(topOriginal);
        bottomFlipped = np.flipud(bottomOriginal);

        for i in range(int(math.floor(self.inputArr.shape[0]/2.0))):
            for j in range(self.inputArr.shape[1]):
                if(topFlipped[i,j] == bottomOriginal[i,j]):
                    totalReflectedElements += 1;
                if(bottomFlipped[i,j] == topOriginal[i,j]):
                    totalReflectedElements += 1;

        self.vSym = totalReflectedElements/totalElements;

        self.output.append(self.vSym);
        #print("topOriginal: \n" + str(topOriginal));
        #print("bottomOriginal: \n" + str(bottomOriginal));
        #print("vSym: " + str(self.vSym));

    def _createHadamardTransformReceptors(self):
        #array must be square, apply Hadamard transform.
        pass

    def _createCavityReceptors(self):
        ho, wo = self.inputArr.shape;
        hn = ho + 2;
        wn = wo + 2;
        self.newArr = np.full([hn, wn], 255);

        for x in range(ho):
            for y in range(wo):
                self.newArr[x + 1,y + 1] = self.inputArr[x,y];

        self.newArr = self.newArr.astype(int);
        #print(self.newArr);

        cCount = -1;
        for x in range(hn):
            for y in range(wn):
                if(self.newArr[x, y] == 255):
                    self._floodFill(x, y);
                    cCount += 1;

        #print(self.newArr);
        #print("cCount: " + str(cCount));
        self.output.append(cCount);

    def _floodFill(self, x, y):
        h, w = self.newArr.shape;
        if(self.newArr[x,y] == 255):
            self.newArr[x, y] = 11;

            if x > 0: # left
                self._floodFill(x-1, y);

            if y > 0: # up
                self._floodFill(x, y-1);

            if x < h-1: # right
                self._floodFill(x+1, y);

            if y < w-1: # down
                self._floodFill(x, y+1);

        else:
            return;

    def _blockshaped(self, arr, nrows, ncols):
        h, w = arr.shape;
        res = (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols));
        return res;

def readFolder(rootDirectory):
    num = 0;
    start = time.time();
    with open('param1.csv', 'wb') as paramfile:
        csv_writer = csv.writer(paramfile);
        for subdir, dirs, files in os.walk(rootDirectory):
            for name in files:
                receptor = Receptor(subdir + name, letter = name[-5]);
                values = receptor.output;
                values[0:0] = name[-5];
                csv_writer.writerow([x for x in values]);
                print("fileNumber: " + str(num) + ", letter: " + name[-5]);
                num += 1;
    end = time.time();
    print("\nTime Elapsed: " + str(end - start) + " seconds");

def demo():
    receptor = Receptor("testdata/tags/11o.png", "p");

if __name__ == '__main__':
    #demo();
    readFolder(rootDirectory = "testdata/tags/");
