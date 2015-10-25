#! /usr/bin/env python

# a tool that creates a .csv file of processed parameters of character images from a directory
# author: Philip Xu

import math
import random
import string
import numpy as np
import scipy as sc
import os
import csv
import sys
import time
import json
from functools import partial
from PIL import Image
from multiprocessing import Pool, cpu_count
from progressbar import Bar, Percentage, ProgressBar, SimpleProgress

class Receptor:
    # constructs a receptor opject from a image file
    def __init__(self, inputURL, letter):
        self.inputArr = np.array(Image.open(inputURL));
        self.inputArr.astype(int);
        self.letter = letter;
        # output is a list of all receptor values
        # to be fed into the neural net
        # x1 - x11 = horizontal values
        # x12 - x22 = vertical values
        # x23 = horizontal symmetry value
        # x24 = vertical symmetry value
        # x25 = cavity count
        # x26 = block count
        self.output = [];

        # set array print options to more readable
        # np.set_printoptions(linewidth = 1000);
        # np.set_printoptions(threshold=np.nan);
        # print("Original Image: \n" + str(self.inputArr));

        # fills output list with receptors param values
        self._createHorizontalValueReceptors();
        self._createVerticalValueReceptors();
        self._createHorizontalSymmetryReceptors();
        self._createVerticalSymmetryReceptors();
        self._createCavityReceptors();
        self._createBlockReceptors();
        # self._createHadamardTransformReceptors();
        # self._createSumReceptors();

        # this prints the output list for debug
        # print("Letter: " + letter + ", Output Array: " + str(self.output));

    # creates 11 horizontal value parameters, at 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% width
    def _createHorizontalValueReceptors(self):
        self.hParams = [];
        self.yIndices = [];
        for x in range(11):
            self.hParams.append(0);
            self.yIndices.append(math.ceil(self.inputArr.shape[0]*(x*0.1)) - 1);
        for i in range(self.inputArr.shape[1]):
            for y in range(11):
                if(self.inputArr[self.yIndices[y], i] == 0):
                    self.hParams[y] += 1;

        self.output.extend(self.hParams);
        #print("yIndices: " + str(self.yIndices));
        #print("hParams: " + str(self.hParams));

    # creates 11 vertical value parameters, at 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% height
    def _createVerticalValueReceptors(self):
        self.vParams = [];
        self.xIndices = [];
        for x in range(11):
            self.vParams.append(0);
            self.xIndices.append(math.ceil(self.inputArr.shape[1]*(x*0.1)) - 1);
        for i in range(self.inputArr.shape[0]):
            for y in range(11):
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

    def _createSumReceptors(self):
        n = 0.0;
        for x in range(0, len(self.output)):
            n += self.output[x];
        self.output.append(n/1000.0);

    def _createCavityReceptors(self):
        ho, wo = self.inputArr.shape;
        hn = ho + 2;
        wn = wo + 2;
        self.newArr = np.full([hn, wn], 255, dtype='int64');

        for x in range(ho):
            for y in range(wo):
                self.newArr[x + 1,y + 1] = self.inputArr[x,y];

        self.newArr = self.newArr.astype(int);
        #print(self.newArr);

        cCount = -1;
        for x in range(hn):
            for y in range(wn):
                if(self.newArr[x, y] == 255):
                    self._cavityFloodFill(x, y);
                    cCount += 1;
        #print(self.newArr);
        #print("cCount: " + str(cCount));
        self.output.append(cCount);

    def _createBlockReceptors(self):
        ho, wo = self.inputArr.shape;
        hn = ho;
        wn = wo;
        self.blockArr = self.inputArr;
        self.blockArr = self.blockArr.astype(int);
        #print(self.blockArr);

        bCount = 0;
        for x in range(hn):
            for y in range(wn):
                if(self.blockArr[x, y] == 0):
                    self._blockFloodFill(x, y);
                    bCount += 1;

        #print(self.blockArr);
        #print("cCount: " + str(cCount));
        self.output.append(bCount);

    def _cavityFloodFill(self, x, y):
        h, w = self.newArr.shape;
        toFill = set();
        toFill.add((x,y));
        while toFill:
            (x,y) = toFill.pop();
            if(self.newArr[x,y] == 255):
                self.newArr[x,y] = 11;

                if x > 0: # left
                    toFill.add((x-1, y));

                if y > 0: # up
                    toFill.add((x, y-1));

                if x < h-1: # right
                    toFill.add((x+1, y));

                if y < w-1: # down
                    toFill.add((x, y+1));

    def _blockFloodFill(self, x, y):
        h, w = self.blockArr.shape;
        toFill = set();
        toFill.add((x,y));
        while toFill:
            (x,y) = toFill.pop();
            if(self.blockArr[x,y] == 0):
                self.blockArr[x, y] = 11;

                if x > 0: # left
                    toFill.add((x-1, y));

                if y > 0: # up
                    toFill.add((x, y-1));

                if x < h-1: # right
                    toFill.add((x+1, y));

                if y < w-1: # down
                    toFill.add((x, y+1));

    def _blockshaped(self, arr, nrows, ncols):
        h, w = arr.shape;
        res = (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols));
        return res;


def readFolderWithName(rootDirectory, filename, multiProcessing):
    num = 0;
    start = time.time();
    with open("encodedcsv/" + filename, 'wb') as paramfile:
        csv_writer = csv.writer(paramfile);
        for subdir, dirs, files in os.walk("data/" + rootDirectory + "/"):
            pbar = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=len(files)).start();
            for name in files:
                receptor = Receptor(subdir + name, letter = name[-5]);
                values = receptor.output;
                values[0:0] = name[-5];
                csv_writer.writerow([x for x in values]);
                # print("fileNumber: " + str(num) + ", letter: " + name[-5]);
                num += 1;
                pbar.update(num);
            pbar.finish();
    end = time.time();
    print("\nSaved data as: " + filename);
    print("Time Elapsed: " + str(end - start) + " seconds");

def readFolderWithJSON(rootDirectory, filename, JSONname, multiProcessing):

    if multiProcessing:
        print("Starting Multiprocessing...");
        start = time.time();
        imgs = [];

        try:
            threadcount = cpu_count();
        except NotImplementedError:
            threadcount = 2;
            print('Error getting core counts, setting threadcount to 2');
        threads = Pool(threadcount);

        print("Creating task list...");
        with open(JSONname) as f:
            data = json.load(f);
            for subdir, dirs, files in os.walk("data/" + rootDirectory + "/"):
                for name in files:
                    imgs.append([name, subdir, str(json.dumps(data["data"][int(name[:-4])])[1])]);

        print("Distributing tasks across " + str(threadcount) + " cores...");
        final = threads.map(partial(mp), imgs);

        print("Writing results to output file...")
        with open("encodedcsv/" + filename, 'wb') as paramfile:
            csv_writer = csv.writer(paramfile);
            for row in final:
                csv_writer.writerow([x for x in row]);

        end = time.time();
        print("\nSaved data as: " + filename);
        print("Time Elapsed: " + str(end - start) + " seconds");

    else:
        num = 0;

        print("Starting sequential processing...");
        start = time.time();
        with open(JSONname) as f:
            data = json.load(f);
            with open("encodedcsv/" + filename, 'wb') as paramfile:
                csv_writer = csv.writer(paramfile);
                for subdir, dirs, files in os.walk("data/" + rootDirectory + "/"):
                    imgs = files;
                    sub = subdir;
                    pbar = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=len(files)).start();
                    for name in files:
                        receptor = Receptor(subdir + name, letter = str(json.dumps(data["data"][int(name[:-4])])[1]));
                        values = receptor.output;
                        values[0:0] = str(json.dumps(data["data"][int(name[:-4])])[1]);
                        # print values;
                        csv_writer.writerow([x for x in values]);
                        # print("fileNumber: " + str(num) + ", letter: " + name[-5]);
                        num += 1;
                        pbar.update(num);
                    pbar.finish();

        end = time.time();
        print("\nSaved data as: " + filename);
        print("Time Elapsed: " + str(end - start) + " seconds");

def mp((name, subdir, letter)):
    receptor = Receptor(subdir + name, letter);
    values = receptor.output;
    values[0:0] = letter;
    return values;

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Parser to create parameter file from directory of character images");
    parser.add_argument('-r', type=str, required=True, help="set name of directory to parse");
    parser.add_argument('-o', type=str, default="param.csv", help="set prefered names of output csv file, default param.csv");
    parser.add_argument('-mp', action='store_true', default=False, help="Multi-processing, default false");
    parser.add_argument('-json', type=str, help="set JSON file name of letters, if none then assumed to be non-JSON");
    opts = parser.parse_args();

    if(opts.json):
        readFolderWithJSON(opts.r, opts.o, opts.json, opts.mp);
    else:
        readFolderWithName(opts.r, opts.o, opts.mp);
