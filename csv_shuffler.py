if __name__ == '__main__':
    import numpy as np
    import argparse
    import csv, sys, operator
    from random import shuffle;
    parser = argparse.ArgumentParser(description="Shuffles CSVs");
    parser.add_argument('filename', type=str, help="an input csv file");
    opts = parser.parse_args();

    with open(opts.filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ');
        # sortedlist = sorted(reader, key=operator.itemgetter(1))
        # print sortedlist
        trainingData = list(reader);

    # trainingData = trainingData.sort()
    shuffle(trainingData);
    np.savetxt(opts.filename, trainingData, delimiter=" ", fmt="%s");
