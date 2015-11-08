if __name__ == '__main__':
    import numpy as np
    import argparse
    import csv
    from random import shuffle;
    parser = argparse.ArgumentParser(description="Shuffles CSVs");
    parser.add_argument('filename', type=str, help="an input csv file");
    opts = parser.parse_args();

    with open(opts.filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ');
        trainingData = list(reader);

    shuffle(trainingData);
    np.savetxt(opts.filename, trainingData, delimiter=" ", fmt="%s");
