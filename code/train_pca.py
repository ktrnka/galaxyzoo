#!/usr/bin/env python

from __future__ import unicode_literals
import argparse
import sys
import numpy
import sklearn.decomposition
import train
import pickle

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="npz file with the input data to model")
    parser.add_argument("output", help="Output pickle file to save the PCA analysis into")
    parser.add_argument("--variance", default=0.95, type=float, help="Fraction of the variance to retain.")
    return parser.parse_args()

def main():
    args = parseArgs()
    X, _, _, _ = train.loadSets(args.input)
    
    pca = sklearn.decomposition.PCA(args.variance)
    pca.fit(X)
    
    numComponents = pca.components_.shape[0]
    numFeatures = pca.components_.shape[1]
    print "{:.1f}% ({} / {}) features to retain {}% of the variance".format(100. * numComponents / numFeatures, numComponents, numFeatures, 100 * args.variance)
    
    with open(args.output, "w") as pcaOut:
        pickle.dump(pca, pcaOut)

if __name__ == '__main__':
    sys.exit(main())