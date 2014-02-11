#!/usr/bin/env python

from __future__ import unicode_literals
import argparse
import sys
import numpy
from sklearn import linear_model
from sklearn import svm
import codecs
import pickle
import sklearn.decomposition
import filehandlers

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Pickled model")
    parser.add_argument("testingInput", help="Numpy archive with testing data.")
    parser.add_argument("outFile", help="Destination file for outputs.")
    parser.add_argument("--headersFrom", default=None, help="Load the CSV header from specified file")
    parser.add_argument("--pca", help="Pickle file containing PCA to use. Number of features MUST match.")
    return parser.parse_args()

def forceNonneg(predictions):
    predictions[predictions < 0] = 0
    return predictions

def loadHeader(filename):
    if not filename:
        return "The header wasn't specified"
    
    with codecs.open(filename, "r") as fileIn:
        for line in fileIn:
            return line.strip()

def main():
    args = parseArgs()
    
    with open(args.model, "r") as modelIn:
        model = pickle.load(modelIn)
        
    testingLabels, testingFeatures = filehandlers.loadTestingData(args.testingInput, pca=args.pca)
    if args.pca:
        pca = filehandlers.loadPca(args.pca)
        testingFeatures = pca.transform(testingFeatures)
    
    predictions = model.predict(testingFeatures)
    testingLabels.shape = (predictions.shape[0], 1)
    numpy.savetxt(args.outFile, numpy.hstack([testingLabels, predictions]), delimiter=",", fmt=["%d"] + ["%f"] * predictions.shape[1], header=loadHeader(args.headersFrom))
    
    print "Don't forget to delete the comment mark on the first line of output"
    

if __name__ == '__main__':
    sys.exit(main())