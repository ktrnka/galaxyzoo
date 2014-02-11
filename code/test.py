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
import train

def parseArgs():
    parser = argparse.ArgumentParser()
    #parser.add_argument("trainingInput", help="Numpy archives containing training and validation data.")
    parser.add_argument("model", help="Pickled model")
    parser.add_argument("testingInput", help="Numpy archive with testing data.")
    parser.add_argument("outFile", help="Destination file for outputs.")
    parser.add_argument("--headersFrom", default=None, help="Load the CSV header from specified file")
    parser.add_argument("--pca", help="Pickle file containing PCA to use. Number of features MUST match.")
    return parser.parse_args()

def loadTestingData(filename, pca=None):
    npzfile = numpy.load(filename)
    testData = npzfile[npzfile.files[0]]
    labels = testData[:,0]
    features = testData[:,1:]
    
    if pca:
        with open(pca, "r") as pcaIn:
            pca = pickle.load(pcaIn)
            return labels, pca.transform(features)
    else:
        return labels, features

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
    
    #trainingInputs, trainingOutputs, validationInputs, validationOutputs = train.loadSets(args.trainingInput, pca=args.pca)
    with open(args.model, "r") as modelIn:
        model = pickle.load(modelIn)
        
    testingLabels, testingFeatures = loadTestingData(args.testingInput, pca=args.pca)
    
    #model = linear_model.Ridge(alpha=60)
    #model.fit(numpy.vstack([trainingInputs, validationInputs]), numpy.vstack([trainingOutputs, validationOutputs]))
    
    #predictions = forceNonneg(model.predict(testingInputs[:,1:]))
    predictions = model.predict(testingFeatures)
    testingLabels.shape = (predictions.shape[0], 1)
    numpy.savetxt(args.outFile, numpy.hstack([testingLabels, predictions]), delimiter=",", fmt=["%d"] + ["%f"] * predictions.shape[1], header=loadHeader(args.headersFrom))
    

if __name__ == '__main__':
    sys.exit(main())