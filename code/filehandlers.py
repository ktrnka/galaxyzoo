#!/usr/bin/env python

import io
import pickle
import numpy

def loadPca(path):
    with io.open(path, "r") as pcaIn:
        return pickle.load(pcaIn)

def savePca(path, pca):
    with io.open(path, "w") as pcaOut:
        pickle.dump(pca, pcaOut)

def loadTrainingSets(npzArchivePath, pcaFile=None):
    npzfile = numpy.load(npzArchivePath)
    return npzfile["trainingInputs"][:,1:], npzfile["trainingOutputs"][:,1:], npzfile["validationInputs"][:,1:], npzfile["validationOutputs"][:,1:]

def loadTestingData(filename, pca=None):
    npzfile = numpy.load(filename)
    testData = npzfile[npzfile.files[0]]
    labels = testData[:,0]
    features = testData[:,1:]
    
    return labels, features