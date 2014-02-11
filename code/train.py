#!/usr/bin/env python

from __future__ import unicode_literals
import argparse
import sys
import numpy
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import filehandlers

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFiles", nargs="+", help="One or more numpy archives containing training and validation data.")
    parser.add_argument("--pca", help="Pickle file containing PCA to use. Number of features MUST match.")
    return parser.parse_args()

def rms(predictions, actuals):
    # see also mean_squared_error, but that doesn't work if predictions is a scalar
    diffs = predictions - actuals
    return (numpy.dot(diffs, diffs)/diffs.size) ** 0.5

def normalizeAnswers(predictionArray, startCol, endCol, sumToCol=None):
    """Normalize the values between start and end to sumToCol"""
    
    sums = predictionArray[:,startCol:endCol].sum(axis=1) + numpy.finfo(predictionArray.dtype).eps
    #print "First 3 sums: {}".format(sums[:3])
    predictionArray[:,startCol:endCol] /= numpy.tile(sums, (endCol-startCol, 1)).transpose()
    
    if sumToCol:
        #print "First 3 target sums: {}".format(predictionArray[:3,sumToCol])
        predictionArray[:,startCol:endCol] *= numpy.tile(predictionArray[:,sumToCol], (endCol-startCol, 1)).transpose()
        #print "After: {}".format(predictionArray[:3,startCol:endCol].sum(axis=1))

def normalizePredictions(predictions):
    
    #print "Before normalization: {}".format(predictions[:3])
    # normalize Class 1.x
    normalizeAnswers(predictions, 0, 3)

    # normalize Class 2.x
    normalizeAnswers(predictions, 3, 5, 1)
    
    # normalize Class 3.x
    normalizeAnswers(predictions, 5, 7, 4)
    
    # normalize Class 4.x (both 3.x answers lead here, so sum to its parent)
    normalizeAnswers(predictions, 7, 9, 4)
    
    # normalize Class 5.x
    normalizeAnswers(predictions, 9, 13, 8)
    
    # normalize Class 7.x
    normalizeAnswers(predictions, 15, 18, 0)
    
    # normalize Class 9.x
    normalizeAnswers(predictions, 25, 28, 3)
    
    # normalize Class 10.x
    normalizeAnswers(predictions, 28, 31, 7)
    
    # normalize Class 11.x (all pathes from 10)
    normalizeAnswers(predictions, 31, 37, 7)
    
    # classes with multiple pathes: 6.x, but luckily it sums to 1
    normalizeAnswers(predictions, 13, 15)
    
    # normalize Class 8.x to output of 6.1
    normalizeAnswers(predictions, 18, 25, 13)

    #print "After normalization: {}".format(predictions[:3])

    return predictions

def transformColsForLR(outputSlice):
    Z = numpy.zeros(outputSlice.shape)
    Z[numpy.arange(Z.shape[0]), outputSlice.argmax(axis=1)] = 1
    
    Z[outputSlice == 0] = 0
    
    outputSlice -= outputSlice
    outputSlice += Z

def transformOutputsForLR(outputs):
    Y = outputs.copy()
    transformColsForLR(Y[:,0:3])
    transformColsForLR(Y[:,3:5])
    transformColsForLR(Y[:,5:7])
    transformColsForLR(Y[:,7:9])
    transformColsForLR(Y[:,9:13])
    transformColsForLR(Y[:,13:15])
    transformColsForLR(Y[:,15:18])
    transformColsForLR(Y[:,18:25])
    transformColsForLR(Y[:,25:28])
    transformColsForLR(Y[:,28:31])
    transformColsForLR(Y[:,31:37])
    return Y

def testLRTransform(outputs):
    print "Before LR transform:\n{}".format(outputs[0:3])
    print "After LR transform:\n{}".format(transformOutputsForLR(outputs[0:3]))

def forceNonneg(predictions):
    predictions[predictions < 0] = 0
    return predictions

def loadSets(trainingFilename, pcaFile=None):
    npzfile = numpy.load(trainingFilename)
    
    if pcaFile:
        with open(pcaFile, "r") as pcaIn:
            pca = pickle.load(pcaIn)
        return pca.transform(npzfile["trainingInputs"][:,1:]), npzfile["trainingOutputs"][:,1:], pca.transform(npzfile["validationInputs"][:,1:]), npzfile["validationOutputs"][:,1:]
    return npzfile["trainingInputs"][:,1:], npzfile["trainingOutputs"][:,1:], npzfile["validationInputs"][:,1:], npzfile["validationOutputs"][:,1:]

def computeRowMinMax(X):
    return X.min(axis=1), X.max(axis=1)

def scale(X, rowMin, rowMax):
    Z = X.copy()
    
    rowScale = rowMax - rowMin
    for i in xrange(Z.shape[1]):
        Z[:,i] = (Z[:,i] - rowMin) / rowScale
    
    return Z

def experimentLogisticRegression(trainingInputs, trainingOutputs, validationInputs, validationOutputs):
    # transform the outputs
    lrTrainingOutputs = transformOutputsForLR(trainingOutputs)
    lrValidationOutputs = transformOutputsForLR(validationOutputs)
    
    for lambder in (1, 10, 50, 100):
        print "\nLogistic regression (lambda={})".format(lambder)
        trainingOutputProbs = numpy.zeros(trainingOutputs.shape)
        validationOutputProbs = numpy.zeros(validationOutputs.shape)
        for outputIndex in xrange(trainingOutputs.shape[1]):
            print "Training output class {}".format(outputIndex)
            lr = linear_model.LogisticRegression(C=1.0/lambder)
            lr.fit(trainingInputs, lrTrainingOutputs[:,outputIndex])
            
            trainingOutputProbs[:,outputIndex] = lr.predict_proba(trainingInputs)[:,1]
            validationOutputProbs[:,outputIndex] = lr.predict_proba(validationInputs)[:,1]
        
        print "\tTraining (unnorm): {}".format(rms(trainingOutputProbs.ravel(), trainingOutputs.ravel()))
        print "\tTraining (norm): {}".format(rms(normalizePredictions(forceNonneg(trainingOutputProbs)).ravel(), trainingOutputs.ravel()))
    
        print "\tValidation (unnorm): {}".format(rms(validationOutputProbs.ravel(), validationOutputs.ravel()))
        print "\tValidation (norm): {}".format(rms(normalizePredictions(forceNonneg(validationOutputProbs)).ravel(), validationOutputs.ravel()))

def experimentLinearRegression(trainingInputs, trainingOutputs, validationInputs, validationOutputs, alpha=60):
    print "\nRegularized linear regression (lambda = {})".format(alpha)
    regr = linear_model.Ridge(alpha=alpha)
    regr.fit(trainingInputs, trainingOutputs)
    print "\tTraining: {}".format(rms(regr.predict(trainingInputs).ravel(), trainingOutputs.ravel()))
    print "\tValidation: {}".format(rms(regr.predict(validationInputs).ravel(), validationOutputs.ravel()))
    
    print "\nRegularized linear regression (with outputs forced to non-negative)"
    print "\tTraining: {}".format(rms(forceNonneg(regr.predict(trainingInputs)).ravel(), trainingOutputs.ravel()))
    print "\tValidation: {}".format(rms(forceNonneg(regr.predict(validationInputs)).ravel(), validationOutputs.ravel()))

    #print "\nRegularized linear regression (with outputs forced to non-negative then normalized)"
    #print "\tTraining: {}".format(rms(normalizePredictions(forceNonneg(regr.predict(trainingInputs))).ravel(), trainingOutputs.ravel()))
    #print "\tValidation: {}".format(rms(normalizePredictions(forceNonneg(regr.predict(validationInputs))).ravel(), validationOutputs.ravel()))
    #
    ## try mean/stddev normalization
    #rowMin, rowMax = computeRowMinMax(trainingInputs)
    #scaledTrainingInputs = scale(trainingInputs, rowMin, rowMax)
    #
    #rowMin, rowMax = computeRowMinMax(validationInputs)
    #scaledValidationInputs = scale(validationInputs, rowMin, rowMax)
    #
    ## try a few different regularization params
    #for lambder in (10, 20, 40, 80, 160):
    #    scaledRegression = linear_model.Ridge(alpha=lambder)
    #    scaledRegression.fit(scaledTrainingInputs, trainingOutputs)
    #    print "\nRegularized linear regression (with feature scaling, lambda={})".format(lambder)
    #    print "\tTraining: {}".format(rms(forceNonneg(scaledRegression.predict(scaledTrainingInputs)).ravel(), trainingOutputs.ravel()))
    #    print "\tValidation: {}".format(rms(forceNonneg(scaledRegression.predict(scaledValidationInputs)).ravel(), validationOutputs.ravel()))


def runTests(filename, pcaFile=None):
    print "TRAINING/TESTING {}".format(filename)
    
    trainingInputs, trainingOutputs, validationInputs, validationOutputs = filehandlers.loadTrainingSets(filename)
    if pcaFile:
        pca = filehandlers.loadPca(pcaFile)
        trainingInputs = pca.transform(trainingInputs)
        validationInputs = pca.transform(validationInputs)

    numExamples = trainingInputs.shape[0]
    numFeatures = trainingInputs.shape[1]
    print "Loaded training data with shape {} and {}".format(trainingInputs.shape, trainingOutputs.shape)
    print "Loaded validation data with shape {} and {}".format(validationInputs.shape, validationOutputs.shape)
    
    avgPredictions = trainingOutputs.mean(axis=0)

    print "Baselines"
    print "\tPredict zero on all outputs (val): {}".format(rms(0, validationOutputs.ravel()))
    print "\tPredict one on all outputs (val): {}".format(rms(1, validationOutputs.ravel()))
    print "\tPredict average on all outputs (val): {}".format(rms(avgPredictions.repeat(validationOutputs.shape[0]), validationOutputs.ravel()))
    
    print "\tPredict one on all outputs then norm (val): {}".format(rms(normalizePredictions(numpy.ones(validationOutputs.shape)).ravel(), validationOutputs.ravel()))
    
    tiledAverage = numpy.tile(avgPredictions, (validationOutputs.shape[0], 1))
    print "\tPredict average on all outputs then norm (val): {}".format(rms(normalizePredictions(tiledAverage).ravel(), validationOutputs.ravel()))

    experimentLinearRegression(trainingInputs, trainingOutputs, validationInputs, validationOutputs)

#    print "\nSVM regression"
#    regr = svm.SVR()
#    regr.fit(trainingInputs, trainingOutputs[:,0])
#    print "\tTraining: {}".format(rms(regr.predict(trainingInputs).ravel(), trainingOutputs.ravel()))
#    print "\tValidation: {}".format(rms(regr.predict(validationInputs).ravel(), validationOutputs.ravel()))
 
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, 5)
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, 5, maxFeatures="auto")
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, None)
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, None, maxFeatures=int(numFeatures * 0.75))
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 40, None, maxFeatures=int(numFeatures * 0.5))
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 80, None, maxFeatures=int(numFeatures * 0.5))
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 160, None, maxFeatures=int(numFeatures * 0.5))

    #for numTrees in [ 40, 80, 160 ]:
    #    experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, numTrees, None, maxFeatures=int(numFeatures * 0.5), minSplit=5)
    
    experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 40, None, maxFeatures=int(numFeatures * 0.5), minSplit=20, saveAsPath="randomForest-40t-0.5f-20mss.pickle")

    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, None, maxFeatures=int(numFeatures * 0.25))
    
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 40, None, maxFeatures="auto")
    #experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, 20, None, maxFeatures="auto", minSplit=5)

def experimentRandomForest(trainingInputs, trainingOutputs, validationInputs, validationOutputs, numTrees, maxDepth, maxFeatures="sqrt", minSplit=20, saveAsPath=None):
    print "\nRandom Forests: {} trees, max depth {}, max features {}, min samples split {}".format(numTrees, maxDepth, maxFeatures, minSplit)
    theta = RandomForestRegressor(n_estimators=numTrees, max_depth=maxDepth, max_features=maxFeatures, min_samples_split=minSplit, n_jobs=3).fit(trainingInputs, trainingOutputs)
    print "\tTraining: {}".format(rms(theta.predict(trainingInputs).ravel(), trainingOutputs.ravel()))
    print "\tValidation: {}".format(rms(theta.predict(validationInputs).ravel(), validationOutputs.ravel()))
    
    if saveAsPath:
        with open(saveAsPath, "w") as modelOut:
            pickle.dump(theta, modelOut)

def main():
    args = parseArgs()

    for inputFile in args.inputFiles:
        runTests(inputFile, args.pca)


if __name__ == '__main__':
    sys.exit(main())