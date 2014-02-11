#!/usr/bin/env python

from __future__ import unicode_literals
import os
import sys
import cv2
import argparse
import numpy
import io
import csv

class Example:
    def __init__(self, exampleId, x, y):
        self.id = exampleId
        self.x = x
        self.y = y
    
    def inputWithId(self):
        return numpy.insert(self.x, 0, self.id)
    
    def outputWithId(self):
        return numpy.insert(self.y, 0, self.id)


def extractNumericId(filename):
    basename = os.path.basename(filename)
    no_ext, ext = basename.split(".")
    return int(no_ext)

def centerCrop(image, w, h):
    shape = image.shape
    
    xMin, xMax = (shape[0] - w)/2, (shape[0] + w)/2
    yMin, yMax = (shape[1] - h)/2, (shape[1] + h)/2
    
    #print "Resizing from {} to {}: {}".format(shape, (w, h), ((xMin, xMax), (yMin, yMax)))
    return image[xMin:xMax,yMin:yMax]

def loadFeatures(imageDir, args):
    """Load the images from specified dir and return a dict of Examples"""
    exampleDict = dict()

    if args.sampledImageDir and not os.path.exists(args.sampledImageDir):
        os.makedirs(args.sampledImageDir)
        
    imageFiles = [ file for file in os.listdir(imageDir) if file.endswith(".jpg") ]
    for i, imageFile in enumerate(imageFiles):
        fullPath = os.path.join(args.inputdir, imageFile).replace("\\", "/")

        # load greyscale
        image = cv2.imread(fullPath, 0)
        
        # center crop
        image = centerCrop(image, args.centerCropSize, args.centerCropSize)
        if args.sampleResolution < args.centerCropSize:
            image = cv2.resize(image, (args.sampleResolution, args.sampleResolution), interpolation=cv2.INTER_CUBIC)
        
        if args.sampledImageDir:
            cv2.imwrite(os.path.join(args.sampledImageDir, imageFile), image)
        
        exampleId = extractNumericId(imageFile)
        exampleDict[exampleId] = Example(exampleId, image.flatten() / 256.0, None)
        
        if args.maxExamples and i >= args.maxExamples:
            break
    
    return exampleDict

def annotateOutputValues(exampleDict, solutionsFile, args):
    """Load and parse training images and their output values."""
    
    with io.open(solutionsFile, "r") as solutionsIn:
        for rowNum, row in enumerate(csv.reader(solutionsIn)):
            if rowNum == 0:
                continue
            
            exampleId = int(row[0])

            if exampleId in exampleDict:
                exampleDict[exampleId].y = numpy.array([ float(val) for val in row[1:] ])
    
    return exampleDict

def splitTrainingData(exampleDict):
    examples = exampleDict.values()
    breakPoint = int(len(examples) * 0.80)
    trainingSet, validationSet = examples[:breakPoint], examples[breakPoint:]
    
    trainingInputs = numpy.vstack([ example.inputWithId() for example in trainingSet ])
    trainingOutputs = numpy.vstack([ example.outputWithId() for example in trainingSet ])
    validationInputs = numpy.vstack([ example.inputWithId() for example in validationSet ])
    validationOutputs = numpy.vstack([ example.outputWithId() for example in validationSet ])
    
    return trainingInputs, trainingOutputs, validationInputs, validationOutputs

def saveTrainingData(trainingInputs, trainingOutputs, validationInputs, validationOutputs, outFilename):
    numpy.savez_compressed(outFilename, trainingInputs=trainingInputs, trainingOutputs=trainingOutputs, validationInputs=validationInputs, validationOutputs=validationOutputs)

def main():
    """Do something"""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir", help="Directory of images to process")
    parser.add_argument("outfile", help="Output filename. Suggested .npz extension")
    parser.add_argument("--sampledImageDir", default=None, help="Directory to store downsampled images.")
    parser.add_argument("--maxExamples", default=None, type=int, help="Maximum number of examples to process")
    parser.add_argument("--sampleResolution", default=56, type=int, help="Height and width of the scaled image. Should be multiple of centerCropSize")
    parser.add_argument("--centerCropSize", default=140, type=int, help="Height/width of a box to use for center cropping.")
    parser.add_argument("--solutionsFile", default=None, help="File with the solutions (enables training data mode)")
    args = parser.parse_args()
    
    exampleDict = loadFeatures(args.inputdir, args)
    
    if args.solutionsFile:
        annotateOutputValues(exampleDict, args.solutionsFile, args)
        trainingInputs, trainingOutputs, validationInputs, validationOutputs = splitTrainingData(exampleDict)
        saveTrainingData(trainingInputs, trainingOutputs, validationInputs, validationOutputs, args.outfile)
    else:
        inputs = numpy.vstack([ example.inputWithId() for example in exampleDict.values() ])
        numpy.savez_compressed(args.outfile, inputs)


if __name__ == '__main__':
    sys.exit(main())
