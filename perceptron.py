# -*- coding: utf-8 -*-
"""
author:       Ben Davis
date:         2020-06-30
description:  creates a network of perceptrons to identify hand written
              numbers read in from a CSV file. These perceptrons are trained
              for 50 epochs. At each epoch, an output file is generated containing 
              a confusion matrix for that epoch.
"""

import numpy as np
    
class Perceptron:
    """
    A class for artificial neurons in the form of perceptrons. Contains
    all the basic functionality for perceptrons.
    """
    
    def __init__(self, inputs, target, clss):
        """
        Accepts a vector of inputs, a target value of 0 or 1, 
        and a vector of weights. Note that the vectors of weights
        and inputs must be the same size
        """
        self.initWeights(inputs.size)
        self.target  = target
        self.inputs  = inputs
        self.clss = clss
    
    
    def output(self):
        """
        Determines the output for the perceptron by finding the dot product
        of the two input vectors. Retruns 1 if the dot product is > 0 and 0 
        otherwise
        """
        y = np.dot(self.inputs, self.weights)
        
        return y
        
    
    def updateWeights(self, learningRate, output):
        """
        Loops through the perceptrons weights updating them according to the 
        perceptron learning rule with the given learningRate. 
        """
        
        if output > 0:
            output = 1
        else:
            output = 0
        
        newWeights = lambda w,x: w + (learningRate*(self.target-output)*x)
        
        vectorUpdate = np.vectorize(newWeights)
        
        self.weights = vectorUpdate(self.weights, self.inputs)     
            
    def initWeights(self, numWeights):
        """
        initializes the weights to be used by the perceptrons before any training 
        has been done.  The weightVector is filled with random values between -.05
        and .05
        """
        self.weights = np.random.uniform(-.05, 0.06, numWeights)
    
    
    def accuracy(self, output):
        """
        returns the accuracy of the perceptron with the givin input data as a 
        fraction of correct targets versus incorrect ones
        """
        if output > 0 and self.target == 1 or (output < 0 and self.target == 0):
            return 1
        else:
            return 0
    
    def setTarget(self, newTarget):
        """
        Sets the value of target to the newTarget value
        """
        self.target = newTarget
   
    def setInput(self, newInput):
        """
        Sets the value of input to a new input value
        """
        self.inputs = newInput
        
        
def trainPerceptrons(perceptrons, numPerceptrons, epochs, valueMatrix, numRows, numCols, learningRate):
    """
    Trains a collection of 10 perceptrons for a given number of epochs. Training 
    and target data will be read in from a csv file and used to initialize
    each perceptron.
    
    """
    
    normalize = lambda t : t/255
    vNormalize = np.vectorize(normalize)
    rightCount = 0
    confuse = np.zeros((10,10))
    
    
    for x in range(0, epochs):
        rightCount = 0
        for y in range(0, numRows):
            
            row = np.append((valueMatrix[y][1:numCols]), 1)
            row = vNormalize(row) #divide row by 255 to normalize
            target = valueMatrix[y][0]
            outputs = np.zeros(numPerceptrons) #the outputs of each perceptron
            
            for z in range(0, numPerceptrons):
                #initialize each perceptron
                perceptrons[target].setTarget(1)
                perceptrons[z].setInput(row)
                output = perceptrons[z].output()
                
                outputs[z] = output
                
                #weights are updated starting with the 1st epoch if y==t
                if x > 0 and perceptrons[z].accuracy(output) == 0:
                    perceptrons[z].updateWeights(learningRate, output)
            
            prediction = searchNumpyArray(outputs, np.amax(outputs))
            
            if target == prediction:
                rightCount += 1
            else: #update the confusion matrix
                confuse[target][prediction] = confuse[target][prediction] + 1
                
            perceptrons[target].setTarget(0) #reset to 0 at the end
        
        #print the number of correct classifications
        print(rightCount/numRows)
        
        #output confustion matricies for each epoch
        fname = "output" + str(x) +".txt"
        np.savetxt(fname, confuse, fmt='%d')
        confuse = np.zeros((10,10))
        
        
        
def searchNumpyArray(toSearch,key):
    """
    Linearly searches a numpy array and returns the index of key if found, -1
    otherwise
    """
    size = (toSearch.size)
    
    for i in range(0, size):
        if toSearch[i] == key:
            return i
        
    return -1

def loadMatrix(csvFile):
    """
    loads the data into a big matrix so that it can be operated on much more easily
    """
    matrix = np.genfromtxt(csvFile, dtype=int, delimiter=",")
    
    return matrix
    
def perceptronExperiment(numPerceptrons, epochs, learningRate, trainFile, validFile, numRowsTrain, numRowsValid, numCols):
    """
    creates a perceptron experiment based on the number of perceptrons, epoch, and 
    learning rate.  A csv file is then used to load a matrix which is used to conduct
    a trial with the other variables
    """
    train = loadMatrix(trainFile)
    validate = loadMatrix(validFile)
    initial = np.zeros(numCols)
    p = Perceptron(initial , 0, 1)
    pArray = np.array([p])
    
    for x in range(0, numPerceptrons):
        j = Perceptron(initial, 0, x)
        pArray = np.append(pArray, j)
        
    print("training " + str(learningRate))
    trainPerceptrons(pArray, numPerceptrons, epochs, train, numRowsTrain, numCols, learningRate)
  
    print("validation " + str(learningRate))
    trainPerceptrons(pArray, numPerceptrons, epochs, validate, numRowsValid, numCols, learningRate)

def main():
    """
    """
    
    perceptronExperiment(10, 50, .00001, 'mnist_train.csv', 'mnist_validation.csv',60000, 10000, 785)
    perceptronExperiment(10, 50, .001, 'mnist_train.csv', 'mnist_validation.csv',60000, 10000, 785)
    perceptronExperiment(10, 50, .1, 'mnist_train.csv', 'mnist_validation.csv',60000, 10000, 785)

    return 0


if __name__ == "__main__":
    # execute only if run as a script
    main()
    
    