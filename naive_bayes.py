"""
author:       Ben Davis
date:         2020-07-24
description:  Implements a naive Basian classifier to work for any two files
              passed in that are formatted like UCI datasets
"""
import numpy as np
import sys
import math
def loadMatrix(file):
    """
    loads the data into a big matrix so that it can be operated on much more easily

    """
    
    matrix = np.genfromtxt(file)
    
    return matrix

def countNumClasses(classArr, numClasses):
    """
    Parameters
    ----------
    classArr : numpy array
        the array of class objects whose number of instances need to be counted

    Returns
    -------
    an array with the cound of each class at the correct indecies for the array
    """
    
    classCounts = np.zeros(numClasses)
    
    for x in range(0, len(classArr)):
        y =int(classArr[x])
        classCounts[y] += 1
        
    return classCounts
def outputTrain(matrixToOutput):
    """
    Parameters
    ----------
    matrixToOutput : numpy array
        the matrix being output to a file called "trainingOutput.txt"

    Returns
    -------
    None
    
    numChars = outFile.write("Class " + str(matrixToOutput[x][0]).format( +
                                 ", attribute " + str(matrixToOutput[x][1]) +
                                 ", mean = " + str(matrixToOutput[x][2]) +
                                 ", std = " + str(matrixToOutput[x][3]) + '\n')
    """    
    outFile = open("trainingOutput.txt", "w")
    for x in range(0, len(matrixToOutput)):
        for y in range(0, len(matrixToOutput[x][:, 0])):
            numChars = outFile.write("Class %d, attribute %d, mean = %.2f,  std = %.2f\n" %
                                     (matrixToOutput[x][y][0],  matrixToOutput[x][y][1],
                                      matrixToOutput[x][y][2],  matrixToOutput[x][y][3]))
    
    outFile.close()

def outputTest(matrixToOutput, accuracy):
    """
    Parameters
    ----------
    
    matrixToOutput : numpy matrix
        the matrix being written to "testOutput.txt" in a proper format
    accuracy : int
        the number of correct classifications, we'll divide it by the size 
        of the matrix to get the classification accuracy

    Returns
    -------
    None

    """

    outFile = open("testOutput.txt", "w")
    for x in range(0, len(matrixToOutput)):
            numChars = outFile.write("ID %5d, predicted %3d, probability %.4f, true %3d, accuracy %4.2f\n" %
                                    (matrixToOutput[x][0],  matrixToOutput[x][1],
                                      matrixToOutput[x][2],  matrixToOutput[x][3],
                                      matrixToOutput[x][4]))
    numChars = outFile.write("\nclassification accuracy=%6.4f" % (accuracy/len(matrixToOutput)))
    outFile.close()
    
def sortMatrix(matrix, maxNumClass, numRows, numCols):
    """
    
    Parameters
    ----------
    matrix : numpytarray
        the matrix being sorted by the first row
    maxNumClass : int
       the number of classes in the matrix to be sorted
    numRows : int
        the nummber of rows in the matrix to be sorted
    numCols : int
        the number of columns in the matrix to be sorted
    Returns
    -------
    output : numpy 3d array
        the matrix sorted into rows categories based on classes
    """
    output = [None] * (maxNumClass)
    
    for x in range((maxNumClass)):
        output[x]  = matrix[matrix[:, 0] == x+1]
        output[x]  = np.array(output[x])
        output[x]  = output[x][np.argsort(output[x][:, 1])]

    return output

def train(toTrain):
    """

    Parameters
    ----------
    toTrain : numpy array
        the array ti be trained
    Returns
    -------
    None.
    """
    
    return 

def test(toTest):
    """

    Parameters
    ----------
    toTest : numpy array
        the array of data to be tested
    outputs : numpy array
        an array of output values to be used to compute the classifications
    Returns
    -------
    None.

    """
            
def naive_bayes(trainingFile, testFile):
    """
    Parameters
    ----------
    trainingFile : string
        the path of the file to use to train the naive bayes classifier
    testFile : string
        the path of the file to use to test the vaive bayes classifier
    Returns
    -------
    None.
    """
    
    train = loadMatrix(trainingFile)
    test = loadMatrix(testFile)
    
    makeInt = lambda x: int(x)
    vmakeInt = np.vectorize(makeInt)
    
    divNumClass = lambda x, y: x/(y)
    vDivNumClass = np.vectorize(divNumClass)
    
    trainClasses = vmakeInt(train[:, len(train[0,:])-1])
    testClasses = vmakeInt(test[:,len(test[0, :])-1])
    numClasses = np.unique(trainClasses)
    
    numEachTrainClass = countNumClasses(trainClasses, len(numClasses)+1)
    numEachTestClass = countNumClasses(testClasses, len(numClasses)+1)
    numEachTrainClass = vDivNumClass(numEachTrainClass, len(train))
    numEachTrainClass = vDivNumClass(numEachTestClass, len(test))

    #trainCLasses, train, 
    for x in range(0, len(trainClasses)): #which is = to the number of rows in the matrix
        row = train[x, :len(train[x,:])-1]
        stdDev = np.std(row)
        if stdDev < 0.01:
            stdDev = 0.01          
        mean = np.mean(row)
        #we need to store std and mean in an array so that it can be used for testing
        if x == 0:
            stdsAndMeans = np.array([stdDev, mean])
            outputs = np.array([int(trainClasses[x]), int(x+1), mean, stdDev]) 
        else:
            stdsAndMeans = np.vstack((stdsAndMeans, np.array([stdDev, mean])))
            outputs = np.vstack((outputs, np.array([int(trainClasses[x]), int(x+1), mean, stdDev]))) 
    
    results = outputs[np.argsort(outputs[:, 0])]
    sortedTrain = sortMatrix(results, len(numClasses), len(results[:, 0]), len(results[0, :]))
    outputTrain(sortedTrain)
    outs = np.zeros(len(stdsAndMeans[:, 0]))
    gaussians = np.zeros(len(stdsAndMeans[:, 0]))
    testOuts = np.zeros(5)

    calStdAndMean = lambda x, z: ((1/(math.sqrt(2*np.pi*x)))*math.exp(-(math.pow(z-x, 2)/math.pow(2*stdsAndMeans[x][1], 2))))
    vCalStdAndMean = np.vectorize(calStdAndMean)
    finalAccuracy = 0.0
    #test classes, train classes, stdsAndMeans, test, numEachTrainClass
    for x in range(0, len(testClasses)):
       row = test[x, :len(test[x,:])-1]

       for w in range(0, len(row)):
           for y in range(0, len(stdsAndMeans)):
                   gaussians[y] = ((1/(math.sqrt(2*np.pi)*stdsAndMeans[y][0]))*math.exp(-1*((math.pow(row[w]-stdsAndMeans[y][1], 2)/(2*math.pow(stdsAndMeans[y][0], 2))))))
                   ind = trainClasses[y]
                   outs[y] = np.log(numEachTrainClass[ind]) + np.sum(np.log(gaussians[y])) 
      
           guess = np.argmax(outs)
           numGuesses = len([guess])
           
           print(trainClasses[guess])
           if testClasses[x] == trainClasses[guess] and numGuesses == 1:
               finalAccuracy += 1
               accuracy = 1.0
    
           elif testClasses[x] == trainClasses[guess] and numGuesses > 1:
               accuracy = 1/len(guess)
               finalAccuracy += accuracy
               
           elif (testClasses[x] != trainClasses[guess] and numGuesses == 1) or (testClasses[x] != trainClasses[guess] and numGuesses > 1):
               accuracy = 0
               
           if x == 0:
               testOuts = np.array([x+1, trainClasses[guess], numEachTrainClass[trainClasses[guess]], testClasses[x], accuracy])
           else:
               testOuts = np.vstack((testOuts, np.array([x+1, trainClasses[guess], numEachTrainClass[trainClasses[guess]], testClasses[x], accuracy])))
           
    outputTest(testOuts, finalAccuracy)
           
def main(argv):
    """
    Parameters 
    ----------
    argv : String
        List of filenames to be entered into the basian classifier

    Returns
    -------
    None.

    """

    if len(argv) < 2:
        return -1
    
    naive_bayes(argv[1], argv[2])

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)


