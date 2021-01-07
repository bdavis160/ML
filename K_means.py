"""
author:       Ben Davis
date:         2020-07-24
description:  Implements a K-Means clustering algorithm using Eucledian
              distance, mean-square-error, mean-square-separation, mean entropy,
              and accuracy to classify elements of the UCI optdigits dataset
"""
import math as mth
from math import log2
import numpy as np
from numpy.random import default_rng

def loadMatrix(csvFile):
    """
    Parameters
    ----------
    csvFile : String
        the name of the csv file to be loaded into a matrix

    Returns
    -------
    matrix : numpy array
        the 2d array containing the data from the csv file

    """
    matrix = np.genfromtxt(csvFile, dtype=int, delimiter=",")
    
    return matrix
    
def initClusters(numMeans, trainFile, maxElemVal, randCol):
    """
    

    Parameters
    ----------
    numMeans : int
        the number of means to generate
    trainFile : numpy array
        the file being used to train k-means
    maxElemVal : int
        the largest element possible in the training file

    Returns
    -------
    seeds : numpy array
        the array of seed values for the k-means clustering
    """
    
    means = np.zeros(numMeans)
    
    for x in range(0, numMeans):
        #print(trainFile[np.where(randCol == x)])
        means[x] = np.around(np.mean(trainFile[np.where(randCol == x)]), 3)
        
    return means

def genGrayscaleImg(clusters, dimension, maxVal, clusterNames):
    """
    

    Parameters
    ----------
   clusters : numpy array of 2d numpy arrays
        the clusters being used to generate the images
    dimension : int
        the dimensions of the images to produce for each cluster (square)
    maxVal : int
        the largest value found in any given cluster index

    Returns
    -------
    gscales : numpy array of 2d numpy arrays
        an array of 8x8 matricies representing each of the matricies in clusters
    """
    for y in range(0, len(clusters)):
        outFile = open("clusterCenter" + str(clusterNames[y]) + ", " + str(y) + ".pgm", "w")
        outFile.write("P2\n")
        outFile.write(str(dimension) + " " + str(dimension) + "\n")
        outFile.write(str(maxVal) + "\n")
        for x in range(0, len(clusters[y])):
            outFile.write(str(clusters[y][x]) + " ")
            if x != 0 and x%8 == 0 and x!= len(clusters[y]):
                outFile.write('\n')
        outFile.close()
    
def kMeans(k, csvFile, maxElem):
    """

    Parameters
    ----------
    k : int
        the number of clusters
        
    csvFile : string
        the name of the csv file to load
    
    maxElem : int
        the largest possible value in the input file
        
    Returns
    -------
    output : numpy array of 2d numpy arrays
        the csvFile clustered around centers

    """
    inFile = loadMatrix(csvFile)
    randCol = np.random.randint(0, k+1, size=len(inFile[:, 0]))
    means = initClusters(k, inFile, maxElem, randCol)
    means = np.around(means, 2)
    newMeans = np.zeros(k)
    distances = np.zeros((len(inFile[:, 0])))
    clusterNdx = np.zeros(len(distances))
    mins = np.zeros(len(distances))
    mse = np.zeros(k)
    mss = 0 
    entropies = [None] * 2
    meanEntropies = [None] * k
    targets = inFile[:, len(inFile[0, :])-1]
    targetFractions = np.zeros(k)
    clusterClass = np.zeros(k)

    while True:
        #claculate euclidean distance
        for y in range(0, k):
            for z in range(0, len(inFile[:, 0])):
                mi = np.subtract(inFile[z], np.full(len(inFile[z]), means[y]))
                sqr = np.power(mi, np.full(len(mi), 2))
                som = np.sum(sqr)
                sqrt = np.sqrt(som)
                distances[z] = sqrt
                
            if y == 0:
                outputs = distances
            else:
                outputs = np.vstack((outputs, distances))
        
        #assign objects to clusters        
        for y in range(0, len(distances)):
                col = outputs[:,y]
                rowMin = np.amin(outputs[:, y])
                ordMin = np.where(col == col.min(axis=0))
                
                if len(ordMin[0]) > 1:
                    randChoice = np.random.choice(ordMin[0], 1, replace=False)
                    mins[y] = col[randChoice]
                    clusterNdx[y] = randChoice
                else:
                    mins[y] = rowMin
                    clusterNdx[y] = ordMin[0]
        
        
        #recalculate means
        for x in range(0, k):
            ndxsForCluster = np.where(clusterNdx == x)
            cluster = inFile[ndxsForCluster, :]
            if len(cluster[0]) < 1:
                randRow = np.random.choice(means, 1, replace=False)
                mse[x] = mth.pow(randRow, 2)
                newMeans[x] = randRow 
               
            else:
                mse[x] = np.sum(np.power(mins[np.where(clusterNdx == x)], np.full(len(mins[np.where(clusterNdx == x)]), 2)))/len(cluster[0])
                newMeans[x] = np.mean(cluster)
                entropies = np.unique(cluster, return_counts=True)
                entropies = np.divide(entropies[0]+1, np.full(len(entropies[1]), len(cluster[0])))
                meanEntropies[x] = np.log2(entropies)
                meanEntropies[x] = np.multiply(meanEntropies[x], entropies)
                meanEntropies[x] = np.sum(meanEntropies[x])
                clusterClass[x] = np.argmax(np.bincount(targets[ndxsForCluster[0]]))
                
        newMeans = np.around(newMeans, 2)
        if np.array_equal(newMeans, means):
            break
        
        else:
            means = np.copy(newMeans)
    
    diff = 0
    for x in range(0, k):
        y = y+diff
        for y in range(0, k):
            if x != y:
                mi = means[x] - means[y]
                sqr = mth.pow(mi, 2)
                mss = mss + sqr
        diff += 1
    
    for y in range(0, k):
        numY = np.where(targets == y)
        ratio = len(numY)/len(targets)
        targetFractions[y] = ratio
    
        
    avgMse = np.sum(mse)/k
    avgMss = mss/((k*(k-1))/2) 
    meanEntropies = (-1) * np.dot(meanEntropies, targetFractions)
    return np.array([newMeans, [avgMse], [avgMss], [meanEntropies], clusterClass])

def testKmeans(clusters, testFile, maxElem, classes):
    """

    Parameters
    ----------
    clusters : numpy array
        the list of cluster centers after training
    testFile : string
         the name of the test file to load into a matrix
    maxElem : int
        the largest element in any object of the test file

    Returns
    -------
    None.

    """
    """
    associate all cluster objs with cluster center
    compute accuracy/create confusion matrix
    generate 8x8 grayscale for each cluster
    """
    inFile = loadMatrix(testFile)
    distances = np.zeros((len(inFile[:, 0])))
    clusterNdx = np.zeros(len(distances))
    mins = np.zeros(len(distances))
    targets = inFile[:, len(inFile[0, :])-1]
    confuse = np.zeros(10,10)
    testClusters = [None] * len(clusters)
    finalAccuracy = 0
    clusterCenters = [None] * len(clusters)
    
    for y in range(0, len(clusters)):
            for z in range(0, len(inFile[:, 0])):
                mi = np.subtract(inFile[z], np.full(len(inFile[z]), clusters[y]))
                sqr = np.power(mi, np.full(len(mi), 2))
                som = np.sum(sqr)
                sqrt = np.sqrt(som)
                distances[z] = sqrt
                
            if y == 0:
                outputs = distances
            else:
                outputs = np.vstack((outputs, distances))
    
    for y in range(0, len(distances)):
                col = outputs[:,y]
                rowMin = np.amin(outputs[:, y])
                ordMin = np.where(col == col.min(axis=0))
                
                if len(ordMin[0]) > 1:
                    randChoice = np.random.choice(ordMin[0], 1, replace=False)
                    mins[y] = col[randChoice]
                    clusterNdx[y] = randChoice
                else:
                    mins[y] = rowMin
                    clusterNdx[y] = ordMin[0]
                    
    for x in range(0, len(clusters)):
            ndxsForCluster = np.where(clusterNdx == x)
            targetNdxs = targets[ndxsForCluster[0]]
            cluster = inFile[ndxsForCluster[0], :]          
            testClusters[x] = cluster
            ndx = mth.floor(len(cluster)/2)
            clusterCenters[x] = cluster[ndx]
            for y in range(0, len(targetNdxs)):
                if len(targetNdxs) > 0:
                    if classes[x] == targetNdxs[y]:
                        finalAccuracy += 1
                    first = mth.floor(classes[x])
                    second = targetNdxs[y]
                    confuse[first][second] += 1
    
    
    genGrayscaleImg(clusterCenters, 8, 16, classes)
    fname = "confuse_k_means.txt"
    np.savetxt(fname, confuse, fmt='%d')
    return finalAccuracy/len(targets)
    
def main():
    """
    ad = 0
    ae = 0
    af = 0
    ag = 0
    ah = 0
    """
    
    k = 10 #change the value of k to get output for each experiment
    
    d = kMeans(k, "optdigits.train", 16)
   
    e = kMeans(k, "optdigits.train", 16)
    
    f = kMeans(k, "optdigits.train", 16)
     
    g = kMeans(k, "optdigits.train", 16)
      
    h = kMeans(k, "optdigits.train", 16)
    
    mse = np.array([d[1], e[1], f[1], g[1], h[1]])
    minMse = np.argmin(mse)
    
    if minMse == 0:
        print("Mean-Square-Error: " + str(d[1]))
        print("Mean-Square-Separation: " + str(d[2]))
        print("Mean Entropy: " + str(d[3]))
        z = testKmeans(d[0], "optdigits.test", 16, d[4])
        print("Final Accuracy: " + str(z))
        
    if minMse == 1:
        print("Mean-Square-Error: " + str(e[1]))
        print("Mean-Square-Separation: " + str(e[2]))
        print("Mean Entropy: " + str(e[3]))
        z = testKmeans(e[0], "optdigits.test", 16, e[4])
        print("Final Accuracy: " + str(z))
        
    if minMse == 2:
        print("Mean-Square-Error: " + str(f[1]))
        print("Mean-Square-Separation: " + str(f[2]))
        print("Mean Entropy: " + str(f[3]))
        z = testKmeans(f[0], "optdigits.test", 16, f[4])
        print("Final Accuracy: " + str(z))
        
    if minMse == 3:
        print("Mean-Square-Error: " + str(g[1]))
        print("Mean-Square-Separation: " + str(g[2]))
        print("Mean Entropy: " + str(g[3]))
        z = testKmeans(g[0], "optdigits.test", 16, g[4])
        print("Final Accuracy: " + str(z))
        
    if minMse == 4:
        print("Mean-Square-Error: " + str(h[1]))
        print("Mean-Square-Separation: " + str(h[2]))
        print("Mean Entropy: " + str(h[3]))
        z = testKmeans(h[0], "optdigits.test", 16, h[4])
        print("Final Accuracy: " + str(z))
    

if __name__ == "__main__":
    # execute only if run as a script
    main()