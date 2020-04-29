from random import randrange
import math

def getCentroidOfDataset(dataset) :
    n = len(dataset)
    sumOfAllData = []
    centroidOfDataset = []
    d = -1
    if (n != 0) :
        d = len(dataset[0])
    for i in range(0, d) :
        sumOfAllData.append(0)
    for i in range(0, n) :
        sumOfAllData = addList(sumOfAllData, dataset[i])
    centroidOfDataset = sumOfAllData
    for i in range(0, d) :
        centroidOfDataset[i] /=  n
    return centroidOfDataset
    
def getAllocationVectorFromProbabilities(allocationProbabilities) :
    allocationVector = []
    n = len(allocationProbabilities)
    d = len(allocationProbabilities[0])
    for i in range(0, n) :
        maxProbability = -1
        assignedCluster = -1
        for j in range(0, d) :
            if allocationProbabilities[i][j]  > maxProbability :
                maxProbability = allocationProbabilities[i][j]
                assignedCluster = j
        allocationVector.append(assignedCluster)
    return allocationVector

def normalizeAllocationProbabilities(allocationProbabilities) :
    n = len(allocationProbabilities)
    d = len(allocationProbabilities[0])
    for i in range(0, n) :
        sumOfProbabilities = 0
        for j in range(0, d) :
            sumOfProbabilities += allocationProbabilities[i][j]
        for j in range(0, d) :
            allocationProbabilities[i][j] /=  sumOfProbabilities
    return allocationProbabilities

def combineAllocationProbabilities(cumulativeAllocationProbabilities, allocationProbabilities, mapOfCenters) :
    n = len(cumulativeAllocationProbabilities)
    d = len(cumulativeAllocationProbabilities[0])
    for i in range(0, n) :
        for j in range(0, d) :
            mappedCenter = mapOfCenters[j]
            cumulativeAllocationProbabilities[i][mappedCenter] *= allocationProbabilities[i][j]
    return cumulativeAllocationProbabilities

def mapCenters(baseCenters, dependentCenters) :
    numOfClusters = len(baseCenters)
    mapOfCenters = {}
    centersMapped = set([])
    for i in range(0, numOfClusters) :
        minDistance = 1000000000
        centerMapped = -1
        for j in range(0, numOfClusters) :
            if (j not in centersMapped) :
                distance = getEuclideanDistance(dependentCenters[i], baseCenters[j])
                if (minDistance > distance) :
                    minDistance = distance
                    centerMapped = j
        centersMapped.add(centerMapped)
        mapOfCenters[i] = centerMapped
    return mapOfCenters

def getAllocationProbabilities(dataset, centers) :
    numOfClusters = len(centers)
    allocationProbabilities = []
    n = len(dataset)
    for i in range(0, n) :
        distancesToCenters = []
        for j in range(0, numOfClusters) :
            distancesToCenters.append(getEuclideanDistance(dataset[i], centers[j]))
        probabilityVector = []
        sumOfProbabilities = 0
        for j in range(0, numOfClusters) :
            probabilityVector.append(math.exp(-1 * distancesToCenters[j]))
            sumOfProbabilities += probabilityVector[j]
        for j in range(0, numOfClusters) :
            probabilityVector[j] /= sumOfProbabilities
        allocationProbabilities.append(probabilityVector)
    return allocationProbabilities

def getStandardDeviation(dataset, allocationVector, centers) :
    variance = 0
    n = len(dataset)
    for i in range(0, n) :
        centerforAllocatedCluster = centers[allocationVector[i]]
        variance += getEuclideanDistance(dataset[i], centerforAllocatedCluster) ** 2
    variance /= n
    return variance ** 0.5

def getFrequency(allocationVector, numOfClusters) :
    frequencies = []
    for j in range(0, numOfClusters) :
        frequencies.append(0)
    n = len(allocationVector)
    for i in range(0, n) :
        frequencies[allocationVector[i]] += 1
    return frequencies

def initCentersRandomly(dataset, numOfClusters) :
    n = len(dataset)
    numOfCentersPicked = 0
    indicesPicked = set([])
    centers = []
    while (numOfCentersPicked != numOfClusters) :
        index = randrange(0, n)
        if (index not in indicesPicked) :
            numOfCentersPicked += 1
            centers.append(dataset[index])
            indicesPicked.add(index)
    return centers

def getFarthestPoint(dataset, indicesPicked, centers) :
    maxOfMinDistances = -1
    index = -1
    n = len(dataset)
    for i in range(0, n) :
        if i not in indicesPicked :
            minDistance = 1000000000
            for center in centers :
                distance = getEuclideanDistance(dataset[i], center)
                if distance < minDistance :
                    minDistance = distance
            if minDistance > maxOfMinDistances :
                index = i
                maxOfMinDistances = minDistance
    return index

def initCentersKmeansPlusPlus(dataset, numOfClusters) :
    numOfCentersPicked = 0
    indicesPicked = set([])
    n = len(dataset)
    centers = []
    while (numOfCentersPicked != numOfClusters) :
        index = getFarthestPoint(dataset, indicesPicked, centers)
        indicesPicked.add(index)
        centers.append(dataset[index])
        numOfCentersPicked += 1
    return centers

def getEuclideanDistance(vector1, vector2) :
    n = len(vector1)
    m = len(vector2)
    if (n != m) :
        return -1
    else :
        distance = 0
        for i in range(0, n) :
            distance += (vector1[i] - vector2[i]) ** 2
        return distance ** 0.5

def getManhattanDistance(vector1, vector2) :
    n = len(vector1)
    m = len(vector2)
    if (n != m) :
        return -1
    else :
        distance = 0
        for i in range(0, n) :
            distance += abs(vector1[i] - vector2[i])
        return distance

def getMinkowskiDistance(vector1, vector2, h) :
    n = len(vector1)
    m = len(vector2)
    if (n != m) :
        return -1
    else :
        distance = 0
        for i in range(0, n) :
            distance += abs(vector1[i] - vector2[i]) ** h
        return distance ** (float(1 / h))

def addList(vector1, vector2) :
    n = len(vector1)
    m = len(vector2)
    if (n != m) :
        return []
    else :
        res = []
        for i in range(0, n) :
            res.append(vector1[i] + vector2[i])
        return res
