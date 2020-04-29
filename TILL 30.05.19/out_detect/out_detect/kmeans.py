import helper

def getAssignedCluster(dataObject, centers) :
    assignedCluster = -1
    minDistance = 100000000
    numOfClusters = len(centers)
    for j in range(0, numOfClusters) :
        distance = helper.getEuclideanDistance(dataObject, centers[j])
        if minDistance > distance :
            minDistance = distance
            assignedCluster = j
    return assignedCluster

def getAllocation(dataset, centers) :
    allocationVector = []
    n = len(dataset)
    for i in range(0, n) :
        allocationVector.append(getAssignedCluster(dataset[i], centers))
    return allocationVector

def updateCenters(dataset, allocationVector, numOfClusters) :
    centers = []
    center = []
    n = len(dataset)
    for j in range(0, len(dataset[0])) :
        center.append(0)
    for k in range(0, numOfClusters) :
        centers.append(center)
    for i in range(0, n) :
        centers[allocationVector[i]] = helper.addList(centers[allocationVector[i]], dataset[i])
    freqsOfClusters = helper.getFrequency(allocationVector, numOfClusters)
    for k in range(0, numOfClusters) :
        for j in range(0, len(dataset[0])) :
            centers[k][j] /= freqsOfClusters[k]
    return centers

def kmeans(dataset, numOfClusters) :
    centers = helper.initCentersRandomly(dataset, numOfClusters)
    allocationVector = getAllocation(dataset, centers)
    numOfIterations = 0
    maxNumOfIterations = 10000
    prevValueOfObjectiveFunc = -1
    currentValueOfObjectiveFunc = helper.getStandardDeviation(dataset, allocationVector, centers)
    while (numOfIterations < maxNumOfIterations and abs(currentValueOfObjectiveFunc - prevValueOfObjectiveFunc) > 0.05) :
        numOfIterations += 1
        centers = updateCenters(dataset, allocationVector, numOfClusters)
        allocationVector = getAllocation(dataset, centers)
        prevValueOfObjectiveFunc = currentValueOfObjectiveFunc
        currentValueOfObjectiveFunc = helper.getStandardDeviation(dataset, allocationVector, centers)
    yield centers
    yield allocationVector
