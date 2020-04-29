from random import randrange
import helper

def initAllocation(dataset, numOfClusters) :
    n = len(dataset)
    allocationProbabilities = []
    for i in range(0, n) :
        allocationProbability = []
        sumOfAllocations = 0
        for j in range(0, numOfClusters) :
            allocationProbability.append(randrange(1, 10))
            sumOfAllocations += allocationProbability[j]
        for j in range(0, numOfClusters) :
            allocationProbability[j] /= sumOfAllocations
        allocationProbabilities.append(allocationProbability)
    return allocationProbabilities

def getCenters(dataset, allocationProbabilities, powerToMembership) :
    numOfClusters = len(allocationProbabilities[0])
    n = len(dataset)
    d = len(dataset[0])
    centers = []
    for j in range(0, numOfClusters) :
        center = []
        sumOfMultiplyingFactors = 0
        for i in range(0, d) :
            center.append(0)
        for i in range(0, n) :
            multiplyingFactor = allocationProbabilities[i][j] ** powerToMembership
            sumOfMultiplyingFactors += multiplyingFactor
            for k in range(0, d) :
                center[k] += dataset[i][k] * multiplyingFactor
        for i in range(0, d) :
            center[i] /= sumOfMultiplyingFactors
        centers.append(center)
    return centers

def getCost(dataset, allocationProbabilities, centers, powerToMembership) :
    totalCost = 0
    n = len(dataset)
    d = len(dataset[0])
    numOfClusters = len(centers)
    for i in range(0, n) :
        for j in range(0, numOfClusters) :
            distance = helper.getEuclideanDistance(dataset[i], centers[j])
            distance *= distance
            distance *= allocationProbabilities[i][j] ** powerToMembership
            totalCost += distance
    return totalCost

def updateAllocations(dataset, centers, powerToMembership) :
    allocationProbabilities = []
    n = len(dataset)
    numOfClusters = len(centers)
    for i in range(0, n) :
        distances = []
        for j in range(0, numOfClusters) :
            distances.append(helper.getEuclideanDistance(dataset[i], centers[j]))
        for j in range(0, numOfClusters) :
            distances[j] = distances[j] ** (2 * (float(1 / (powerToMembership - 1))))
        denominatorSum = 0
        for j in range(0, numOfClusters) :
            denominatorSum += (1 / distances[j])
        allocationVector = []
        allocationSum = 0
        for j in range(0, numOfClusters) :
            allocationProbability = distances[j] * denominatorSum
            allocationProbability = 1 / allocationProbability
            allocationSum += allocationProbability
            allocationVector.append(allocationProbability)
        for j in range(0, numOfClusters) :
            allocationVector[j] /= allocationSum
        allocationProbabilities.append(allocationVector)
    return allocationProbabilities

def fuzzyCmeans(dataset, numOfClusters, powerToMembership, epsilon) :
    allocationProbabilities = initAllocation(dataset, numOfClusters)
    centers = getCenters(dataset, allocationProbabilities, powerToMembership)
    costValue = getCost(dataset, allocationProbabilities, centers, powerToMembership)
    prevCostValue = -1
    maxNumOfIterations = 10000
    numOfIterations = 0
    while (abs(costValue - prevCostValue) > epsilon and numOfIterations != maxNumOfIterations) :
        numOfIterations += 1
        prevCostValue = costValue
        allocationProbabilities = updateAllocations(dataset, centers, powerToMembership)
        centers = getCenters(dataset, allocationProbabilities, powerToMembership)
        costValue = getCost(dataset, allocationProbabilities, centers, powerToMembership)
    yield centers
    yield allocationProbabilities
