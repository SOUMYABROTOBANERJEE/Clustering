import helper

def getRMatrix(centers, intraClusterScatters) :
    numOfClusters = len(centers)
    RMatrix = []
    for i in range(0, numOfClusters) :
        row = []
        for j in range(0, numOfClusters) :
            if (i == j) :
                row.append(-1)
            else :
                row.append((intraClusterScatters[i] + intraClusterScatters[j]) / helper.getEuclideanDistance(centers[i], centers[j]))
        RMatrix.append(row)
    return RMatrix

def getIntraClusterScatters(dataset, allocationVector, centers) :
    intraClusterScatters = []
    numOfClusters = len(centers)
    n = len(dataset)
    for j in range(0, numOfClusters) :
        intraClusterScatters.append(0)
    for i in range(0, n) :
        assignedCluster = allocationVector[i]
        intraClusterScatters[assignedCluster] += helper.getEuclideanDistance(dataset[i], centers[assignedCluster])
    freqOfClusters = helper.getFrequency(allocationVector, numOfClusters)
    for j in range(0, numOfClusters) :
        intraClusterScatters[j] /= freqOfClusters[j]
    return intraClusterScatters

def getDBIndex(dataset, allocationVector, centers) :
    n = len(dataset)
    numOfClusters = len(centers)
    dbValues = []
    intraClusterScatters = getIntraClusterScatters(dataset, allocationVector, centers)
    RMatrix = getRMatrix(centers, intraClusterScatters)
    sumOfDBValues = 0
    for i in range(0, numOfClusters) :
        maxRValue = -1
        for j in range(0, numOfClusters) :
            if (i != j and maxRValue < RMatrix[i][j]) :
                maxRValue = RMatrix[i][j]
        dbValues.append(maxRValue)
        sumOfDBValues += dbValues[i]
    return float(sumOfDBValues) / numOfClusters
