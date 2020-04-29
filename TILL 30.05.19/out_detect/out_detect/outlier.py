import kmeans

def removeOutliers(dataset, allocationProbabilities, numOfClusters, theta) :
    n = len(dataset)
    allocationVector = []
    normalData = []
    outlierIndices = set([])
    normalDataIndices = []
    for i in range(0, n) :
        clusterAssigned = -1
        maxAssignmentProbability = -1
        for j in range(0, numOfClusters) :
            if allocationProbabilities[i][j] > maxAssignmentProbability :
                maxAssignmentProbability = allocationProbabilities[i][j]
                clusterAssigned = j
        allocationVector.append(clusterAssigned)
    for i in range(0, n) :
        if (allocationProbabilities[i][allocationVector[i]] >= (float(1 / numOfClusters) + theta)) :
            normalData.append(dataset[i])
            normalDataIndices.append(i)
        else :
            outlierIndices.add(i)
    normalDataAllocationVector = []
    normalDataAllocationProbabilities = []
    for index in normalDataIndices :
        normalDataAllocationVector.append(allocationVector[index])
        normalDataAllocationProbabilities.append(allocationProbabilities[index])
    centers = kmeans.updateCenters(normalData, normalDataAllocationVector, numOfClusters)
    yield normalData
    yield normalDataAllocationVector
    yield normalDataAllocationProbabilities
    yield normalDataIndices
    yield outlierIndices
    yield centers
