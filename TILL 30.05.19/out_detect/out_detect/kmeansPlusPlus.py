import helper
import kmeans

def kmeansPlusPlus(dataset, numOfClusters) :
    centers = helper.initCentersKmeansPlusPlus(dataset, numOfClusters)
    allocationVector = kmeans.getAllocation(dataset, centers)
    numOfIterations = 0
    maxNumOfIterations = 10000
    prevValueOfObjectiveFunc = -1
    currentValueOfObjectiveFunc = helper.getStandardDeviation(dataset, allocationVector, centers)
    while (numOfIterations < maxNumOfIterations and abs(currentValueOfObjectiveFunc - prevValueOfObjectiveFunc) > 0.05) :
        numOfIterations += 1
        centers = kmeans.updateCenters(dataset, allocationVector, numOfClusters)
        allocationVector = kmeans.getAllocation(dataset, centers)
        prevValueOfObjectiveFunc = currentValueOfObjectiveFunc
        currentValueOfObjectiveFunc = helper.getStandardDeviation(dataset, allocationVector, centers)
    yield centers
    yield allocationVector
