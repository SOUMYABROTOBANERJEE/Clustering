import helper
from sklearn import metrics

def getTotalSumOfSquares(dataset) :
    centroidOfDataset = helper.getCentroidOfDataset(dataset)
    totalSumOfSquares = 0
    n = len(dataset)
    for i in range(0, n) :
        totalSumOfSquares += helper.getEuclideanDistance(dataset[i], centroidOfDataset) ** 2
    return totalSumOfSquares

def getTotalIntraClusterVariance(dataset, centers, allocationVector) :
    totalIntraClusterVariance = 0
    n = len(dataset)
    for i in range(0, n) :
        assignedCenter = centers[allocationVector[i]]
        totalIntraClusterVariance += helper.getEuclideanDistance(dataset[i], assignedCenter) ** 2
    return totalIntraClusterVariance

''' Calinski-Harabasz Index
 Higher the value, better is the clustering quality
 refer to: http://ethen8181.github.io/machine-learning/clustering_old/clustering/clustering.html'''
 
def getChIndex(dataset, centers, allocationVector) :
    n = len(dataset)
    numOfClusters = len(centers)
    totalIntraClusterVariance = getTotalIntraClusterVariance(dataset, centers, allocationVector)
    totalSumOfSquares = getTotalSumOfSquares(dataset)
    varianceOfClusterCentroidsFromDatasetCentroid =  totalSumOfSquares - totalIntraClusterVariance
    chIndexValue = varianceOfClusterCentroidsFromDatasetCentroid / totalIntraClusterVariance
    chIndexValue *= (n - numOfClusters)
    chIndexValue /= (numOfClusters - 1)
    return chIndexValue

def getSklearnChIndex(dataset, allocationVector) :
    return metrics.calinski_harabaz_score(dataset, allocationVector)

