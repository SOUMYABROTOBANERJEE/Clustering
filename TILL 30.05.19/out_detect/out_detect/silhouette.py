import helper

def getSilhouetteIndexForSinglePoint(index, dataset, allocationVector, freqOfClusters) :
	averageDistanceToClusters = []
	numOfClusters = len(freqOfClusters)
	n = len(dataset)
	for k in range(0, numOfClusters) :
		averageDistanceToClusters.append(0)
	for i in range(0, n) :
		distance = helper.getEuclideanDistance(dataset[index], dataset[i])
		averageDistanceToClusters[allocationVector[i]] += distance
	for k in range(0, numOfClusters) :
		averageDistanceToClusters[k] /= freqOfClusters[k]
	dissimilarityForOwnCluster = averageDistanceToClusters[allocationVector[index]]
	minimumDissimilarityForOtherClusters = 1000000
	for k in range(0, numOfClusters) :
		if k != allocationVector[index] :
			if averageDistanceToClusters[k] < minimumDissimilarityForOtherClusters :
				minimumDissimilarityForOtherClusters = averageDistanceToClusters[k]
	maxBetweenDissimilarities = dissimilarityForOwnCluster if dissimilarityForOwnCluster > minimumDissimilarityForOtherClusters else minimumDissimilarityForOtherClusters
	return (minimumDissimilarityForOtherClusters - dissimilarityForOwnCluster) / maxBetweenDissimilarities

def getSilhouetteIndex(dataset, centers, allocationVector) :
	numOfClusters = len(centers)
	freqOfClusters = helper.getFrequency(allocationVector, numOfClusters)
	n = len(dataset)
	silhoutteIndexValues = []
	for i in range(0, n) :
		silhoutteIndexValue = getSilhouetteIndexForSinglePoint(i, dataset, allocationVector, freqOfClusters)
		silhoutteIndexValues.append(silhoutteIndexValue)
	averageSilhoutteIndexValue = 0
	for i in range(0, n) :
		averageSilhoutteIndexValue += silhoutteIndexValues[i]
	averageSilhoutteIndexValue /= n
	return averageSilhoutteIndexValue
