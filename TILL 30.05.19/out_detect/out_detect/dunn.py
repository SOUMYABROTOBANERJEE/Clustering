import helper

def getMinimumInterclusterDistance(centers) :
	minimumInterclusterDistance = 10000000
	numOfClusters = len(centers)
	for i in range(0, numOfClusters - 1) :
		for j in range(i + 1, numOfClusters) :
			interClusterDistance = helper.getEuclideanDistance(centers[i], centers[j])
			if interClusterDistance < minimumInterclusterDistance :
				minimumInterclusterDistance = interClusterDistance
	return minimumInterclusterDistance

def getMaximumIntraClusterDistance(dataset, centers, allocationVector) :
	maximumIntraClusterDistance = -1
	numOfClusters = len(centers)
	n = len(dataset)
	freqOfClusters = helper.getFrequency(allocationVector, numOfClusters)
	intraClusterDistances = []
	for k in range(0, numOfClusters) :
		intraClusterDistances.append(0)
	for i in range(0, n) :
		clusterAssigned = allocationVector[i]
		distanceToCenter = helper.getEuclideanDistance(dataset[i], centers[clusterAssigned])
		intraClusterDistances[clusterAssigned] += (distanceToCenter / freqOfClusters[clusterAssigned])
	for k in range(0, numOfClusters) :
		if intraClusterDistances[k] > maximumIntraClusterDistance :
			maximumIntraClusterDistance = intraClusterDistances[k]
	return maximumIntraClusterDistance

def getDunnIndex(dataset, allocationVector, centers) :
	minimumInterclusterDistance = getMinimumInterclusterDistance(centers)
	maximumIntraClusterDistance = getMaximumIntraClusterDistance(dataset, centers, allocationVector)
	return float(minimumInterclusterDistance / maximumIntraClusterDistance)
