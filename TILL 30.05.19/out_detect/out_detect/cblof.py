import kmeans
import kmeansPlusPlus
import helper
import loadDataset

def getLargeClusters(freqOfClusters, alpha, n) :
	numOfClusters = len(freqOfClusters)
	largeClusters = set([])
	for i in range(0, numOfClusters) :
		if (freqOfClusters[i] >= alpha * n) :
			largeClusters.add(i)
	return largeClusters

def getSmallClusters(freqOfClusters, alpha, n) :
	numOfClusters = len(freqOfClusters)
	smallClusters = set([])
	for i in range(0, numOfClusters) :
		if (freqOfClusters[i] < alpha * n) :
			smallClusters.add(i)
	return smallClusters

def getLargeAndSmallClusters(freqOfClusters, n, alpha, beta) :
	numOfClusters = len(freqOfClusters)
	clusterEnums = range(0, numOfClusters)
	freqOfClusters, clusterEnums = (list(t) for t in zip(*sorted(zip(freqOfClusters, clusterEnums))))
	freqOfClusters.reverse()
	clusterEnums.reverse()
	boundaryFreq = int(n * alpha)
	largeClusters = set([])
	smallClusters = set([])
	cumulativeFreq = 0
	boundaryReached = False
	for i in range(0, numOfClusters) :
		cumulativeFreq += freqOfClusters[i]
		if boundaryReached == False :
			largeClusters.add(clusterEnums[i])
		elif boundaryReached == True :
			smallClusters.add(clusterEnums[i])
		if cumulativeFreq >= boundaryFreq and (i < numOfClusters - 1) and (float(freqOfClusters[i]) / freqOfClusters[i+1] >= beta) :
			boundaryReached = True
	return [largeClusters, smallClusters]

def getNearestLargeCluster(datum, centers, largeClusters) :
	minDistance = 10000000
	minDistantLargeCluster = -1
	for cluster in largeClusters :
		distance = helper.getEuclideanDistance(datum, centers[cluster])
		if distance < minDistance :
			minDistance = distance
			minDistantLargeCluster = cluster
	return minDistance

def getcblofs(dataset, centers, largeClusters, smallClusters, allocationVector, freqOfClusters) :
	cblofs = []
	n = len(dataset)
	for i in range(0, n) :
		assignedCluster = allocationVector[i]
		cblof = -1
		if assignedCluster in largeClusters :
			cblof = freqOfClusters[assignedCluster] * (1.0 / helper.getEuclideanDistance(dataset[i], centers[assignedCluster]))
		else :
			nearestLargeClusterDistance = getNearestLargeCluster(dataset[i], centers, largeClusters)
			cblof = freqOfClusters[assignedCluster] * (1.0 / nearestLargeClusterDistance)
		cblofs.append(cblof)
	return cblofs


def cblof(dataset, numOfClusters, alpha, beta) :
	'''clusteringResultFromKmeans = []
				for result in kmeans.kmeans(dataset, numOfClusters) :
					clusteringResultFromKmeans.append(result)
				centersFromKmeans = clusteringResultFromKmeans[0]
				allocationVectorFromKmeans = clusteringResultFromKmeans[1]
				freqOfClusters = helper.getFrequency(allocationVectorFromKmeans, numOfClusters)
				print(freqOfClusters)
				n = len(dataset)
				[largeClusters, smallClusters] = getLargeAndSmallClusters(freqOfClusters, n, alpha, beta)
				print([largeClusters, smallClusters])
				#largeClusters = getLargeClusters(freqOfClusters, alpha, n)
				#smallClusters = getSmallClusters(freqOfClusters, alpha, n)
				cblofsOfAllDatum = getcblofs(dataset, centersFromKmeans, largeClusters, smallClusters, allocationVectorFromKmeans, freqOfClusters)
				return cblofsOfAllDatum'''
	clusteringResultFromKmeansPlusPlus = []
	for result in kmeansPlusPlus.kmeansPlusPlus(dataset, numOfClusters) :
		clusteringResultFromKmeansPlusPlus.append(result)
	centersFromKmeansPlusPlus = clusteringResultFromKmeansPlusPlus[0]
	allocationVectorFromKmeansPlusPlus = clusteringResultFromKmeansPlusPlus[1]
	freqOfClusters = helper.getFrequency(allocationVectorFromKmeansPlusPlus, numOfClusters)
	print(freqOfClusters)
	n = len(dataset)
	[largeClusters, smallClusters] = getLargeAndSmallClusters(freqOfClusters, n, alpha, beta)
	print([largeClusters, smallClusters])
	#largeClusters = getLargeClusters(freqOfClusters, alpha, n)
	#smallClusters = getSmallClusters(freqOfClusters, alpha, n)
	cblofsOfAllDatum = getcblofs(dataset, centersFromKmeansPlusPlus, largeClusters, smallClusters, allocationVectorFromKmeansPlusPlus, freqOfClusters)
	return [cblofsOfAllDatum, allocationVectorFromKmeansPlusPlus]

def getCountOfCblofsBelowThresold(cblofs, thresold) :
	countOfOutliers = 0
	n = len(cblofs)
	for i in range(0, n) :
		if cblofs[i] < thresold :
			countOfOutliers += 1
	return countOfOutliers

def getFreqInTopKOutliers(cblofs, allocationVector, numOfClusters, topKPercent) :
    cblofs, allocationVector = (list(t) for t in zip(*sorted(zip(cblofs, allocationVector))))	
    n = len(allocationVector)
    countOfTopKOutliers = int(n * topKPercent)
    prunedAllocationVector = allocationVector[: countOfTopKOutliers]
    freqOfClusters = helper.getFrequency(prunedAllocationVector, numOfClusters)
    return freqOfClusters

def main() :
	dataset = loadDataset.loadWdbcForCblof()
	numOfClusters = 2
	'''dataset = loadDataset.loadLymphography()
	numOfClusters = 4'''
	alpha = 0.9
	beta = 5
	[cblofs, allocationVector] = cblof(dataset, numOfClusters, alpha, beta)
	print(cblofs)
	#print(getCountOfCblofsBelowThresold(cblofs, 100))
	topKPercent = 0.1
	freqInTopKOutliers = getFreqInTopKOutliers(cblofs, allocationVector, numOfClusters, topKPercent)
	print(freqInTopKOutliers)

if __name__ == "__main__" :
	main()