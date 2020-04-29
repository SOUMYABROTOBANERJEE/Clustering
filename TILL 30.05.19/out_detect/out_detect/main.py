import loadDataset
import kmeans
import kmeansPlusPlus
import fcm
import dunn
import silhouette
import db
import ch
import helper
import constants
import outlier

def main() :
    '''dataset = loadDataset.loadWdbc()
                numOfClusters = 2'''
    '''dataset = loadDataset.loadYeast()
    numOfClusters = 7'''
    '''dataset = loadDataset.loadSatimage()
                numOfClusters = 7'''
    dataset = loadDataset.loadIonoSphere()
    numOfClusters = 2
    '''dataset = loadDataset.loadShuttle()
    numOfClusters = 7'''

    clusteringResultsFromKmeansPlusPlus = []
    for result in kmeansPlusPlus.kmeansPlusPlus(dataset, numOfClusters) :
        clusteringResultsFromKmeansPlusPlus.append(result)
    centersFromKmeansPlusPlus = clusteringResultsFromKmeansPlusPlus[0]
    allocationVectorFromKmeansPlusPlus =  clusteringResultsFromKmeansPlusPlus[1]
    allocationProbabilitiesFromKmeansPlusPlus = helper.getAllocationProbabilities(dataset, centersFromKmeansPlusPlus)

    clusteringResultsFromKmeans = []
    for result in kmeans.kmeans(dataset, numOfClusters) :
        clusteringResultsFromKmeans.append(result)
    centersFromKmeans = clusteringResultsFromKmeans[0]
    allocationVectorFromKmeans =  clusteringResultsFromKmeans[1]
    allocationProbabilitiesFromKmeans = helper.getAllocationProbabilities(dataset, centersFromKmeans)
    mapKmeansCentersToKmeansPlusPlus = helper.mapCenters(centersFromKmeansPlusPlus, centersFromKmeans)

    clusteringResultsFromFcm = []
    for result in fcm.fuzzyCmeans(dataset, numOfClusters, constants.powerToMembershipInFcm, constants.epsilon) :
        clusteringResultsFromFcm.append(result)
    centersFromFcm = clusteringResultsFromFcm[0]
    allocationProbabilitiesFromFcm =  clusteringResultsFromFcm[1]
    mapFcmCentersToKmeansPlusPlus = helper.mapCenters(centersFromKmeansPlusPlus, centersFromFcm)

    cumulativeAllocationProbabilities = allocationProbabilitiesFromKmeansPlusPlus
    cumulativeAllocationProbabilities = helper.combineAllocationProbabilities(cumulativeAllocationProbabilities, allocationProbabilitiesFromKmeans, mapKmeansCentersToKmeansPlusPlus)
    cumulativeAllocationProbabilities = helper.combineAllocationProbabilities(cumulativeAllocationProbabilities, allocationProbabilitiesFromFcm, mapFcmCentersToKmeansPlusPlus)
    cumulativeAllocationProbabilities = helper.normalizeAllocationProbabilities(cumulativeAllocationProbabilities)

    centersBeforeOutlierAnalysis = centersFromKmeansPlusPlus
    allocationVectorBeforeOutlierAnalysis = helper.getAllocationVectorFromProbabilities(cumulativeAllocationProbabilities)

    dunnIndexValueBeforeOutlierAnalysis = dunn.getDunnIndex(dataset, allocationVectorBeforeOutlierAnalysis, centersBeforeOutlierAnalysis)
    print('dunnIndexValueBeforeOutlierAnalysis', dunnIndexValueBeforeOutlierAnalysis, sep = constants.labelValueSeparator)
    silhouetteIndexValueBeforeOutlierAnalysis = silhouette.getSilhouetteIndex(dataset, centersBeforeOutlierAnalysis, allocationVectorBeforeOutlierAnalysis)
    print('silhouetteIndexValueBeforeOutlierAnalysis', silhouetteIndexValueBeforeOutlierAnalysis, sep = constants.labelValueSeparator)
    dbIndexValueBeforeOutlierAnalysis = db.getDBIndex(dataset, allocationVectorBeforeOutlierAnalysis, centersBeforeOutlierAnalysis)
    print('dbIndexValueBeforeOutlierAnalysis', dbIndexValueBeforeOutlierAnalysis, sep = constants.labelValueSeparator)
    #chIndexValueBeforeOutlierAnalysis = ch.getChIndex(dataset, centersBeforeOutlierAnalysis, allocationVectorBeforeOutlierAnalysis)
    chIndexValueBeforeOutlierAnalysis = ch.getSklearnChIndex(dataset, allocationVectorBeforeOutlierAnalysis)
    print('chIndexValueBeforeOutlierAnalysis', chIndexValueBeforeOutlierAnalysis, sep = constants.labelValueSeparator)

    allocationProbabilitiesBeforeOutlierAnalysis = cumulativeAllocationProbabilities
    clusteringResultsAfterOutlierAnalysis = []
    for result in outlier.removeOutliers(dataset, allocationProbabilitiesBeforeOutlierAnalysis, numOfClusters, constants.theta) :
        clusteringResultsAfterOutlierAnalysis.append(result)
    normalData = clusteringResultsAfterOutlierAnalysis[0]
    normalDataAllocationVector = clusteringResultsAfterOutlierAnalysis[1]
    normalDataAllocationProbabilities = clusteringResultsAfterOutlierAnalysis[2]
    normalDataIndices = clusteringResultsAfterOutlierAnalysis[3]
    outlierIndices = clusteringResultsAfterOutlierAnalysis[4]
    centersAfterOutlierAnalysis = clusteringResultsAfterOutlierAnalysis[5]

    dunnIndexValueAfterOutlierAnalysis = dunn.getDunnIndex(normalData, normalDataAllocationVector, centersAfterOutlierAnalysis)
    print('dunnIndexValueAfterOutlierAnalysis', dunnIndexValueAfterOutlierAnalysis, sep = constants.labelValueSeparator)
    silhouetteIndexValueAfterOutlierAnalysis = silhouette.getSilhouetteIndex(normalData, centersAfterOutlierAnalysis, normalDataAllocationVector)
    print('silhouetteIndexValueAfterOutlierAnalysis', silhouetteIndexValueAfterOutlierAnalysis, sep = constants.labelValueSeparator)
    dbIndexValueAfterOutlierAnalysis = db.getDBIndex(normalData, normalDataAllocationVector, centersAfterOutlierAnalysis)
    print('dbIndexValueAfterOutlierAnalysis', dbIndexValueAfterOutlierAnalysis, sep = constants.labelValueSeparator)
    #chIndexValueAfterOutlierAnalysis = ch.getChIndex(normalData, centersAfterOutlierAnalysis, normalDataAllocationVector)
    chIndexValueAfterOutlierAnalysis = ch.getSklearnChIndex(normalData, normalDataAllocationVector)
    print('chIndexValueAfterOutlierAnalysis', chIndexValueAfterOutlierAnalysis, sep = constants.labelValueSeparator)

if __name__ == "__main__" :
    main()
