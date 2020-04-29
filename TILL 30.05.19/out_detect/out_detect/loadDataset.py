import random
import helper
from sys import platform

def loadWdbc() :
    if platform == "linux" or platform == "linux2" :
        path = '/home/akash/Desktop/Code/outlier/datasets/breast-cancer-wisconsin.data'
    elif platform == "win32" :
        path = 'datasets/breast-cancer-wisconsin.data'
    datafile = open(path, 'r')
    dataset = []
    for line in datafile :
        dataTuple = line.split(',')
        d = len(dataTuple)
        dataset.append(dataTuple[1 : d - 1])
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] != '?') :
                dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] != '?' and minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (dataset[i][j] != '?' and maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] == '?') :
                dataset[i][j] = (minValuesOfAttributes[j] + maxValuesOfAttributes[j]) / (2 * maxValuesOfAttributes[j])
            else :
                dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    return dataset

def loadWdbcForCblof() :
    if platform == "linux" or platform == "linux2" :
        path = '/home/akash/Desktop/Code/outlier/datasets/breast-cancer-wisconsin.data'
    elif platform == "win32" :
        path = 'datasets/breast-cancer-wisconsin.data'
    datafile = open(path, 'r')
    dataset = []
    labels = []
    numOfClasses = 2
    for line in datafile :
        dataTuple = line.split(',')
        d = len(dataTuple)
        dataset.append(dataTuple[1 : d - 1])
        label = dataTuple[d-1]
        label = int(label)
        label = label / 2 - 1
        labels.append(int(label))
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] != '?') :
                dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] != '?' and minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (dataset[i][j] != '?' and maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            if (dataset[i][j] == '?') :
                dataset[i][j] = (minValuesOfAttributes[j] + maxValuesOfAttributes[j]) / (2 * maxValuesOfAttributes[j])
            else :
                dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    labelFreq = helper.getFrequency(labels, numOfClasses)
    print(labelFreq)
    print(labels)
    prunedDataset = []
    prunedLabels = []
    pruneCountForClasses = [14, 202]
    for i in range(0, n) :
        if pruneCountForClasses[labels[i]] > 0 :
            pruneCountForClasses[labels[i]] -= 1
        else :
            prunedDataset.append(dataset[i])
            prunedLabels.append(labels[i])
    prunedLabelFreq = helper.getFrequency(prunedLabels, numOfClasses)
    print(prunedLabelFreq)
    print(prunedLabels)
    return prunedDataset

def loadYeast() :
    if platform == "linux" or platform == "linux2" :
        path = '/home/akash/Desktop/Code/outlier/datasets/yeast.data'
    elif platform == "win32" :
        path = 'datasets/yeast.data'
    datafile = open(path, 'r')
    dataset = []
    for line in datafile :
        dataTuple = line.split()
        d = len(dataTuple)
        dataset.append(dataTuple[1 : d - 1])
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    return dataset

def loadSatimage() :
    if platform == "linux" or platform == "linux2" :
        path = '/home/akash/Desktop/Code/outlier/datasets/satimage.trn'
    elif platform == "win32" :
        path = 'datasets/satimage.trn'
    datafile = open(path, 'r')
    dataset = []
    for line in datafile :
        dataTuple = line.split()
        d = len(dataTuple)
        dataset.append(dataTuple[: d - 1])
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    return dataset

def loadIonoSphere() :
    if platform == "linux" or platform == "linux2" :
        path = '/home/akash/Desktop/Code/outlier/datasets/ionosphere.data'
    elif platform == "win32" :
        path = 'datasets/ionosphere.data'
    datafile = open(path, 'r')
    dataset = []
    for line in datafile :
        dataTuple = line.split(',')
        d = len(dataTuple)
        dataset.append(dataTuple[: d - 1])
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] == maxValuesOfAttributes[j]) :
                dataset[i][j] = random.random()
            else :
                dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    return dataset

def loadShuttle() :
    if platform == "linux" or platform == "linux2" :
        path = 'home/akash/Desktop/Code/outlier/datasets/shuttle.tst'
    elif platform == "win32" :
        path = "datasets/shuttle.tst"
    datafile = open(path, 'r')
    dataset = []
    for line in datafile :
        dataTuple = line.split()
        d = len(dataTuple)
        dataset.append(dataTuple[1 : d - 1])
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] == maxValuesOfAttributes[j]) :
                dataset[i][j] = random.random()
            else :
                dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    return dataset

def loadLymphography() :
    if platform == "linux" or platform == "linux2" :
        path = 'home/akash/Desktop/Code/outlier/datasets/lymphography.data'
    elif platform == "win32" :
        path = "datasets/lymphography.data"
    datafile = open(path, 'r')
    dataset = []
    labels = []
    for line in datafile :
        dataTuple = line.split(',')
        d = len(dataTuple)
        dataset.append(dataTuple[1 : d])
        label = int(dataTuple[0]) - 1
        labels.append(label)
    n = len(dataset)
    d = len(dataset[0])
    for i in range(0, n) :
        for j in range(0, d) :
            dataset[i][j] = float(dataset[i][j])
    minValuesOfAttributes = []
    maxValuesOfAttributes = []
    for i in range(0, d) :
        minValuesOfAttributes.append(1000000000)
        maxValuesOfAttributes.append(-1)
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] > dataset[i][j]) :
                minValuesOfAttributes[j] = dataset[i][j]
            if (maxValuesOfAttributes[j] < dataset[i][j]) :
                maxValuesOfAttributes[j] = dataset[i][j]
    for i in range(0, n) :
        for j in range(0, d) :
            if (minValuesOfAttributes[j] == maxValuesOfAttributes[j]) :
                dataset[i][j] = random.random()
            else :
                dataset[i][j] = (dataset[i][j] - minValuesOfAttributes[j]) / (maxValuesOfAttributes[j] - minValuesOfAttributes[j])
    numOfClasses = 4
    labelFreq = helper.getFrequency(labels, numOfClasses)
    print(labelFreq)
    return dataset

#loadWdbcForCblof()
#loadLymphography()