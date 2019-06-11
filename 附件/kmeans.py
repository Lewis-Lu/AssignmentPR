from numpy import *
import matplotlib.pyplot as plt

def loadfile(filename):
    data = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        floatline = map(float,curline)
        data.append(floatline)
    return data


# Euclid distance
def distE(a, b):
    return sqrt(sum(square(a-b)))


# Manhattan Distance
def distM(a, b):
    return sum(abs(a-b))


def randomcenter(dataMat, k):
    n = int(shape(dataMat)[1])
    centers = mat(zeros((k, n)))
    for i in range(n):

        MIN = min(dataMat[:, i])
        MAX = max(dataMat[:, i])
        RANGE = float(MAX - MIN)
        centers[:, i] = MIN + RANGE * random.rand(k, 1)
    print(centers)
    return centers


def Kmeans(datamat, k, dist = distM, randcenter = randomcenter):
    m = shape(datamat)[0]
    center = randcenter(datamat,k)
    # clusters [0,1]
    #           index , minDist
    clusters = mat(zeros((m,2)))
    clusterchanged = True
    while clusterchanged:
        clusterchanged = False
        for i in range(m):
            minDist = inf
            clusterindex = -1
            for j in range(k):
                distance = dist(datamat[i,:],center[j,:])
                if(distance < minDist):
                    minDist = distance
                    clusterindex = j
            if(clusters[i,0]!=clusterindex):
                clusterchanged = True
            clusters[i, :] = clusterindex, minDist

        for i in range(k):
            index_all = clusters[:,0].A
            values = nonzero(index_all == i)
            samples = datamat[values[0]]
            center[i,:] = mean(samples,axis=0)
    return center,clusters


if __name__ == '__main__':
    data = loadfile('Data/dataset.txt')
    print(data[:, 0])
    # center, cluster = Kmeans(data, 5)
