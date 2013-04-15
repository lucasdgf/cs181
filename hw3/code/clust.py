# clust.py
# -------
# Lucas Freitas and Angela Li

import sys
import random
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATAFILE = "adults-small.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 3:
        return False
    if sys.argv[1] <= 0:
        return False
    if sys.argv[2] <= 0:
        return False
    return True


#-----------


def parseInput(datafile):
    """
    params datafile: a file object, as obtained from function `open`
    returns: a list of lists

    example (typical use):
    fin = open('myfile.txt')
    data = parseInput(fin)
    fin.close()
    """
    data = []
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
    return data


def printOutput(data, numExamples):
    for instance in data[:numExamples]:
       print ','.join([str(x) for x in instance])

def HAC (numClusters, numExamples, data, metric):
    """
    Performs hierarchical agglomerative clustering (HAC)

    Args:
        numClusters: number of clusters (integer)
        numExamples: number of examples to use (integer)
        data: data to cluster
        metric: distance metric used for HAC

    Returns:
        None
    Raises:
        In case of invalid metric, prints error and exits.
    """

    # clusters to be reduced with HAC
    E = [[list(x)] for x in data[:numExamples]]

    while len(E) > numClusters:
        # shortest distance between clusters
        closest_clusters = sys.maxint
        # two closest clusters, which we should merge
        A = None
        B = None

        # check all possible permutations
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                # distance between x_i and x_j
                distance = None

                # check which method we want to use
                if metric == "min":
                    distance = cmin(E[i], E[j], squareDistance)
                elif metric == "max":
                    distance = cmax(E[i], E[j], squareDistance)
                elif metric == "mean":
                    distance = cmean(E[i], E[j], squareDistance)
                elif metric == "cent":
                    distance = ccent(E[i], E[j], squareDistance)
                else:
                    print "Invalid metric system"
                    sys.exit()

                # check if distance is shorter, and if so, updae closest clusters
                if distance < closest_clusters:
                    closest_clusters = distance
                    A = i
                    B = j
        
        # merge closes clusters by adding B to A and popping B
        E[A] = E[A] + E[B]
        E.pop(B)

    for i in range(len(E)):
        print "Cluster " + str(i + 1) + ": " + str(len(E[i])) + " instances"

    if numClusters == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for cluster, color in zip(E, ['r', 'g', 'b', 'y']):
            for i in range(len(cluster)):
                ax.scatter(cluster[i][0], cluster[i][1], cluster[i][2], c=color)

        filename = "3dplot_" + str(numClusters) + "_" + str(numExamples) + "_" + metric+ ".pdf"
        plt.savefig(filename)



def autoClass (numClusters, numExamples, data):
    """
    Performs Autoclass using naive Bayes clustering

    Args:
        numClusters: number of clusters
        numExamples: number of examples
    Returns:
        None
    Raises:
        None
    """

    # dictionaries for expectations (E) and parameters (P):
    E = {}
    P = {}

    # initialize parameters to some (random) initial values
    # for each index, the subindex 'c' represents theta_c, 1
    # represents theta_i^1, and 0, theta_i^0

    for i in range(1, numClusters + 1):
        P[i] = {}
        # random.random returns a float between 0 and 1, aka a probability
        P[i]['c'] = random.random()
        for j in {0, 1}:
            P[i][j] = random.random()

    # repeat until convergence
    for _ in range (10):
        # expectation step (set all expectations to zero)
        for i in range(1, numClusters + 1):
            E[i] = {}
            E[i]['c'] = 0.
            for j in {0, 1}:
                E[i][j] = 0.
 
# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples"
        sys.exit(1);

    numClusters = int(sys.argv[1])
    numExamples = int(sys.argv[2])

    #Initialize the random seed
    
    random.seed()

    #Initialize the data
    dataset = file(DATAFILE, "r")
    if dataset == None:
        print "Unable to open data file"


    data = parseInput(dataset)
    
    
    dataset.close()
    printOutput(data,numExamples)

    HAC(numClusters, numExamples, data, "mean")

if __name__ == "__main__":
    validateInput()
    main()
