"""
Implementation of k-nearest-neighbor classifier
"""
#Andy Zhang
#Michael Liu
import numpy as np
from pylab import *

from binary import *


class KNN(BinaryClassifier):
    """
    This class defines a nearest neighbor classifier, that support
    _both_ K-nearest neighbors _and_ epsilon ball neighbors.
    """

    def __init__(self, opts):
        # remember the options
        self.opts = opts
        # just call reset
        self.reset()

    def reset(self):
        self.trX = zeros((0,0))    # where we will store the training examples
        self.trY = zeros((0))      # where we will store the training labels

    def online(self):
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)

    def euclidean(a,b):
        dist = np.linalg.norm(a-b)
        return dist

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the 'vote' in favor of a positive
        or negative label.  In particular, if, in our neighbor set,
        there are 5 positive training examples and 2 negative
        examples, we return 5-2=3.
        """

        isKNN = self.opts['isKNN']
        N     = self.trX.shape[0]      # number of training examples

        if self.trY.size == 0:
            return 0                   # if we haven't trained yet, return 0
        elif isKNN:
            # this is a K nearest neighbor model
            K = self.opts['K']         # how many NN to use

            val = 0                    # this is our return value: #pos - #neg of the K nearest neighbors of X
            #calculate eucildean distance for each X, store in an array,
            #make a dictionary with distance -> Y
            #sort dictionary, get first K things
            if K > len(self.trX):
                for v in self.trY:
                    val = val + v
                if val >= 0:
                    return 1
                else:
                    return -1
            distances = []
            for i in self.trX:
                #distances.append(euclidean(i,x))
                distances.append(np.linalg.norm(i-X))
            sorted = np.argsort(distances)
            for i in range(K):
                val = val + self.trY[sorted[i]]
            return val
        else:
            # this is an epsilon ball model
            eps = self.opts['eps']     # how big is our epsilon ball

            val = 0                    # this is our return value: #pos - #neg within an epsilon ball of X
            distances = []
            for i in self.trX:
                distances.append(np.linalg.norm(i-X))
            #sorted = np.argsort(distances)
            #iterate through training data
            for i in range(len(self.trX)):
                if distances[i] <= eps:
                    val = val + self.trY[i]

            return val

    def getRepresentation(self):
        """
        Return the weights
        """
        return (self.trX, self.trY)

    def train(self, X, Y):
        """
        Just store the data.
        """
        self.trX = X
        self.trY = Y
