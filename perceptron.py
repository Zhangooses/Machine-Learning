"""
Implementation of a perceptron classifier
"""
#Andy Zhang
#Michael Liu
from numpy import *
from binary import *
import util

class Perceptron(BinaryClassifier):

    def __init__(self, opts):
        BinaryClassifier.__init__(self, opts)
        self.opts = opts
        self.reset()

    def reset(self):
        """
        Reset the internal state of the classifier.
        """

        self.weights = 0    # our weight vector
        self.bias    = 0    # our bias
        self.numUpd  = 0    # number of updates made

    def online(self):
        """
        Our perceptron is online
        """
        return True

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)   +  ", b=" + repr(self.bias)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the margin at this point.
        Semantically, a return value <0 means class -1 and a return
        value >=0 means class +1
        """

        if self.numUpd == 0:
            return 0          # failure
        else:
            return dot(self.weights, X) + self.bias   # this is done for you!

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # check to see if we've made an error
        if Y * self.predict(X) <= 0:   ### SOLUTION-AFTER-IF
            self.numUpd  = self.numUpd  + 1

            if self.weights == 0:
                self.weights = [0] * len(X)

            # perform an update
            for i in enumerate(X):
                self.weights[i] = self.weights[i] + Y * X[i]    ### TODO: YOUR CODE HERE

            self.bias = self.bias + Y    ### TODO: YOUR CODE HERE


    def nextIteration(self):
        return   # don't need to do anything here


    def getRepresentation(self):
        """
        Return a tuple of the form (number-of-updates, weights, bias)
        """

        return (self.numUpd, self.weights, self.bias)
