"""
Implementation of a decision tree classifier
"""

#Andy Zhang
#Michael Liu
from numpy import *
import numpy as np
from binary import *
import util

class DT(BinaryClassifier):

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)

        self.isLeaf = True
        self.label  = 1

    def online(self):
        return False

    def __repr__(self):
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        if self.isLeaf:
            return self.label
        else:
            if X[self.feature] >= .5:
                return self.right.predict(X)
            else:
                return self.left.predict(X)

    def trainDT(self, X, Y, maxDepth, used):
        """
        recursively build the decision tree
        """

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True
            self.label  = util.mode(Y)

        else:
            # get the size of the data set
            N,D = X.shape

            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error
            bestError   = N      # the number of errors for this feature
            for d in range(D):
                # have we used this feature yet
                if d in used:
                    continue

                #put negative values on the left and positive on the right
                negInd = [i for i, x in enumerate(X) if x[d] <0.5] #indices
                negVal = [x for i, x in enumerate(X) if x[d] <0.5] #entire x values
                posInd = [i for i ,x in enumerate(X) if x[d] >=0.5] #indices
                posVal = [x for i ,x in enumerate(X) if x[d] >=0.5] #entire y values
                # negX = [X[i][d] for i in negInd]
                # posX = [X[i][d] for i in posInd]
                leftY = [Y[i] for i in negInd]
                #leftY  = util.raiseNotDefined()    ### TODO: YOUR CODE HERE
                rightY = [Y[i] for i in posInd]
                #rightY = util.raiseNotDefined()    ### TODO: YOUR CODE HERE

                #calculating guesses
                left_guess = 0
                if len(leftY) != 0:
                    if np.mean(leftY) >=0:
                        left_guess = 1
                    else:
                        left_guess = -1
                right_guess = 0
                if len(rightY) != 0:
                    if np.mean(rightY) >=0:
                        right_guess = 1
                    else:
                        right_guess = -1

                # calculating error by looking at mislabeled points
                num_errors = 0.0
                for y in leftY:
                    if y != left_guess:
                        num_errors = num_errors + 1
                for y in rightY:
                    if y != right_guess:
                        num_errors = num_errors + 1
                error = num_errors/N

                # check to see if this is a better error rate
                if error <= bestError:
                    permNeg = array(negVal)
                    permPos = array(posVal)
                    permLeft = array(leftY)
                    permRight = array(rightY)
                    permRight
                    bestFeature = d
                    bestError   = error

            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False    ### TODO: YOUR CODE HERE

                self.feature = bestFeature    ### TODO: YOUR CODE HERE


                self.left  = DT({'maxDepth': maxDepth-1})
                self.right = DT({'maxDepth': maxDepth-1})
                # recurse on our children by calling
                #   self.left.trainDT(...)
                # and
                #   self.right.trainDT(...)
                # with appropriate arguments
                used.append(bestFeature)
                self.left.trainDT(permNeg,permLeft,maxDepth-1,used)
                self.right.trainDT(permPos,permRight,maxDepth-1,used)


    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.
        """

        self.trainDT(X, Y, self.opts['maxDepth'], [])


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        return self
