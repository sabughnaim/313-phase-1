#!/usr/bin/env python
#code taken and edited from https://github.com/Honghe/perceptron

# performance measure is the accuracy and runtime 

import random
import numpy 
from numpy import genfromtxt
from matplotlib.pylab import *
x = [0] 

data = genfromtxt('Training_Dataset.csv', delimiter=',')
testset = genfromtxt('Validation_Dataset.csv', delimiter=',')

#print x for testing 
print data[1] #prints the second row
print data[1,2] #prints the third element of second row 

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = [random() * 2 - 1 for _ in xrange(2)] # weights
        self.learningRate = 0.1

    def response(self, x):
        """perceptron output"""
        #y = x[0] * self.w[0] + x[1] * self.w[1] # dot product between w and x
        y = sum([i * j for i, j in zip(self.w, x)]) # more pythonic
        if y >= 0 : 
            return 1
        else:
            return -1

    def updateWeights(self, x, iterError):
        """
        upates the wights status, w at time t+1 is
        w(t+1) = w(t) + learningRate * (d - r) * x
        iterError is (d - r)
        """
        # self.w[0] += self.learningRate * iterError * x[0]
        # self.w[1] += self.learningRate * iterError * x[1]
        self.w = \
            [i + self.learningRate * iterError * j for i, j in zip(self.w, x)]

    def train(self, data):
        """
        trains all the vector in data.
        Every vector in data must three elements.
        the third eclemnt(x[2]) must be the label(desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            globalError = 0.0
            for data[x] in data: # for each sample's label classification
                r = self.response(x)
                #starting at row 1, the second row 
                if data[x,1] != r: # if have a wrong response
                    iterError = data[x,1] - r # desired response - actual response
                    self.updateWeights(x, iterError)
                    globalError += abs(iterError)
            iteration += 1
            if globalError == 0.0 or iteration >= 100: # stop criteria
                print 'iterations: %s' % iteration
                learned = True # stop learing

p = Perceptron()
p.train(data)

#Perceptron test
i = 0
for i in testset:
    r = p.response(i)
    if r != data[i]: # if the response is not correct
        print 'not hit.'
    if r == 1:
        plot(data[0], data[1], 'ob')
    else:
        plot(data[0], data[1], 'or')

# plot of the separation line. 
# The centor of line is the coordinate origin
# So the length of line is 2
# The separation line is orthogonal to w

n = norm(p.w) # aka the length of p.w vector
ww = p.w / n # a unit vector
ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
show()
