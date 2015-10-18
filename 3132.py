#!/usr/bin/env python
#coding: utf-8

import random
import numpy 
from numpy import genfromtxt


data = genfromtxt('Training_Dataset.csv', delimiter=',')
#print x
print data[1] #prints the second row
print data[1,2] #prints the third element of second row 

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = [random.random() * 2 - 1 for _ in xrange(2)] # weights
        self.learningRate = 0.1

    def response(self, x):
        """perceptron output"""
        # y = x[0] * self.w[0] + x[1] * self.w[1] # dot product between w and x
        y = sum([i * j for i, j in zip(self.w, x)]) # more pythonic
        if y >= 0:
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

#two/multi dimensional arrays 
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
            for data[i,1] in data: # for each sample's label classification
                r = self.response(x)
                #starting at row 1, the second row 
                if data[i,1] != r: # if have a wrong response
                    iterError = data[i,1] - r # desired response - actual response
                    self.updateWeights(x, iterError)
                    globalError += abs(iterError)
            iteration += 1
            if globalError == 0.0 or iteration >= 100: # stop criteria
                print 'iterations: %s' % iteration
                learned = True # stop learing




