

import 3132.py

p = perceptron.Perceptron() # use a short
p.train(data)

#Perceptron test
for x in testset:
    r = p.response(x)
    if r != x[2]: # if the response is not correct
        print 'not hit.'
    if r == 1:
        plot(x[0], x[1], 'ob')
    else:
        plot(x[0], x[1], 'or')
