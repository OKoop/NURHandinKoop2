import numpy as np

#This function scales a given array to a region where the logarithmic
#regression will work better. It calculates the mean and variance
#and then subtracts the mean from the array and divides by sqrt(sigma).
def Scalefeat(arr):
    n = len(arr)
    muxo = sum(arr)/n
    sigxo = sum((arr-muxo)**2.)/n
    arr = (arr - muxo)/(sigxo**(1./2.))
    return arr

#The sigmoid activation function to use for the logistic regression.
def sigmoid(x):
    return 1./(1. + np.exp(-x))

#This defines the standard cost function for lgistic regression.
def cost(labels, yhat):
    loss = -(labels * np.log(yhat) + (1. - labels) * np.log(1. - yhat))
    return sum(loss)/len(labels)

#This function returns the predicted values for each data-point, using a linear
#combination of the data-columns with the parameters theta as vector.
def ht(data, theta):
    s = theta[0]
    for i in range(data.shape[1]):
        s += theta[i + 1] * data[:,i]
    return sigmoid(s)

#A logistic regression algorithm with a first-order it takes the data, the
#known labels, a learning parameter, a target accuracy and maximal amount of
#iterations.
def logreg1storder(data,labels,alph=.1,tareps=10**-6.,maxit=100):
    #Initialize the needed arrays.
    n = len(data[:,0])
    no_of_parms = len(data[0,:]) + 1
    theta = [i for i in range(no_of_parms)]
    #Find the initial cost-function
    yhat = ht(data, theta)
    c = cost(labels, yhat)
    i = 0
    eps = 1000000000.
    costs = np.zeros(maxit)
    #For each iteration, find how we need to update the parameters using the
    #difference between the predicted values and the labels (0 or 1)
    while eps > tareps and i < maxit:
        b = np.ones((n,no_of_parms))
        b[:,1:] = data
        for j in range(no_of_parms):
            update = (yhat - labels) * b[:,j]
            theta[j] -= alph * sum(update)/n
        #Find the new predicted values and the new accuracy.
        yhat = ht(data, theta)
        cn = cost(labels, yhat)
        eps = abs(cn-c)
        c = cn
        costs[i] = c
        i += 1
    return theta, eps, i, costs
