import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import functions26 as f6

#Load in the data.
data = np.loadtxt('GRBs.txt', usecols=(2,3,4,5,6,7,8))

#Delete the rows where T_90==-1, because we can not use these.
mask = (data[:,1]!=-1)
data = data[mask]

#For each non-log column, set all -1's to 0, and Scale the features.
listofcols = [data[:,0],data[:,3],data[:,5],data[:,6]]
for l in listofcols:
    mask = (l==-1)
    l[~mask] = f6.Scalefeat(l[~mask])
    l[mask] = 0
   
#For the log-columns, take the exponential, set all previously -1's to 0
#And again scale the feature.
mask = (data[:,4]==-1)
data[:,4] = np.exp(data[:,4])
data[:,4][~mask] = f6.Scalefeat(data[:,4][~mask])
data[:,4][mask] = 0
mask = (data[:,2]==-1)
data[:,2] = np.exp(data[:,2])
data[:,2][~mask] = f6.Scalefeat(data[:,2][~mask])
data[:,2][mask] = 0

#Create the array with labels for the data-points, and select the columns to
#use in the analysis.
#Short==0, Long==1
labels = np.asarray(data[:,1]>=10,dtype='int')
data = data[:,[0,2,4,5]]

#Perform the logistic regression.
x = f6.logreg1storder(data,labels,alph=.309,tareps=10**-6,maxit=5000)
print('The best fit parameters theta are',x[0])
print('The accuracy is:', x[1])
print('The amount of steps taken:',x[2])

#Plot the predicted labels and the threshold for decision.
preds = f6.ht(data,x[0])
mask0 = (labels==0)
xs1 = np.arange(0,len(preds[mask0]),1)
xs2 = np.arange(0,len(preds[~mask0]),1)
plt.scatter(xs1, preds[mask0],s=5,label='short')
plt.scatter(xs2, preds[~mask0],s=5,label='long')
plt.plot([0,190],[.5,.5])
plt.legend()
plt.xlabel('Arbitrary index')
plt.ylabel('Predicted value')
plt.title('Predicted values from logistic regression')
plt.savefig('6')
plt.clf()

#Calculate the accuracy before and after the logistic regression.
mask = (np.asarray((preds+.49),dtype='int')==labels)
print('The initial accuracy if setting all labels to 1:',len(preds[mask])/len(labels))
print('The accuracy after logistic regression:s',sum(labels)/len(labels))