import csv
import pandas as pf
import scipy
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import csv
import pandas as pf
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix

df=pf.read_csv("/home/cb/HRanalytics.csv")
df = df.sample(frac=1).reset_index(drop=True)
TempTrainingDataFrame=df[:7500]
TempTestingDataFrame=df[7500:]

NNoutput=TempTrainingDataFrame['left'] # training output
testout=TempTestingDataFrame['left']

dicSales={'technical':1,'support':2,'IT':3,'product_mng':4,'marketing':5,
         'RandD':6,'accounting':7,'hr':8,'management':9,'sales':10}
dicSalary={'high':1,'medium':2,'low':3}

temp=TempTrainingDataFrame
temp1=TempTestingDataFrame

# replace values with sales and salary help of the dictionary

for i in range(7500):
    temp.iat[i,8]=dicSales[TempTrainingDataFrame.iat[i,8]]
    temp.iat[i,9]=dicSalary[TempTrainingDataFrame.iat[i,9]]
temp.drop('left',axis=1,inplace=True)

for i in range(7499):
    temp1.iat[i,8]=dicSales[TempTestingDataFrame.iat[i,8]]
    temp1.iat[i,9]=dicSalary[TempTestingDataFrame.iat[i,9]]
temp1.drop('left',axis=1,inplace=True)


X=np.array(temp)        # input for model or classifier
Y=np.array(NNoutput)    # output for model or classifer
Y=np.reshape(Y,(NNoutput.size,1))

testX=np.array(temp1)   # test input
testY=np.array(testout) # test output
testY=np.reshape(testY,(testout.size,1))

#Decision tree model is used for classification
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)

print("Accuracy_of_the_model--> "+ str(accuracy_score(testY,clf.predict(testX))))

print(confusion_matrix(testY,clf.predict(testX)))
