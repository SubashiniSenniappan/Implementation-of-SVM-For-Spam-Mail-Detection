# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages.
Import the dataset to operate on.
Split the dataset.
Predict the required output.
End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Subashini.S
RegisterNumber:  212222240106
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
# Data Head:
![282235099-3e3f4271-a6c3-49bb-9936-730d42a3e720](https://github.com/SubashiniSenniappan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404951/3948f123-d18d-4a63-8b1f-e875c290cf2b)

# Data Info:
![282235110-193edd0f-c6f9-419a-a816-8699e98ddcc4](https://github.com/SubashiniSenniappan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404951/490fe6f2-4490-47b7-a9fc-43a24b3f231e)

# Data isnull():

![282235137-4216195e-44eb-4238-98fa-8956aca2b0bc](https://github.com/SubashiniSenniappan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404951/d1536950-ca65-4780-8f88-5d49fa5ec7ce)

# y_pred:

![282235160-a21733be-5292-4c6f-b739-8c7c9900e936](https://github.com/SubashiniSenniappan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404951/13378bde-731c-4c81-90f9-c6ae932f14c2)

# Accuracy:

![282235169-7e843899-38d5-412b-9701-fcdbbe06b68d](https://github.com/SubashiniSenniappan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404951/8489733f-b08a-4362-84e8-d902dac3cb26)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
