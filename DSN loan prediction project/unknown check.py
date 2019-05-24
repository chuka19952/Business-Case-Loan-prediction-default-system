import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series,DataFrame 
from datetime import date
import datetime as DT
import io


#importing the datasets
df12=pd.read_csv('train.csv')
df13=pd.read_csv('test.csv')


#DATA PRE-PROCESSING
#for train set
df12['Gender'].value_counts()
df12.Gender = df12.Gender.fillna('Male')

df12['Married'].value_counts()
df12.Married = df12.Married.fillna('Yes')

df12['Dependents'].value_counts()
df12.Dependents = df12.Dependents.fillna('0')

df12['Self_Employed'].value_counts()
df12.Self_Employed = df12.Self_Employed.fillna('No')

df12['LoanAmount'].value_counts()
df12.LoanAmount = df12.LoanAmount.fillna(df12['LoanAmount'].mean())

df12['Loan_Amount_Term'].value_counts()
df12.Loan_Amount_Term = df12.Loan_Amount_Term.fillna(360)

df12['Credit_History'].value_counts()
df12.Credit_History = df12.Credit_History.fillna(1.0)

#for test set
df13['Gender'].value_counts()
df13.Gender = df13.Gender.fillna('Male')

df13['Married'].value_counts()
df13.Married = df13.Married.fillna('Yes')

df13['Dependents'].value_counts()
df13.Dependents = df13.Dependents.fillna('0')

df13['Self_Employed'].value_counts()
df13.Self_Employed = df13.Self_Employed.fillna('No')

df13['LoanAmount'].value_counts()
df13.LoanAmount = df13.LoanAmount.fillna(df12['LoanAmount'].mean())

df13['Loan_Amount_Term'].value_counts()
df13.Loan_Amount_Term = df13.Loan_Amount_Term.fillna(360)

df13['Credit_History'].value_counts()
df13.Credit_History = df13.Credit_History.fillna(1.0)






#we now encode our categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
df12["Loan_ID"] = labelencoder_X.fit_transform(df12["Loan_ID"])


labelencoder_X = LabelEncoder()
df12["Gender"] = labelencoder_X.fit_transform(df12["Gender"])

labelencoder_X = LabelEncoder()
df12["Dependents"] = labelencoder_X.fit_transform(df12["Dependents"])

labelencoder_X = LabelEncoder()
df12["Married"] = labelencoder_X.fit_transform(df12["Married"])

labelencoder_X = LabelEncoder()
df12["Education"] = labelencoder_X.fit_transform(df12["Education"])

labelencoder_X = LabelEncoder()
df12["Self_Employed"] = labelencoder_X.fit_transform(df12["Self_Employed"])

labelencoder_X = LabelEncoder()
df12["Property_Area"] = labelencoder_X.fit_transform(df12["Property_Area"])

labelencoder_X = LabelEncoder()
df12["Loan_Status"] = labelencoder_X.fit_transform(df12["Loan_Status"])





#encoding categorical variables for test.csv
labelencoder_X = LabelEncoder()
df13["Loan_ID"] = labelencoder_X.fit_transform(df13["Loan_ID"])


labelencoder_X = LabelEncoder()
df13["Gender"] = labelencoder_X.fit_transform(df13["Gender"])


labelencoder_X = LabelEncoder()
df13["Married"] = labelencoder_X.fit_transform(df13["Married"])


labelencoder_X = LabelEncoder()
df13["Education"] = labelencoder_X.fit_transform(df13["Education"])

labelencoder_X = LabelEncoder()
df13["Self_Employed"] = labelencoder_X.fit_transform(df13["Self_Employed"])

labelencoder_X = LabelEncoder()
df13["Property_Area"] = labelencoder_X.fit_transform(df13["Property_Area"])

labelencoder_X = LabelEncoder()
df13["Dependents"] = labelencoder_X.fit_transform(df13["Dependents"])



#splitting our dataset 2
X=df12.iloc[:, 1:12].values
y=df12.iloc[:, 12:].values




from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 5], X[y_kmeans == 0, 5], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 5], X[y_kmeans == 1, 5], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 5], X[y_kmeans == 2, 5], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 5], X[y_kmeans == 3, 5], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 9], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()