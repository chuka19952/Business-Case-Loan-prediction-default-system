from sklearn import metrics
import scipy as sp
import numpy as np
import math
import pandas as pd


def perturbation(model,X,y,regression):
    errors=[]
    
    for i in range (X.shape[1]):
        hold=np.array(X[:,i])
        np.random.shuffle(X[:,i])
        
        if regression :
            pred= model.predict(X)
            error=metrics.mean_squared_error(y,pred)
        else:
            pred= model.predict_proba(X)
            error=metrics.log_loss(y,pred)
        
        errors.append(error)
        X[:, i]=hold
    
    max.error=np.max(errors)
    importance= [e/max_error for e in errors]
    
    data ={'name':names, 'error':errors, 'importance':importance}
    result=pd.DataFrame(data,columns=['name','error','importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result

#To display your result
from IPython.display import display, HTML
names = list(df.columns)  #X and y column names
names.remove("target var") #remove the target var (y)
rank= perturbation_rank(model,X_test, y_test,names, False)
display(rank)



# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])



# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_





# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)