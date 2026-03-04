import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
def dataload(datapath):
    df = pd.read_csv(datapath)
    return df

def cleaning(df):
    print(df.head)
    print("checking null values",df.isnull().sum())
    df.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
    print("data clean")
    return df
    
def stats(df):
    print("stats of the data")
    print("shape of the data",df.shape)

#def polt_fig(df.corr()):
 #   plt.figure(figsize=(8,8))
  #  sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
   # plt.show()

def spilt(df):
    X = df.drop('sales',axis= 1)
    Y = df['sales']
    return X,Y 

def trainig_split(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return  x_train,x_test,y_train,y_test

def pipeling():
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('linreg',LinearRegression())
    ])
    return pipe

def main():
    data = dataload("Advertising.csv")
    data = cleaning(data)
    stats(data)
    X,Y = spilt(data)
    X_train,X_test,Y_train,Y_test = trainig_split(X,Y)
    pipeing = pipeling()
    pipeing.fit(X_train,Y_train)
    y_pred = pipeing.predict(X_test)
    score = root_mean_squared_error(Y_test,y_pred)

    print("score of the model",score)
    print("r2 scor",r2_score(Y_test,y_pred))
    print("mean square error",mean_squared_error(Y_test,y_pred)) 

if __name__ =="__main__":
    main()