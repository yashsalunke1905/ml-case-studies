import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ARTIFACT = Path("winepredictKNN")
ARTIFACT.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT/"wine_knn.joblib"

def Data_path(datapath):
    df = pd.read_csv(datapath)
    return df

def Data_Stats(df):
    print("Shape of the dataset :",df.shape)
    print("Statistical imformation :",df.describe())
    print("Is ther null value:",df.isnull().any())

def split_data(df):
    x = df.drop("Class",axis = 1)
    y = df['Class']
    return x,y

def Scaler(x):
    scaler  = StandardScaler()
    X_scale = scaler.fit_transform(x)
    return X_scale

def training_split(x_scaled,y,RANDOM_STATE = 42,SPLIT_SIZE = 0.2):
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size = SPLIT_SIZE,random_state = RANDOM_STATE)
    return x_train,x_test,y_train,y_test

def max_k(x_train,x_test,y_train,y_test):
    k_range = range(1,11)
    accu = []
    for i in k_range:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        accu.append(accuracy)
    
    best_k = k_range[accu.index(max(accu))]
    return  best_k
      
def pipeing(k):
    pipe = Pipeline([
        ('knn',KNeighborsClassifier(n_neighbors=k))
    ])
    return pipe

def dumping(pipe,MODEL_PATH):
    joblib.dump(pipe,MODEL_PATH)

def main():
    data = Data_path("WinePredictor.csv")
    print(data)

    Data_Stats(data)

    X,Y = split_data(data)

    x_scaled = Scaler(X)

    X_train,X_test,Y_train,Y_test = training_split(x_scaled,Y)
    print("train shape :",X_train.shape)
    print("test shape :",X_test.shape)
    print("train shape :",Y_train.shape)
    print("test shape :",Y_test.shape)

    k_value = max_k(X_train,X_test,Y_train,Y_test)

    pipelineing = pipeing(k_value)
    pipelineing.fit(X_train,Y_train)

    dumping(pipelineing,MODEL_PATH)

if __name__ =="__main__":
    main()