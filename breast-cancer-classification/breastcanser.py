import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ABSTRACT = Path("cancerlogreg")
ABSTRACT.mkdir(exist_ok=True)
MODEL_PATH = ABSTRACT/"cancerlogreg.joblib"

def load_data(datapath):
    df = pd.read_csv(datapath)
    return df

def stats(df):
    print("first five rows of the data:",df.head)
    print("shape of the dataset:",df.shape)
    print("stats of the data are :",df.describe)

def cleaning_data(df):
    print("columns with zero values:",df.isnull().sum())
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].replace('?',np.nan)
        df.dropna(inplace = True)
    return df

def data_spilt(df):
    X = df.drop('CancerType',axis = 1)
    Y = df['CancerType']
    return X,Y

def training_spilt(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

def pipelineg():
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('logreg',LogisticRegression())
    ])
    return pipe

def evaluation(Y_test,y_pred):
    print("accuracy of the model :",accuracy_score(Y_test,y_pred))
    print("confusion matrix")
    conf = confusion_matrix(Y_test,y_pred)
    print(conf)
    print("classifiacation report:")
    print(classification_report(Y_test,y_pred))

def visual(df):
    plt.figure(figsize=(8,8))
    sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
    plt.show()

def dumping(pipe,MODEL_PATH):
    joblib.dump(pipe,MODEL_PATH)

def main():
    data = load_data("breast-cancer-wisconsin.csv")
    stats(data)
    cleaning_data(data)

    X,Y = data_spilt(data)
    print("shape of X:",X.shape)
    print("shape of Y:",Y.shape)

    X_train,X_test,Y_train,Y_test = training_spilt(X,Y)
    print("shape of x_train:",X_train.shape)
    print("shape of x_test:",X_test.shape)
    print("shape of y_train:",Y_train.shape)
    print("shape of y_train:",Y_test.shape)

    pipeline = pipelineg()
    pipeline.fit(X_train,Y_train)
    y_pred = pipeline.predict(X_test)

    evaluation(Y_test,y_pred)

    print("visual of the model :")
    visual(data)

    dumping(pipeline,MODEL_PATH)

if __name__=="__main__":
    main()