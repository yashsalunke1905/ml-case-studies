import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import root_mean_squared_error,r2_score,mean_squared_error

def load_data(datapath):
    df = pd.read_csv(datapath)
    return df

def clean_data(df):
    df.isnull().sum()
    df['Price'] = df['Price'].str.replace('$','',regex = False)
    df['Price'] = df['Price'].str.replace(',', '', regex=False)
    df['Price'] = df['Price'].str.replace('.00', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'])
    return df

def stats(df):
    print("first five element of the dataset :",df.head())
    print("stats of the datasets :",df.describe())
    print("shape of the datasets :",df.shape)

def split(df):
    X = df.drop('Price',axis = 1)
    Y = df["Price"]
    return X,Y

def encoding(X):
    categorical_col = X.select_dtypes(include=['object', 'category']).columns
    numeric_col = X.select_dtypes(include = np.number).columns

    return categorical_col,numeric_col

def training_spilt(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test


def pipeing(preprocessor):
    pipe = Pipeline([
        ('preprocessor',preprocessor),
        ('linreg',LinearRegression())
    ])
    return pipe

def evaluation(y_test,y_pred):
    print("root mean square error is:",root_mean_squared_error(y_test,y_pred))
    print("r2 square :",r2_score(y_test,y_pred))
    print("mean square error :",mean_squared_error(y_test,y_pred))

def main():
    Data = load_data("car-sales.csv")
    Data = clean_data(Data)
    stats(Data)

    X,Y = split(Data)

    categorical_col = X.select_dtypes(include=['object', 'category']).columns
    numeric_col = X.select_dtypes(include = np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',MinMaxScaler(),numeric_col),
            ('cat',OneHotEncoder(handle_unknown='ignore',drop='first'),categorical_col)
        ],
        remainder='passthrough'
    )

    X_train,X_test,Y_train,Y_test = training_spilt(X,Y)

    print("shape of the x train",X_train.shape)
    print("shape of the x test",X_test.shape)
    print("shape of the y train",Y_train.shape)
    print("shape of the y test",Y_test.shape)

    pipeline = pipeing(preprocessor)
    pipeline.fit(X_train,Y_train)
    Y_pred = pipeline.predict(X_test)

    evaluation(Y_test,Y_pred)

if __name__=="__main__":
    main()

