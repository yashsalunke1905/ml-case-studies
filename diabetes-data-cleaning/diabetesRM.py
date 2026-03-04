import pandas as pd
import numpy as np
def data_load(datapath):
    df = pd.read_csv(datapath)
    return df

def cleaning(df):
    for col in df.select_dtypes(include =np.number).columns:
        mean_value = df[col].mean()
        df[col] = df[col].replace(0,mean_value)

    return df

def main():
    data = data_load("diabetes.csv")
    print(data.head())
    cleaning(data)
    print(data)

if __name__ =="__main__":
    main()