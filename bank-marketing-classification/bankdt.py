import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def main():
    data = pd.read_csv("bank-full.csv")
    print(data.head())

    # Encode all categorical columns
    for col in data.select_dtypes(include=['object','category']).columns:
        data[col]= data[col].astype('str')
        data[col] = LabelEncoder().fit_transform(data[col])

    print(data.head())

    # Now split features and target
    X = data.drop(columns=['y'], axis=1)
    Y = data['y']  
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    importance = pd.Series(model.feature_importances_,index=X.columns)
    importance.sort_values(ascending=False,inplace=True)

    importance.plot(kind='bar',figsize=(10,8))
    plt.show()

if __name__ == "__main__":
    main()
