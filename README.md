# Machine Learning Case Studies

This repository contains multiple machine learning case studies demonstrating data preprocessing, model training, evaluation, and pipeline creation using Python and Scikit-Learn.

The goal of this repository is to practice building **end-to-end machine learning workflows** on real datasets.

---

# Projects Included

## 1. Advertising Sales Prediction

Predicts product sales based on advertising budgets.

**Techniques Used**

* Data Cleaning
* Train/Test Split
* StandardScaler
* Linear Regression
* Model Evaluation (RMSE, R², MSE)

**Libraries**

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## 2. Bank Marketing Classification

Predicts whether a customer will subscribe to a bank term deposit.

**Model**
Decision Tree Classifier

**Steps**

* Encode categorical features
* Train/Test Split
* Model training
* Accuracy evaluation
* Feature importance visualization

---

## 3. Breast Cancer Classification

Predicts cancer type using logistic regression.

**Techniques**

* Data cleaning
* Feature scaling
* Logistic Regression
* Confusion Matrix
* Classification Report
* Model serialization using Joblib

---

## 4. Car Price Prediction

Predicts the price of cars based on multiple features.

**Techniques**

* Feature engineering
* OneHotEncoding
* MinMax Scaling
* Linear Regression
* Pipeline using ColumnTransformer

---

## 5. Diabetes Data Cleaning

Demonstrates preprocessing of medical data.

**Techniques**

* Replace invalid zero values with mean
* Data inspection and cleaning

---

## 6. Wine Classification

Classifies wine categories using K-Nearest Neighbors.

**Techniques**

* Feature scaling
* KNN model training
* Hyperparameter search for best K
* Model saving using Joblib

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

# Repository Structure

ml-case-studies

advertising.py
bankdt.py
breastcanser.py
carsales.py
diabetesRM.py
winepredictor.py

README.md

---

# How to Run the Projects

Install required libraries

pip install pandas numpy scikit-learn matplotlib seaborn joblib

Run any project

python advertising.py
python bankdt.py
python breastcanser.py
python carsales.py
python diabetesRM.py
python winepredictor.py

---

# Skills Demonstrated

* Data preprocessing
* Feature engineering
* Machine learning pipelines
* Model evaluation
* Data visualization
* Model persistence

---

# Future Improvements

* Model deployment using FastAPI / Flask
* Hyperparameter tuning
* Cross validation
* Docker containerization

---

# Author

Yash Salunke
Machine Learning Enthusiast
