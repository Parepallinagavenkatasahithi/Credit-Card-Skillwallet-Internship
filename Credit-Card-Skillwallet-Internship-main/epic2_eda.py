import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import os

def perform_eda():
    app = pd.read_csv('application_record.csv')
    credit = pd.read_csv('credit_record.csv')
    
    print(app.head())
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Univariate Analysis
    print("Number of people working status :")
    print(app['OCCUPATION_TYPE'].value_counts())
    
    sns.set(rc={'figure.figsize':(18,6)})
    sns.countplot(x='OCCUPATION_TYPE', data=app, palette='Set2')
    plt.title('Distribution of Occupation Types')
    plt.xticks(rotation=45)
    plt.savefig('plots/univariate_occupation.png')
    plt.close()
    
    # Multivariate Analysis
    fig, ax = plt.subplots(figsize=(8,6))
    numeric_app = app.select_dtypes(include=[np.number])
    sns.heatmap(numeric_app.corr(), annot=True, ax=ax)
    plt.title('Correlation Heatmap')
    plt.savefig('plots/multivariate_heatmap.png')
    plt.close()
    
    # Descriptive Analysis
    print(app.describe())

if __name__ == "__main__":
    perform_eda()
