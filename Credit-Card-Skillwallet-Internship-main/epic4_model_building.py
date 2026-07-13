import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(name, y_test, y_pred):
    print(f"--- {name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

def train_models():
    df = pd.read_csv('processed_dataset.csv')
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop('STATUS_BIN', axis=1)
    y = df['STATUS_BIN']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    evaluate_model("Logistic Regression", y_test, lr.predict(X_test))
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", y_test, rf.predict(X_test))
    
    # 3. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    evaluate_model("Decision Tree", y_test, dt.predict(X_test))

if __name__ == "__main__":
    train_models()
