import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

def create_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Generate synthetic features
    income = np.random.normal(60000, 20000, num_samples)
    employment_duration = np.random.normal(5, 3, num_samples)
    loan_balance = np.random.normal(10000, 5000, num_samples)
    credit_inquiries = np.random.poisson(2, num_samples)
    past_due_records = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.7, 0.15, 0.1, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Income': income,
        'Employment_Duration': employment_duration,
        'Loan_Balance': loan_balance,
        'Credit_Inquiries': credit_inquiries,
        'Past_Due_Records': past_due_records
    })
    
    # Feature Engineering (Multi-class to Binary for Past Due Records)
    # 0 -> 0 (No past due), 1, 2, 3 -> 1 (Has past due records)
    df['High_Risk_Past_Due'] = df['Past_Due_Records'].apply(lambda x: 1 if x > 0 else 0)
    
    # Define Target Variable (Approval: 1 = Approved, 0 = Rejected)
    # Let's create a logical relationship for the target
    target = []
    for i in range(num_samples):
        score = (df['Income'].iloc[i] / 10000) - (df['Loan_Balance'].iloc[i] / 5000) + df['Employment_Duration'].iloc[i] - (df['Credit_Inquiries'].iloc[i] * 2) - (df['High_Risk_Past_Due'].iloc[i] * 5)
        # Add some noise
        score += np.random.normal(0, 2)
        if score > 0:
            target.append(1)
        else:
            target.append(0)
            
    df['Approved'] = target
    return df

def main():
    print("Generating synthetic data...")
    df = create_synthetic_data()
    
    # Features and Target
    X = df[['Income', 'Employment_Duration', 'Loan_Balance', 'Credit_Inquiries', 'High_Risk_Past_Due']]
    y = df['Approved']
    
    # Train / Test Split (80% Train / 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data Pre-processing (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print("\nTraining and Evaluating Models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Save the best model and preprocessing objects
    print("Saving model and scaler...")
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Files saved successfully as model.pkl and scaler.pkl.")

if __name__ == "__main__":
    main()
