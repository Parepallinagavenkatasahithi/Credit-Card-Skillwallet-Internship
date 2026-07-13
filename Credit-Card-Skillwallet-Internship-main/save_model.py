import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

def save_model():
    print("Training balanced lightweight model for deployment...")
    df = pd.read_csv('processed_dataset.csv')
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop('STATUS_BIN', axis=1)
    y = df['STATUS_BIN']
    
    # Train a Decision Tree Model with class_weight='balanced' so it approves applications too
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=20, class_weight='balanced', random_state=42)
    dt_model.fit(X, y)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
        
    print("Lightweight Balanced Model saved successfully as 'model.pkl'.")

if __name__ == "__main__":
    save_model()
