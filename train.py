import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os

if not os.path.exists('data.csv'):
    print("❌ ERROR: data.csv missing")
else:
    df = pd.read_csv('data.csv')
    
    # Logic: Keep all Fraud cases, and only a tiny bit of Safe cases
    fraud = df[df.iloc[:, -1] == 1]
    safe = df[df.iloc[:, -1] == 0].sample(n=len(fraud)) 
    
    balanced_df = pd.concat([fraud, safe])
    
    X = balanced_df.iloc[:, :-1]
    y = balanced_df.iloc[:, -1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using DecisionTree for sharper detection
    model = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    model.fit(X_scaled, y)
    
    joblib.dump(model, 'final_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ SUCCESS: High-Sensitivity Decision Model Saved!")