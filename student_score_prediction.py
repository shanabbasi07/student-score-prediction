
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_PATH = 'student_data.csv' 

def load_or_generate(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
    else:
       
        rng = np.random.RandomState(42)
        hours = rng.uniform(0, 10, 200)
        scores = 5 * hours + rng.normal(0, 5, 200) + 30  
        df = pd.DataFrame({'Study_Hours': hours, 'Exam_Score': scores})
        df.to_csv(path, index=False)
        print(f"Generated synthetic dataset and saved to {path}")
    return df

def main():
    df = load_or_generate(DATA_PATH)
    print(df.head())


    print(df.describe())


    X = df[['Study_Hours']]        
    y = df['Exam_Score']          


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")

  
    plt.figure(figsize=(8,5))
    plt.scatter(X_test, y_test, label='Actual', alpha=0.7)
    plt.scatter(X_test, y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Study Hours')
    plt.ylabel('Exam Score')
    plt.legend()
    plt.title('Actual vs Predicted (Test set)')
    plt.show()

  
    joblib.dump(model, 'student_score_model.joblib')
    print("Saved trained model to student_score_model.joblib")

if __name__ == '__main__':
    main()
