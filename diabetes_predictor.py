import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle 

class DiabetesPredictor:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.columns = None

    def load_and_clean_data(self):
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)

        cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)

        self.columns = df.drop("Outcome", axis=1).columns
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:\nAccuracy: {round(accuracy * 100, 2)}%")

    def predict(self, input_data: dict):
        if self.columns is None:
            raise Exception("Model not trained yet.")
        data_df = pd.DataFrame([input_data])[self.columns]
        data_scaled = self.scaler.transform(data_df)
        prob = self.model.predict_proba(data_scaled)[0][1]
        return round(prob * 100, 2)

if __name__ == "__main__":
    predictor = DiabetesPredictor()
    X_train, X_test, y_train, y_test = predictor.load_and_clean_data()
    predictor.train(X_train, y_train)
    predictor.evaluate(X_test, y_test)

    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump({
            "model": predictor.model,
            "scaler": predictor.scaler,
            "columns": predictor.columns
        }, f)

    print("\n--- Model Saved Successfully ---")

   

    low_risk_input = {
        'Pregnancies': 0,
        'Glucose': 95,
        'BloodPressure': 70,
        'SkinThickness': 22,
        'Insulin': 85,
        'BMI': 21.0,
        'DiabetesPedigreeFunction': 0.2,
        'Age': 25
    }

    high_risk_input = {
        'Pregnancies': 6,
        'Glucose': 180,
        'BloodPressure': 88,
        'SkinThickness': 35,
        'Insulin': 200,
        'BMI': 40.0,
        'DiabetesPedigreeFunction': 1.2,
        'Age': 50
    }

    low_risk = predictor.predict(low_risk_input)
    high_risk = predictor.predict(high_risk_input)

    print(f"Low-risk prediction: {low_risk}% chance of diabetes")
    print(f"High-risk prediction: {high_risk}% chance of diabetes")