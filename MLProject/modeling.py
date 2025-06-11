import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import random
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "flood_prepro.csv")
    data = pd.read_csv(file_path)


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('FloodProbability', axis=1),
        data['FloodProbability'],
        test_size=0.2,
        random_state=42
    )

    # Input example for logging
    input_example = X_train.iloc[0:5]
    n_estimators = int(sys.argv[1] if len(sys.argv) > 1 else 100)
    learning_rate = float(sys.argv[2] if len(sys.argv) > 2 else 0.01)
    random_state = int(sys.argv[3] if len(sys.argv) > 3 else 42)

    # MLflow tracking
    with mlflow.start_run():

        model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train model
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        mae_gbr = mean_absolute_error(y_test, y_pred)
        mse_gbr = mean_squared_error(y_test, y_pred)
        r2_gbr = r2_score(y_test, y_pred)

        # Log model manually
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            input_example=input_example
        )
