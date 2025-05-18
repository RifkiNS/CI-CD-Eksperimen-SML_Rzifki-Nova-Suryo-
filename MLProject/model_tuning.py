import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("GradientBoosting_Tuning")

# Load dataset
df = pd.read_csv(r"d:\Laskar Ai\Tugas\Eksperimen SML_Rifki Nova Suryo\membangun_model\flood_prepro.csv")

X = df.drop("FloodProbability", axis=1)
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_dist = {
    "n_estimators": np.linspace(50, 500, 5, dtype=int),
    "learning_rate": np.linspace(0.01, 0.3, 5)
}

# Randomized Search CV
gb = GradientBoostingRegressor(random_state=42)
search = RandomizedSearchCV(gb, param_distributions=param_dist, n_iter=5, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)

# MLflow logging
with mlflow.start_run(run_name="GB_Hyperparameter_Tuning"):
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Save model
    mlflow.sklearn.log_model(best_model, artifact_path="gradient_boosting_model")

print(f"Best model params: {search.best_params_}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")
