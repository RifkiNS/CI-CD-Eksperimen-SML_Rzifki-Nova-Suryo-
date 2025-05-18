import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("SML_experiment")

# Load dataset
df = pd.read_csv(r"d:\Laskar Ai\Tugas\Eksperimen SML_Rifki Nova Suryo\membangun_model\flood_prepro.csv")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('FloodProbability', axis=1),
    df['FloodProbability'],
    test_size=0.2,
    random_state=42
)

# Input example for logging
input_example = X_train.iloc[0:5]

# MLflow tracking
with mlflow.start_run():
    mlflow.autolog()

    # Define hyperparameters correctly
    learning_rate = 0.01
    n_estimators = 100
    random_state = 42

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

    # Log metrics
    mlflow.log_metric('MAE', mae_gbr)
    mlflow.log_metric('MSE', mse_gbr)
    mlflow.log_metric('R2_Score', r2_gbr)

    # Log model manually
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        input_example=input_example
    )