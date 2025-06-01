import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Parameters 
data_dir = "."  # Current directory
forecast_horizons = list(range(72, 72 + 24 * 10))

# Load CSV files 
temp_df = pd.read_csv(os.path.join(data_dir, "temperature.csv"))
humidity_df = pd.read_csv(os.path.join(data_dir, "humidity.csv"))
pressure_df = pd.read_csv(os.path.join(data_dir, "pressure.csv"))
wind_df = pd.read_csv(os.path.join(data_dir, "wind_speed.csv"))

# Melt all to long format
value_vars = temp_df.columns[1:]
temp_melt = temp_df.melt(id_vars=["datetime"], value_vars=value_vars, var_name="City", value_name="temp")
humidity_melt = humidity_df.melt(id_vars=["datetime"], value_vars=value_vars, var_name="City", value_name="humidity")
pressure_melt = pressure_df.melt(id_vars=["datetime"], value_vars=value_vars, var_name="City", value_name="pressure")
wind_melt = wind_df.melt(id_vars=["datetime"], value_vars=value_vars, var_name="City", value_name="wind")

# Merge numeric features 
merged_df = temp_melt.merge(humidity_melt, on=["datetime", "City"]) \
                     .merge(pressure_melt, on=["datetime", "City"]) \
                     .merge(wind_melt, on=["datetime", "City"])

# Feature engineering 
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])
merged_df["temp"] = merged_df["temp"] - 273.15
merged_df["hour"] = merged_df["datetime"].dt.hour
merged_df["day"] = merged_df["datetime"].dt.day
merged_df["month"] = merged_df["datetime"].dt.month
merged_df["day_of_week"] = merged_df["datetime"].dt.dayofweek
merged_df["is_weekend"] = (merged_df["day_of_week"] >= 5).astype(int)

# Sort and clean 
merged_df = merged_df.sort_values(by=["City", "datetime"]).dropna()

# Add future targets for ALL parameters 
print("Creating target variables for all parameters...")

# Temperature targets
for h in forecast_horizons:
    merged_df[f"target_temp_{h}h"] = merged_df.groupby("City")["temp"].shift(-h)

# Humidity targets
for h in forecast_horizons:
    merged_df[f"target_humidity_{h}h"] = merged_df.groupby("City")["humidity"].shift(-h)

# Pressure targets
for h in forecast_horizons:
    merged_df[f"target_pressure_{h}h"] = merged_df.groupby("City")["pressure"].shift(-h)

# Wind speed targets
for h in forecast_horizons:
    merged_df[f"target_wind_{h}h"] = merged_df.groupby("City")["wind"].shift(-h)

# Create target column lists
temp_target_cols = [f"target_temp_{h}h" for h in forecast_horizons]
humidity_target_cols = [f"target_humidity_{h}h" for h in forecast_horizons]
pressure_target_cols = [f"target_pressure_{h}h" for h in forecast_horizons]
wind_target_cols = [f"target_wind_{h}h" for h in forecast_horizons]

all_target_cols = temp_target_cols + humidity_target_cols + pressure_target_cols + wind_target_cols

# Drop rows with NaN in target columns
merged_df = merged_df.dropna(subset=all_target_cols)

# Feature selection 
features = ["temp", "humidity", "pressure", "wind", "hour", "day", "month", "day_of_week", "is_weekend"]
X = merged_df[features]

# Train models for each parameter type
X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Features: {features}")
print(f"Total targets to predict: {len(all_target_cols)}")

# Dictionary to store all models
all_models = {
    "temperature": {},
    "humidity": {},
    "pressure": {},
    "wind": {}
}

# Train Temperature Models 
print("\n=== Training Temperature Models ===")
for i, col in enumerate(temp_target_cols):
    y_train = merged_df.loc[X_train.index, col]
    y_test = merged_df.loc[X_test.index, col]
    
    print(f"Training model for {col}...")
    
    model = xgb.XGBRegressor(
        device="cpu",
        tree_method="hist",
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10  
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    hours = int(col.split('_')[-1][:-1])
    all_models["temperature"][hours] = model
    
    # Calculate RMSE
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  RMSE: {rmse:.4f}")

# === Train Humidity Models ===
print("\n=== Training Humidity Models ===")
for i, col in enumerate(humidity_target_cols):
    y_train = merged_df.loc[X_train.index, col]
    y_test = merged_df.loc[X_test.index, col]
    
    print(f"Training model for {col}...")
    
    model = xgb.XGBRegressor(
        device="cpu",
        tree_method="hist",
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    hours = int(col.split('_')[-1][:-1])
    all_models["humidity"][hours] = model
    
    # Calculate RMSE
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  RMSE: {rmse:.4f}")

# Train Pressure Models 
print("\n=== Training Pressure Models ===")
for i, col in enumerate(pressure_target_cols):
    y_train = merged_df.loc[X_train.index, col]
    y_test = merged_df.loc[X_test.index, col]
    
    print(f"Training model for {col}...")
    
    model = xgb.XGBRegressor(
        device="cpu",
        tree_method="hist",
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    hours = int(col.split('_')[-1][:-1])
    all_models["pressure"][hours] = model
    
    # Calculate RMSE
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  RMSE: {rmse:.4f}")

# Train Wind Speed Models
print("\n=== Training Wind Speed Models ===")
for i, col in enumerate(wind_target_cols):
    y_train = merged_df.loc[X_train.index, col]
    y_test = merged_df.loc[X_test.index, col]
    
    print(f"Training model for {col}...")
    
    model = xgb.XGBRegressor(
        device="cpu",
        tree_method="hist",
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    hours = int(col.split('_')[-1][:-1])
    all_models["wind"][hours] = model
    
    # Calculate RMSE
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  RMSE: {rmse:.4f}")

# Save all models and metadata 
print("\n=== Saving Models ===")
joblib.dump(all_models, "weather_multi_parameter_models.joblib")
# joblib.dump(features, "weather_model_features.joblib")
joblib.dump(forecast_horizons, "weather_forecast_horizons.joblib")

print("All models saved successfully!")

# Create prediction function 
def predict_weather_multi_params(input_features, hours_ahead):
    """
    Predict multiple weather parameters for a given number of hours ahead.
    
    Args:
        input_features: DataFrame with columns matching 'features'
        hours_ahead: Number of hours to predict ahead
    
    Returns:
        Dictionary with predicted values for all parameters
    """
    # Find closest available forecast horizon
    closest_horizon = min(forecast_horizons, key=lambda h: abs(h - hours_ahead))
    
    predictions = {}
    
    # Predict each parameter
    if closest_horizon in all_models["temperature"]:
        predictions["temperature"] = all_models["temperature"][closest_horizon].predict(input_features)[0]
    
    if closest_horizon in all_models["humidity"]:
        predictions["humidity"] = all_models["humidity"][closest_horizon].predict(input_features)[0]
    
    if closest_horizon in all_models["pressure"]:
        predictions["pressure"] = all_models["pressure"][closest_horizon].predict(input_features)[0]
    
    if closest_horizon in all_models["wind"]:
        predictions["wind_speed"] = all_models["wind"][closest_horizon].predict(input_features)[0]
    
    return predictions

# Save the prediction function
joblib.dump(predict_weather_multi_params, "weather_prediction_function.joblib")

# Visualization 
print("\n=== Creating Performance Visualization ===")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Model Performance for All Weather Parameters", fontsize=16)

# Temperature subplot
ax = axes[0, 0]
temp_rmse = []
for h in forecast_horizons:
    model = all_models["temperature"][h]
    y_test = merged_df.loc[X_test.index, f"target_temp_{h}h"]
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    temp_rmse.append(rmse)

ax.bar([f"{h}h" for h in forecast_horizons], temp_rmse, color='red', alpha=0.7)
ax.set_title("Temperature Prediction RMSE")
ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("RMSE (°C)")
ax.grid(True, alpha=0.3)

# Humidity subplot
ax = axes[0, 1]
humidity_rmse = []
for h in forecast_horizons:
    model = all_models["humidity"][h]
    y_test = merged_df.loc[X_test.index, f"target_humidity_{h}h"]
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    humidity_rmse.append(rmse)

ax.bar([f"{h}h" for h in forecast_horizons], humidity_rmse, color='blue', alpha=0.7)
ax.set_title("Humidity Prediction RMSE")
ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("RMSE (%)")
ax.grid(True, alpha=0.3)

# Pressure subplot
ax = axes[1, 0]
pressure_rmse = []
for h in forecast_horizons:
    model = all_models["pressure"][h]
    y_test = merged_df.loc[X_test.index, f"target_pressure_{h}h"]
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    pressure_rmse.append(rmse)

ax.bar([f"{h}h" for h in forecast_horizons], pressure_rmse, color='green', alpha=0.7)
ax.set_title("Pressure Prediction RMSE")
ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("RMSE (mb)")
ax.grid(True, alpha=0.3)

# Wind speed subplot
ax = axes[1, 1]
wind_rmse = []
for h in forecast_horizons:
    model = all_models["wind"][h]
    y_test = merged_df.loc[X_test.index, f"target_wind_{h}h"]
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    wind_rmse.append(rmse)

ax.bar([f"{h}h" for h in forecast_horizons], wind_rmse, color='orange', alpha=0.7)
ax.set_title("Wind Speed Prediction RMSE")
ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("RMSE (kph)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("multi_parameter_model_performance.png", dpi=300)
plt.show()

print("\nTraining completed! Multi-parameter weather prediction models are ready.")