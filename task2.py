import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("uav_data.csv")

# Convert 'Observation' column if it's categorical
if data["Observation"].dtype == "object":
    label_encoder = LabelEncoder()
    data["Observation"] = label_encoder.fit_transform(data["Observation"])

# Define features (X) and target (y)
X = data[["Observation"]]  # Feature
y = data["Reward"]  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')
