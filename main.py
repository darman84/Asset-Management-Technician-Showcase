import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read hydrant data from JSON file
with open('hydrant_data.json') as file:
    data = json.load(file)

# Create a DataFrame from the hydrant data
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df[['age', 'maintenance_count', 'water_main_size', 'max_pressure', 'days_since_last_maintenance',
        'accessibility', 'leaks', 'outlet_condition', 'operating_nut_condition', 'frequency_of_use']]
y = df['days_since_last_maintenance']

# Handle missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create and train the model (Gradient Boosting Regressor)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Function to predict days until next maintenance
def predict_maintenance(age, maintenance_count, water_main_size, max_pressure, days_since_last_maintenance,
                        accessibility, leaks, outlet_condition, operating_nut_condition, frequency_of_use):
    input_data = [[age, maintenance_count, water_main_size, max_pressure, days_since_last_maintenance,
                   accessibility, leaks, outlet_condition, operating_nut_condition, frequency_of_use]]
    input_data_imputed = imputer.transform(input_data)
    days_until_maintenance = model.predict(input_data_imputed)
    return int(days_until_maintenance[0])

# Example usage
new_hydrant_data = [
    {'hydrant_id': 6, 'age': 7, 'maintenance_count': 2, 'water_main_size': 6, 'max_pressure': 105,
     'days_since_last_maintenance': 400, 'accessibility': 1, 'leaks': 0, 'outlet_condition': 1,
     'operating_nut_condition': 1, 'frequency_of_use': 7},
    {'hydrant_id': 7, 'age': 15, 'maintenance_count': 5, 'water_main_size': 8, 'max_pressure': 140,
     'days_since_last_maintenance': 800, 'accessibility': 1, 'leaks': 0, 'outlet_condition': 0,
     'operating_nut_condition': 1, 'frequency_of_use': 15}
]

for hydrant in new_hydrant_data:
    days_until_maintenance = predict_maintenance(
        hydrant['age'],
        hydrant['maintenance_count'],
        hydrant['water_main_size'],
        hydrant['max_pressure'],
        hydrant['days_since_last_maintenance'],
        hydrant.get('accessibility', 1),  # Use get() to handle missing values
        hydrant.get('leaks', 0),
        hydrant.get('outlet_condition', 1),
        hydrant.get('operating_nut_condition', 1),
        hydrant.get('frequency_of_use', 0)
    )
    print(f"Hydrant {hydrant['hydrant_id']} will need maintenance in approximately {days_until_maintenance} days.")
