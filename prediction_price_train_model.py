# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load your historical data
# For the sake of this example, let's assume your data is in a CSV file
data = pd.read_csv('historical_data.csv')

# Preprocess the data
# Assuming your data has columns: 'commodity', 'town', 'date', 'price'
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Convert categorical variables to dummy/indicator variables
data = pd.get_dummies(data, columns=['commodity', 'town'], drop_first=True)

# Define features and target
X = data.drop(['price', 'date'], axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model
joblib.dump(model, 'price_model.pkl')
