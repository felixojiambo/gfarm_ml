from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data('data/commodity_prices.csv')
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    joblib.dump(model, 'models/commodity_price_model.pkl')
    joblib.dump(X_train.columns, 'models/model_columns.pkl')

if __name__ == '__main__':
    train_model()
