import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = pd.get_dummies(data, columns=['commodity', 'market'])
    
    X = data.drop(columns=['date', 'price'])
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
