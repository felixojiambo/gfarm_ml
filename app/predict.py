import joblib
import pandas as pd

def predict_price(input_data):
    model = joblib.load('models/commodity_price_model.pkl')
    
    # Assuming input_data is a dictionary
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['commodity', 'market'])
    
    # Ensure the input data has the same columns as the training data
    model_columns = joblib.load('models/model_columns.pkl')
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    predicted_price = model.predict(input_df)[0]
    return predicted_price
