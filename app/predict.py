import joblib
import pandas as pd

def predict_price(input_data):
    model = joblib.load('models/commodity_price_model.pkl')
    
    # input_data is a dictionary
    input_df = pd.DataFrame([input_data])
    
    # Ensure 'commodity' and 'market' keys are in input_data
    required_columns = ['commodity', 'market']
    for col in required_columns:
        if col not in input_df.columns:
            raise KeyError(f"The input data must include the column '{col}'")

    input_df = pd.get_dummies(input_df, columns=['commodity', 'market'])
    
    # Ensure the input data has the same columns as the training data
    model_columns = joblib.load('models/model_columns.pkl')
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    predicted_price = model.predict(input_df)[0]
    return predicted_price

if __name__ == '__main__':
    input_data = {
        'year': 2023,
        'month': 7,
        'day': 10,
        'commodity': 'wheat',   # Updated to include commodity
        'market': 'Nairobi',    # Updated to include market
        'quantity': 1000,
        'supply': 1200,
        'demand': 1100
    }
    print(f'Predicted Price: {predict_price(input_data)}')
