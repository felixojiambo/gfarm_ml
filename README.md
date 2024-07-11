
# Gfarm Commodity Price Prediction

Gfarm is a Nairobi-based startup offering a web and mobile app for real-time market prices of agricultural commodities. This project contains the machine learning component for predicting the prices of these commodities based on historical data.

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── data/
│   └── commodity_prices.csv
├── models/
│   ├── commodity_price_model.pkl
│   └── model_columns.pkl
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.6+
- Virtualenv (optional, but recommended)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/gfarm-ml.git
   cd gfarm-ml
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your data**

   Place your historical commodity prices data in the `data/` directory as `commodity_prices.csv`. The data should have the following columns: `date`, `commodity`, `market`, `price`, `quantity`, `supply`, `demand`.

   Example:

   ```csv
   date,commodity,market,price,quantity,supply,demand
   2021-01-01,tomato,Nairobi,50,1000,1200,1100
   2021-01-02,tomato,Nairobi,52,1050,1300,1150
   ```

## Training the Model

To train the machine learning model, run the following command:

```bash
python app/train.py
```

This script will:

- Load and preprocess the historical data.
- Train a Linear Regression model.
- Save the trained model and the columns used for training in the `models/` directory.

## Making Predictions

To make a prediction using the trained model, run the `predict.py` script:

```bash
python app/predict.py
```

The script includes a sample input data dictionary which you can modify as needed. The script will output the predicted price.

## Files Description

- **app/train.py**: Script to train the machine learning model.
- **app/predict.py**: Script to make predictions using the trained model.
- **app/preprocess.py**: Functions for data preprocessing.
- **data/commodity_prices.csv**: Example dataset for training the model.
- **models/commodity_price_model.pkl**: Trained machine learning model.
- **models/model_columns.pkl**: Columns used for training the model.
- **requirements.txt**: List of required Python packages.

## Dependencies

- pandas
- scikit-learn
- joblib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
