import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from data_processing import load_data, convert_number
from model import make_predictions, evaluate_model
from visualization import visualize_results
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_from_file(filepath):
    """
    Loads a saved model from a file.
    
    Args:
        filepath (str): Path to the model file.
    
    Returns:
        Sequential: Loaded model.
    """
    try:
        model = load_model(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def preprocess_data_for_prediction(df_test, features, target, timesteps=90):
    """
    Preprocesses the test data for prediction.
    
    Args:
        df_test (pd.DataFrame): Test DataFrame.
        features (list): List of column names to use as features.
        target (str): Name of the target column.
        timesteps (int): Number of timesteps for the sequence.
    
    Returns:
        tuple: X_test, real_stock_price, sc (MinMaxScaler)
    """
    # Select features for testing
    test_set = df_test[features].values

    # Feature scaling (fitted on the test set)
    sc = MinMaxScaler(feature_range=(0, 1))
    test_set_scaled = sc.fit_transform(test_set)

    # Prepare inputs for the test set
    X_test = []
    for i in range(timesteps, len(test_set_scaled)):
        X_test.append(test_set_scaled[i-timesteps:i, :])
    X_test = np.array(X_test)

    # Adjust real_stock_price size to match X_test
    real_stock_price = test_set[timesteps:, features.index(target)].reshape(-1, 1)

    return X_test, real_stock_price, sc


def main():
    """
    Script to load an existing model and make predictions on a test set.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Stock price prediction using a pre-trained model.')
    parser.add_argument('--company', type=str, required=True, help='Company name.')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the testing CSV file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the pre-trained model file.')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'es'], help='Data language.')
    parser.add_argument('--timesteps', type=int, default=60, help='Number of timesteps for the sequence.')
    args = parser.parse_args()

    company_name = args.company
    test_csv = args.test_csv
    model_path = args.model
    language = args.language
    timesteps = args.timesteps

    features_mapping = {
        'es': ["Price", "Open", "High", "Low", "Vol.", "Change %"],
        'en': ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    }
    target_mapping = {
        'es': "Price",
        'en': "Price"
    }

    features = features_mapping[language]
    target = target_mapping[language]

    # Load the test dataset
    logging.info("Loading test dataset...")
    df_test = load_data(test_csv, language=language)

    # Preprocess the test data
    logging.info("Preprocessing test data...")
    X_test, real_stock_price, sc = preprocess_data_for_prediction(
        df_test, features, target, timesteps
    )

    # Extract dates from the test set (adjusted for timesteps)
    dates_test = df_test['Date'].values[timesteps:]

    # Load the model
    logging.info(f"Loading model from {model_path}...")
    model = load_model_from_file(model_path)

    # Make predictions
    logging.info("Making predictions on the test set...")
    predicted_stock_price_scaled = model.predict(X_test)

    # Inverse scaling of predictions
    # Create an array of zeros for the other features
    dummy = np.zeros((predicted_stock_price_scaled.shape[0], sc.n_features_in_ - 1))
    predicted_scaled_full = np.concatenate((predicted_stock_price_scaled, dummy), axis=1)
    predicted_stock_price = sc.inverse_transform(predicted_scaled_full)[:, 0].reshape(-1, 1)

    # Evaluate the model
    logging.info("Evaluating the model...")
    mae, rmse = evaluate_model(real_stock_price, predicted_stock_price)

    # Visualize the results
    logging.info("Visualizing the results...")
    error = real_stock_price - predicted_stock_price
    visualize_results(dates_test, real_stock_price, predicted_stock_price, company_name, error)


if __name__ == "__main__":
    main()
