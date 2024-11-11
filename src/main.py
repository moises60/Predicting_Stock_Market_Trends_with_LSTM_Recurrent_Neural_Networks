import os
import sys
import argparse
import logging
from data_processing import load_data, preprocess_data
from model import build_model, train_model, make_predictions, evaluate_model, save_model, load_model_func
from visualization import visualize_results

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Main function that runs the entire project workflow.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Stock price prediction model using LSTM.')
    parser.add_argument('--company', type=str, required=True, help='Company name.')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file.')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the testing CSV file.')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'es'], help='Data language.')
    parser.add_argument('--timesteps', type=int, default=90, help='Number of timesteps for the sequence.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    args = parser.parse_args()

    company_name = args.company
    train_csv = args.train_csv
    test_csv = args.test_csv
    language = args.language
    timesteps = args.timesteps
    epochs = args.epochs
    batch_size = args.batch_size

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

    # Load datasets
    logging.info("Loading training and testing datasets...")
    df_train = load_data(train_csv, language=language)
    df_test = load_data(test_csv, language=language)

    # Preprocess data
    logging.info("Preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, real_stock_price, sc = preprocess_data(
        df_train, df_test, features, target, timesteps
    )

    # Extract dates from the test set
    dates_test = df_test['Date'].values

    # Build the model
    logging.info("Building the model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Train the model
    logging.info("Training the model...")
    train_model(model, X_train, y_train, X_val, y_val, company_name, epochs, batch_size)

    # Load the best saved model
    logging.info("Loading the best saved model...")
    model = load_model_func(company_name, best=True)

    # Make predictions
    logging.info("Making predictions on the test set...")
    predicted_stock_price = make_predictions(model, X_test, sc, feature_index=features.index(target))

    # Evaluate the model
    logging.info("Evaluating the model...")
    mae, rmse = evaluate_model(real_stock_price, predicted_stock_price)

    # Visualize the results
    logging.info("Visualizing the results...")
    error = real_stock_price - predicted_stock_price
    visualize_results(dates_test, real_stock_price, predicted_stock_price, company_name, error)

    # Save the final model
    save_model(model, company_name)


if __name__ == "__main__":
    main()
