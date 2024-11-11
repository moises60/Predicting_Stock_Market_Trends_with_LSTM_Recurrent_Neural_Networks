import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_number(x, language='en'):
    """
    Converts a number from Spanish or English format to float.
    Handles 'B', 'M', 'K', and '%' if present.
    
    Args:
        x (str or float): Number in Spanish or English format.
        language (str): Language of the format ('es' for Spanish, 'en' for English).
    
    Returns:
        float: Converted number as a float.
    """
    try:
        if isinstance(x, str):
            x = x.strip()  # Remove leading and trailing whitespace
            is_percentage = False
            multiplier = 1
            if x.endswith('%'):
                is_percentage = True
                x = x.replace('%', '')
            if 'B' in x or 'b' in x:
                multiplier = 1_000_000_000  # Multiply by one billion
                x = x.replace('B', '').replace('b', '')
            elif 'M' in x or 'm' in x:
                multiplier = 1_000_000  # Multiply by one million
                x = x.replace('M', '').replace('m', '')
            elif 'K' in x or 'k' in x:
                multiplier = 1_000  # Multiply by one thousand
                x = x.replace('K', '').replace('k', '')
            
            if language == 'es':
                # Spanish format: '.' as thousand separator and ',' as decimal
                x = x.replace('.', '').replace(',', '.')
            elif language == 'en':
                # English format: ',' as thousand separator and '.' as decimal
                x = x.replace(',', '')  # Remove thousand separators
                # The dot is already the decimal separator
            else:
                raise ValueError(f"Unsupported language: {language}")
            
            num = float(x) * multiplier
            if is_percentage:
                num = num / 100  # Convert percentage to decimal
            return num
        return float(x)
    except Exception as e:
        logging.error(f"Error converting number: {x} - {e}")
        return np.nan


def load_data(filepath, language='en'):
    """
    Loads a CSV file and performs basic preprocessing based on the language.
    
    Args:
        filepath (str): Path to the CSV file.
        language (str): Column language ('es' or 'en').
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Define column mappings based on the language
    column_mapping = {
        'es': {
            "Fecha": "Date",
            "Último": "Price",
            "Apertura": "Open",
            "Máximo": "High",
            "Mínimo": "Low",
            "% var.": "Change %",
            "Vol.": "Vol."
        },
        'en': {
            "Date": "Date",
            "Price": "Price",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Change %": "Change %",
            "Vol.": "Vol."
        }
    }
    
    try:
        df = pd.read_csv(filepath)
        # Rename columns according to the mapping
        df.rename(columns=column_mapping[language], inplace=True)
        
        # Convert relevant columns to float
        columns_to_convert = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
        for col in columns_to_convert:
            df[col] = df[col].apply(lambda x: convert_number(x, language=language))
        
        # Convert "Date" column to datetime
        date_format = '%m/%d/%Y' if language == 'en' else '%d.%m.%Y'
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        
        # Sort the dataset by date in ascending order
        df = df.sort_values('Date')
        df.reset_index(drop=True, inplace=True)
        
        # Handle missing values if any
        df.ffill(inplace=True)
        
        return df
    except FileNotFoundError:
        logging.error(f"The file {filepath} was not found.")
        raise
    except KeyError as e:
        logging.error(f"Mapping key not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise


def preprocess_data(df_train, df_test, features, target, timesteps=80):
    """
    Preprocesses the data for training and testing.
    
    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_test (pd.DataFrame): Testing DataFrame.
        features (list): List of column names to use as features.
        target (str): Name of the target column.
        timesteps (int): Number of timesteps for the sequence.
    
    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, real_stock_price, sc (MinMaxScaler)
    """
    # Select features for training
    training_set = df_train[features].values
    test_set = df_test[features].values

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set_scaled = sc.transform(test_set)
    
    # Create data structure for training
    X_train = []
    y_train = []
    for i in range(timesteps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-timesteps:i, :])
        y_train.append(training_set_scaled[i, features.index(target)])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Split into training and validation sets
    split_index = int(len(X_train) * 0.9)
    X_val = X_train[split_index:]
    y_val = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    
    # Prepare inputs for the test set
    dataset_total = pd.concat((df_train[features], df_test[features]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(df_test) - timesteps:].values
    inputs_scaled = sc.transform(inputs)
    
    X_test = []
    for i in range(timesteps, timesteps + len(df_test)):
        X_test.append(inputs_scaled[i-timesteps:i, :])
    X_test = np.array(X_test)
    
    # Get real target values
    real_stock_price = df_test[target].values.reshape(-1, 1)
    
    return X_train, y_train, X_val, y_val, X_test, real_stock_price, sc
