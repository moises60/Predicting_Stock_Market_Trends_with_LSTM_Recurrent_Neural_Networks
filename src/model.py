import os
import numpy as np
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the project's root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def build_model(input_shape):
    """
    Builds and compiles the LSTM model.
    
    Args:
        input_shape (tuple): Input shape (timesteps, features).
    
    Returns:
        Sequential: Compiled model.
    """
    model = Sequential()
    
    # Add LSTM and Dropout layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Add output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, company_name, epochs=200, batch_size=32):
    """
    Trains the model using Early Stopping and ModelCheckpoint.
    
    Args:
        model (Sequential): Compiled model.
        X_train (np.array): Training data.
        y_train (np.array): Training labels.
        X_val (np.array): Validation data.
        y_val (np.array): Validation labels.
        company_name (str): Company name.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
    
    Returns:
        History: Training history.
    """
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    checkpoint_path = os.path.join(MODELS_DIR, f'{company_name}_best_model.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard_log_dir = os.path.join(LOGS_DIR, company_name)
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, tensorboard],
        verbose=1
    )
    
    return history


def make_predictions(model, X_test, sc, feature_index=0):
    """
    Makes predictions on the test set and scales back the results.
    
    Args:
        model (Sequential): Trained model.
        X_test (np.array): Test data.
        sc (MinMaxScaler): Scaler used for features.
        feature_index (int): Index of the target feature.
    
    Returns:
        np.array: Scaled back predictions.
    """
    predicted_scaled = model.predict(X_test)
    # Inverse scaling only for the target feature
    dummy = np.zeros((predicted_scaled.shape[0], sc.n_features_in_ - 1))
    predicted_scaled_full = np.concatenate((predicted_scaled, dummy), axis=1)
    predicted = sc.inverse_transform(predicted_scaled_full)[:, feature_index]
    return predicted.reshape(-1, 1)


def evaluate_model(real, pred):
    """
    Calculates and displays evaluation metrics.
    
    Args:
        real (np.array): Real values.
        pred (np.array): Predicted values.
    
    Returns:
        tuple: MAE and RMSE.
    """
    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    logging.info(f'MAE: {mae}')
    logging.info(f'RMSE: {rmse}')
    return mae, rmse


def save_model(model, company_name):
    """
    Saves the trained model to a file.
    
    Args:
        model (Sequential): Trained model.
        company_name (str): Company name.
    """
    filepath = os.path.join(MODELS_DIR, f'{company_name}_final_model.keras')
    model.save(filepath)
    logging.info(f"Model saved at {filepath}")


def load_model_func(company_name, best=True):
    """
    Loads a saved model from a file.
    
    Args:
        company_name (str): Company name.
        best (bool): Indicates whether to load the best model or the final one.
    
    Returns:
        Sequential: Loaded model.
    """
    try:
        if best:
            filepath = os.path.join(MODELS_DIR, f'{company_name}_best_model.keras')
        else:
            filepath = os.path.join(MODELS_DIR, f'{company_name}_final_model.keras')
        model = load_model(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
