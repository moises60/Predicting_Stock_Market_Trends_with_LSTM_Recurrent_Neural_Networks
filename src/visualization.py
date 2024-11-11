import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the project's root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure the directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def visualize_results(dates, real, pred, company_name, error=None):
    """
    Visualizes prediction results and saves the graphs.
    
    Args:
        dates (np.array): Array of dates corresponding to the test data.
        real (np.array): Real values.
        pred (np.array): Predicted values.
        company_name (str): Company name.
        error (np.array, optional): Prediction errors.
    """
    # Prediction graph
    plt.figure(figsize=(14, 5))
    plt.plot(dates, real, color='red', label='Real Stock Price')
    plt.plot(dates, pred, color='blue', label='Predicted Stock Price')
    plt.title(f'{company_name} Stock Price Prediction')
    plt.xlabel('Month and Year')
    plt.ylabel('Stock Price (EUR)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    prediction_plot_path = os.path.join(OUTPUTS_DIR, f'{company_name}_prediction.png')
    plt.savefig(prediction_plot_path)
    logging.info(f"Prediction graph saved at {prediction_plot_path}")
    plt.close()
    
    # Error graph
    if error is not None:
        plt.figure(figsize=(14, 5))
        plt.plot(dates, error, color='green', label='Prediction Error')
        plt.title(f'Prediction Error - {company_name}')
        plt.xlabel('Month and Year')
        plt.ylabel('Error')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        error_plot_path = os.path.join(OUTPUTS_DIR, f'{company_name}_error.png')
        plt.savefig(error_plot_path)
        logging.info(f"Error graph saved at {error_plot_path}")
        plt.close()
