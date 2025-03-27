# stock-price-prediction-lstm-atl
Stock price prediction using LSTM with baseline and with Adaptive Transfer Learning performance and learning efficiency 
# Stock Price Prediction with LSTM and Adaptive Transfer Learning

This repository contains Python code for forecasting stock prices using Long Short-Term Memory (LSTM) neural networks. The project compares a baseline LSTM model with an enhanced version that incorporates an Adaptive Torodial Learning (ATL) mechanism to dynamically adjust the learning rate during training.

## Key Features

* **LSTM Neural Network:** Utilizes an LSTM model to capture temporal dependencies in stock price data.
* **Adaptive Transfer Learning (ATL):** Implements a strategy to modulate the learning rate based on the model's learning gain.
* **Multiple Stock Analysis:** Capable of training and evaluating models for multiple stock tickers (currently set to the top 10).
* **Future Price Prediction:** Predicts the closing price for a specified future horizon (default is 30 days).
* **Performance Evaluation:** Calculates and summarizes key metrics like training loss, test loss, energy consumption (simulated), and prediction latency.
* **Visualization:** Generates plots comparing actual stock prices with baseline and ATL model predictions, including future forecasts.
* **Experiment Tracking:** Uses TensorBoard for detailed logging of training progress.

## Getting Started

### Prerequisites

* Python 3.6 or higher
* pip (Python package installer)

### Installation

1.  Clone this repository (once you've created it on GitHub - see instructions below).
2.  Navigate to the repository directory:
    ```bash
    cd your-repository-name
    ```
3.  Install the required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

Execute the Python script:

```bash
python stock_price_prediction.py
The script will:

Download historical stock data for the specified tickers.
Preprocess the data for the LSTM model.
Train both the baseline and ATL-enhanced LSTM models.
Evaluate the models on a test set.
Generate plots of the stock price forecasts for each ticker.
Print a summary table of the performance metrics.
Provide instructions on how to launch TensorBoard to view detailed training logs.
Viewing TensorBoard
To view the TensorBoard logs, open a new terminal, navigate to the repository directory, and run:

Bash

tensorboard --logdir runs
Then, open your web browser and go to the address provided (usually http://localhost:6006/).

Code Structure
stock_price_prediction.py: Contains the main Python script with all the code for data loading, preprocessing, model definition, training, evaluation, and plotting.
requirements.txt: Lists the Python packages required to run the script.
LICENSE: Specifies the licensing terms for the code.
runs/: This directory will be created during the execution of the script and will contain the TensorBoard log files for each stock and model.
Key Components Explained
compute_learning_gain(loss, prev_loss, alpha=0.1): Calculates the "learning gain" (g(t)), which reflects the improvement in the model's performance based on the loss reduction.
atl_lr_modulation(base_lr, learning_gain): Implements the Adaptive Transfer Learning strategy by adjusting the learning rate based on the calculated learning gain.
load_stock_data(ticker, period, interval): Downloads historical stock data from Yahoo Finance.
preprocess_data(data, sequence_length, prediction_horizon): Prepares the stock data for the LSTM model by scaling and creating sequences.
LSTMModel(nn.Module): Defines the architecture of the LSTM neural network.
train_with_gt(model, optimizer, criterion, dataloader, writer, use_atl, epoch): Trains the LSTM model for one epoch, with the option to use ATL.
evaluate_model(model, dataloader, criterion, scaler, scaled_data_shape): Evaluates the trained model on the test dataset.
The main part of the script iterates through a list of top 10 stock tickers, trains and evaluates both baseline and ATL models, makes future predictions, and generates plots.
Interpretation of Results
The script will output a summary table showing the final training loss, final test loss, simulated energy consumption, prediction latency, and the predicted next-day price for both the baseline and ATL models for each stock. The generated plots will visualize the model's performance in forecasting stock prices compared to the actual prices. TensorBoard provides more detailed insights into the training process, including loss curves and learning gain values over epochs.

Potential Improvements
Explore different LSTM architectures (e.g., adding more layers, dropout).
Incorporate other relevant features beyond OHLCV data (e.g., technical indicators, news sentiment).
Implement more sophisticated evaluation metrics for time series forecasting.
Perform hyperparameter tuning to optimize the model's performance.
Implement a more accurate energy consumption monitoring mechanism if possible.

