# Stock Price Prediction with LSTM and Adaptive Toroidal Learning (ATL): Neural Network Optimization and Learning Efficiency

This repository contains Python code for forecasting stock prices using Long Short-Term Memory (LSTM) neural networks. The project compares a baseline LSTM model with an enhanced version that incorporates an **Adaptive Toroidal Learning (ATL)** mechanism. This experiment serves as a **toy example** to explore and test new learning paradigms where **self-reflection** enables emergent learning and intelligence through dynamic performance improvement updates.

## Key Features

* **LSTM Neural Network:** Utilizes an LSTM model to capture temporal dependencies in stock price data.
* **Adaptive Toroidal Learning (ATL):** Implements a novel strategy to modulate the learning rate based on the model's **self-reflection coefficient of "learning gain" or "learning Efficiency"**.
* **Self-Reflection for Emergent Learning:** The core idea behind ATL, as explored in this experiment, is to enable the model to dynamically adapt its learning process based on its own performance, mimicking a form of self-reflection that can lead to emergent improvements.
* **Multiple Stock Analysis:** Capable of training and evaluating models for multiple stock tickers (currently set to the top 10).
* **Future Price Prediction:** Predicts the closing price for a specified future horizon (default is 30 days).
* **Performance and Learning Efficiency Evaluation:** Calculates and summarizes key metrics like training loss, test loss, energy consumption (simulated), and prediction latency, alongside the learning gain.
* **Visualization:** Generates plots comparing actual stock prices with baseline and ATL model predictions, including future forecasts.
* **Experiment Tracking:** Uses TensorBoard for detailed logging of training progress, including the learning efficiency.

## Code Structure

- **`stock_price_prediction.py`:** Contains the main Python script with all the code for data loading, preprocessing, model definition, training, evaluation, and plotting.
- **`requirements.txt`:** Lists the Python packages required to run the script.
- **`LICENSE`:** Specifies the licensing terms for the code.
- **`runs/`:** This directory will be created during the execution of the script and will contain the TensorBoard log files for each stock and model.

## Key Components Explained

- **`compute_learning_gain(loss, prev_loss, alpha=0.1)`:** Calculates the **self-reflection coefficient of "learning gain" or "learning Efficiency"** (`g(t)`). This metric reflects the rate at which the model's loss is decreasing. A higher value indicates a greater efficiency in learning. It serves as a form of self-assessment for the learning process.
- **`atl_lr_modulation(base_lr, learning_gain)`:** Implements the **Adaptive Toroidal Learning (ATL)** strategy. It dynamically adjusts the learning rate based on the calculated learning gain. The learning rate is modulated to potentially enhance convergence and performance based on the model's perceived learning efficiency.
- **`load_stock_data(ticker, period, interval)`:** Downloads historical stock data from Yahoo Finance.
- **`preprocess_data(data, sequence_length, prediction_horizon)`:** Prepares the stock data for the LSTM model by scaling and creating sequences.
- **`LSTMModel(nn.Module)`:** Defines the architecture of the LSTM neural network.
- **`train_with_gt(model, optimizer, criterion, dataloader, writer, use_atl, epoch)`:** Trains the LSTM model for one epoch, with the option to use ATL. The training process with ATL allows the model to update its learning rate based on its self-measured learning efficiency.
- **`evaluate_model(model, dataloader, criterion, scaler, scaled_data_shape)`:** Evaluates the trained model on the test dataset.
- The main part of the script iterates through a list of top 10 stock tickers, trains and evaluates both baseline and ATL models, makes future predictions, and generates plots to visually compare their performance.

## Interpretation of Results

The script will output a summary table showing the final training loss, final test loss, simulated energy consumption, prediction latency, and the predicted next-day price for both the baseline and ATL models for each stock. Additionally, the learning efficiency (as tracked through the learning gain) can be observed in the TensorBoard logs. The generated plots will visually compare the forecasting performance of the baseline and ATL models. The goal of this **toy experiment** is to observe if the self-reflective learning approach of ATL leads to noticeable differences in performance and learning efficiency compared to the standard baseline.

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
