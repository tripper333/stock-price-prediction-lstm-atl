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
