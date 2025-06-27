import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import json

# Define LSTM model class (same as during training)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Normalize prices on a rolling basis resetting at the start of each day
def normalize_daily_rolling(data):
    data['date'] = data.index.date
    data['rolling_high'] = data.groupby('date')['high'].transform(lambda x: x.expanding(min_periods=1).max())
    data['rolling_low'] = data.groupby('date')['low'].transform(lambda x: x.expanding(min_periods=1).min())

    data['norm_open'] = (data['open'] - data['rolling_low']) / (data['rolling_high'] - data['rolling_low'])
    data['norm_high'] = (data['high'] - data['rolling_low']) / (data['rolling_high'] - data['rolling_low'])
    data['norm_low'] = (data['low'] - data['rolling_low']) / (data['rolling_high'] - data['rolling_low'])
    data['norm_close'] = (data['close'] - data['rolling_low']) / (data['rolling_high'] - data['rolling_low'])

    data.fillna(0, inplace=True)
    return data[['norm_open', 'norm_high', 'norm_low', 'norm_close']]

# Load the saved model
input_size = 5  # time_token, norm_open, norm_high, norm_low, norm_close
hidden_layer_size = 100
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, output_size)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# Connect to MetaTrader 5
if not mt5.initialize():
    print("Initialize failed")
    mt5.shutdown()

# Load the latest 160 bars of market data
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
bars = 160
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
mt5.shutdown()

# Convert to DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

# Normalize the new data
data[['norm_open', 'norm_high', 'norm_low', 'norm_close']] = normalize_daily_rolling(data)

# Tokenize time
data['time_token'] = (data.index.hour * 3600 + data.index.minute * 60 + data.index.second) / 86400

# Drop unnecessary columns
data = data[['time_token', 'norm_open', 'norm_high', 'norm_low', 'norm_close']]

# Fetch the last 100 sequences for evaluation
seq_length = 60
evaluation_steps = 100

# Initialize lists for storing evaluation results
all_true_prices = []
all_predicted_prices = []

model.eval()

for step in range(evaluation_steps, 0, -1):
    seq = data.values[-step-seq_length:-step]
    seq = torch.tensor(seq, dtype=torch.float32)

    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        prediction = model(seq).item()
    
    all_true_prices.append(data['norm_close'].values[-step])
    all_predicted_prices.append(prediction)

# Calculate percent changes and convert to percentages
true_prices = np.array(all_true_prices)
predicted_prices = np.array(all_predicted_prices)

true_pct_change = (np.diff(true_prices) / (true_prices[:-1] + 1e-10)) * 100
predicted_pct_change = (np.diff(predicted_prices) / (predicted_prices[:-1] + 1e-10)) * 100

# Make next prediction
next_seq = data.values[-seq_length:]
next_seq = torch.tensor(next_seq, dtype=torch.float32)

with torch.no_grad():
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))
    next_prediction = model(next_seq).item()

# Calculate percent change for the next prediction
next_price_pct_change = ((next_prediction - all_predicted_prices[-1]) / all_predicted_prices[-1]) * 100

# Save results to JSON for web dashboard
results = {
    "predicted_prices": list(map(float, all_predicted_prices)),
    "true_prices": list(map(float, all_true_prices)),
    "percent_change": float(next_price_pct_change),
    "next_prediction": float(next_prediction)
}

with open("prediction_results.json", "w") as f:
    json.dump(results, f)