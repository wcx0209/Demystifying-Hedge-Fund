import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading the model
import pytz

# Define the start and end dates for the data download
end_date = datetime.now()
start_date = end_date - timedelta(days=59)

# Download USD/JPY hourly price data from yfinance
ticker = 'JPY=X'
data = yf.download(ticker, start=start_date, end=end_date, interval='30m')

# Calculate the RSI and ADX indicators using the ta library
data['RSI'] = ta.momentum.RSIIndicator(data['Open'], window=9).rsi()

adx_indicator = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
data['ADX'] = adx_indicator.adx()
data['plus_di'] = adx_indicator.adx_pos()
data['minus_di'] = adx_indicator.adx_neg()

# Drop NaN values after calculating the indicators
data.dropna(inplace=True)

# Define ADX, plus_di, and minus_di as the features for the HMM model
features = data[['ADX','plus_di', 'minus_di']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets (last 260 hours as test)
train_data = features_scaled[:-260]  # All data except the last 260 hours

# Train HMM model with 2 hidden states using the training set
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model.fit(train_data)

# Save the trained model
joblib.dump(model, 'USDJPY_hmm_model_DMI_1h.pkl')

# Predict the hidden states for the entire data
hidden_states_train = model.predict(features_scaled)

# Add hidden states to the DataFrame
data['State'] = np.nan
data.iloc[:len(hidden_states_train), -1] = hidden_states_train  # Assign train states

# Extract the transition matrix
transition_matrix = model.transmat_

# Plot the transition matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=[f'State {i}' for i in range(2)],
            yticklabels=[f'State {i}' for i in range(2)])
plt.title('Transition Matrix Heatmap - Training Data')
plt.xlabel('To State')
plt.ylabel('From State')
plt.show()

# Calculate hourly returns using Open price
data['Return'] = data['Open'].pct_change().shift(-1)
data.dropna(inplace=True)

# Calculate cumulative returns for each state
cumulative_returns_by_state = {}
for state in range(2):
    state_data = data[data['State'] == state]
    cumulative_returns = (1 + state_data['Return']).cumprod() - 1
    cumulative_returns_by_state[state] = cumulative_returns

# Highlight points according to states
colors = ['green', 'red','yellow','purple']

# Plot cumulative returns by state for the training set
plt.figure(figsize=(14, 8))
for state, cumulative_returns in cumulative_returns_by_state.items():
    plt.plot(cumulative_returns.index, cumulative_returns, color=colors[state], label=f'State {state}')

plt.axhline(y=0, color='orange', linestyle='--', alpha=0.4)
plt.title('Cumulative Returns by State - Training Data (Using ADX, DI+)')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# Plot the price trend with states highlighted for the training set
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Open'], label='Open Price', color='black')

for state in range(2):
    state_data = data[data['State'] == state]
    plt.scatter(state_data.index, state_data['Open'], color=colors[state], label=f'State {state}')

plt.title('USDJPY Price Trend with Hidden States Highlighted - Training Data (Using ADX, DI+)')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.legend()
plt.grid(True)
plt.show()

# Display the last data point and its associated state
last_data_point = data.iloc[-1]

# Convert the last data point's time to Singapore time zone if it is already timezone-aware
sg_timezone = pytz.timezone('Asia/Singapore')

if last_data_point.name.tzinfo is not None:
    last_data_point_time_sg = last_data_point.name.tz_convert(sg_timezone)
else:
    last_data_point_time_sg = last_data_point.name.tz_localize('UTC').tz_convert(sg_timezone)

print(f"Last Data Point:")
print(f"Date: {last_data_point_time_sg}")
print(f"Open Price: {last_data_point['Open']}")
print(f"State: {last_data_point['State']}")

# Plot the DMI (ADX, +DI, -DI) trend with states highlighted

# Create a plot for ADX, +DI, and -DI
plt.figure(figsize=(14, 8))

# Plot the ADX, +DI, and -DI trends
# plt.plot(data.index, data['ADX'], label='ADX', color='blue', linestyle='--')
plt.plot(data.index, data['plus_di'], label='+DI', color='green', linestyle='-')
plt.plot(data.index, data['minus_di'], label='-DI', color='red', linestyle='-')

# Highlight the ADX, +DI, and -DI points according to hidden states
for state in range(2):
    state_data = data[data['State'] == state]
    # plt.scatter(state_data.index, state_data['ADX'], color=colors[state], label=f'State {state} - ADX', marker='o')
    plt.scatter(state_data.index, state_data['plus_di'], color=colors[state], label=f'State {state} - +DI', marker='x')
    plt.scatter(state_data.index, state_data['minus_di'], color=colors[state], label=f'State {state} - -DI', marker='^')

# Add title and labels
plt.title('ADX, +DI, -DI Trends with Hidden States Highlighted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.show()
