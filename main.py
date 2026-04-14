import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/bitcoin_price_history_2010_2025.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=True)
df.set_index('date', inplace=True)
df.columns = df.columns.str.lower()
df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

print(df.head()) 
print(df.shape)
print(df.describe())
print(df.info())

plt.figure(figsize=(15, 5))
plt.plot(df['close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

df['open-close'] = df['open'] - df['close']
df['low-high'] = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0) # Directional change

df['Close_Lag1'] = df['close'].shift(1)
df['Close_Lag5'] = df['close'].shift(5)
df['year'] = df.index.year
df['month'] = df.index.month

df.dropna(inplace=True)

plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

data_grouped = df.groupby(df.index.year).mean(numeric_only=True)

plt.subplots(figsize=(20,10))
for i, col in enumerate(['open', 'high', 'low', 'close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar(title=f'Mean {col.title()} Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Mean Price in Dollars')
plt.tight_layout()
plt.show()

X = df[['Close_Lag1', 'Close_Lag5', 'year', 'month']]
Y = df['close']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, shuffle=False
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("R2 Score:", model.score(X_valid, Y_valid))

y_pred = model.predict(X_valid)
print("MSE:", mean_squared_error(Y_valid, y_pred))

plt.figure(figsize=(12, 6))

plt.plot(Y_valid.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")

plt.title("Actual vs Predicted Bitcoin Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.savefig("prediction_vs_actual.png")
plt.show()

last_known_close = df['close'].iloc[-1]
last_known_date = df.index[-1]

def get_single_price_prediction_recursive(date_str, scaler, trained_model, last_price, last_date):
    try:
        prediction_date = pd.to_datetime(date_str)
        current_date = last_date + pd.Timedelta(days=1)
        predicted_price = last_price
        
        while current_date <= prediction_date:
            
            pred_year = current_date.year
            pred_month = current_date.month
            close_lag_1 = predicted_price 
            
            close_lag_5_date = current_date - pd.Timedelta(days=5)
            try:
                close_lag_5 = df.loc[df.index <= close_lag_5_date, 'close'].iloc[-1]
            except IndexError:
                close_lag_5 = last_price 

            input_data = np.array([[close_lag_1, close_lag_5, pred_year, pred_month]])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_input = scaler.transform(input_data)
            
            predicted_price = trained_model.predict(scaled_input)[0]
            current_date += pd.Timedelta(days=1)
        
        return f"{predicted_price:,.2f}"

    except Exception:
        return "Error: Invalid date format. Use YYYY-MM-DD."

while True:
    user_date_input = input("Enter the date (YYYY-MM-DD) you want to PREDICT the price for (or 'exit'): ")
    
    if user_date_input.lower() == 'exit':
        break
    
    predicted_price_str = get_single_price_prediction_recursive(
        user_date_input, scaler, model, last_known_close, last_known_date
    )
    
    print(f"\nPredicted BTC Price: ${predicted_price_str}")

print("Exiting prediction tool.")