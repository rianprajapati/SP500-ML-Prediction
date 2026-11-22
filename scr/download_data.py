import yfinance as yf
from pathlib import Path

# 1. Download historical S&P 500 daily prices
data = yf.download(
    "^GSPC",
    start="2016-01-01",
    interval="1d",
    auto_adjust=False,
    multi_level_index=False,
    progress=False
)

# 2. Check that data was downloaded
if data.empty:
    raise ValueError("No market data was downloaded.")

# 3. Keep the four candlestick prices
data = data[["Open", "High", "Low", "Close"]].copy()

# 4. Identify bullish, bearish and neutral candles
data["Direction"] = "Neutral"
data.loc[data["Close"] > data["Open"], "Direction"] = "Bullish"
data.loc[data["Close"] < data["Open"], "Direction"] = "Bearish"

# 5. Create a folder and save the data
folder = Path("data")
folder.mkdir(exist_ok=True)

data.to_csv(folder / "sp500_historical.csv")

# 6. Display the latest five candles
print(data.tail())
print("\nHistorical data saved successfully!")