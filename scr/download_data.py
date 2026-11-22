from pathlib import Path
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Download S&P 500 daily candles
root = Path(__file__).resolve().parent.parent
data_dir = root / "data"
results_dir = root / "results"
data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

prices = yf.download(
    "^GSPC", start="2016-01-01", interval="1d",
    auto_adjust=False, multi_level_index=False, progress=False
)
if prices.empty:
    raise ValueError("No S&P 500 data downloaded. Try again later.")
prices = prices[["Open", "High", "Low", "Close"]].dropna().copy()
prices.to_csv(data_dir / "sp500_historical.csv")

# 2. Describe today's candle using only information known at today's close
prices["Return"] = prices["Close"].pct_change()
prices["Body"] = (prices["Close"] - prices["Open"]) / prices["Open"]
prices["Range"] = (prices["High"] - prices["Low"]) / prices["Open"]
prices["MA5"] = prices["Close"].rolling(5).mean() / prices["Close"]
prices["MA10"] = prices["Close"].rolling(10).mean() / prices["Close"]
features = ["Return", "Body", "Range", "MA5", "MA10"]

# Tomorrow's candle is the prediction target (1 = bullish, 0 = bearish)
prices["Target"] = (prices["Close"].shift(-1) > prices["Open"].shift(-1)).astype(int)
history = prices.iloc[:-1].dropna(subset=features).copy()

# 3. Train on the earlier 80%, test on the later 20%
split = int(len(history) * 0.8)
train, test = history.iloc[:split], history.iloc[split:]
model = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42
)
model.fit(train[features], train["Target"])
predicted = model.predict(test[features])
accuracy = accuracy_score(test["Target"], predicted)

# Simple benchmark: always predict the direction most common in training
baseline_direction = int(train["Target"].mean() >= 0.5)
baseline = accuracy_score(test["Target"], [baseline_direction] * len(test))

backtest = pd.DataFrame({
    "Prediction_made_after": test.index,
    "Actual_direction": test["Target"].map({1: "Bullish", 0: "Bearish"}).to_numpy(),
    "Predicted_direction": pd.Series(predicted).map({1: "Bullish", 0: "Bearish"}).to_numpy(),
    "Correct": predicted == test["Target"].to_numpy(),
})
backtest.to_csv(results_dir / "backtest_predictions.csv", index=False)

# 4. Save a simple performance chart
rolling_accuracy = backtest["Correct"].rolling(50).mean() * 100
plt.figure(figsize=(10, 5))
plt.plot(test.index, rolling_accuracy, label="Model: last 50 predictions")
plt.axhline(baseline * 100, linestyle="--", label="Simple baseline")
plt.title("S&P 500 next-day candle prediction: historical backtest")
plt.ylabel("Accuracy (%)")
plt.xlabel("Date prediction was made after")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "backtest_chart.png", dpi=160)
plt.close()

# 5. Evaluate any previously recorded forward predictions
log_path = results_dir / "forward_predictions.csv"
columns = ["Based_on_date", "Prediction", "Model_probability", "Actual", "Correct"]
if log_path.exists():
    log = pd.read_csv(log_path).fillna("")
else:
    log = pd.DataFrame(columns=columns)

for i, entry in log.iterrows():
    if str(entry["Actual"]).strip():
        continue
    later = prices.loc[prices.index > pd.Timestamp(entry["Based_on_date"])]
    if not later.empty:
        candle = later.iloc[0]
        actual = "Bullish" if candle["Close"] > candle["Open"] else "Bearish"
        log.loc[i, "Actual"] = actual
        log.loc[i, "Correct"] = str(actual == entry["Prediction"])

# 6. Retrain on all available completed historical targets, predict next candle
model.fit(history[features], history["Target"])
latest = prices.dropna(subset=features).iloc[[-1]]
prob_up = model.predict_proba(latest[features])[0, 1]
direction = "Bullish" if prob_up >= 0.5 else "Bearish"
probability = prob_up if direction == "Bullish" else 1 - prob_up
based_on = latest.index[-1].strftime("%Y-%m-%d")

# Do not overwrite a prediction already recorded for this observation date
if based_on not in log["Based_on_date"].astype(str).values:
    log.loc[len(log)] = [based_on, direction, round(float(probability), 4), "", ""]
log.to_csv(log_path, index=False)

# 7. Display the results
print(f"Historical backtest: {accuracy:.2%} ({sum(predicted == test['Target'])}/{len(test)})")
print(f"Simple baseline:     {baseline:.2%}")
print(f"Latest candle date:  {based_on}")
print(f"Next candle:         {direction}")
print(f"Model probability:   {probability:.2%} (not calibrated)")
checked = log[log["Actual"].astype(str).str.len() > 0]
if not checked.empty:
    forward_accuracy = (checked["Correct"].astype(str) == "True").mean()
    print(f"Forward accuracy:    {forward_accuracy:.2%} ({len(checked)} checked predictions)")
print("Files saved to data/ and results/")