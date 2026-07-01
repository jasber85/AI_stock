import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. 多變量資料準備 (加入成交量與 5日均線)
# ==========================================
print("正在載入多變量資料...")
df = yf.download('2330.TW', start='2021-01-01', end='2026-06-25')

# 計算 5 日移動平均線 (MA5)
df['MA5'] = df['Close'].rolling(window=5).mean()
df.dropna(inplace=True) # 刪除因為計算 MA 產生的空值

# 挑選 3 個特徵：收盤價、成交量、MA5
features = ['Close', 'Volume', 'MA5']
df_features = df[features]
dataset = df_features.values

# 切分訓練集與測試集 (80% / 20%)
training_data_len = int(np.ceil(len(dataset) * 0.8))

# 對 3 個欄位獨立進行歸一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# ==========================================
# 2. 建立 3D 時間窗口 (特徵數量 = 3)
# ==========================================
time_step = 60
feature_count = len(features) # 3

# 建立訓練集
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
for i in range(time_step, len(train_data)):
    x_train.append(train_data[i-time_step:i, :]) # 抓取 3 個欄位過去 60 天
    y_train.append(train_data[i, 0])             # 預測目標依然是「收盤價」(索引 0)

x_train, y_train = np.array(x_train), np.array(y_train)

# 建立測試集
test_data = scaled_data[training_data_len - time_step:, :]
x_test = []
y_test = dataset[training_data_len:, 0] # 測試集真實的收盤價答案

for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, :])

x_test = np.array(x_test)

print(f"訓練集形狀 (樣本數, 時間步長, 特徵數): {x_train.shape}") # 應該是 (N, 60, 3)

# ==========================================
# 3. 升級 LSTM 模型架構
# ==========================================
model = Sequential([
    # input_shape 調整為 (60, 3)
    LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], feature_count)),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation='relu'), # 增加一個密集的特徵融合層
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1)

# ==========================================
# 4. 預測與多變量反歸一化 (核心難點)
# ==========================================
predictions = model.predict(x_test)

# 因為 scaler 當初是吃 3 個欄位，反轉換時也必須給它 3 個欄位
# 我們建立一個虛擬的矩陣，把預測值放第一欄，其餘補 0
prediction_copies = np.repeat(predictions, feature_count, axis=-1)
# 透過 inverse_transform 還原，再取出第一欄 (收盤價)
predictions_inverse = scaler.inverse_transform(prediction_copies)[:, 0]

# ==========================================
# 5. 科學數據評估
# ==========================================
rmse = np.sqrt(mean_squared_error(y_test, predictions_inverse))
mape = mean_absolute_percentage_error(y_test, predictions_inverse)

print("\n====== 模型定量評估 ======")
print(f"均方根誤差 (RMSE): {rmse:.2f} TWD (平均預測誤差幾元)")
print(f"平均絕對百分比誤差 (MAPE): {mape * 100:.2f} %")

# ==========================================
# 終極修正版：強制扁平化為 1D 陣列，防止維度爆炸
# ==========================================

# 1. 確保 valid_plot 建立時，Close 欄位被強制降維成 1D 陣列
valid_plot = pd.DataFrame(index=df.iloc[training_data_len:].index)

# 使用 .flatten() 確保它們都是形狀為 (264,) 的 1D 陣列
valid_plot['Actual_Close'] = dataset[training_data_len:, 0].flatten()
valid_plot['Predictions'] = predictions_inverse.flatten()

# 為了讓你看程式碼比較直覺，我們宣告成獨立的 1D 變數
actual_price = valid_plot['Actual_Close'].values
predicted_price = valid_plot['Predictions'].values

# ==========================================
# 1. 產生交易訊號 (Signal)
# ==========================================
# 策略：如果預測明天的價格 > 今天的實際價格 -> 訊號 = 1
# 因為我們要拿「今天的實際價格」來比，所以實際價格要挪動一位 (.shift(1) 或用切片)
# 這裡用 numpy 處理 1D 陣列最安全：
actual_today = actual_price[:-1]       # 今天
predicted_tomorrow = predicted_price[1:] # 明天

# 建立一個與 valid_plot 長度相同的訊號陣列，預設為 0
signals = np.zeros(len(valid_plot))
# 如果明天預測會漲，今天收盤就發出買入訊號 (1)
signals[:-1] = np.where(predicted_tomorrow > actual_today, 1, 0)
valid_plot['Signal'] = signals

# ==========================================
# 2. 計算勝率 (Directional Accuracy)
# ==========================================
# 實際市場漲跌：明天實際收盤 > 今天實際收盤
actual_direction = np.where(actual_price[1:] > actual_price[:-1], 1, 0)

# 模型是否預測對方向 (比較預測方向與實際方向)
predicted_direction = signals[:-1]
correct_predictions = (predicted_direction == actual_direction)

win_rate = np.mean(correct_predictions)

# ==========================================
# 3. 計算策略報酬率 vs. 買進持有 (Buy & Hold)
# ==========================================
# 計算市場每日報酬率 (使用 pandas 處理百分比變動)
valid_plot['Market_Return'] = valid_plot['Actual_Close'].pct_change()

# 策略報酬率：昨天的訊號 * 今天的市場報酬
valid_plot['Strategy_Return'] = valid_plot['Signal'].shift(1) * valid_plot['Market_Return']

# 計算累積報酬率
valid_plot['Cum_Market_Return'] = (1 + valid_plot['Market_Return'].fillna(0)).cumprod() - 1
valid_plot['Cum_Strategy_Return'] = (1 + valid_plot['Strategy_Return'].fillna(0)).cumprod() - 1

market_roi = valid_plot['Cum_Market_Return'].iloc[-1]
strategy_roi = valid_plot['Cum_Strategy_Return'].iloc[-1]

# ==========================================
# 4. 輸出量化報告與繪圖
# ==========================================
print("\n========== LSTM 量化回測報告 (維度修正版) ==========")
print(f"測試交易天數: {len(valid_plot) - 1} 天")
print(f"模型預測漲跌勝率: {win_rate * 100:.2f} %")
print(f"【買進持有】總投資報酬率 (ROI): {market_roi * 100:.2f} %")
print(f"【LSTM 策略】總投資報酬率 (ROI): {strategy_roi * 100:.2f} %")
print("====================================================")

plt.figure(figsize=(14, 6))
plt.plot(valid_plot.index, valid_plot['Cum_Market_Return'] * 100, label='Buy & Hold (Market)', color='blue')
plt.plot(valid_plot.index, valid_plot['Cum_Strategy_Return'] * 100, label='LSTM AI Strategy', color='green')
plt.title('Backtesting: 1D Aligned LSTM Strategy vs. Market', fontsize=14)
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()