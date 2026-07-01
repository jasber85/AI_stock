import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. 資料載入與預處理
# ==========================================
print("正在下載股票資料...")
# 下載台積電 (2330.TW) 資料
df = yf.download('2330.TW', start='2020-01-01', end='2026-06-25')

# 只要收盤價，並確保保留日期索引方便後續繪圖
df_close = df[['Close']]
dataset = df_close.values

# 將資料切分為：80% 訓練集，20% 測試集
training_data_len = int(np.ceil(len(dataset) * 0.8))

# 建立歸一化工具 (僅用訓練集去 fit，避免資訊洩漏)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# ==========================================
# 2. 建立時間窗口資料集 (Time Step = 60)
# ==========================================
time_step = 60

# 訓練集
train_data = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []
for i in range(time_step, len(train_data)):
    x_train.append(train_data[i-time_step:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 測試集 (需要包含前 60 天的資料作為基礎線索)
test_data = scaled_data[training_data_len - time_step:, :]
x_test = []
y_test = dataset[training_data_len:, :] # 這是真實的股價答案 (未歸一化)

for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# ==========================================
# 3. 建立並訓練 LSTM 模型
# ==========================================
print("開始建立並訓練模型...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
# 為了示範速度，這裡設 10 個 epochs，實戰中可增加
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

# ==========================================
# 4. 進行預測與反歸一化
# ==========================================
print("模型訓練完成，正在進行預測...")
predictions = model.predict(x_test)
# 關鍵：將預測出的 0~1 數值轉換回真正的股價
predictions = scaler.inverse_transform(predictions)

# ==========================================
# 5. 使用 Matplotlib 繪製對比圖表
# ==========================================
# 切分出繪圖所需的 DataFrame 區段
train_plot = df_close.iloc[:training_data_len].copy()
valid_plot = df_close.iloc[training_data_len:].copy()
valid_plot['Predictions'] = predictions # 將預測結果塞入對應的測試集區段

plt.figure(figsize=(14, 7))
plt.title('LSTM Stock Price Prediction - TSMC (2330)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (TWD)', fontsize=12)

# 畫出三條線：訓練歷史、測試集真實股價、測試集預測股價
plt.plot(train_plot.index, train_plot['Close'], label='Train Data', color='gray', alpha=0.6)
plt.plot(valid_plot.index, valid_plot['Close'], label='Actual Price', color='blue', linewidth=1.5)
plt.plot(valid_plot.index, valid_plot['Predictions'], label='Predicted Price', color='orange', linestyle='--', linewidth=1.5)

plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()