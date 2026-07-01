import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 確保 valid_plot 欄位正確，並去除任何可能的空值
# 欄位應包含: 'Close' (當天實際收盤價), 'Predictions' (模型預測的明天股價)
backtest_df = valid_plot.copy()

# ==========================================
# 1. 產生交易訊號 (Signal)
# ==========================================
# 策略：如果預測明天的價格 > 今天的實際價格 -> 明天看漲 (訊號 = 1)，否則看跌 (訊號 = 0)
# 註：這裡使用 shift(1) 是因為預測值是針對「明天」的
backtest_df['Signal'] = np.where(backtest_df['Predictions'] > backtest_df['Close'].shift(1), 1, 0)

# ==========================================
# 2. 計算勝率 (Directional Accuracy)
# ==========================================
# 實際市場漲跌方向：今天實際收盤 > 昨天實際收盤
backtest_df['Actual_Direction'] = np.where(backtest_df['Close'] > backtest_df['Close'].shift(1), 1, 0)

# 模型是否預測對方向？ (當預測方向與實際漲跌一致時為 True)
# 注意：訊號是昨天預測今天的，所以要跟今天的實際漲跌比對
backtest_df['Predict_Correct'] = (backtest_df['Signal'].shift(1) == backtest_df['Actual_Direction'])

# 計算勝率 (排除第一天因為 shift 產生的 NaN)
valid_days = backtest_df.dropna()
win_rate = valid_days['Predict_Correct'].mean()

# ==========================================
# 3. 計算策略報酬率 vs. 買進持有 (Buy & Hold)
# ==========================================
# 計算市場每日報酬率
backtest_df['Market_Return'] = backtest_df['Close'].pct_change()

# 計算策略每日報酬率：如果昨天發出買入訊號(1)，今天就享受市場報酬；若為(0)則手握現金，報酬為 0
backtest_df['Strategy_Return'] = backtest_df['Signal'].shift(1) * backtest_df['Market_Return']

# 計算累積報酬率 (利用連乘效應 cumprod)
backtest_df['Cum_Market_Return'] = (1 + backtest_df['Market_Return'].fillna(0)).cumprod() - 1
backtest_df['Cum_Strategy_Return'] = (1 + backtest_df['Strategy_Return'].fillna(0)).cumprod() - 1

# 提取最終投資報酬率 (ROI)
market_roi = backtest_df['Cum_Market_Return'].iloc[-1]
strategy_roi = backtest_df['Cum_Strategy_Return'].iloc[-1]

# ==========================================
# 4. 輸出量化報告
# ==========================================
print("\n========== LSTM 量化回測報告 ==========")
print(f"測試交易天數: {len(valid_days)} 天")
print(f"模型預測漲跌勝率: {win_rate * 100:.2f} %")
print(f"【買進持有】總投資報酬率 (ROI): {market_roi * 100:.2f} %")
print(f"【LSTM 策略】總投資報酬率 (ROI): {strategy_roi * 100:.2f} %")
print("=======================================")

# ==========================================
# 5. 繪製績效走勢圖
# ==========================================
plt.figure(figsize=(14, 6))
plt.plot(backtest_df.index, backtest_df['Cum_Market_Return'] * 100, label='Buy & Hold (Market)', color='blue')
plt.plot(backtest_df.index, backtest_df['Cum_Strategy_Return'] * 100, label='LSTM AI Strategy', color='green', linestyle='-')
plt.title('Backtesting: LSTM AI Strategy vs. Buy & Hold', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()