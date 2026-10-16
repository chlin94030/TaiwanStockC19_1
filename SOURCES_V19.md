# 資料來源與成本設定參考
查閱：2026-10-16。下列是來源文件，不代表已實測所有API可達、具有商業再散布授權或完整PIT保證。

- FinMind官方快速開始：https://finmind.github.io/quickstart/
- FinMind日線及資料集說明：https://finmind.github.io/tutor/TaiwanMarket/Technical/
- 證交所股票個股日成交資訊：https://www.twse.com.tw/zh/trading/historical/stock-day.html
- 本包21筆真實解析樣本：https://www.twse.com.tw/exchangeReport/STOCK_DAY?date=20250901&response=json&stockNo=2330
- 證交所ETF交易規則（0050賣出稅設定參考）：https://www.twse.com.tw/zh/products/securities/etf/overview/rules.html

`v19_twse_sample.json`只作解析測試；來源回覆日期、價格及數量保留，不能據以計算十日期多股票研究。Yahoo備援為第三方日線資料，非證交所即時認證。個人研究請依資料供應商使用及授權條件；遇權限錯誤或限流，程式停止該來源，不嘗試繞過。
