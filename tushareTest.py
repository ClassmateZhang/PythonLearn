import tushare as ts
import pandas as pd

'''
股票相关
'''
ts.set_token("a37ea9575adf46d3794c5edee49fbc98631cc80a7c6f7e5b3cf90f19")
pro = ts.pro_api()
#获取A股2019年6月份交易日
# df = pro.trade_cal(exchange='', start_date='20190601', end_date='20190701', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
# print(df)

# 获取当前所有正常交易的股票列表
#stockData = pro.query('stock_basic',exchang='SSE',list_status='L',fields='ts_code,symbol,name,area,industry,list_date')
#print(stockData)
#获取当日行情数据
df = pro.query('daily', ts_code='000001.SZ', start_date='20190610', end_date='20190612')
print(df)

