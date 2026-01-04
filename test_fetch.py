import requests
import pandas as pd
from io import StringIO
import datetime

def fetch_data_csv_yahoo(ticker: str, start: str, end: str):
    try:
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int(pd.Timestamp(end).timestamp())
    except Exception as e:
        print('Invalid date', e)
        return
    download_url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        session.get(f'https://finance.yahoo.com/quote/{ticker}/history?p={ticker}', headers=headers, timeout=10)
    except Exception:
        pass
    try:
        resp = session.get(download_url, headers=headers, timeout=10)
    except Exception as e:
        print('Request error', e)
        return
    print('STATUS', resp.status_code)
    txt = resp.text
    print('LEN', len(txt))
    print(txt[:800])

if __name__ == '__main__':
    fetch_data_csv_yahoo('AAPL', '2020-01-01', datetime.datetime.today().strftime('%Y-%m-%d'))
