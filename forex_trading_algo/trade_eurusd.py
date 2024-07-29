import json
import oandapyV20
from oandapyV20 import API   
from oandapyV20.endpoints import instruments, accounts, orders, trades, positions
import oandapyV20.endpoints.pricing as pricing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import time
import pickle
import xgboost as xgb


ACCESS_TOKEN = "4aa51e7711418ba1aa356b835fab3afd-53be4f4d5aa21768bf421b354527dc96"
ACCOUNT_ID = "101-004-16909090-001"
client = API(access_token=ACCESS_TOKEN)


def get_prices(to_date: str, count: int = 5,instrument: str = "EUR_USD", granularity: str = "H1"):  
  params = {
    "count": count,
    "granularity": granularity,
    "to": to_date
    }
  r = instruments.InstrumentsCandles(instrument=instrument, params=params)
  return client.request(r)



def make_df(data: dict):
  data = pd.DataFrame(data["candles"])
  candles = pd.json_normalize(data["mid"]).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
  data = data.join(candles, how = "left").drop(["mid"], axis = 1)
  data["time"] = pd.to_datetime(data["time"])
  data = data.apply(lambda x: x.astype(float) if x.name in ["open", "high", "low", "close"] else x, axis = 0)
  return data

def get_to_date_in_utc(day_diff: int = 0):
    return (datetime.utcnow().replace(microsecond=0,second=0,minute=0) - timedelta(days=day_diff)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def is_market_open():
  return datetime.now().weekday() not in [5, 6]

def get_signal(data):
  # calculate features
  data["1h_spread"] = data["close"].diff() * 10000
  data["2h_spread"] = data["close"].diff(2) * 10000
  data["3h_spread"] = data["close"].diff(3) * 10000
  data["ma2_1h_spread"] = data["1h_spread"].rolling(window=2).mean()
  # load the model
  models = pickle.load(open("models/models.pkl", "rb"))
  # predict signal
  ensemble_prediction = 0
  for model in models:
     feature = model["feature"]
     mod = model["model"]
     feature_data = xgb.DMatrix(data[feature].iloc[-1].reshape(1, -1))
     ensemble_prediction += mod.predict(feature_data)[-1]
  return np.sign(ensemble_prediction / len(models))


def trade():
    if is_market_open():
        open_trade_request = trades.OpenTrades(accountID=ACCOUNT_ID)
        open_trade = client.request(open_trade_request)
        open_trade = open_trade["trades"]
        if open_trade:
            # Get the ID of the first open trade
            trade_id = open_trade[0]["id"]
            
            # Step 2: Close the trade
            close_trade_request = trades.TradeClose(accountID=ACCOUNT_ID, tradeID=trade_id)
            close_trade_response = client.request(close_trade_request)
            
            # Print the response
            print(f'TRADE CLOSED: {close_trade_response}')

        
        to_date = get_to_date_in_utc()
        data = make_df(
            get_prices(to_date=to_date)
            )
        signal = get_signal(data)
        quantity = 10000 if signal > 0 else -10000
        # Order details
        order_data = {
            "order": {
                "instrument": "EUR_USD",  # Replace with your desired instrument
                "units": quantity,  # Number of units to buy (positive for buy, negative for sell)
                "type": "MARKET",  # Order type
                "positionFill": "DEFAULT"  # Order position fill option
            }
        }
        trade_request = orders.OrderCreate(accountID=ACCOUNT_ID, data=order_data)
        trade = client.request(trade_request)
        action = "Bought" if signal > 0 else "Sold"
        print(f'{action}: {trade}')


schedule.every(4).hours.at(":00").do(trade)
#schedule.every(2).minutes.do(trade)
print("Trading algorithm started.")
while True:
    # Run all pending jobs
    schedule.run_pending()
    time.sleep(1)