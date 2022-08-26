import pandas as pd
import elexon
import json
from pathlib import Path

with open('secrets.json', 'r') as f:
    api_key = json.load(f)['elexon-scripting-key']

api = elexon.ElexonRawClient(api_key)
elexon_data = Path('./elexon_data')
elexon_data.mkdir(exist_ok=True)

def get_imbalance_data(date):
    res = api.Transparency.B1770(SettlementDate=date, Period='*')
    df = pd.DataFrame(res)
    return df

def get_volume_data(date):
    res = api.Transparency.B1780(SettlementDate=date, Period='*')
    df = pd.DataFrame(res)
    return df

# from_date = '2022-01-01'; to_date = '2022-07-05'
from_date = '2022-01-01'; to_date = '2022-08-01'
def fetch_data():
    daddy_df = None
    ds = pd.date_range(from_date, to_date)
    lends = len(ds)
    for i, date in enumerate(ds):
        try:
            print(f"{date} done, {i}/{lends}")
            df_im = get_imbalance_data(date)
            df_vol = get_volume_data(date)
            df = pd.concat([df_im, df_vol], sort=False)
            if daddy_df is None:
                daddy_df = df
            else:
                daddy_df = pd.concat([daddy_df, df], sort=False)
        except Exception as e:
            print(f"{date} failed: {e}")
            return daddy_df
    return daddy_df

def save_df(daddy_df, fname):
    try:
        daddy_df.to_csv(fname)
    except:
        print("Failed to save data")
