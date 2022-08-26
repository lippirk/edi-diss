import pandas as pd

df_21_im = pd.read_csv('./elexon_data/elexon-data-2021.csv')
df_22_im = pd.read_csv('./elexon_data/elexon-data-2022-up-to-july.csv')
df_21_vol = pd.read_csv('./elexon_data/elexon-data-2021-volume.csv')
df_22_jan_both = pd.read_csv('./elexon_data/jan-22-dirty.csv')

jan_22_is_im = df_22_jan_both['imbalanceQuantityMAW'].isna()
jan_22_is_vol = df_22_jan_both['imbalancePriceAmountGBP'].isna()
df_22_jan_im = df_22_jan_both[jan_22_is_im]
df_22_jan_vol = df_22_jan_both[jan_22_is_vol]

def clean_im_df(df):
    groups = df.groupby(['settlementPeriod', 'settlementDate'])

    for _tup, g in groups:
        price = g['imbalancePriceAmountGBP']
        categories = set(g['priceCategory'])
        # check prices are consistent
        assert len(price) == 2
        assert price.iloc[0] == price.iloc[1]
        # check categories are as expected
        assert len(categories) == 2 and categories == {'Excess balance', 'Insufficient balance'}

    df = (groups.first()
            .reset_index()
            .sort_values(['settlementDate', 'settlementPeriod'])
            .reset_index())
    df = df[['settlementDate',
             'settlementPeriod',
             'imbalancePriceAmountGBP']]
    return df

def clean_vol_df(df_vol):
    groups = df_vol.groupby(['settlementPeriod', 'settlementDate'])

    for _tup, g in groups:
        quantity = g['imbalanceQuantityMAW']
        direction = g['imbalanceQuantityDirection']
        assert len(quantity) == 1
        assert len(direction) == 1
        quantity = list(quantity)[0]
        direction = list(direction)[0]
        if quantity > 0:
            assert direction == 'SURPLUS'
        elif quantity < 0:
            assert direction == 'DEFICIT'

    df_vol = (groups.first()
                    .reset_index()
                    .sort_values(['settlementDate', 'settlementPeriod'])
                    .reset_index())
    df_vol = df_vol[['settlementDate',
                 'settlementPeriod',
                 'imbalanceQuantityMAW']]
    return df_vol

def clean_dfs(df_im, df_vol):
    df_im = clean_im_df(df_im)
    df_vol = clean_vol_df(df_vol)
    df = pd.merge(df_im, df_vol, on=['settlementDate', 'settlementPeriod'],
                  how='outer')
    return df

# clean_dfs(df_21_im,df_21_vol).to_csv('./elexon_data/elexon-clean-2021.csv')
# clean_im_df(df_22_im).to_csv('./elexon_data/elexon-clean-2022.csv')
clean_dfs(df_22_jan_im, df_22_jan_vol).to_csv('./elexon_data/elexon-clean-jan-22.csv')
