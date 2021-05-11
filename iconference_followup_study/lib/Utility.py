import pandas as pd

seed = 77


def shape(df):
    return len(df), len(df.columns)


def print_unique_count(df):
    df_unique = pd.DataFrame()
    for col_name in df.columns:
        df_unique[col_name] = [len(df[col_name].unique())]

    df_unique['total'] = [len(df)]
    df_unique.index = ['unique count']
    return df_unique.T.iloc[:, 0]


def print_na_count(df):
    df_na = pd.DataFrame()
    for col_name in df.columns:
        df_na[col_name] = [df[col_name].isna().sum()]

    df_na['total'] = [len(df)]
    df_na.index = ['na count']
    return df_na.T.iloc[:, 0]


