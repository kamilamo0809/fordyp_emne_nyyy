import pandas as pd
import numpy as np

def read_spotprice(filename):
    data = pd.read_excel(filename)
    spotprice = data['NO3']

    # Multiply all values by 1000
    spotprice = spotprice * 1000

    return spotprice


def read_consumption(filename):
    data=pd.read_csv(filename)

    # Drop columns where all elements are NaN to clean up the data
    data = data.dropna(axis = 1, how = 'all')

    # Include only the consumption in the dataframe
    consumption_df = data[['Consumption']]
    consumption_add = consumption_df[7800:8280] * 1.2
    consumption_add.index = range(8280, 8760)
    consumption_df_updated = pd.concat([consumption_df, consumption_add])

    return consumption_df_updated


def merge_lists(file1, file2, file3):
    data1 = pd.read_excel(file1)
    data2 = pd.read_excel(file2)
    data3 = pd.read_excel(file3)

    df = pd.concat([data1['NO3'], data2['NO3'], data3['NO3']], axis=1)
    df.columns = [1, 2, 3]

    df = df * 1000

    return df

def consumption_3_scenarios(filename):
    df_demand = read_consumption(filename)
    # Step 1: Create a new DataFrame by duplicating the 'Consumption' column three times
    df_demand = pd.concat([df_demand['Consumption']] * 3, axis = 1)

    # Step 2: Rename the columns to 1, 2, and 3
    df_demand.columns = [1, 2, 3]

    return df_demand

