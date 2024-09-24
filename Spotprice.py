import pandas as pd
import numpy as np

def read_spotprice(filename):
    data = pd.read_excel(filename)
    spotprice = data['NO3']

    return spotprice

def read_consumption(filename):
    data=pd.read_csv(filename)
    consumption=data['Consumption']
    consumption_add = consumption[7800:8280] * 1.2
    consumption = np.append(consumption, consumption_add)

    return consumption



def merge_lists(file1, file2, file3):
    data1 = pd.read_excel(file1)
    data2 = pd.read_excel(file2)
    data3 = pd.read_excel(file3)
    df = pd.concat([data1['NO3'], data2['NO3'], data3['NO3']], axis=1)
    df.columns = [1, 2, 3]

    return df