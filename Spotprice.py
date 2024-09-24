
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

