import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- MAKE CSV FILES ----------------------------------

def Generate_consumption(filename):

    # Generate a list of years from 2020 to 2060
    years = list(range(2020, 2061))

    def generate_peak_profile(start_value, mid_value, end_value, years):
        midpoint = len(years) // 3
        profile = np.concatenate(
            [np.linspace(start_value, mid_value, midpoint), np.linspace(mid_value, end_value, len(years) - midpoint)])
        return profile

    # Setting lower and upper bounds in TWh
    start_consumption = 1.3
    mid_consumption = 2.3
    end_consumption = 2.6

    # Generates gradually increasing consumption between 1.3 TWh and 2.6 TWh
    consumption = generate_peak_profile(start_consumption, mid_consumption, end_consumption, years)

    # Adds some random variation to simulate realistic oscillations
    random_variation = np.random.uniform(-0.04, 0.04, len(years))  # tilfeldig variasjon mellom -0.05 og 0.05 TWh
    consumption = consumption + random_variation
    consumption = np.clip(consumption, start_consumption, end_consumption)  # Sikrer at verdiene holder seg mellom 1.3 og 2.0

    # Round to two decimals
    consumption = np.round(consumption, 2)

    # Make a dataframe
    data = pd.DataFrame({
        'Year': years,
        'Consumption (TWh)': consumption
    })

    # Write to CSV-file
    data.to_csv(filename, index=False)

    print(f"CSV-file {filename} is generated.")

def Generate_spot_prices(filename):
    # Genrate years from 2020 to 2060
    years = list(range(2020, 2061))

    def generate_peak_profile(start_value, peak_value, end_value, years):
        midpoint = len(years) // 3
        profile = np.concatenate(
            [np.linspace(start_value, peak_value, midpoint), np.linspace(peak_value, end_value, len(years) - midpoint)])
        return profile

    # Setter start-, topp- og sluttverdier for prisene
    low_price_start, low_price_peak, low_price_end = 0.50, 0.60, 0.40
    normal_price_start, normal_price_peak, normal_price_end = 0.50, 0.80, 0.50
    high_price_start, high_price_peak, high_price_end = 0.50, 1.0, 0.60

    # Genererer prisprofiler med topp i 2040
    low_prices = generate_peak_profile(low_price_start, low_price_peak, low_price_end, years)
    normal_prices = generate_peak_profile(normal_price_start, normal_price_peak, normal_price_end, years)
    high_prices = generate_peak_profile(high_price_start, high_price_peak, high_price_end, years)

    # Legger til litt tilfeldige variasjoner for å simulere realistiske svingninger
    low_prices += np.random.uniform(-0.01, 0.01, len(years))  # tilfeldig variasjon mellom -1 og 1 øre/kWh
    normal_prices += np.random.uniform(-0.02, 0.02, len(years))  # tilfeldig variasjon mellom -2 og 2 øre/kWh
    high_prices += np.random.uniform(-0.03, 0.03, len(years))  # tilfeldig variasjon mellom -3 og 3 øre/kWh

    # Runder av tallene til 2 desimaler
    low_prices = np.round(low_prices, 2)
    normal_prices = np.round(normal_prices, 2)
    high_prices = np.round(high_prices, 2)

    # Make a DataFrame
    data = pd.DataFrame(
        {'Year': years, 'Low prices (kr/kWh)': low_prices, 'Normal prices (kr/kWh)': normal_prices,
            'High prices (kr/kWh)': high_prices})

    # Write to CSV-file
    data.to_csv(filename, index = False)

    print(f"CSV-file {filename} is generated.")

# --------------------------- READ CSV FILES ----------------------------------

def Read_spot(filename):
    '''
    Reads data from specified csv file and processes it into structured data frames.

    :param filename: The name of the csv file containing the data.
    :return: A pandas DataFrames with the time stamp as indeces to the consumption data
    '''

    # Read data into dataframe
    df = pd.read_csv(filename)

    # Drop columns where all elements are NaN to clean up the data
    df = df.dropna(axis=1, how='all')

    # Set the time stamp as the index
    df.set_index('Year', inplace = True)

    # Include only the consumption in the dataframe
    consumption_df = df[['Low prices (kr/kWh)', 'Normal prices (kr/kWh)', 'High prices (kr/kWh)']]

    # Return the dataframe
    return consumption_df

def Read_consumption(filename):
    '''
    Reads data from specified csv file and processes it into structured data frames.

    :param filename: The name of the csv file containing the data.
    :return: A pandas DataFrames with the time stamp as indeces to the consumption data
    '''

    # Read data into dataframe
    df = pd.read_csv(filename)

    # Drop columns where all elements are NaN to clean up the data
    df = df.dropna(axis=1, how='all')

    # Set the time stamp as the index
    df.set_index('Year', inplace = True)

    # Include only the consumption in the dataframe
    consumption_df = df[['Consumption (TWh)']]

    # Return the dataframe
    return consumption_df


# --------------------------- PLOT DATA ----------------------------------

# Make data frames
df_consumption = Read_consumption('consumption_data.csv')
df_spot = Read_spot('spot_data.csv')

# Choose a color palet
sns.set_palette('Set2')

# --------------------------- CONSUMPTION ----------------------------------

sns.lineplot(df_consumption)
plt.title('Estimated consumption data for Trondheim from 2020 to 2060')
plt.xlabel('Year')
plt.ylabel('Consumption [TWh]')
plt.show()

# ----------------------------- PRICES ------------------------------------

sns.lineplot(df_spot)
plt.title('Estimated spot prices in Trondheim from 2020 to 2060')
plt.xlabel('Year')
plt.ylabel('Price [kr/kWh]')
plt.show()