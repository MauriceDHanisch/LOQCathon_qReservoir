# Samuel Rey
# Maurice Hanisch mhanisc@eth.ch 

import numpy as np
import pandas as pd



def load_data_into_matrix_edf(file_name : str, m : int) -> np.array:
    """Loads the data from the file and returns a matrix of shape (t, (m-1)*5)"""
    try:
        df = pd.read_csv("wind_data_fr_2021.csv", sep=";")
    except FileNotFoundError:
        print("File not found. Please make sure the file is in the same folder as the Notebook.")
        return None
    
    df = df.iloc[:, 1:]
    da = df.to_numpy()
    # rescale of the data
    data_scaled_down = da / max(da) * 2*np.pi # because mapped to angles

    n = (m-1)*5
    new_data = np.zeros((data_scaled_down.shape[0], data_scaled_down.shape[1]*n))
    for i in range(n):
        new_data[:, i*data_scaled_down.shape[1]:(i+1)*data_scaled_down.shape[1]] = data_scaled_down

    return new_data

