import numpy as np
import pandas as pd
import random
import time
from scipy import stats


def floor_outliers(df, cap):
    df['appliance_power'] = df['appliance_power'].mask(df['appliance_power'] > cap, 0)
    return df


def load_all_houses_with_device(path, appliance):
    df = pd.read_csv(path, usecols=['dataid', 'grid', 'solar', 'solar2', str(appliance)])
    df = df.fillna(0)
    grouped = df[['dataid', str(appliance)]].groupby(df.dataid).sum()
    house_ids = grouped[grouped[str(appliance)] > 0].dataid.index.unique()
    df = df[df.dataid.isin(house_ids)]
    df = df.reset_index(drop=True)

    df['net_power'] = df.grid + df.solar + df.solar2
    df['net_power'] = np.clip(df['net_power']*1000, a_min=0, a_max=None)
    df['net_power'] = df['net_power'].mask(df['net_power'] > 30000, 0)

    df[str(appliance)] = np.clip(df[str(appliance)]*1000, a_min=0, a_max=None)
    df['appliance_power'] = df[str(appliance)]

    df = df.drop(columns=['grid', 'solar', 'solar2', str(appliance)])
    df['appliance_power'] = df['appliance_power'].values.astype('float32')
    df['net_power'] = df['net_power'].values.astype('float32')

    remove = []
    cap = []
    if appliance == "air1":
        cap = 5000
        # remove = [1249, 7365, 8849]
    if appliance == 'clotheswasher1':
        cap = 1300
        # remove = [6240]
    # if appliance == 'drye1':
    #    remove = [183]
    if appliance == 'furnace1':
        cap = 1000
        # remove = [6703, 3403, 7365, 2126, 10089]
    if appliance == 'oven1':
        cap = 5000
        # remove = [792]
    if appliance == 'range1':
        cap = 8000
        # remove = [7367]
    if appliance == 'dishwasher1':
        cap = 1500
        # remove = [8849, 6240, 7367]
    # if appliance == 'freezer1':
    # remove = [9973, 7159]
    # if appliance == 'heater1':
    # remove = [2561]
    if appliance == 'waterheater1':
        cap = 6000
        # remove = [10983, 9973, 8627, 2096]
    if appliance == 'waterheater2':
        cap = 5000
        # remove = [8627]
    if appliance == 'wellpump1':
        cap = 1600
        # remove = [10983]
    if appliance == 'clotheswasher_dryg1':
        cap = 800
    if appliance == 'refrigerator1':
        cap = 2000
        # remove = [145,
        #           3344,
        #           4628,
        #           10554,
        #           950,
        #           984,
        #           1240,
        #           1249,
        #           5192,
        #           5367,
        #           5982,
        #           6069,
        #           6564,
        #           6594,
        #           6703,
        #           6907,
        #           7935,
        #           8162,
        #           8627,
        #           8849,
        #           9002,
        #           11421,
        #           11785
        #           ]
        #remove = [145, 3344, 4628, 10554]
        remove = [
            145,
            690,
            950,
            984,
            1249,
            2561,
            3344,
            3700,
            4628,
            5367,
            5982,
            6907,
            7021,
            8162,
            8849,
            9973,
            10164,
            10554
        ]
        # keep = [  142,   183,   335,   387,   526,  1240,  1417,  2126,  2358,
        #      3383,  3488,  3976,  3996,  5058,  5192,  6069,  6178,  6240,
        #      6526,  6564,  6594,  6672,  6703,  7069,  7365,  7935,  8627,
        #      8825,  9002,  9004,  9053,  9290, 10182, 10811, 10983, 11421,
        #     11785, 11878]
    if remove:
        for house in remove:
            df = df[df['dataid'] != house]

    if cap:
        df = floor_outliers(df, cap)
    return df


def normalize_x(x_sequence):
    temp_x = np.copy(x_sequence)
    temp_x = stats.boxcox(np.add(temp_x, 1))[0]
    std = np.std(temp_x)
    if std != 0:
        temp_x = (temp_x - np.mean(temp_x)) / std
        return temp_x
    else:
        std = np.std(x_sequence)
        return (x_sequence - np.mean(x_sequence)) / std


def normalize_y(y_sequence):
    y_sequence = np.log(np.add(y_sequence, 1))
    sequence_min = np.min(y_sequence)
    sequence_max = np.max(y_sequence)
    output = (y_sequence - sequence_min) / (sequence_max - sequence_min)
    return output, sequence_min, sequence_max


def create_activations(path, appliance, window_length, buildings):
    data = load_all_houses_with_device(path, appliance)
    data = [data.loc[data['dataid'] == i] for i in buildings]
    data = pd.concat(data)
    data_size = len(data)
    remainder = data_size % window_length
    y_data = data['appliance_power'].values
    x_data = data['net_power'].values
    #lmax = stats.boxcox_normmax(np.add(x_data, 1), brack=(-1.9, 2), method='mle')
    #x_data = stats.boxcox(np.add(x_data, 1))[0]
    x_data = normalize_x(x_data)

    if remainder != 0:
        y_data = np.append(y_data, np.zeros(window_length - remainder))
        x_data = np.append(x_data, np.zeros(window_length - remainder))
    y_data = np.reshape(y_data, (-1, window_length))
    x_data = np.reshape(x_data, (-1, window_length))

    y_data = y_data[0:-1]
    x_data = x_data[0:-1]

    #x_data = normalize_x(x_data)

    nan_tracker = []

    for i in range(len(y_data)):
        if max(y_data[i]) - min(y_data[i]) == 0:
            nan_tracker.append(i)
        #elif max(y_data[i]) - min(y_data[i]) < 30:
            #leave_out = np.random.binomial(1, 0.2)
            #if leave_out == 0:
            #nan_tracker.append(i)

    y_data = [elem for i, elem in enumerate(y_data) if i not in nan_tracker]
    x_data = [elem for i, elem in enumerate(x_data) if i not in nan_tracker]

    y_data, sequence_min, sequence_max = normalize_y(y_data)
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32), np.array(sequence_min, dtype=np.float32), np.array(sequence_max, dtype=np.float32)
