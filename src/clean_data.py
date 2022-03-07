import numpy as np
import pandas as pd


def get_home_ids(df):
    return df['dataid'].unique()


def load_all_houses_with_device(path, appliance):
    df = pd.read_csv(path, usecols=['dataid', 'grid', 'solar', 'solar2', str(appliance)])
    dataids = get_home_ids(df)
    columns = {
        'dataid': df.dataid,
        'grid': df.grid,
        'solar': df.solar,
        'solar2': df.solar2,
        str(appliance): df[str(appliance)]
    }
    df = pd.DataFrame(columns)
    df['net_power'] = df.fillna(0).grid + df.fillna(0).solar + df.fillna(0).solar2
    df = df.fillna(0)
    df['net_power'] = np.clip(df['net_power'] * 1000, a_min=0, a_max=None)
    df[str(appliance)] = np.clip(df[str(appliance)] * 1000, a_min=0, a_max=None)
    df['appliance_power'] = df[str(appliance)]
    df = df.drop(columns=['grid', 'solar', 'solar2', str(appliance)])
    df['appliance_power'] = df['appliance_power'].values.astype('float32')
    df['net_power'] = df['net_power'].values.astype('float32')

    for house in dataids:
        house_of_interest = df.loc[df['dataid'] == house]
        if max(house_of_interest['appliance_power']) - min(house_of_interest['appliance_power']) < 60:
            df = df[df['dataid'] != house]

    remove = []
    if appliance == "air1":
        df['appliance_power'][df['appliance_power'] > 5000] = 0
        remove = [1249, 7365, 8849]
    if appliance == 'clotheswasher1':
        df['appliance_power'][df['appliance_power'] > 1300] = 0
        remove = [6240]
    if appliance == 'drye1':
        remove = [3000, 3840, 3976, 4550, 6526, 9004, 9053, 183]
    if appliance == 'furnace1':
        df['appliance_power'][df['appliance_power'] > 1000] = 0
        remove = [6703, 3403, 7365, 2126, 10089]
    if appliance == 'oven1':
        df['appliance_power'][df['appliance_power'] > 5000] = 0
        remove = [792]
    if appliance == 'range1':
        df['appliance_power'][df['appliance_power'] > 8000] = 0
        remove = [7367]
    if appliance == 'dishwasher1':
        df['appliance_power'][df['appliance_power'] > 1500] = 0
        remove = [8849, 6240, 7367]
    if appliance == 'freezer1':
        remove = [9973, 7159]
    if appliance == 'heater1':
        remove = [2561]
    if appliance == 'waterheater1':
        df['appliance_power'][df['appliance_power'] > 6000] = 0
        remove = [10983, 9973, 8627, 2096]
    if appliance == 'waterheater2':
        df['appliance_power'][df['appliance_power'] > 5000] = 0
        remove = [8627]
    if appliance == 'wellpump1':
        df['appliance_power'][df['appliance_power'] > 1600] = 0
        remove = [10983]
    if appliance == 'clotheswasher_dryg1':
        df['appliance_power'][df['appliance_power'] > 800] = 0
    if appliance == 'refrigerator1':
        df['appliance_power'][df['appliance_power'] > 600] = 0
        remove = [9973, 142, 145, 1417, 2561, 2786, 5367, 6594, 9053, 984, 11878, 526, 6564, 7935]
    if remove:
        for house in remove:
            df = df[df['dataid'] != house]
    return df


def normalize_x(x_sequence):
    std = np.std(x_sequence)
    x_sequence = (x_sequence - np.mean(x_sequence)) / std
    return x_sequence


def normalize_y(y_sequence):
    y_sequence = np.log(np.add(y_sequence, 1))
    return (y_sequence - np.min(y_sequence)) / (np.max(y_sequence) - np.min(y_sequence))


def create_activations(path, appliance, window_length, buildings):
    data = load_all_houses_with_device(path, appliance)
    data = [data.loc[data['dataid'] == i] for i in buildings]
    data = pd.concat(data)
    data_size = len(data)
    remainder = data_size % window_length
    y_data = data['appliance_power'].values
    x_data = data['net_power'].values

    if remainder != 0:
        y_data = np.append(y_data, np.zeros(window_length - remainder))
        x_data = np.append(x_data, np.zeros(window_length - remainder))

    y_data = np.reshape(y_data, (-1, window_length))
    x_data = np.reshape(x_data, (-1, window_length))

    data_size = len(y_data)

    y_data = y_data[0:data_size - 1]
    x_data = x_data[0:data_size - 1]

    x_data = normalize_x(x_data)

    nan_tracker = []

    for i in range(len(y_data)):
        if max(y_data[i]) - min(y_data[i]) == 0:
            nan_tracker.append(i)
        elif max(y_data[i]) - min(y_data[i]) < 30:
            nan_tracker.append(i)

    y_data = [elem for i, elem in enumerate(y_data) if i not in nan_tracker]
    x_data = [elem for i, elem in enumerate(x_data) if i not in nan_tracker]

    y_data = normalize_y(y_data)

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)
